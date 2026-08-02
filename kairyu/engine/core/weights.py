"""Safetensors checkpoint reader (design m8 D5; extended by M12 loader).

Reads HF-format checkpoints: ``model.safetensors.index.json`` + shards, or a
single ``model.safetensors``/``*.safetensors`` file. Deferred import of
``safetensors`` (optional ``[hf]`` extra). ``get_slice`` is the seam the M16
per-rank sharded loader uses to read tensor slices without materializing the
full tensor.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def _strict_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _import_safetensors():
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - exercised only without the dep
        raise RuntimeError(
            "checkpoint loading requires the 'safetensors' package (uv sync --extra hf)"
        ) from error
    return safe_open


class CheckpointReader:
    """Name → tensor access over a (possibly sharded) safetensors checkpoint."""

    def __init__(self, path: str | Path, framework: str = "pt") -> None:
        self._safe_open = _import_safetensors()
        self._framework = framework
        directory = Path(path)
        if directory.is_file():
            self._shards = {directory.name: directory}
            self._index = None
        elif directory.is_dir():
            self._shards, self._index = self._discover(directory)
        else:
            raise ValueError(f"no checkpoint at {path}")
        self._name_to_shard = self._build_name_map()

    @staticmethod
    def _discover(directory: Path) -> tuple[dict[str, Path], dict | None]:
        index_file = directory / "model.safetensors.index.json"
        if index_file.is_file():
            try:
                index = json.loads(
                    index_file.read_text(),
                    object_pairs_hook=_strict_json_object,
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid safetensors index at {index_file}"
                ) from error
            if not isinstance(index, dict):
                raise ValueError("safetensors index must contain a JSON object")
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    "safetensors index requires a non-empty object weight_map"
                )
            for tensor_name, shard_name in weight_map.items():
                if not isinstance(tensor_name, str) or not tensor_name:
                    raise ValueError(
                        "safetensors index tensor names must be non-empty strings"
                    )
                if (
                    not isinstance(shard_name, str)
                    or not shard_name
                    or Path(shard_name).name != shard_name
                ):
                    raise ValueError(
                        f"unsafe safetensors shard path {shard_name!r}"
                    )
            shard_names = sorted(set(weight_map.values()))
            shards = {name: directory / name for name in shard_names}
            missing = [name for name, path in shards.items() if not path.is_file()]
            if missing:
                raise ValueError(
                    "safetensors index references missing shard(s): "
                    f"{missing[:8]}"
                )
            resolved_directory = directory.resolve()
            escaped = [
                name
                for name, path in shards.items()
                if path.resolve().parent != resolved_directory
            ]
            if escaped:
                raise ValueError(
                    "safetensors shard path escapes the checkpoint directory: "
                    f"{escaped[:8]}"
                )
            return shards, index
        files = sorted(directory.glob("*.safetensors"))
        if not files:
            raise ValueError(f"no *.safetensors files in {directory}")
        return {file.name: file for file in files}, None

    def _build_name_map(self) -> dict[str, str]:
        if self._index is not None:
            mapping = dict(self._index["weight_map"])
            expected_by_shard: dict[str, set[str]] = {}
            for tensor_name, shard_name in mapping.items():
                expected_by_shard.setdefault(shard_name, set()).add(tensor_name)
            for shard_name, shard_path in self._shards.items():
                with self._safe_open(
                    shard_path,
                    framework=self._framework,
                ) as reader:
                    actual = set(reader.keys())
                expected = expected_by_shard[shard_name]
                missing = sorted(expected - actual)
                unexpected = sorted(actual - expected)
                if missing or unexpected:
                    raise ValueError(
                        f"safetensors index/header mismatch in {shard_name!r}: "
                        f"missing={missing[:8]} ({len(missing)} total), "
                        f"unexpected={unexpected[:8]} "
                        f"({len(unexpected)} total)"
                    )
            return mapping
        mapping: dict[str, str] = {}
        for shard_name, shard_path in self._shards.items():
            with self._safe_open(shard_path, framework=self._framework) as reader:
                for tensor_name in reader.keys():
                    mapping[tensor_name] = shard_name
        return mapping

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._name_to_shard))

    def shapes(self, names: Iterable[str]) -> dict[str, tuple[int, ...]]:
        """Read tensor shapes from shard headers without materializing tensors."""

        names_by_shard: dict[str, list[str]] = {}
        for name in names:
            shard_name = self._name_to_shard.get(name)
            if shard_name is None:
                raise KeyError(f"tensor {name!r} not in checkpoint")
            names_by_shard.setdefault(shard_name, []).append(name)
        shapes: dict[str, tuple[int, ...]] = {}
        for shard_name, shard_names in names_by_shard.items():
            shard_path = self._shards[shard_name]
            with self._safe_open(
                shard_path,
                framework=self._framework,
            ) as reader:
                actual = set(reader.keys())
                for name in shard_names:
                    if name not in actual:
                        raise KeyError(
                            f"tensor {name!r} not in shard {shard_name!r}"
                        )
                    shapes[name] = tuple(reader.get_slice(name).get_shape())
        return shapes

    def specs(
        self,
        names: Iterable[str],
    ) -> dict[str, tuple[tuple[int, ...], str]]:
        """Read shape and safetensors dtype from headers without tensor data."""

        names_by_shard: dict[str, list[str]] = {}
        for name in names:
            shard_name = self._name_to_shard.get(name)
            if shard_name is None:
                raise KeyError(f"tensor {name!r} not in checkpoint")
            names_by_shard.setdefault(shard_name, []).append(name)
        specs: dict[str, tuple[tuple[int, ...], str]] = {}
        for shard_name, shard_names in names_by_shard.items():
            shard_path = self._shards[shard_name]
            with self._safe_open(
                shard_path,
                framework=self._framework,
            ) as reader:
                actual = set(reader.keys())
                for name in shard_names:
                    if name not in actual:
                        raise KeyError(
                            f"tensor {name!r} not in shard {shard_name!r}"
                        )
                    view = reader.get_slice(name)
                    specs[name] = (
                        tuple(view.get_shape()),
                        view.get_dtype(),
                    )
        return specs

    def __contains__(self, name: str) -> bool:
        return name in self._name_to_shard

    def tensor(self, name: str):
        shard = self._shard_for(name)
        with self._safe_open(shard, framework=self._framework) as reader:
            return reader.get_tensor(name)

    def get_slice(self, name: str, dim: int, start: int, end: int):
        """Read a slice along ``dim`` without loading the full tensor (M16 seam)."""
        shard = self._shard_for(name)
        with self._safe_open(shard, framework=self._framework) as reader:
            view = reader.get_slice(name)
            index: list[slice] = [slice(None)] * len(view.get_shape())
            index[dim] = slice(start, end)
            return view[tuple(index)]

    def items(self) -> Iterator[tuple[str, object]]:
        """Iterate (name, tensor) shard by shard — mmap-friendly load order."""
        by_shard: dict[str, list[str]] = {}
        for name, shard in self._name_to_shard.items():
            by_shard.setdefault(shard, []).append(name)
        for shard_name in sorted(by_shard):
            with self._safe_open(self._shards[shard_name], framework=self._framework) as reader:
                for name in sorted(by_shard[shard_name]):
                    yield name, reader.get_tensor(name)

    def selected_items(self, names: Iterable[str]) -> Iterator[tuple[str, object]]:
        """Iterate a selected tensor set while opening every shard only once.

        Expert-parallel ranks own only a subset of a large checkpoint. Opening
        a safetensors shard once per tensor turns Qwen3-235B's ~146k names into
        avoidable filesystem overhead; grouping the selected names by shard
        preserves mmap-friendly loading without materializing remote experts.
        """

        by_shard: dict[str, list[str]] = {}
        for name in names:
            shard_name = self._name_to_shard.get(name)
            if shard_name is None:
                raise KeyError(f"tensor {name!r} not in checkpoint")
            by_shard.setdefault(shard_name, []).append(name)
        for shard_name in sorted(by_shard):
            with self._safe_open(
                self._shards[shard_name],
                framework=self._framework,
            ) as reader:
                actual = set(reader.keys())
                for name in sorted(by_shard[shard_name]):
                    if name not in actual:
                        raise KeyError(
                            f"tensor {name!r} not in shard {shard_name!r}"
                        )
                    yield name, reader.get_tensor(name)

    def _shard_for(self, name: str) -> Path:
        shard = self._name_to_shard.get(name)
        if shard is None:
            raise KeyError(f"tensor {name!r} not in checkpoint")
        return self._shards[shard]
