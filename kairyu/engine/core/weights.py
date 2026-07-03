"""Safetensors checkpoint reader (design m8 D5; extended by M12 loader).

Reads HF-format checkpoints: ``model.safetensors.index.json`` + shards, or a
single ``model.safetensors``/``*.safetensors`` file. Deferred import of
``safetensors`` (optional ``[hf]`` extra). ``get_slice`` is the seam the M16
per-rank sharded loader uses to read tensor slices without materializing the
full tensor.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


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
            index = json.loads(index_file.read_text())
            shard_names = sorted(set(index["weight_map"].values()))
            return {name: directory / name for name in shard_names}, index
        files = sorted(directory.glob("*.safetensors"))
        if not files:
            raise ValueError(f"no *.safetensors files in {directory}")
        return {file.name: file for file in files}, None

    def _build_name_map(self) -> dict[str, str]:
        if self._index is not None:
            return dict(self._index["weight_map"])
        mapping: dict[str, str] = {}
        for shard_name, shard_path in self._shards.items():
            with self._safe_open(shard_path, framework=self._framework) as reader:
                for tensor_name in reader.keys():
                    mapping[tensor_name] = shard_name
        return mapping

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._name_to_shard))

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

    def _shard_for(self, name: str) -> Path:
        shard = self._name_to_shard.get(name)
        if shard is None:
            raise KeyError(f"tensor {name!r} not in checkpoint")
        return self._shards[shard]
