"""Measured restore-versus-recompute policy for the DRAM KV tier (#187).

The tier is deliberately not enabled by an arbitrary token threshold.  A
deployment loads a retained benchmark profile, independently re-derives its
crossover from paired raw samples, and checks that the model/KV/runtime identity
matches the process that will use it.  A stale profile therefore fails at
startup instead of silently applying a decision measured for another layout.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import ssl
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kairyu.engine.core.kv_pool import PagedKVPool

if TYPE_CHECKING:
    from kairyu.engine.core.kv_tier import PinnedKVPageTier

SCHEMA_VERSION = "kairyu.dram-kv-crossover.v2"
PAIRED_REPEATS_PER_LENGTH = 9
REQUIRED_RESTORE_WINS = 8
POLICY_RULE = "median_lt_1_and_wins_at_least_8_of_9"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def model_config_sha256(model_path: str | Path) -> str:
    path = Path(model_path) / "config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model config for DRAM KV policy: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("model config for DRAM KV policy must be a JSON object")
    return _canonical_sha256(raw)


def cpu_model_identity() -> str:
    """Stable CPU model cohort used by host-side KV checksum work."""

    try:
        models = {
            value.strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if ":" in line
            and line.split(":", 1)[0].strip() in {"model name", "Processor"}
            and (value := line.split(":", 1)[1].strip())
        }
    except (OSError, RuntimeError) as error:
        raise ValueError("cannot read CPU model identity from /proc/cpuinfo") from error
    if not models:
        raise ValueError("CPU model identity is missing from /proc/cpuinfo")
    return ";".join(sorted(models))


def engine_source_inventory(
    package_root: str | Path | None = None,
) -> dict[str, str]:
    """Hash every installed ``kairyu/**/*.py`` by stable logical path.

    The crossover's recompute arm spans model, runner, scheduler, and attention
    implementations, so a hand-maintained module allowlist is unsafe.  Symlinks
    are rejected instead of following an installed package path outside the
    attested root.
    """

    if package_root is None:
        package = importlib.import_module("kairyu")
        raw_file = getattr(package, "__file__", None)
        if not isinstance(raw_file, str) or not raw_file:
            raise ValueError("installed kairyu package has no source root")
        root = Path(raw_file).parent
    else:
        root = Path(package_root)
    if root.is_symlink():
        raise ValueError("installed kairyu package root must not be a symlink")
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"cannot resolve installed kairyu package root: {root}") from error
    if not resolved_root.is_dir():
        raise ValueError(f"installed kairyu package root is not a directory: {root}")

    try:
        entries = sorted(resolved_root.rglob("*"), key=lambda path: path.as_posix())
    except (OSError, RuntimeError) as error:
        raise ValueError(f"cannot enumerate installed kairyu package: {root}") from error
    if any(path.is_symlink() for path in entries):
        raise ValueError("installed kairyu package contains a symlink")

    files: dict[str, str] = {}
    for path in entries:
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(resolved_root)
            payload = resolved.read_bytes()
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(
                f"installed kairyu Python source escapes its package root: {path}"
            ) from error
        logical_name = f"kairyu/{relative.as_posix()}"
        files[logical_name] = hashlib.sha256(payload).hexdigest()
    if not files:
        raise ValueError("installed kairyu package contains no Python source")
    return dict(sorted(files.items()))


def engine_source_sha256(package_root: str | Path | None = None) -> str:
    """Canonical full-package Python-source rollup for runtime policy."""

    return _canonical_sha256(engine_source_inventory(package_root))


def _pci_bus_id(properties: object) -> str:
    try:
        domain = int(properties.pci_domain_id)
        bus = int(properties.pci_bus_id)
        device = int(properties.pci_device_id)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("CUDA device does not expose a PCI identity") from error
    if domain < 0 or bus < 0 or device < 0:
        raise ValueError("CUDA device exposes an invalid PCI identity")
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"


def pcie_transport_identity(
    pci_bus_id: str,
    *,
    sysfs_root: Path = Path("/sys"),
) -> tuple[str, int, int, int]:
    """Read stable PCIe transport facts and the GPU's NUMA node from sysfs."""

    device_path = sysfs_root / "bus/pci/devices" / pci_bus_id.lower()

    def read(name: str) -> str:
        path = device_path / name
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError(f"cannot read CUDA PCIe identity field: {path}") from error
        if not value:
            raise ValueError(f"CUDA PCIe identity field is empty: {path}")
        return value

    max_speed = read("max_link_speed")
    try:
        max_width = int(read("max_link_width"))
        current_width = int(read("current_link_width"))
        numa_node = int(read("numa_node"))
    except ValueError as error:
        raise ValueError(f"CUDA PCIe identity is malformed for {pci_bus_id}") from error
    if max_width < 1 or current_width < 1 or numa_node < 0:
        raise ValueError(f"CUDA PCIe identity is invalid for {pci_bus_id}")
    return max_speed, max_width, current_width, numa_node


@dataclass(frozen=True)
class DramKVRuntimeIdentity:
    model_config_sha256: str
    engine_source_sha256: str
    tensor_parallel_size: int
    attention_backend_identity: str
    transfer_backend_identity: str
    max_num_batched_tokens: int
    host_memory_placement_policy: str
    os_page_size: int
    pcie_max_link_speed: str | None
    pcie_max_link_width: int | None
    pcie_current_link_width: int | None
    local_numa_gpu_count: int | None
    cpu_model: str
    python_version: str
    openssl_version: str
    page_size: int
    num_layers: int
    num_kv_heads_per_rank: int
    head_dim: int
    v_head_dim: int
    kv_dtype: str
    device_type: str
    accelerator_name: str
    compute_capability: tuple[int, int] | None
    torch_version: str
    cuda_version: str | None

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(asdict(self))

    @classmethod
    def from_pool(
        cls,
        pool: PagedKVPool,
        *,
        model_config_digest: str,
        tensor_parallel_size: int,
        tensor_parallel_rank: int,
        attention_backend_identity: str,
        max_num_batched_tokens: int,
    ) -> DramKVRuntimeIdentity:
        import torch

        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if (
            type(tensor_parallel_rank) is not int
            or not 0 <= tensor_parallel_rank < tensor_parallel_size
        ):
            raise ValueError("tensor_parallel_rank must be inside its TP degree")
        if (
            not isinstance(attention_backend_identity, str)
            or not attention_backend_identity
        ):
            raise ValueError("attention_backend_identity must be a non-empty string")
        from kairyu.engine.core.attention_selector import (
            validate_attention_backend_execution_identity,
        )

        validate_attention_backend_execution_identity(attention_backend_identity)
        if type(max_num_batched_tokens) is not int or max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be a positive integer")
        device = pool.k.device
        from kairyu.engine.core.kv_tier import resolve_transfer_backend_identity

        transfer_backend_identity = resolve_transfer_backend_identity(device)
        if device.type == "cuda":
            from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY

            index = (
                torch.cuda.current_device()
                if device.index is None
                else device.index
            )
            properties = torch.cuda.get_device_properties(index)
            accelerator_name = properties.name
            capability: tuple[int, int] | None = (
                properties.major,
                properties.minor,
            )
            host_memory_placement_policy = GPU_LOCAL_FIRST_TOUCH_POLICY
            pci_bus_id = _pci_bus_id(properties)
            (
                pcie_max_link_speed,
                pcie_max_link_width,
                pcie_current_link_width,
                local_numa_node,
            ) = pcie_transport_identity(pci_bus_id)
            base_index = index - tensor_parallel_rank
            if base_index < 0:
                raise ValueError(
                    "CUDA device index and TP rank do not describe a contiguous TP group"
                )
            cohort_numa_nodes = tuple(
                pcie_transport_identity(
                    _pci_bus_id(torch.cuda.get_device_properties(base_index + rank))
                )[3]
                for rank in range(tensor_parallel_size)
            )
            local_numa_gpu_count = cohort_numa_nodes.count(local_numa_node)
        else:
            accelerator_name = "cpu"
            capability = None
            host_memory_placement_policy = "not-applicable"
            pcie_max_link_speed = None
            pcie_max_link_width = None
            pcie_current_link_width = None
            local_numa_gpu_count = None
        return cls(
            model_config_sha256=model_config_digest,
            engine_source_sha256=engine_source_sha256(),
            tensor_parallel_size=tensor_parallel_size,
            attention_backend_identity=attention_backend_identity,
            transfer_backend_identity=transfer_backend_identity,
            max_num_batched_tokens=max_num_batched_tokens,
            host_memory_placement_policy=host_memory_placement_policy,
            os_page_size=os.sysconf("SC_PAGE_SIZE"),
            pcie_max_link_speed=pcie_max_link_speed,
            pcie_max_link_width=pcie_max_link_width,
            pcie_current_link_width=pcie_current_link_width,
            local_numa_gpu_count=local_numa_gpu_count,
            cpu_model=cpu_model_identity(),
            python_version=platform.python_version(),
            openssl_version=ssl.OPENSSL_VERSION,
            page_size=pool.page_size,
            num_layers=pool.num_layers,
            num_kv_heads_per_rank=pool.num_kv_heads,
            head_dim=pool.head_dim,
            v_head_dim=pool.v_head_dim,
            kv_dtype=str(pool.k.dtype),
            device_type=device.type,
            accelerator_name=accelerator_name,
            compute_capability=capability,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )

    @classmethod
    def from_json(cls, raw: object) -> DramKVRuntimeIdentity:
        if not isinstance(raw, dict):
            raise ValueError("DRAM KV runtime_identity must be an object")
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(raw) != expected:
            raise ValueError("DRAM KV runtime_identity fields are incomplete or unknown")
        values = dict(raw)
        capability = values["compute_capability"]
        if capability is not None:
            if (
                not isinstance(capability, list)
                or len(capability) != 2
                or any(type(item) is not int or item < 0 for item in capability)
            ):
                raise ValueError("compute_capability must be [major, minor] or null")
            values["compute_capability"] = tuple(capability)
        identity = cls(**values)
        from kairyu.engine.core.attention_selector import (
            validate_attention_backend_execution_identity,
        )

        validate_attention_backend_execution_identity(
            identity.attention_backend_identity
        )
        if (
            len(identity.model_config_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in identity.model_config_sha256
            )
        ):
            raise ValueError("runtime model_config_sha256 is invalid")
        if (
            len(identity.engine_source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in identity.engine_source_sha256
            )
        ):
            raise ValueError("runtime engine_source_sha256 is invalid")
        for name in (
            "tensor_parallel_size",
            "max_num_batched_tokens",
            "os_page_size",
            "page_size",
            "num_layers",
            "num_kv_heads_per_rank",
            "head_dim",
        ):
            value = getattr(identity, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"runtime {name} must be a positive integer")
        if type(identity.v_head_dim) is not int or identity.v_head_dim < 0:
            raise ValueError("runtime v_head_dim must be a non-negative integer")
        for name in (
            "attention_backend_identity",
            "transfer_backend_identity",
            "host_memory_placement_policy",
            "cpu_model",
            "python_version",
            "openssl_version",
            "kv_dtype",
            "device_type",
            "accelerator_name",
            "torch_version",
        ):
            if not isinstance(getattr(identity, name), str) or not getattr(
                identity, name
            ):
                raise ValueError(f"runtime {name} must be a non-empty string")
        if identity.cuda_version is not None and (
            not isinstance(identity.cuda_version, str) or not identity.cuda_version
        ):
            raise ValueError("runtime cuda_version must be a non-empty string or null")
        transport_values = (
            identity.pcie_max_link_width,
            identity.pcie_current_link_width,
            identity.local_numa_gpu_count,
        )
        if identity.device_type == "cuda":
            from kairyu.engine.core.kv_tier import (
                CUDA_CONTIGUOUS_BACKEND,
            )
            from kairyu.engine.core.numa import GPU_LOCAL_FIRST_TOUCH_POLICY

            if identity.host_memory_placement_policy != GPU_LOCAL_FIRST_TOUCH_POLICY:
                raise ValueError("runtime CUDA host-memory placement policy is invalid")
            if identity.transfer_backend_identity != CUDA_CONTIGUOUS_BACKEND:
                raise ValueError("runtime CUDA transfer backend identity is invalid")
            if (
                not isinstance(identity.pcie_max_link_speed, str)
                or not identity.pcie_max_link_speed
                or any(type(value) is not int or value < 1 for value in transport_values)
            ):
                raise ValueError("runtime CUDA PCIe transport identity is invalid")
        elif (
            identity.host_memory_placement_policy != "not-applicable"
            or identity.transfer_backend_identity != "torch-cpu-page-copy-v1"
            or identity.pcie_max_link_speed is not None
            or any(value is not None for value in transport_values)
        ):
            raise ValueError("runtime non-CUDA transport identity must be not-applicable")
        return identity


@dataclass(frozen=True)
class DramKVCrossoverPolicy:
    min_restore_tokens: int
    runtime_identity: DramKVRuntimeIdentity
    profile_sha256: str
    median_ratios: tuple[tuple[int, float], ...]
    wins: tuple[tuple[int, int], ...]
    evidence_sha256: str


@dataclass(frozen=True)
class DramKVTierBinding:
    """One validated runtime policy bound to one physical rank-local tier."""

    tier: PinnedKVPageTier
    policy: DramKVCrossoverPolicy

    @property
    def handshake_identity(self) -> str:
        """Rank-invariant identity used by TP startup agreement."""

        return _canonical_sha256(
            {
                "runtime_fingerprint": self.policy.runtime_identity.fingerprint,
                "profile_sha256": self.policy.profile_sha256,
                "min_restore_tokens": self.policy.min_restore_tokens,
                "capacity_pages": self.tier.capacity_pages,
            }
        )


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _derive_crossover(
    samples: object,
    *,
    page_size: int,
) -> tuple[
    int,
    tuple[tuple[int, float], ...],
    tuple[tuple[int, int], ...],
]:
    if not isinstance(samples, list) or not samples:
        raise ValueError("DRAM KV profile samples must be a non-empty list")
    by_length: dict[int, list[float]] = {}
    orders: dict[int, set[tuple[str, str]]] = {}
    repeats: dict[int, set[int]] = {}
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("DRAM KV sample rows must be objects")
        required = {
            "prefix_tokens",
            "repeat",
            "order",
            "restore_ns",
            "recompute_ns",
            "restore_sha256",
            "recompute_sha256",
        }
        if set(row) != required:
            raise ValueError("DRAM KV sample fields are incomplete or unknown")
        prefix_tokens = row["prefix_tokens"]
        repeat = row["repeat"]
        order = row["order"]
        restore_ns = row["restore_ns"]
        recompute_ns = row["recompute_ns"]
        if (
            type(prefix_tokens) is not int
            or prefix_tokens < page_size
            or prefix_tokens % page_size
        ):
            raise ValueError("sample prefix_tokens must be positive and page-aligned")
        if type(repeat) is not int or repeat < 0:
            raise ValueError("sample repeat must be a non-negative integer")
        if (
            not isinstance(order, list)
            or tuple(order)
            not in {("restore", "recompute"), ("recompute", "restore")}
        ):
            raise ValueError("sample order must contain each arm exactly once")
        if (
            type(restore_ns) is not int
            or restore_ns <= 0
            or type(recompute_ns) is not int
            or recompute_ns <= 0
        ):
            raise ValueError(
                "sample controller wall durations must be positive integer nanoseconds"
            )
        if (
            not _valid_digest(row["restore_sha256"])
            or row["restore_sha256"] != row["recompute_sha256"]
        ):
            raise ValueError("sample restore/recompute KV digests must match")
        if repeat in repeats.setdefault(prefix_tokens, set()):
            raise ValueError("sample repeat is duplicated within a prefix length")
        repeats[prefix_tokens].add(repeat)
        orders.setdefault(prefix_tokens, set()).add(tuple(order))
        by_length.setdefault(prefix_tokens, []).append(restore_ns / recompute_ns)

    ratios: list[tuple[int, float]] = []
    wins: list[tuple[int, int]] = []
    for prefix_tokens in sorted(by_length):
        if len(by_length[prefix_tokens]) != PAIRED_REPEATS_PER_LENGTH:
            raise ValueError(
                f"prefix {prefix_tokens} must have exactly "
                f"{PAIRED_REPEATS_PER_LENGTH} paired repeats"
            )
        if orders[prefix_tokens] != {
            ("restore", "recompute"),
            ("recompute", "restore"),
        }:
            raise ValueError(
                f"prefix {prefix_tokens} does not contain both paired arm orders"
            )
        ratios.append(
            (
                prefix_tokens,
                statistics.median(by_length[prefix_tokens]),
            )
        )
        wins.append(
            (
                prefix_tokens,
                sum(ratio < 1.0 for ratio in by_length[prefix_tokens]),
            )
        )
    wins_by_length = dict(wins)
    crossover = next(
        (
            prefix_tokens
            for index, (prefix_tokens, _ratio) in enumerate(ratios)
            if all(
                later_ratio < 1.0
                and wins_by_length[later_tokens] >= REQUIRED_RESTORE_WINS
                for later_tokens, later_ratio in ratios[index:]
            )
        ),
        None,
    )
    if crossover is None:
        raise ValueError(
            "DRAM restore has no stable measured crossover: each cell at and "
            "above it needs median ratio < 1 and at least 8/9 paired wins"
        )
    return crossover, tuple(ratios), tuple(wins)


def load_crossover_policy(
    path: str | Path,
    *,
    expected_identity: DramKVRuntimeIdentity,
) -> DramKVCrossoverPolicy:
    profile_path = Path(path)
    try:
        payload_bytes = profile_path.read_bytes()
        raw = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read DRAM KV crossover profile: {profile_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("DRAM KV crossover profile must be a JSON object")
    required = {
        "schema_version",
        "runtime_identity",
        "runtime_fingerprint",
        "samples",
        "policy",
        "evidence_sha256",
        "passed",
    }
    if set(raw) != required:
        raise ValueError("DRAM KV crossover profile fields are incomplete or unknown")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported DRAM KV crossover schema {raw['schema_version']!r}")
    identity = DramKVRuntimeIdentity.from_json(raw["runtime_identity"])
    if raw["runtime_fingerprint"] != identity.fingerprint:
        raise ValueError("DRAM KV profile runtime fingerprint does not match its identity")
    if identity != expected_identity:
        raise ValueError(
            "DRAM KV crossover profile does not match the current model/KV/runtime identity"
        )
    if raw["passed"] is not True:
        raise ValueError("DRAM KV crossover profile is not a passing measurement")
    if not _valid_digest(raw["evidence_sha256"]):
        raise ValueError("DRAM KV crossover profile evidence_sha256 is invalid")
    policy = raw["policy"]
    if not isinstance(policy, dict) or set(policy) != {
        "rule",
        "min_restore_tokens",
    }:
        raise ValueError("DRAM KV crossover policy fields are invalid")
    if policy["rule"] != POLICY_RULE:
        raise ValueError("DRAM KV crossover profile changed the binding decision rule")
    crossover, ratios, wins = _derive_crossover(
        raw["samples"],
        page_size=identity.page_size,
    )
    if policy["min_restore_tokens"] != crossover:
        raise ValueError("DRAM KV policy threshold differs from the raw-sample crossover")
    return DramKVCrossoverPolicy(
        min_restore_tokens=crossover,
        runtime_identity=identity,
        profile_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        median_ratios=ratios,
        wins=wins,
        evidence_sha256=raw["evidence_sha256"],
    )


def build_dram_kv_tier(
    pool: PagedKVPool,
    *,
    model_path: str | Path,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    capacity_pages: int,
    profile_path: str | Path,
    attention_backend_identity: str,
    max_num_batched_tokens: int,
) -> DramKVTierBinding:
    """Fail-closed startup binding for the measured DRAM policy.

    The profile is checked against the actual rank-local KV layout and runtime
    before the pinned slab is allocated.  A stale model, TP degree, dtype,
    accelerator, or software runtime therefore cannot silently enable a policy
    measured for different bytes.
    """

    from kairyu.engine.core.kv_tier import PinnedKVPageTier

    if type(capacity_pages) is not int or capacity_pages < 1:
        raise ValueError("DRAM KV tier capacity_pages must be a positive integer")
    if (
        type(tensor_parallel_size) is not int
        or tensor_parallel_size < 1
        or type(tensor_parallel_rank) is not int
        or not 0 <= tensor_parallel_rank < tensor_parallel_size
    ):
        raise ValueError("DRAM KV tier TP rank must be inside its positive TP degree")
    config_digest = model_config_sha256(model_path)
    identity = DramKVRuntimeIdentity.from_pool(
        pool,
        model_config_digest=config_digest,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        attention_backend_identity=attention_backend_identity,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    policy = load_crossover_policy(
        profile_path,
        expected_identity=identity,
    )
    minimum_capacity_pages = (
        policy.min_restore_tokens + pool.page_size - 1
    ) // pool.page_size
    if capacity_pages < minimum_capacity_pages:
        raise ValueError(
            "DRAM KV tier capacity_pages must be at least "
            f"{minimum_capacity_pages} to hold the measured restore crossover "
            f"of {policy.min_restore_tokens} tokens"
        )
    tier = PinnedKVPageTier(
        pool,
        model_id=config_digest,
        tp_size=tensor_parallel_size,
        tp_rank=tensor_parallel_rank,
        capacity_pages=capacity_pages,
    )
    if (
        tier.transfer_backend != identity.transfer_backend_identity
    ):
        raise RuntimeError(
            "DRAM KV transfer backend changed between profile validation and "
            "tier construction"
        )
    return DramKVTierBinding(tier=tier, policy=policy)


__all__ = [
    "DramKVCrossoverPolicy",
    "DramKVTierBinding",
    "DramKVRuntimeIdentity",
    "PAIRED_REPEATS_PER_LENGTH",
    "POLICY_RULE",
    "REQUIRED_RESTORE_WINS",
    "SCHEMA_VERSION",
    "build_dram_kv_tier",
    "cpu_model_identity",
    "engine_source_sha256",
    "engine_source_inventory",
    "load_crossover_policy",
    "model_config_sha256",
    "pcie_transport_identity",
]
