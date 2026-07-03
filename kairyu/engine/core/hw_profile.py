"""Hardware capability profiles (roadmap §2, design m8 D5).

Nothing in the engine hard-codes a GPU: strategy is a function of the probed
profile. ``probe()`` returns a ``cpu`` profile without CUDA and reads
``torch.cuda`` when present; measured numbers (bandwidth, P2P matrix) are
filled by the GPU-day env scripts and recorded via ``write_env_record``.
The decision logic (``best_format``, ``kernel_tier``) is pure and CPU-tested.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kairyu.engine.core.quant_config import QuantConfig, QuantMethod

# SM -> supported compute formats (roadmap §2 quantization matrix)
_FORMATS_BY_SM: tuple[tuple[int, tuple[str, ...]], ...] = (
    (120, ("bf16", "fp8", "nvfp4")),  # RTX PRO 6000 Blackwell (SM120)
    (100, ("bf16", "fp8", "nvfp4")),  # B200 (SM100)
    (89, ("bf16", "fp8")),  # Ada / Hopper FP8 floor
    (80, ("bf16", "int8")),  # A100 (SM80): no FP8/FP4 tensor cores
)

_METHOD_FORMAT = {
    QuantMethod.NONE: "bf16",
    QuantMethod.FP8: "fp8",
    QuantMethod.INT8: "int8",
    QuantMethod.AWQ: "w4a16",
    QuantMethod.GPTQ: "w4a16",
    QuantMethod.NVFP4: "nvfp4",
}


@dataclass(frozen=True)
class HardwareProfile:
    """One node's capability profile; ``arch='cpu'`` for GPU-less machines."""

    arch: str  # "cpu" | "cuda"
    device_name: str = "cpu"
    sm: int | None = None  # compute capability major*10+minor (e.g. 120)
    device_count: int = 0
    memory_gb: float = 0.0
    measured_bandwidth_gbs: float | None = None  # GPU-day measurement
    p2p_matrix: tuple[tuple[float, ...], ...] | None = None  # GPU-day measurement
    formats: tuple[str, ...] = ("bf16",)
    interconnect: str = "none"  # "nvlink" | "pcie" | "none"

    @property
    def kernel_tier(self) -> str:
        """Which attention/GEMM backend family exists for this SM (roadmap §2)."""
        if self.sm is None:
            return "torch"  # CPU / unknown: pure-torch paths only
        if self.sm in (90, 100):
            return "full"  # FA2/FA3, FlashMLA, DeepGEMM, CUTLASS
        if self.sm == 120:
            return "fa2"  # FA2-class only; Triton-first FP8; 99KB smem
        return "fa2" if self.sm >= 80 else "torch"

    def supports(self, fmt: str) -> bool:
        return fmt in self.formats or fmt in ("bf16", "w4a16")

    def best_format(self, quant: QuantConfig) -> str:
        """The loader-picks-best-format rule: the checkpoint's format if this
        profile supports it, else fail loudly (silent dequant hides perf)."""
        fmt = _METHOD_FORMAT[quant.method]
        if self.supports(fmt):
            return fmt
        raise ValueError(
            f"checkpoint format {fmt!r} is not supported on {self.device_name} "
            f"(sm={self.sm}); supported: {self.formats}"
        )


def _formats_for_sm(sm: int) -> tuple[str, ...]:
    for floor, formats in _FORMATS_BY_SM:
        if sm >= floor:
            return formats
    return ("bf16",)


def probe() -> HardwareProfile:
    """Probe this machine; CPU profile when torch/CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return HardwareProfile(arch="cpu")
    if not torch.cuda.is_available():
        return HardwareProfile(arch="cpu")
    return _probe_cuda(torch)


def _probe_cuda(torch) -> HardwareProfile:  # pragma: no cover - needs CUDA hardware
    properties = torch.cuda.get_device_properties(0)
    sm = properties.major * 10 + properties.minor
    return HardwareProfile(
        arch="cuda",
        device_name=properties.name,
        sm=sm,
        device_count=torch.cuda.device_count(),
        memory_gb=properties.total_memory / 1e9,
        formats=_formats_for_sm(sm),
        interconnect="pcie",  # NVLink detection is a GPU-day env-script concern
    )


@dataclass(frozen=True)
class EnvRecord:
    """Schema of ``bench/results/env-<date>.json`` (G2 §8 evidence rules)."""

    date: str
    profile: HardwareProfile
    driver: str | None = None
    cuda: str | None = None
    library_versions: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def write_env_record(record: EnvRecord, results_dir: str | Path) -> Path:
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"env-{record.date}.json"
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n")
    return path


def load_env_record(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
