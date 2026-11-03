"""SM120 runtime adaptation for EmbeddingGemma 2 on RTX PRO 6000 GPUs.

The edit replaces exactly one known source anchor and fails the image build
when the anchor is missing or repeated, or when vLLM is not one of the pinned
releases (``VLLM_VERSIONS``), so an upstream change is never patched silently.

Encoder prefill tiles: vLLM's Triton encoder attention
(``triton_prefill_attention.context_attention_fwd``, which runs EmbeddingGemma
2's bidirectional layers) uses 128-row tiles on every GPU of compute capability
8.0 or newer. For EmbeddingGemma 2's 256-dimensional sliding-window heads that
tile needs 160 KiB of shared memory, while an SM120 GPU offers 99 KiB per
block, so the embedding engine cannot start. Upstream already drops
512-dimensional heads to 32-row tiles; the edit drops 256-dimensional heads to
64-row tiles whenever the device's opt-in shared memory per block is below
160 KiB. Other head sizes and GPUs with more shared memory keep upstream's
tiles. Gemma 4 chat replicas use the unified Triton attention kernel, which
this edit does not touch.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

VLLM_VERSIONS = {"0.33.0", "0.32.0"}
TARGET = "v1/attention/ops/triton_prefill_attention.py"
ANCHOR = "    if Lk >= 512:\n        BLOCK = min(BLOCK, 32)\n"
REPLACEMENT = ANCHOR + (
    "    elif Lk >= 256 and torch.cuda.get_device_properties(\n"
    "        q.device\n"
    "    ).shared_memory_per_block_optin < 160 * 1024:\n"
    "        # Kairyu SM120 overlay: a 128-row tile of 256-dimensional heads\n"
    "        # needs 160 KiB of shared memory; SM120 has 99 KiB per block.\n"
    "        BLOCK = min(BLOCK, 64)\n"
)


def main() -> None:
    version = importlib.metadata.version("vllm")
    if version not in VLLM_VERSIONS:
        raise SystemExit(f"vLLM {version} is not a patched release {sorted(VLLM_VERSIONS)}")
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vLLM is not installed")
    path = Path(spec.submodule_search_locations[0]) / TARGET
    source = path.read_text(encoding="utf-8")
    if source.count(ANCHOR) != 1:
        raise SystemExit(f"{TARGET}: expected exactly one encoder tile anchor")
    path.write_text(source.replace(ANCHOR, REPLACEMENT, 1), encoding="utf-8")
    print(f"patched {TARGET} for vLLM {version}")


if __name__ == "__main__":
    main()
