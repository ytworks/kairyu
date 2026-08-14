"""Qwen3-32B TP gate for typed text and pre-tokenized prompts.

Run on the reviewed eight-GPU host:

    uv run python verification/l1/correctness/typed_prompt_qwen.py \
      --model-path /models/qwen3-32b \
      --output bench/results/issue-227-typed-prompt-qwen3-32b-tp8-2026-07-30.json \
      --assert-gate

The gate is deterministic and structural.  It does not use wall-clock timing:
the same externally tokenized prompt must produce the same greedy output as
the text path, report exact prompt-token usage, and reuse the native KV cache.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from kairyu.engine.backend import GenerationRequest
from kairyu.engine.kairyu_backend import KairyuBackend
from kairyu.engine.prompt import TokensPrompt
from kairyu.engine.tokenizer import resolve_tokenizer
from kairyu.sampling_params import SamplingParams

SCHEMA_VERSION = 1
TP = 8
PAGE_SIZE = 16
EXPECTED_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
EXPECTED_CHECKPOINT = {
    "revision": "9216db5781bf21249d130ec9da846c4624c16137",
    "config.json": "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb",
    "model.safetensors.index.json": (
        "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
    ),
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
SOURCE_FILES = (
    "verification/l1/correctness/typed_prompt_qwen.py",
    "kairyu/engine/backend.py",
    "kairyu/engine/engine_loop.py",
    "kairyu/engine/kairyu_backend.py",
    "kairyu/engine/prompt.py",
    "kairyu/engine/tokenizer.py",
    "kairyu/engine/core/model_runner.py",
    "kairyu/engine/core/radix_kv.py",
    "kairyu/engine/core/scheduler.py",
    "kairyu/sampling_params.py",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT = " ".join(
    [
        "Typed prompt parity must preserve every caller supplied token id.",
        "The text and token paths must produce identical greedy output.",
    ]
    * 12
)
DISPLAY_PROMPT = "Display metadata is intentionally not the execution prompt."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(model_path: Path) -> dict[str, str]:
    revision_file = (
        model_path / ".cache/huggingface/download/config.json.metadata"
    )
    revision = revision_file.read_text(encoding="utf-8").splitlines()[0]
    return {
        "revision": revision,
        **{
            name: _sha256(model_path / name)
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
            )
        },
    }


def _source_identity() -> dict[str, object]:
    try:
        git_head = subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tracked_tree_clean = (
            subprocess.run(
                ("git", "diff", "--quiet", "HEAD", "--"),
                cwd=REPO_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except (OSError, subprocess.CalledProcessError):
        git_head = None
        tracked_tree_clean = False
    return {
        "git_head": git_head,
        "tracked_tree_clean": tracked_tree_clean,
        "files": {name: _sha256(REPO_ROOT / name) for name in SOURCE_FILES},
    }


async def _measure(model_path: Path) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != TP:
        raise RuntimeError(f"typed-prompt gate requires exactly {TP} visible CUDA devices")

    checkpoint = _checkpoint_identity(model_path)
    gpu_names = [torch.cuda.get_device_name(index) for index in range(TP)]
    source = _source_identity()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen3ForCausalLM"] or config.get(
        "num_hidden_layers"
    ) != 64:
        raise RuntimeError("--model-path is not a Qwen3-32B checkpoint")

    tokenizer = resolve_tokenizer(str(model_path))
    prompt_token_ids = tokenizer.encode(PROMPT)
    if len(prompt_token_ids) < 32:
        raise RuntimeError("fixed typed-prompt fixture encoded to too few tokens")

    backend = KairyuBackend(
        model_path=str(model_path),
        tokenizer=str(model_path),
        tensor_parallel_size=TP,
        num_pages=256,
        page_size=PAGE_SIZE,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        decode_mode="eager",
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=8,
        ignore_eos=True,
        seed=0,
    )
    try:
        text_result = await backend.generate(
            GenerationRequest("typed-text", PROMPT, params)
        )
        token_result = await backend.generate(
            GenerationRequest(
                "typed-token-ids",
                TokensPrompt(prompt_token_ids, prompt=DISPLAY_PROMPT),
                params,
            )
        )
    finally:
        await backend.shutdown()

    text_output = text_result.completions[0]
    token_output = token_result.completions[0]
    text_usage = text_result.usage
    token_usage = token_result.usage
    expected_cached_tokens = (
        (len(prompt_token_ids) - 1) // PAGE_SIZE * PAGE_SIZE
    )
    checks = {
        "source_files_bound": (
            isinstance(source["git_head"], str)
            and len(source["git_head"]) in (40, 64)
            and all(character in "0123456789abcdef" for character in source["git_head"])
            and source["tracked_tree_clean"] is True
            and set(source["files"]) == set(SOURCE_FILES)
            and all(
                isinstance(digest, str) and len(digest) == 64
                for digest in source["files"].values()
            )
        ),
        "exact_pinned_checkpoint": checkpoint == EXPECTED_CHECKPOINT,
        "eight_reviewed_blackwell_gpus": gpu_names == [EXPECTED_GPU] * TP,
        "qwen3_32b_tp8": (
            config.get("hidden_size") == 5120
            and config.get("vocab_size") == 151936
        ),
        "external_token_ids_preserved": (
            token_result.prompt_token_ids == prompt_token_ids
        ),
        "display_text_is_non_authoritative": (
            DISPLAY_PROMPT != PROMPT
            and isinstance(token_result.prompt, TokensPrompt)
            and token_result.prompt.prompt == DISPLAY_PROMPT
        ),
        "text_and_token_output_ids_match": (
            text_output.token_ids == token_output.token_ids
            and len(text_output.token_ids) == params.max_tokens
        ),
        "text_and_token_output_text_match": text_output.text == token_output.text,
        "exact_prompt_usage": (
            text_usage is not None
            and token_usage is not None
            and text_usage.prompt_tokens == len(prompt_token_ids)
            and token_usage.prompt_tokens == len(prompt_token_ids)
        ),
        "native_cache_reused": (
            text_usage is not None
            and token_usage is not None
            and text_usage.cached_tokens == 0
            and token_usage.cached_tokens == expected_cached_tokens
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": "qwen3-32b-tp8-typed-prompt-parity",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "checkpoint": checkpoint,
        "hardware": {
            "gpu_names": gpu_names,
            "torch_cuda": torch.version.cuda,
        },
        "config": {
            "tensor_parallel_size": TP,
            "page_size": PAGE_SIZE,
            "model_architecture": config["architectures"][0],
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "vocab_size": config["vocab_size"],
            "prompt_tokens": len(prompt_token_ids),
            "expected_cached_tokens": expected_cached_tokens,
            "max_tokens": params.max_tokens,
            "timing_is_binding": False,
        },
        "text": {
            "output_token_ids": list(text_output.token_ids),
            "usage": asdict(text_usage) if text_usage is not None else None,
        },
        "tokens": {
            "output_token_ids": list(token_output.token_ids),
            "usage": asdict(token_usage) if token_usage is not None else None,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args()

    payload = asyncio.run(_measure(args.model_path.resolve()))
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.assert_gate and payload["passed"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
