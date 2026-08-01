#!/usr/bin/env python3
"""Real Qwen3-32B gate for dense versus packed-FP8 EAGLE-3 draft heads.

The gate uses one fixed target trace per prompt.  Every arm receives identical
target embeddings, three-layer auxiliary hidden states, target KV contents, and
verification geometry.  Draft proposals are greedily verified by the real
target model, so acceptance and committed-token goodput remain output-correct
even when the draft changes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
PUBLIC_DRAFT_REPOSITORY = "thoughtworks/Qwen3-32B-Eagle3"
PUBLIC_DRAFT_REVISION = "ba10360e72cc5208048695e713c2d45781921013"
PUBLIC_DRAFT_WEIGHTS_SHA256 = "65fd3a6ad0f78f82e44e948e61096c914159912c31948bbfd90a73af5c973562"
PUBLIC_DRAFT_CONFIG_SHA256 = "47c995bf777ed9cc9d66cd96f688d726e6e4d3965d807f4023b6d06894e23996"
TARGET_REPOSITORY = "Qwen/Qwen3-32B"
TARGET_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
TARGET_CONFIG_SHA256 = "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"
TARGET_INDEX_SHA256 = "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"
TARGET_WEIGHTS_ROLLUP_SHA256 = "6aa37b7da4e37d45b277d0bca47b00c2bfb58bb60986ba93e4c3e667df123955"
SOURCE_PATHS = (
    "bench/draft_quant_qwen.py",
    "kairyu/models/draft_quant.py",
    "kairyu/models/eagle.py",
    "kairyu/models/llama.py",
)

PROMPTS = (
    "Explain why speculative decoding can improve language-model throughput.",
    "Write a concise checklist for reviewing a concurrent Python service.",
    "A farmer has seventeen sheep and all but nine leave. How many remain?",
    "Describe the tradeoffs between tensor parallelism and data parallelism.",
    "Give three practical ways to make a GPU benchmark reproducible.",
    "Summarize how a paged key-value cache avoids external fragmentation.",
    "What should an inference server do when checkpoint metadata is unsupported?",
    "Compare dynamic per-token FP8 activation scaling with static scaling.",
)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    ignored_layers: tuple[str, ...] | None

    @property
    def quantized(self) -> bool:
        return self.ignored_layers is not None


ARMS = (
    ArmSpec("dense", None),
    ArmSpec("fp8_all", ()),
    ArmSpec("fp8_dense_fc", ("fc",)),
    ArmSpec("fp8_dense_lm_head", ("lm_head",)),
    ArmSpec("fp8_dense_fc_head", ("fc", "lm_head")),
)


@dataclass
class PromptTrace:
    prompt_index: int
    prompt_tokens: int
    sequence: tuple[int, ...]
    aux_hidden: Any
    pool: Any
    page_table: list[int]
    anchors: tuple[int, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def checkpoint_snapshot(model_path: Path) -> dict[str, object]:
    """Full-hash the exact pinned target checkpoint used by the gate."""

    weight_paths = sorted(model_path.glob("model-*.safetensors"))
    if len(weight_paths) != 17:
        raise RuntimeError(
            f"pinned Qwen3-32B target requires 17 weight shards, got {len(weight_paths)}"
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        weight_hashes = list(executor.map(sha256_file, weight_paths))
    weights = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(weight_paths, weight_hashes, strict=True)
    ]
    weights_rollup = hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()
    revision_path = model_path / ".cache/huggingface/download/config.json.metadata"
    revision_lines = (
        revision_path.read_text(encoding="utf-8").splitlines() if revision_path.is_file() else []
    )
    snapshot = {
        "repository": TARGET_REPOSITORY,
        "revision": revision_lines[0].strip() if revision_lines else None,
        "config_sha256": sha256_file(model_path / "config.json"),
        "index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
        "weights": weights,
        "weights_rollup_sha256": weights_rollup,
    }
    expected = {
        "revision": TARGET_REVISION,
        "config_sha256": TARGET_CONFIG_SHA256,
        "index_sha256": TARGET_INDEX_SHA256,
        "weights_rollup_sha256": TARGET_WEIGHTS_ROLLUP_SHA256,
    }
    mismatches = {
        key: {"expected": value, "actual": snapshot[key]}
        for key, value in expected.items()
        if snapshot[key] != value
    }
    if mismatches:
        raise RuntimeError(f"pinned Qwen3-32B target identity mismatch: {mismatches}")
    return snapshot


def _checkout_head() -> str:
    git_dir = Path(".git")
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise RuntimeError("unsupported .git file")
        git_dir = Path(marker.removeprefix("gitdir: "))
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    raise RuntimeError(f"cannot resolve checkout HEAD reference {reference!r}")


def source_snapshot(expected_commit: str | None) -> dict[str, object]:
    """Bind the relevant implementation while ignoring unrelated untracked files."""

    files = [{"path": path, "sha256": sha256_file(path)} for path in SOURCE_PATHS]
    head = _checkout_head()
    try:
        tracked_status = _git_output(
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--",
            *SOURCE_PATHS,
        )
    except FileNotFoundError:
        tracked_status = None
    return {
        "git_head": head,
        "expected_git_head": expected_commit,
        "commit_matches": expected_commit is not None and head == expected_commit,
        "tracked_status_porcelain": tracked_status,
        "tracked_clean_when_git_available": (
            tracked_status == "" if tracked_status is not None else None
        ),
        "files": files,
        "files_sha256": sha256_json(files),
    }


def fp8_config(ignored_layers: tuple[str, ...]) -> dict[str, object]:
    """Validated dynamic per-channel/per-token compressed-tensors dialect."""

    return {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "type": "float",
                    "num_bits": 8,
                    "strategy": "channel",
                    "symmetric": True,
                },
                "input_activations": {
                    "type": "float",
                    "num_bits": 8,
                    "dynamic": True,
                    "strategy": "token",
                    "symmetric": True,
                },
            }
        },
        "ignore": list(ignored_layers),
    }


def accepted_prefix(proposals: list[int], target_tokens: list[int]) -> int:
    if len(proposals) != len(target_tokens):
        raise ValueError("proposal and target verification widths differ")
    for index, (proposal, target) in enumerate(zip(proposals, target_tokens, strict=True)):
        if proposal != target:
            return index
    return len(proposals)


def eagle_shifted_context(sequence: tuple[int, ...], anchor: int) -> tuple[int, ...]:
    """Pair aux[t] with token[t+1], ending in the target-produced root."""

    if not 0 < anchor < len(sequence):
        raise ValueError("EAGLE anchor must have both context and a target root")
    shifted = sequence[1 : anchor + 1]
    if len(shifted) != anchor:
        raise AssertionError("EAGLE shifted context width drifted")
    return shifted


def eagle_verification_geometry(
    sequence: tuple[int, ...],
    anchor: int,
    proposals: list[int],
) -> tuple[list[int], tuple[int, ...]]:
    """Return the root-prefixed target input and its absolute positions."""

    if not 0 <= anchor < len(sequence):
        raise ValueError("EAGLE verification anchor falls outside the teacher sequence")
    tokens = [sequence[anchor], *proposals]
    return tokens, tuple(range(anchor, anchor + len(tokens)))


def score_eagle_verification(
    sequence: tuple[int, ...],
    anchor: int,
    proposals: list[int],
    target_tokens: list[int],
    bonus_token: int,
) -> tuple[int, int, bool]:
    """Score proposals and validate rejection/all-accepted correction semantics."""

    if not 0 <= anchor < len(sequence):
        raise ValueError("EAGLE verification anchor falls outside the teacher sequence")
    accepted = accepted_prefix(proposals, target_tokens)
    correction = target_tokens[accepted] if accepted < len(proposals) else bonus_token
    teacher_index = anchor + 1 + accepted
    if teacher_index >= len(sequence):
        raise ValueError("teacher sequence does not contain the EAGLE correction token")
    teacher_prefix = list(sequence[anchor + 1 : teacher_index])
    valid = (
        proposals[:accepted] == teacher_prefix
        and correction == sequence[teacher_index]
    )
    return accepted, correction, valid


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def summarize_arm(
    name: str,
    *,
    module_bytes: int,
    cuda_bytes: int,
    forward_peak_extra_bytes: int,
    draft_logits_finite: bool,
    draft_tokens: int,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    draft_ms = [float(row["draft_ms"]) for row in rows]
    verify_ms = [float(row["verify_ms"]) for row in rows]
    cycle_ms = [float(row["cycle_ms"]) for row in rows]
    accepted = sum(int(row["accepted"]) for row in rows)
    proposed = sum(int(row["proposed"]) for row in rows)
    committed = sum(int(row["committed"]) for row in rows)
    total_seconds = sum(cycle_ms) / 1000.0
    return {
        "arm": name,
        "samples": len(rows),
        "module_bytes": module_bytes,
        "cuda_bytes": cuda_bytes,
        "forward_peak_extra_bytes": forward_peak_extra_bytes,
        "draft_latency_ms": {
            "median": statistics.median(draft_ms),
            "p95": _percentile(draft_ms, 0.95),
        },
        "verify_latency_ms": {
            "median": statistics.median(verify_ms),
            "p95": _percentile(verify_ms, 0.95),
        },
        "cycle_latency_ms": {
            "median": statistics.median(cycle_ms),
            "p95": _percentile(cycle_ms, 0.95),
        },
        "accepted": accepted,
        "proposed": proposed,
        "acceptance_rate": accepted / proposed,
        "acceptance_by_position": [
            sum(int(row["accepted"]) > position for row in rows) / len(rows)
            for position in range(draft_tokens)
        ],
        "mean_accepted_per_cycle": accepted / len(rows),
        "committed": committed,
        "cycle_committed_tokens_per_second": committed / total_seconds,
        "draft_logits_finite": draft_logits_finite,
        "target_logits_finite": all(bool(row["target_logits_finite"]) for row in rows),
        "target_corrections_valid": all(bool(row["target_correction_valid"]) for row in rows),
    }


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _module_bytes(module) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in module.state_dict().values())


def _new_pool(config, *, total_tokens: int, page_size: int, device, dtype):
    from kairyu.engine.core.kv_pool import PagedKVPool

    num_pages = -(-total_tokens // page_size) + 1
    return PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=config.kv_cache_num_heads,
        head_dim=config.kv_cache_head_dim,
        v_head_dim=config.kv_cache_v_head_dim,
        dtype=dtype,
        device=device,
    )


def _clone_pool(source):
    from kairyu.engine.core.kv_pool import PagedKVPool

    clone = PagedKVPool(
        num_layers=source.num_layers,
        num_pages=source.num_pages,
        page_size=source.page_size,
        num_kv_heads=source.num_kv_heads,
        head_dim=source.head_dim,
        v_head_dim=source.v_head_dim,
        dtype=source.k.dtype,
        device=source.k.device,
    )
    clone.k.copy_(source.k)
    clone.v.copy_(source.v)
    return clone


def _build_trace(
    target,
    config,
    *,
    prompt_index: int,
    prompt_ids: tuple[int, ...],
    continuation_tokens: int,
    draft_tokens: int,
    aux_layer_ids: tuple[int, ...],
    page_size: int,
    device,
    dtype,
) -> PromptTrace:
    import torch

    # EAGLE consumes one target-produced root token before its first proposal.
    # Two extra teacher tokens make that root and the all-accepted bonus
    # independently checkable instead of declaring either valid by construction.
    teacher_tokens = continuation_tokens + 2
    total_tokens = len(prompt_ids) + teacher_tokens + draft_tokens
    pool = _new_pool(
        config,
        total_tokens=total_tokens,
        page_size=page_size,
        device=device,
        dtype=dtype,
    )
    page_table = list(range(pool.num_pages))
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    hidden, aux_hidden = target.forward_tokens_with_aux(
        prompt,
        torch.arange(prompt.numel(), device=device),
        pool,
        page_table,
        seq_len=prompt.numel(),
        aux_layer_ids=aux_layer_ids,
    )
    sequence = list(prompt_ids)
    for _ in range(teacher_tokens):
        next_token = int(torch.argmax(target.logits(hidden[-1])).item())
        position = len(sequence)
        sequence.append(next_token)
        hidden, aux_row = target.forward_tokens_with_aux(
            torch.tensor([next_token], dtype=torch.long, device=device),
            torch.tensor([position], dtype=torch.long, device=device),
            pool,
            page_table,
            seq_len=position + 1,
            aux_layer_ids=aux_layer_ids,
            write_from=position,
        )
        aux_hidden = torch.cat([aux_hidden, aux_row], dim=0)
    first_anchor = len(prompt_ids)
    anchors = tuple(range(first_anchor, first_anchor + continuation_tokens - draft_tokens + 1))
    if not anchors:
        raise ValueError("continuation_tokens must be at least draft_tokens")
    return PromptTrace(
        prompt_index=prompt_index,
        prompt_tokens=len(prompt_ids),
        sequence=tuple(sequence),
        aux_hidden=aux_hidden,
        pool=pool,
        page_table=page_table,
        anchors=anchors,
    )


def _evaluate_trace(
    head,
    target,
    trace: PromptTrace,
    *,
    draft_tokens: int,
    aux_layer_ids: tuple[int, ...],
    check_draft_finite: bool = False,
    device,
):
    import torch

    rows: list[dict[str, object]] = []
    verify_pool = _clone_pool(trace.pool)
    embedding_weight = target.model.embed_tokens.weight

    # Descending anchors preserve the teacher-forced KV prefix needed by the
    # next measurement even though each verification overwrites its suffix.
    for anchor in reversed(trace.anchors):
        torch.cuda.synchronize(device)
        cycle_started = time.perf_counter()
        root = trace.sequence[anchor]
        # EAGLE3 training and serving pair aux[t] with embedding(token[t+1]).
        # Drop the first context token and append the target-produced root.
        shifted_ids = torch.tensor(
            eagle_shifted_context(trace.sequence, anchor),
            dtype=torch.long,
            device=device,
        )
        embeds = target.model.embed_tokens(shifted_ids)
        aux_hidden = trace.aux_hidden[:anchor]

        torch.cuda.synchronize(device)
        started = time.perf_counter()
        fused = head.fuse(aux_hidden)
        finite_flags: list[torch.Tensor] = []
        hook = None
        if check_draft_finite:
            hook = head.lm_head.register_forward_hook(
                lambda _module, _inputs, output, _flags=finite_flags: _flags.append(
                    torch.isfinite(output).all()
                )
            )
        try:
            proposals = head.rollout(
                embeds,
                fused,
                lambda token: embedding_weight[token],
                k=draft_tokens,
            )
        finally:
            if hook is not None:
                hook.remove()
        torch.cuda.synchronize(device)
        draft_seconds = time.perf_counter() - started
        draft_logits_finite = (
            len(finite_flags) == draft_tokens and all(bool(flag.item()) for flag in finite_flags)
            if check_draft_finite
            else None
        )

        verification_tokens, verification_positions = eagle_verification_geometry(
            trace.sequence,
            anchor,
            proposals,
        )
        verification_tensor = torch.tensor(
            verification_tokens,
            dtype=torch.long,
            device=device,
        )
        positions = torch.tensor(
            verification_positions,
            dtype=torch.long,
            device=device,
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        verify_hidden, _verify_aux = target.forward_tokens_with_aux(
            verification_tensor,
            positions,
            verify_pool,
            trace.page_table,
            seq_len=anchor + draft_tokens + 1,
            aux_layer_ids=aux_layer_ids,
            write_from=anchor,
        )
        verify_logits = target.logits(verify_hidden)
        torch.cuda.synchronize(device)
        verify_seconds = time.perf_counter() - started

        target_tokens = [int(token) for token in torch.argmax(verify_logits[:-1], dim=-1).tolist()]
        accepted, correction, target_correction_valid = score_eagle_verification(
            trace.sequence,
            anchor,
            proposals,
            target_tokens,
            int(torch.argmax(verify_logits[-1]).item()),
        )
        target_logits_finite = bool(torch.isfinite(verify_logits).all())
        torch.cuda.synchronize(device)
        cycle_seconds = time.perf_counter() - cycle_started
        rows.append(
            {
                "prompt_index": trace.prompt_index,
                "anchor": anchor,
                "context_tokens": anchor,
                "target_root": root,
                "proposed": draft_tokens,
                "accepted": accepted,
                "committed": accepted + 1,
                "draft_ms": draft_seconds * 1000.0,
                "verify_ms": verify_seconds * 1000.0,
                "cycle_ms": cycle_seconds * 1000.0,
                "proposals": proposals,
                "verification_tokens": verification_tokens,
                "target_tokens": target_tokens,
                "correction": correction,
                "draft_logits_finite": draft_logits_finite,
                "target_logits_finite": target_logits_finite,
                "target_correction_valid": target_correction_valid,
            }
        )
    return rows


def _prepare_quant_checkpoint(
    source: Path,
    output: Path,
    *,
    dense_head,
    eagle_config,
    ignored_layers: tuple[str, ...],
) -> dict[str, str]:
    import torch
    from safetensors.torch import save_file

    from kairyu.models.draft_quant import (
        draft_linear_factory,
        pack_dynamic_fp8_draft_state,
        parse_draft_quantization,
    )
    from kairyu.models.eagle import EagleDraftHead
    from kairyu.quant.linear import ModelScope

    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    raw["quantization_config"] = fp8_config(ignored_layers)
    metadata = parse_draft_quantization(
        raw,
        expected_scope=ModelScope.EAGLE_DRAFT,
    )
    quantized = EagleDraftHead(
        eagle_config,
        linear_factory=draft_linear_factory(
            metadata,
            dtype=torch.bfloat16,
        ),
    ).eval()
    packed = pack_dynamic_fp8_draft_state(dense_head.state_dict(), quantized)
    output.mkdir(parents=True)
    config_path = output / "config.json"
    weights_path = output / "model.safetensors"
    config_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_file(packed, weights_path)
    return {
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
    }


def _load_heads(source: Path, scratch: Path, *, device):
    import torch

    from kairyu.models.eagle import EagleConfig, load_eagle_head

    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    eagle_config = EagleConfig.from_checkpoint_config(raw)
    dense_head = load_eagle_head(source, eagle_config, dtype=torch.bfloat16)
    checkpoints: dict[str, Path] = {"dense": source}
    checkpoint_digests: dict[str, dict[str, str]] = {
        "dense": {
            "config_sha256": sha256_file(source / "config.json"),
            "weights_sha256": sha256_file(source / "model.safetensors"),
        }
    }
    for spec in ARMS:
        if not spec.quantized:
            continue
        checkpoint = scratch / spec.name
        checkpoints[spec.name] = checkpoint
        assert spec.ignored_layers is not None
        checkpoint_digests[spec.name] = _prepare_quant_checkpoint(
            source,
            checkpoint,
            dense_head=dense_head,
            eagle_config=eagle_config,
            ignored_layers=spec.ignored_layers,
        )
    del dense_head
    gc.collect()

    heads: dict[str, object] = {}
    memory: dict[str, dict[str, int]] = {}
    for spec in ARMS:
        before = torch.cuda.memory_allocated(device)
        head = load_eagle_head(
            checkpoints[spec.name],
            eagle_config,
            dtype=torch.bfloat16,
            target_device=device,
        ).to(device)
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_allocated(device)
        heads[spec.name] = head
        memory[spec.name] = {
            "module_bytes": _module_bytes(head),
            "cuda_bytes": after - before,
        }
    return eagle_config, heads, memory, checkpoint_digests


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.tokenizer import HFTokenizer
    from kairyu.models.eagle import default_eagle3_aux_layer_ids
    from kairyu.models.loader import load_model

    if not torch.cuda.is_available():
        raise RuntimeError("the formal draft-quant gate requires CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    draft_path = args.draft_path.resolve()
    model_path = args.model_path.resolve()
    source_provenance = source_snapshot(args.source_commit)
    actual_draft_config_sha = sha256_file(draft_path / "config.json")
    if actual_draft_config_sha != PUBLIC_DRAFT_CONFIG_SHA256:
        raise RuntimeError(
            "public draft config SHA-256 mismatch: "
            f"expected {PUBLIC_DRAFT_CONFIG_SHA256}, got {actual_draft_config_sha}"
        )
    actual_draft_sha = sha256_file(draft_path / "model.safetensors")
    if actual_draft_sha != PUBLIC_DRAFT_WEIGHTS_SHA256:
        raise RuntimeError(
            "public draft checkpoint SHA-256 mismatch: "
            f"expected {PUBLIC_DRAFT_WEIGHTS_SHA256}, got {actual_draft_sha}"
        )
    target_provenance = checkpoint_snapshot(model_path)

    tokenizer = HFTokenizer(model_path)
    prompt_ids = [
        tokenizer.encode(prompt)[: args.max_prompt_tokens]
        for prompt in PROMPTS[: args.prompt_count]
    ]
    if any(len(tokens) < args.draft_tokens for tokens in prompt_ids):
        raise RuntimeError("every formal prompt must be at least draft_tokens long")

    backend = select_backend(probe(device), device=device)
    target, target_config, _generation = load_model(
        model_path,
        dtype=torch.bfloat16,
        attention_backend=backend,
        target_device=device,
    )
    target = target.to(device)
    if target_config.architecture != "Qwen3ForCausalLM":
        raise RuntimeError(
            "public Qwen3-32B EAGLE gate requires Qwen3ForCausalLM, "
            f"got {target_config.architecture}"
        )
    aux_layer_ids = default_eagle3_aux_layer_ids(target_config.num_hidden_layers)

    with tempfile.TemporaryDirectory(
        prefix="kairyu-draft-quant-",
        dir=args.scratch_dir,
    ) as temporary:
        scratch = Path(temporary)
        eagle_config, heads, memory, checkpoint_digests = _load_heads(
            draft_path,
            scratch,
            device=device,
        )
        if eagle_config.hidden_size != target_config.hidden_size:
            raise RuntimeError("draft and target hidden sizes differ")
        dense_d2t = heads["dense"].d2t
        mapped_target_ids = torch.arange(eagle_config.draft_vocab_size, device=device) + dense_d2t
        mapped_min = int(mapped_target_ids.min().item())
        mapped_max = int(mapped_target_ids.max().item())
        if mapped_min < 0 or mapped_max >= target_config.vocab_size:
            raise RuntimeError(
                "draft d2t map falls outside target vocabulary: "
                f"range=[{mapped_min}, {mapped_max}], vocab={target_config.vocab_size}"
            )
        if any(
            not torch.equal(head.d2t, dense_d2t) for name, head in heads.items() if name != "dense"
        ):
            raise RuntimeError("generated quantized checkpoints changed the draft d2t map")

        traces = [
            _build_trace(
                target,
                target_config,
                prompt_index=index,
                prompt_ids=tokens,
                continuation_tokens=args.continuation_tokens,
                draft_tokens=args.draft_tokens,
                aux_layer_ids=aux_layer_ids,
                page_size=args.page_size,
                device=device,
                dtype=torch.bfloat16,
            )
            for index, tokens in enumerate(prompt_ids)
        ]

        # Warm every measured context and multi-token target-verify shape. The
        # same discarded pass also checks every draft-step logit for finiteness.
        draft_finite: dict[str, bool] = {}
        for spec in ARMS:
            warm_rows: list[dict[str, object]] = []
            for trace in traces:
                warm_rows.extend(
                    _evaluate_trace(
                        heads[spec.name],
                        target,
                        trace,
                        draft_tokens=args.draft_tokens,
                        aux_layer_ids=aux_layer_ids,
                        check_draft_finite=True,
                        device=device,
                    )
                )
            draft_finite[spec.name] = all(row["draft_logits_finite"] is True for row in warm_rows)
        torch.cuda.synchronize(device)

        raw_rows: dict[str, list[dict[str, object]]] = {spec.name: [] for spec in ARMS}
        forward_peak_extra = {spec.name: 0 for spec in ARMS}
        measurement_schedule: list[dict[str, object]] = []
        block = 0
        for repeat in range(args.repeats):
            for trace in traces:
                offset = block % len(ARMS)
                arm_order = ARMS[offset:] + ARMS[:offset]
                measurement_schedule.append(
                    {
                        "repeat": repeat,
                        "prompt_index": trace.prompt_index,
                        "arms": [spec.name for spec in arm_order],
                    }
                )
                for order_index, spec in enumerate(arm_order):
                    baseline = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    rows = _evaluate_trace(
                        heads[spec.name],
                        target,
                        trace,
                        draft_tokens=args.draft_tokens,
                        aux_layer_ids=aux_layer_ids,
                        device=device,
                    )
                    peak = torch.cuda.max_memory_allocated(device) - baseline
                    forward_peak_extra[spec.name] = max(
                        forward_peak_extra[spec.name],
                        peak,
                    )
                    for row in rows:
                        row["repeat"] = repeat
                        row["arm_order_index"] = order_index
                    raw_rows[spec.name].extend(rows)
                block += 1

        summaries: dict[str, dict[str, object]] = {}
        for spec in ARMS:
            summaries[spec.name] = summarize_arm(
                spec.name,
                module_bytes=memory[spec.name]["module_bytes"],
                cuda_bytes=memory[spec.name]["cuda_bytes"],
                forward_peak_extra_bytes=forward_peak_extra[spec.name],
                draft_logits_finite=draft_finite[spec.name],
                draft_tokens=args.draft_tokens,
                rows=raw_rows[spec.name],
            )

    dense = summaries["dense"]
    eligible = [
        summary
        for name, summary in summaries.items()
        if name != "dense"
        and float(summary["acceptance_rate"])
        >= float(dense["acceptance_rate"]) * args.min_acceptance_retention
    ]
    selected = (
        max(
            eligible,
            key=lambda summary: float(summary["cycle_committed_tokens_per_second"]),
        )
        if eligible
        else None
    )

    def ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    comparisons = {
        name: {
            "memory_ratio": int(summary["module_bytes"]) / int(dense["module_bytes"]),
            "acceptance_ratio": ratio(
                float(summary["acceptance_rate"]),
                float(dense["acceptance_rate"]),
            ),
            "draft_latency_ratio": float(summary["draft_latency_ms"]["median"])
            / float(dense["draft_latency_ms"]["median"]),
            "cycle_goodput_ratio": ratio(
                float(summary["cycle_committed_tokens_per_second"]),
                float(dense["cycle_committed_tokens_per_second"]),
            ),
        }
        for name, summary in summaries.items()
        if name != "dense"
    }
    selected_summary = selected or {}
    selected_available = selected is not None
    backend_components = backend.selection_decision.components
    gate = {
        "selected_arm": selected_summary.get("arm"),
        "selected_arm_available": selected_available,
        "source_commit_matches": bool(source_provenance["commit_matches"]),
        "balanced_arm_order": len(measurement_schedule) % len(ARMS) == 0,
        "at_least_three_repeats": args.repeats >= 3,
        "flashinfer_target_verification": "flashinfer" in backend_components.values(),
        "dense_acceptance_sane": float(dense["acceptance_rate"]) >= args.min_dense_acceptance,
        "all_finite": all(
            bool(summary["draft_logits_finite"]) and bool(summary["target_logits_finite"])
            for summary in summaries.values()
        ),
        "target_corrections_valid": all(
            bool(summary["target_corrections_valid"]) for summary in summaries.values()
        ),
        "memory_reduced": selected_available
        and int(selected_summary["module_bytes"]) < int(dense["module_bytes"]),
        "acceptance_retained": selected_available
        and float(selected_summary["acceptance_rate"])
        >= float(dense["acceptance_rate"]) * args.min_acceptance_retention,
        "cycle_goodput_retained": selected_available
        and float(selected_summary["cycle_committed_tokens_per_second"])
        >= float(dense["cycle_committed_tokens_per_second"]) * args.min_goodput_retention,
    }
    gate["passed"] = all(value for key, value in gate.items() if key != "selected_arm")

    properties = torch.cuda.get_device_properties(device)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "configuration": {
            "draft_tokens": args.draft_tokens,
            "continuation_tokens": args.continuation_tokens,
            "teacher_bonus_tokens": 1,
            "prompt_count": len(prompt_ids),
            "repeats": args.repeats,
            "prompt_token_counts": [len(tokens) for tokens in prompt_ids],
            "page_size": args.page_size,
            "aux_layer_ids": list(aux_layer_ids),
            "min_acceptance_retention": args.min_acceptance_retention,
            "min_goodput_retention": args.min_goodput_retention,
            "min_dense_acceptance": args.min_dense_acceptance,
            "measurement_scope": (
                "standalone speculative cycle from context tensor construction "
                "through target correction; excludes scheduler and serving"
            ),
        },
        "provenance": {
            "source": source_provenance,
            "public_draft": {
                "repository": PUBLIC_DRAFT_REPOSITORY,
                "revision": PUBLIC_DRAFT_REVISION,
                "config_sha256": actual_draft_config_sha,
                "weights_sha256": actual_draft_sha,
            },
            "target": {"path": str(model_path), **target_provenance},
            "generated_checkpoints": checkpoint_digests,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory": properties.total_memory,
            "attention_backend": asdict(backend.selection_decision),
        },
        "arms": [asdict(spec) for spec in ARMS],
        "draft_target_compatibility": {
            "target_architecture": target_config.architecture,
            "target_vocab_size": target_config.vocab_size,
            "draft_vocab_size": eagle_config.draft_vocab_size,
            "mapped_target_id_min": mapped_min,
            "mapped_target_id_max": mapped_max,
        },
        "measurement_schedule": measurement_schedule,
        "summaries": summaries,
        "comparisons": comparisons,
        "raw_rows": raw_rows,
    }
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--draft-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--continuation-tokens", type=int, default=8)
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--min-dense-acceptance", type=float, default=0.20)
    parser.add_argument("--min-acceptance-retention", type=float, default=0.95)
    parser.add_argument("--min-goodput-retention", type=float, default=0.95)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.prompt_count <= len(PROMPTS):
        parser.error(f"--prompt-count must be within [1, {len(PROMPTS)}]")
    if args.draft_tokens < 1:
        parser.error("--draft-tokens must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.continuation_tokens < args.draft_tokens:
        parser.error("--continuation-tokens must be >= --draft-tokens")
    if args.max_prompt_tokens < args.draft_tokens:
        parser.error("--max-prompt-tokens must be >= --draft-tokens")
    if args.page_size < 1:
        parser.error("--page-size must be positive")
    for name in (
        "min_dense_acceptance",
        "min_acceptance_retention",
        "min_goodput_retention",
    ):
        value = getattr(args, name)
        if not 0 < value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be within (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gate": artifact["gate"], "output": str(args.output)}))
    return int(args.assert_gate and not artifact["gate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
