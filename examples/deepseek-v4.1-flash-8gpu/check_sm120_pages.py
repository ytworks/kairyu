"""GPU numerical gate for the example's V4.1 sparse-MLA page adaptations.

Run inside the pinned L1 image with one SM120 GPU. Uses real packed FP8 caches
with padded block strides, independent PyTorch dequantization/attention, both
compression ratios, decode/prefill, masking, sinks, and DSpark-width indices.

Default: retain the original floating-reference fidelity gate. --arithmetic
models the pinned decode/prefill intermediate formats and reports that gate
separately; it never relabels a failed floating-reference comparison as PASS.
"""

import argparse
import itertools
import json
import math

import torch
from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner
from flashinfer.mla._sparse_mla_sm120_plan import (
    _MODEL_TYPE_DSV4,
    KernelVariant,
    plan,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache


def cache(page_size):
    n = 512
    source = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16)
    source[:, :448] *= torch.linspace(0.1, 2, 7, device="cuda").repeat_interleave(64)
    storage = torch.zeros(n // page_size, page_size * 584 + 128, device="cuda", dtype=torch.uint8)
    packed = storage[:, : page_size * 584]
    quantize_and_insert_k_cache(source, packed, torch.arange(n, device="cuda"), page_size)
    # Decode the actual stored bytes independently of the kernel under test.
    data = packed[:, : page_size * 576].reshape(-1, page_size, 576)
    exponents = packed[:, page_size * 576 :].reshape(-1, page_size, 8)[..., :7]
    scales = torch.exp2(exponents.float() - 127).repeat_interleave(64, dim=-1)
    nope = data[..., :448].view(torch.float8_e4m3fn).float() * scales
    rope = data[..., 448:].view(torch.bfloat16).float()
    reference = torch.cat((nope, rope), dim=-1).reshape(n, 512)
    return packed.view(-1, page_size, 1, 584), reference


def cache_components(packed):
    """Read the stored FP8 bytes/UE8M0 footer/BF16 RoPE without runtime helpers."""
    page = packed.shape[1]
    rows = packed.detach().cpu().reshape(packed.shape[0], page * 584)
    data = rows[:, : page * 576].reshape(-1, page, 576)
    scales = torch.exp2(rows[:, page * 576 :].reshape(-1, page, 8)[..., :7].float() - 127).reshape(
        -1, 7
    )
    nope = data[..., :448].contiguous().view(torch.float8_e4m3fn).float().reshape(-1, 7, 64)
    rope = data[..., 448:].contiguous().view(torch.bfloat16).float().reshape(-1, 64)
    return nope, scales, rope


def fp8(value):
    return value.clamp(-448, 448).to(torch.float8_e4m3fn).float()


def quantized_reference(
    q, main_cache, extra_cache, indices, extra_indices, sinks, *, decode, chunks_per_block=1
):
    """Return CPU FP32 reference and pre-merge splits for this probe's envelope.

    This models arithmetic formats, not the kernel's threads, memory movement
    or MMA implementation. Independent torch contractions use decoded stored
    bytes. FP32 reduction/exp2 instruction rounding may differ from CUDA.
    Source: FlashInfer60b49158 common/fp8_quant.cuh:208-230;
    decode_dsv4_kernel.cuh:712-886,965-1034; prefill_mg_kernel.cuh:1700-2104.
    Decode uses FP8 Q and BF16 split outputs. Dual-cache prefill dispatches
    BF16 QK, retains one FP32 running output, and rounds only at final store.
    """
    q = q.detach().cpu().float()
    sinks = sinks.detach().cpu().float()
    assert q.shape[0] in (1, 128) and q.shape[-1] == 512
    assert not decode or q.shape[0] == 1
    parts = [cache_components(main_cache), cache_components(extra_cache)]
    idx = [indices.cpu().long(), extra_indices.cpu().long()]
    raw, scale, rope = [
        torch.cat([p[i][ix.clamp_min(0)] for p, ix in zip(parts, idx, strict=True)], dim=1)
        for i in range(3)
    ]
    valid = torch.cat([ix >= 0 for ix in idx], dim=1)
    values = torch.cat(((raw * scale[..., None]).flatten(-2), rope), dim=-1)
    if decode:
        tiles = q[..., :448].reshape(*q.shape[:-1], 7, 64)
        qscale = torch.exp2(torch.ceil(torch.log2(tiles.abs().amax(-1).clamp_min(1e-4) / 448)))
        q = torch.cat(
            ((fp8(tiles / qscale[..., None]) * qscale[..., None]).flatten(-2), q[..., 448:]), dim=-1
        )
    # Prefill converts stored KV to BF16 before QK; UE8M0 × FP8 is exact
    # for the finite normal scale range in this fixture, but keep the cast.
    keys = values if decode else values.to(torch.bfloat16).float()
    log2e = torch.tensor(math.log2(math.e), dtype=torch.float32)
    sm_scale_log2e = torch.tensor(1 / math.sqrt(512), dtype=torch.float32) * log2e
    logits = torch.einsum("thd,tkd->thk", q, keys) * sm_scale_log2e
    logits.masked_fill_(~valid[:, None, :], -torch.inf)
    chunks = []
    for start in range(0, valid.shape[-1], 64):
        end = start + 64
        if decode and not bool(valid[:, start:end].any()):
            continue  # Decode packs only ceil(topk_length/64) chunks per cache.
        chunks.append((start, end))
    groups = (
        [chunks[i : i + chunks_per_block] for i in range(0, len(chunks), chunks_per_block)]
        if decode
        else [chunks]
    )
    partials = []
    lses = []
    for group in groups:
        maximum = torch.full(q.shape[:-1], -torch.inf)
        denom = torch.zeros_like(maximum)
        numer = torch.zeros_like(q)
        for start, end in group:
            chunk = logits[..., start:end]
            next_max = torch.maximum(maximum, chunk.amax(-1))
            alpha = torch.where(torch.isfinite(maximum), torch.exp2(maximum - next_max), 0)
            weight = torch.where(
                valid[:, None, start:end], torch.exp2(chunk - next_max[..., None]), 0
            )
            numer *= alpha[..., None]
            denom = denom * alpha + weight.sum(-1)
            maximum = next_max
            scaled = weight[..., None] * scale[:, None, start:end, :]
            weight_scale = scaled.abs().amax(-2).clamp_min(1e-10) / 448
            rounded = fp8(scaled / weight_scale[..., None, :])
            nope = (
                torch.einsum("thkc,tkcd->thcd", rounded, raw[:, start:end])
                * weight_scale[..., None]
            )
            rope_out = torch.einsum(
                "thk,tkd->thd", weight.to(torch.bfloat16).float(), rope[:, start:end]
            )
            numer += torch.cat((nope.flatten(-2), rope_out), dim=-1)
        if not decode:
            total = denom + torch.exp2(sinks[None, :] * log2e - maximum)
            return (numer / total[..., None]).to(torch.bfloat16).float(), None, None
        partials.append((numer / denom[..., None]).to(torch.bfloat16).float())
        lses.append(torch.log2(denom) + maximum)
    lse = torch.stack(lses, -1)
    maximum = lse.amax(-1)
    weights = torch.exp2(lse - maximum[..., None])
    total = weights.sum(-1) + torch.exp2(sinks[None, :] * log2e - maximum)
    split_output = torch.stack(partials, -2)
    output = (split_output * weights[..., None]).sum(-2) / total[..., None]
    return output.to(torch.bfloat16).float(), split_output, lse


def comparison(actual, reference):
    actual, reference = actual.detach().cpu().float(), reference.detach().cpu().float()
    error = (actual - reference).abs()
    allowed = 0.05 + 0.05 * reference.abs()
    return {
        "passed": bool(torch.isfinite(error).all() and (error <= allowed).all()),
        "mismatched_elements": int((error > allowed).sum()),
        "max_abs_error": float(error.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, choices=(8, 32), default=8)
    parser.add_argument(
        "--arithmetic",
        action="store_true",
        help="check pinned arithmetic; retain the original floating-reference verdict separately",
    )
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    args = parser.parse_args()
    if args.gpu_memory_fraction is not None:
        if not 0 < args.gpu_memory_fraction <= 1:
            parser.error("--gpu-memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    torch.set_num_threads(2)
    torch.manual_seed(4175)
    runner = _SparseMLAPagedAttentionRunner(d_v=512)
    results = []
    for extra_page, tokens, topk, masked in itertools.product(
        (32, 64), (1, 128), (128, 192), (False, True)
    ):
        main_cache, main_ref = cache(64)
        extra_cache, extra_ref = cache(extra_page)
        q = torch.randn(tokens, args.heads, 512, device="cuda", dtype=torch.bfloat16)
        sink = torch.linspace(-1, 3, args.heads, device="cuda")
        indices = torch.randint(0, 512, (tokens, topk), device="cuda", dtype=torch.int32)
        extra_indices = torch.randint(0, 512, (tokens, 128), device="cuda", dtype=torch.int32)
        lens = extra_lens = None
        if masked:
            lens = torch.randint(1, topk, (tokens,), device="cuda", dtype=torch.int32)
            extra_lens = torch.randint(1, 128, (tokens,), device="cuda", dtype=torch.int32)
            indices.masked_fill_(torch.arange(topk, device="cuda")[None] >= lens[:, None], -1)
            extra_indices.masked_fill_(
                torch.arange(128, device="cuda")[None] >= extra_lens[:, None], -1
            )
        output = torch.empty_like(q)
        runner.run(
            q,
            main_cache,
            indices,
            output,
            1 / math.sqrt(512),
            topk_length=lens,
            attn_sink=sink,
            extra_kv_cache=extra_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_lens,
        )
        values = torch.cat(
            (main_ref[indices.clamp_min(0).long()], extra_ref[extra_indices.clamp_min(0).long()]),
            dim=1,
        )
        valid = torch.cat((indices >= 0, extra_indices >= 0), dim=1)
        logits = torch.einsum("thd,tkd->thk", q.float(), values) / math.sqrt(512)
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        weights = torch.cat((logits, sink[None, :, None].expand(tokens, -1, -1)), dim=-1).softmax(
            -1
        )[..., :-1]
        reference = torch.einsum("thk,tkd->thd", weights, values)
        floating = comparison(output, reference)
        row = {
            "extra_page": extra_page,
            "tokens": tokens,
            "heads": args.heads,
            "topk": topk,
            "masked": masked,
            "max_abs_error": floating["max_abs_error"],
            "float_reference": floating,
        }
        gate_reference = reference
        if args.arithmetic:
            # Attest the actual planner branch before selecting its arithmetic.
            # The oracle does not silently assume a calibrated dispatch choice.
            selected = plan(
                num_tokens=tokens,
                num_heads=args.heads,
                topk=topk,
                model_type=_MODEL_TYPE_DSV4,
                page_block_size=64,
                has_extra=True,
                prefill_impl_pref=0,
                device=q.device,
                extra_topk=128,
            )
            expected_variant = (
                KernelVariant.DECODE_SPLITK if tokens == 1 else KernelVariant.PREFILL_MG_DUAL
            )
            assert selected is not None and selected.variant is expected_variant
            if tokens == 1:
                assert selected.cpb in (-1, 1), "arithmetic oracle requires one chunk per split"
                if selected.cpb == -1:
                    # With all chunks inside one SM wave, the pinned C++
                    # minimum-tail-gap heuristic uniquely selects cpb=1.
                    blocks = ((args.heads + 15) // 16) * ((topk + 63) // 64 + 2)
                    assert (
                        blocks <= torch.cuda.get_device_properties(q.device).multi_processor_count
                    )
            row["kernel_variant"] = selected.variant.name
            gate_reference, _, _ = quantized_reference(
                q,
                main_cache,
                extra_cache,
                indices,
                extra_indices,
                sink,
                decode=tokens == 1,
            )
            row["quantized_arithmetic"] = comparison(output, gate_reference)
        results.append(row)
        print(json.dumps(row), flush=True)
        # The original floating gate remains the default. Arithmetic mode
        # changes the reference to the source-defined operation, not its
        # tolerance; floating fidelity failures are retained in each row.
        torch.testing.assert_close(
            output.detach().cpu().float(),
            gate_reference.detach().cpu().float(),
            atol=0.05,
            rtol=0.05,
        )
    print(
        json.dumps(
            {
                "passed": True,
                "gate": "quantized_arithmetic" if args.arithmetic else "float_reference",
                "float_reference_passed": all(row["float_reference"]["passed"] for row in results),
                "cases": len(results),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
