"""GPU oracle for invalid sparse-KV rows that contain NaN bytes.

Only cache slots 128..383 are populated and referenced; unused bytes are zero
or 0xFF. Paired seeds keep the valid workload identical. The independent
reference zeros invalid gathered rows before both QK and XV. Source amplitude
is 0.25 and the existing FP8 tolerance remains atol=rtol=0.05; run the sibling
check_sm120_pages.py unchanged as a separate full-amplitude regression gate.

Set MASKED_KV_BASELINE_DIR when running the original image to retain its
finite-cache output tensors (the 12 poisoned/masked rows are expected to fail).
Set MASKED_KV_COMPARE_DIR when running the patched image to additionally
require bitwise equality to those tensors for all 48 zero/poison cases.
MASKED_KV_RESULTS selects the JSON report path (default /tmp/masked-kv-results.json).
"""

import itertools
import json
import math
import os
from pathlib import Path

import torch
from flashinfer.mla._sparse_mla_sm120 import _SparseMLAPagedAttentionRunner
from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache


def cache(page_size, poison):
    n = 512
    source = torch.randn(n, 512, device="cuda", dtype=torch.bfloat16) * 0.25
    source[:, :448] *= torch.linspace(0.1, 2, 7, device="cuda").repeat_interleave(64)
    storage = torch.full(
        (n // page_size, page_size * 584 + 128),
        255 if poison else 0,
        device="cuda",
        dtype=torch.uint8,
    )
    packed = storage[:, : page_size * 584]
    quantize_and_insert_k_cache(
        source[128:384], packed, torch.arange(128, 384, device="cuda"), page_size
    )
    # Decode the actual stored bytes independently of the kernel under test.
    data = packed[:, : page_size * 576].reshape(-1, page_size, 576)
    exponents = packed[:, page_size * 576 :].reshape(-1, page_size, 8)[..., :7]
    scales = torch.exp2(exponents.float() - 127).repeat_interleave(64, dim=-1)
    nope = data[..., :448].view(torch.float8_e4m3fn).float() * scales
    rope = data[..., 448:].view(torch.bfloat16).float()
    reference = torch.cat((nope, rope), dim=-1).reshape(n, 512)
    return packed.view(-1, page_size, 1, 584), reference


def main():
    torch.manual_seed(4175)
    runner = _SparseMLAPagedAttentionRunner(d_v=512)
    results = []
    baseline_dir = os.environ.get("MASKED_KV_BASELINE_DIR")
    compare_dir = os.environ.get("MASKED_KV_COMPARE_DIR")
    if baseline_dir:
        Path(baseline_dir).mkdir(parents=True, exist_ok=True)
    for extra_page, tokens, heads, poison, masked in itertools.product(
        (32, 64), (1, 22, 48), (8, 32), (False, True), (False, True)
    ):
        # Paired zero/poison runs use identical populated rows, queries and indices.
        torch.manual_seed(4175 + extra_page + tokens * 100 + heads * 10000 + int(masked))
        topk = 128
        main_cache, main_ref = cache(64, poison)
        extra_cache, extra_ref = cache(extra_page, poison)
        q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
        sink = torch.linspace(-1, 3, heads, device="cuda")
        indices = torch.randint(128, 384, (tokens, topk), device="cuda", dtype=torch.int32)
        extra_indices = torch.randint(128, 384, (tokens, 128), device="cuda", dtype=torch.int32)
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
        # Zero invalid fetched rows before BOTH QK and XV; 0 * NaN is NaN.
        values = torch.where(valid[..., None], values, 0.0)
        logits = torch.einsum("thd,tkd->thk", q.float(), values) / math.sqrt(512)
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        weights = torch.cat((logits, sink[None, :, None].expand(tokens, -1, -1)), dim=-1).softmax(
            -1
        )[..., :-1]
        reference = torch.einsum("thk,tkd->thd", weights, values)
        # Match FlashInfer's pinned DSV4 tests (FP8 QK/XV intermediates):
        # tests/attention/test_sparse_mla_sm120.py at 60b49158, dual-cache gates.
        key = f"page{extra_page}-t{tokens}-h{heads}-masked{int(masked)}.pt"
        if baseline_dir and not poison:
            torch.save(output.cpu(), Path(baseline_dir) / key)
        baseline_equal = None
        if compare_dir:
            baseline = torch.load(Path(compare_dir) / key, weights_only=True).to(output.device)
            baseline_equal = torch.equal(output, baseline)
        passed = True
        error = None
        try:
            torch.testing.assert_close(output.float(), reference, atol=0.05, rtol=0.05)
        except AssertionError as exc:
            passed = False
            error = str(exc)
        if baseline_equal is False:
            passed = False
            error = (error or "") + " Output differs from original finite-cache baseline."
        results.append(
            {
                "baseline_bitwise_equal": baseline_equal,
                "valid_cache_slots": [128, 383],
                "invalid_index": -1,
                "unused_byte_value": 255 if poison else 0,
                "heads": heads,
                "poison_unused_bytes": poison,
                "passed": passed,
                "nonfinite_output": int((~torch.isfinite(output)).sum()),
                "nonfinite_reference": int((~torch.isfinite(reference)).sum()),
                "error": error,
                "extra_page": extra_page,
                "tokens": tokens,
                "topk": topk,
                "masked": masked,
                "max_abs_error": (output.float() - reference).abs().max().item(),
            }
        )
        print(json.dumps(results[-1]), flush=True)

    Path(os.environ.get("MASKED_KV_RESULTS", "/tmp/masked-kv-results.json")).write_text(
        json.dumps(results, indent=2) + "\n"
    )
    passed = all(row["passed"] for row in results)
    print(json.dumps({"passed": passed, "cases": len(results)}), flush=True)
    raise SystemExit(int(not passed))


if __name__ == "__main__":
    main()
