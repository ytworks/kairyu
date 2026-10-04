# Measurements: winnow-12b-q8-1gpu

Status: **GPU verification pending.** No GPU run has been recorded for this
environment yet. All L1 values in `compose.yaml` (8 slots x 65,536 tokens,
q8_0 KV, 8 decision branches, `--memory auto`) are provisional.

What is verified so far (CPU, 2026-10-04):

- `l1.correctness.llamacpp_upstream_contract` passes against stock llama.cpp
  b11391. That build is not this example's runtime; it uses a synthetic Gemma 4
  vocabulary model (`docs/design/llamacpp-upstream.md`, Evidence).
- Every behavior that gate relies on was re-read in this example's runtime base
  (llama.cpp `911f6cd`, b11036) and is unchanged. Winnow's patches add routes
  and model fixes; they leave the chat request schema unchanged.
- `kairyu.yaml` loads and builds its engine and System One entry.

To record the GPU evidence on the target host:

```sh
./run.sh
./verify.sh all
```

Then replace this file with:

- the `attest` `/props` snapshot (build_info, slots, per-slot context, sampling);
- the contract gate report;
- the tool-calling, vision and systemone gate results;
- the serving rows (TTFT, output tok/s per concurrency);
- the image digests of `winnow-server` and Kairyu;
- GPU memory at idle and under the c8 serving row, to confirm or retune the
  slot/context geometry.
