# Issue #364 Design: Opt-in FP32 Final Logits

Status: **Implemented and locally/GPU verified; paired real-model measurement
pending** (2026-08-05).

Related contracts: M12 D2/D5/D6, M8 D2, M14 D3, and G2 A1/A2.

## 1. Goal and scope

Kairyu currently returns the output-head result in the model's compute dtype.
On the production GPU path that is normally BF16. Greedy selection therefore
sees BF16 logits directly, while the general sampler can only upcast values
after the final output rounding has already happened. This is compatible with
the existing Hugging Face and vLLM behavior and is not a parity defect, but it
can move an argmax or a filtering boundary when two logits are close.

Issue #364 adds a construction-time precision choice for the final dense
output-head GEMM and measures it against the already-closed G2 A1/A2 parity
contracts. It does not change hidden-state, checkpoint-weight, KV-cache, or
sampler-processing precision, and it does not introduce a per-request option.
The generic config-versus-config experiment framework in issue #365 is a later
meta-issue item and is explicitly outside this change.

## 2. Public contract

All native real-model construction and deployment surfaces expose:

```python
logits_dtype: Literal["model", "float32"] = "model"
```

- `model` is the compatibility default. `DenseDecoder.logits` executes the
  existing output-head operation and returns its natural model/output-head
  dtype. No new kernel, cast, or rounding boundary is inserted.
- `float32` requests FP32 output from the final output-head GEMM. On CUDA with
  BF16 or FP16 hidden states and dense head weights, the GEMM itself produces
  FP32 output through `out_dtype=torch.float32`; converting an already-rounded
  BF16 result with `.float()` does not satisfy this contract.
- If hidden states and the head are already FP32, `float32` resolves to the
  ordinary FP32 output-head operation. In particular, the existing CPU
  real-model path remains FP32 without a CUDA-only `out_dtype` call.
- Values have exact string validation. Unknown values and wrong types fail
  before model or distributed resources are constructed.
- This setting belongs to engine/model construction, not `SamplingParams`, the
  OpenAI request schema, or an individual scheduled request. A captured CUDA
  graph and every rank in a distributed model have one stable logits mode.

The value is propagated consistently through single-process, process-isolated,
TP, EP, and P-D native real-model builders. Process startup attestation and the
distributed handshake bind the requested and resolved mode so a child or rank
cannot silently use a different precision. Backend health reports both the
requested value and resolved tensor dtype; a homogeneous pool may promote the
field only after all local workers agree.

For example, the process-isolated deployment form is:

```yaml
engines:
  precise:
    backend: kairyu-proc
    options:
      model_path: /models/llama
      logits_dtype: float32
```

The repository's internal pipeline-stage wrapper inherits the mode from the
already-constructed full decoder and applies it at the final stage. Pipeline
parallelism is not currently a standalone production deployment surface, so
this issue does not introduce a separate PP launcher option.

An explicit `float32` request must never be silently ignored. Any future custom
or non-dense output-head implementation that cannot provide FP32 at the final
GEMM boundary must either implement this same semantic contract or fail during
construction. The current real-model output head remains dense and unquantized,
including for quantized checkpoints, under the default linear-selection policy.

## 3. Numerical and compatibility contract

For an input shaped `[..., hidden_size]`, the CUDA BF16/FP16 implementation
flattens all leading dimensions, multiplies by the transposed dense head weight
with FP32 output, and restores `[..., vocab_size]`. One-dimensional terminal
prefill rows, two-dimensional batched/decode rows, and higher-rank compatibility
inputs therefore preserve the existing shape contract.

The option does not cast or duplicate the output-head weight. Model weights
remain in their configured dtype; tied embedding/output-head identity remains
intact; parameter and buffer names, `state_dict`, checkpoint loading, sharding,
and memory ownership are unchanged. No scalar or tensor is copied to the host
before the established public result boundary.

The `model` arm must remain the exact former path and is expected to preserve
existing tokens, raw logprobs, CUDA graph behavior, deterministic replay, and
performance. The `float32` arm intentionally may change:

- a greedy token at a near tie;
- membership at min-p, top-k, or top-p boundaries;
- temperature-scaled selection probabilities and reported raw logprobs.

Those changes are configuration semantics, not regressions by themselves.
Within one chosen mode, overlap execution, TP rank ownership, speculative target
verification, seeded replay, and eager/CUDA-graph execution remain subject to
their existing determinism and correctness contracts. The option must affect
the target output head before greedy argmax, raw-logprob capture, penalties, or
filtering; it must not alter the stateless random stream itself.

## 4. Required local and GPU verification

P0 correctness tests are binding:

1. A synthetic CUDA BF16 near-tie head demonstrates that the former BF16 output
   can collapse or reorder candidates, while `float32` matches an FP32-output
   GEMM oracle and differs from post-hoc `bf16_logits.float()`.
2. One-, two-, and higher-dimensional hidden-state inputs preserve shape and
   produce the requested dtype. `model` is exactly equal to the former direct
   output-head call.
3. Tied and untied heads retain model-dtype weights, tied identity, and the
   exact parameter/buffer and `state_dict` structure.
4. Sequential prefill, batched prefill, eager tensor/list decode, and CUDA graph
   capture/replay consume FP32 logits before direct greedy selection and raw
   selected/top-logprob reporting.
5. Greedy and temperature/min-p/top-k/top-p fixtures prove there is no hidden
   downcast before selection. Intended boundary changes are recorded, while the
   `model` arm remains unchanged.
6. A steady CUDA profiler gate retains zero host-synchronizing scalar reads and
   keeps the selected token on device.
7. Single-process, TP, EP, P-D, and process-isolated construction propagate one
   requested/resolved mode. Rank-handshake and child-startup mismatches fail
   closed; backend health reports the constructed decision.
8. Both native CLI/config validators cover defaults, explicit values, wrong
   types, and unknown values. Existing CPU, GPU, TP, CUDA-graph, speculative,
   and process suites remain green.

P1 evidence and documentation are required for issue closure: the narrow paired
A1/A2 runner, raw position-level evidence, complete provenance, public examples,
and the final result amendment described below. Final-head latency or peak-output
memory measurements are optional P2 diagnostics; issue #364 sets no new
performance threshold, so an ad-hoc timing cannot become an unwritten pass/fail
gate. P0 and P1 are the blocking completion levels. Any P2 rows that are collected
must still be retained without post-hoc selection.

## 5. Existing A1/A2 baselines (not issue #364 results)

The following retained closures are immutable reference context. This note does
not rewrite their artifacts, criteria, or verdicts.

| Gate | Reference self-agreement | Retained Kairyu result | Other binding facts |
|---|---:|---:|---|
| A1, Llama-3.1-8B | 1010/1024 | TP1 1014/1024; TP2 1014/1024 | overlap ON equals OFF 64/64 at both degrees; zero substantive and zero missing; agreeing-position max logprob delta 0.10440/0.10331, both at or below 0.25 |
| A2, pinned Llama-3.3-70B FP8 | 1005/1024; tie gap 0.5 nat | TP2 1006/1024; TP4 1005/1024; TP8 1006/1024 | zero substantive and zero missing; TP4/TP8 versus TP2 each 1004/1024 with zero substantive differences |

A2's retained agreeing-position maximum logprob deltas
0.56900/0.54147/0.25563 are diagnostics, not a third A2 criterion. Free-running
cross-implementation equality is likewise diagnostic; A1's same-engine overlap
ON/OFF continuation equality remains binding.

Reference artifacts:

- `bench/results/g2-a1-llama31-8b-rtxpro6000-2026-07-26.json`
- `bench/results/g2-a2-llama33-70b-fp8-rtxpro6000-2026-07-27.json`

## 6. Predeclared paired real-model measurement

Both arms are measured from the same clean implementation commit, on the same
hardware/runtime, with identical model and tokenizer revisions, prompt token
IDs, BF16 model/KV settings, and one immutable raw HF reference. Each retained
artifact carries full checkpoint/config/tokenizer digests, source commit, CUDA,
NCCL, device/topology, command/config, and per-position token/logprob evidence.
Separate reference generation per arm is not allowed.

The narrow issue #364 matrix is:

| Gate | Arm | Required TP degrees and observations | Result |
|---|---|---|---|
| A1 | `model` | TP1/2, 64 prompts x 16 teacher-forced positions; complete overlap OFF/ON continuations | pending |
| A1 | `float32` | TP1/2, the same 64 x 16 rows and overlap continuations | pending |
| A2 | `model` | TP2/4/8, 64 prompts x 16 teacher-forced positions; TP4/8 versus TP2 | pending |
| A2 | `float32` | TP2/4/8, the same 64 x 16 rows and TP comparisons | pending |

The four formal arm artifacts are assembled by
`bench/gate_logits_dtype.py`; it fails closed on mixed arm identity,
provenance, references, checkpoints, prompts, commits, or incomplete raw rows.

Every A1 arm must independently preserve exact overlap ON/OFF continuations,
zero substantive disagreements, zero missing observations, agreement at or
above the paired run's shared reference self-agreement, and the existing 0.25-nat
agreeing-position logprob bound. Every A2 arm must independently have zero
substantive disagreements, zero missing observations, agreement at or above
the reference self-agreement, and zero substantive TP4/8-versus-TP2 differences.
No threshold is added for A2's diagnostic agreeing-position logprob delta.

The paired summary reports each checkpoint and TP degree separately:

- agreement count and change from `model`;
- changed argmax positions, including moves toward and away from HF;
- substantive and missing counts;
- agreeing-position absolute selected-logprob deviation from the shared HF
  reference, while retaining every raw selected and top-logprob observation;
- the immutable reference self-agreement floor and measured tie gap;
- any separately collected non-binding latency or memory diagnostics.

Results are not pooled across models or TP degrees to manufacture one global
improvement. A cell is a positive quality result only when agreement improves
against the same reference and all existing safety gates still pass. A broader
claim requires no measured cell to worsen and at least one cell to improve;
otherwise the conclusion is `mixed`, `no_measurable_improvement`, or `negative`
as the retained numbers require. A truthful non-positive result is still a
valid outcome of the issue proposal. If the `float32` arm fails an existing A1
or A2 correctness gate, however, the public feature is not ready for closure.

## 7. Completion record (fill after the retained run)

- Implementation commit: pending
- Paired artifact (`bench/results/issue-364-fp32-logits-a1-a2-<date>.json`)
  and independent replay command: pending
- A1 `model` / `float32` verdicts: pending / pending
- A2 `model` / `float32` verdicts: pending / pending
- Quality classification: pending
- CI and focused GPU tests: pending

After measurement, this section and `PROGRESS.md` receive the result without
editing the historical A1/A2 closure text or artifacts.
