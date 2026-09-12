# Six-GPU V4.1 deployment investigation

GPU validation is **in progress** after the owner released the host on
2026-09-13. The initial GPU-resident Engram candidate failed memory profiling;
the current trial uses the existing Engram CPU-offload option.
The sibling eight-GPU example's measurements do not establish correctness,
memory fit or performance for this topology.

## Selected topology

DeepSeek uses GPUs 0–5 as TP2 × internal attention-DP3, EP6, PP1 behind one
vLLM service. Qwen3.8-27B-FP8 uses GPUs 6 and 7 as two TP1 replicas. Kairyu's
thinking and direct DeepSeek pools point at that same native chat endpoint.
The L1 middleware supplies the role's `thinking` flag; the native tokenizer,
reasoning and tool parser are all `deepseek_v41`. No rendered text-only
DeepSeek completion template is used.

The pinned checkpoint has 64 attention heads, 8 output groups, 16 vision
heads and 384 routed experts. TP2 divides all those head/group counts;
EP6 assigns 64 routed experts per rank. Its engram configuration supplies
8 heads for each of three n-gram sizes, or 24 hash heads; TP2 × DP3 divides
those into four hash heads per rank. The fixed implementation can share
the engram table over local DP ranks. These arithmetic and source checks
are necessary conditions, not successful initialization or numerical evidence.

TP6 is invalid because neither 64 attention heads nor 8 output groups divides
by six. TP2 × PP3 is not selected: although the model advertises `SupportsPP`,
`attention.py` raises `NotImplementedError` when a pipeline stage lacks the
source of its shared indexer/compressed KV cache (lines 390–396 and 478–489).
The source groups begin at layers 2, 8, 14 and 20. Layer 20 also publishes
candidate blocks used by subsequent indexers; `nvidia/model.py` creates this
buffer locally (lines 430–440) and PP transfers hidden states and pre-mix,
not those candidates (lines 675–678). Therefore a prospective PP split must
respect these groups and retain layers 20–39 together. The two large engram
tables and uneven layer groups make PP stage capacity an additional open issue.

DSpark is disabled. The checkpoint's draft has 128 routed experts, which does
not divide EP6. No claim is made that the pinned draft implementation handles
this case or that the eight-GPU DSpark result transfers. Adding DSpark or
changing PP/DP requires a separate source review and GPU experiment.

The initial limits are 1,048,576 context tokens, 32 sequences per internal DP
engine, 16,384 batched tokens, 90% GPU memory, FP8 KV, 64-token manager blocks,
MXFP4 indexer cache. The initial GPU-resident Engram trial loaded 84.09 GiB
per GPU, then failed a 512 MiB sparse-indexer profiling allocation with only
377 MiB free. CPU offload reduces GPU model memory to 52.62 GiB and places
approximately 189 GiB of tables in pinned host memory. It retains the same
TP/DP collectives and UVA lookup implementation (engram.py lines 735–784).
No other serving limit changed for this trial. Do not silently lower limits or
switch topology after a failure;
record the failure, revise the candidate explicitly and rerun its full gates.

## Fixed source and image provenance

The source was inspected in the already running container
`deepseek-v4-1-flash-8gpu-deepseek-0-1`, image
`sha256:027bf47b2bd6f0d0abe54b296e7e9e3d31ee103bb6e46fa0a9807117681c2359`.
Its configured user is empty (Docker default root). Source version is
`0.1.dev20904+g179dd0fa9`; current GitHub main was not substituted for it.

The parent overlay reuses `../deepseek-v4.1-flash-8gpu/vllm-sm120.Dockerfile`
and `patch_runtime.py` unchanged. The lifecycle builds/attests that parent using
the sibling directory as context, then builds this directory's child overlay.
The existing parent supplies SM120 sparse-cache compatibility and the official
high reasoning mapping (75). The child applies `patch_masked_kv.py`, checking
the exact FlashInfer header hashes before replacing invalid candidate gathers
with a zero device row in decode and prefill. Copy sizes and asynchronous barrier
accounting remain unchanged; valid candidate addresses remain unchanged.
Generated kernels use a new persistent cache namespace.

The original offloaded runtime starts but produces intermittent NaN logprobs,
corrupt text and grammar errors. Eager and NCCL-only diagnostics reproduce it.
An instrumented temporary image locates the first nonfinite values at layer 0
Attention. A separate sparse-attention oracle reproduces the masked-index failure
with NaN-poisoned unused KV memory. See `MEASUREMENTS.md` for the failed candidates
and subsequent numerical/serving evidence. Diagnostic images are never accepted
as the pinned runtime by the verification launcher.

Model revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
Model tree SHA256: `d21211ca29ad7eba1fda84e49b1a34a73214ec9b84b23928cde63902c3318bfd`.
Inspected files, relative to `/usr/local/lib/python3.12/dist-packages/`:

| File | SHA256 |
| --- | --- |
| `vllm/models/deepseek_v4_1/nvidia/model.py` | `530ed24c8fd2e9daeb5c3d340ef52246618786f19eace8ec217c8d3bdf271110` |
| `vllm/models/deepseek_v4_1/attention.py` | `521f3aedd5711e3c49ea0716192a87e99bcc585436ec4909cc63c031aec8175d` |
| `vllm/models/deepseek_v4_1/common/engram.py` | `41c5bdf25cf8337088247be768e3d43fa41e6358397800f8b602ebf1ede69480` |
| `/models/deepseek-v4.1-flash/config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |

`common/engram.py` lines 704–719 compute `num_shards = tp_size * dp_size`,
ceil-divide hash columns and reject ranks left without heads. `nvidia/model.py`
lines 158–166 enable sequence parallelism for PP1 + EP + TP>1 with DP>1.
These are static constraints; collectives, sequence parallelism and SM120
Marlin with this six-rank shape still need GPU validation.

## Native requirements request contract

The fixed image's `entrypoints/openai/chat_completion/protocol.py` declares
`thinking_token_budget` at line 257 and `structured_outputs` at line 375 and
forwards them to sampling at lines 741–744. Schema acceptance alone does not
prove enforcement. The inspected enforcement chain is:

- `reasoning/deepseek_v41_engine_reasoning_parser.py` exports the registered
  `DeepSeekV41ParserReasoningAdapter`.
- `parser/engine/registered_adapters.py` constructs that adapter from
  `DeepSeekV41Parser`, which inherits V4's `<think>` / `</think>` markers.
- `parser/engine/adapters.py` exposes start/end strings;
  `config/reasoning.py` tokenizes them when initializing the reasoning config.
- `v1/sample/thinking_budget_state.py` tracks budgeted requests and forces
  the configured end tokens when the budget is exhausted.

Native GPU probes now confirm completed JSON for all 12 API/nested-effort
combinations, with exact request-hash correlation to fixed-high hook records.
A forced 16-token thinking budget reports exactly 16 reasoning tokens and a
completed public answer. These probes do not establish arbitrary task quality.

The pinned Qwen v0.23 protocol forwards `thinking_token_budget` to sampling
(protocol lines 231/643). Its reasoning parser uses token IDs 248068/248069
for `<think>`/`</think>`; the sampler forces the latter at the budget boundary.
The example now reserves half the draft/answer cap for body text, keeping all
effort and sampling defaults. Composed GPU revalidation is required because
the initial unrestricted answer consumed all 4096 tokens in reasoning.

## Required GPU follow-up

On a machine with all eight GPUs explicitly available, verify the checkpoint,
image and config hashes first. Start DeepSeek alone on GPUs 0–5 and capture
complete startup logs. A healthy endpoint must follow actual model loading
and cache allocation. Preserve any startup, OOM, collective or kernel failure;
there is no fallback to an old model/image or a different GPU count.

Then verify native default/high/low/max and non-thinking calls, requirements
JSON plus thinking-budget enforcement, tool calls, images, multi-turn state,
cancellation and normal restart. Add Qwen on GPUs 6–7 and test the public
ensemble's requirements → two-policy fan-out alongside draft critique
→ three-candidate synthesis → audit graph and direct routes. Finally measure same-topology
DeepSeek-direct baselines and ensemble concurrency 1/8/16/32, with original
request/response traces and exact image/config/model provenance. Long-context
capacity/retrieval is a separate gate. Update validation status only for gates
actually run; never import the old example's throughput or TTFT evidence.
