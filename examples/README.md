# Kairyu L3 + vLLM L1 examples

`examples/` contains these complete environments:

| Environment | GPU layout | Model/context |
|---|---|---|
| [`qwen3.8-27b-1gpu`](qwen3.8-27b-1gpu/README.md) | one selected RTX PRO 6000 Blackwell | official FP8, 262,144 tokens |
| [`quyet-1.0-large-1gpu`](quyet-1.0-large-1gpu/README.md) | Quyet-1.0-Large (Gemma-4-31B-it decision fine-tune, bf16) on vLLM, one selected RTX PRO 6000 Blackwell | System One only: calibrated typed decisions on `/v1/systemone` (the `quyet` package's prompt and calibration, read on vLLM), verified with JevBench and TypeSafe's SDK; Jev-style playground, no chat |
| [`qwen3.8-deepseek-v4.1-8gpu`](qwen3.8-deepseek-v4.1-8gpu/README.md) | DeepSeek-V4.1 DP6/EP6 (GPU 0-5) + Qwen TP1 x 2 replicas (GPU 6, 7) | Qwen-judged two-route Kairyu L2 (one thinking DeepSeek answer, or a dual-track ensemble with two policies) with native image input on every role |
| [`deepseek-v4.1-quyet-8gpu`](deepseek-v4.1-quyet-8gpu/README.md) | DeepSeek-V4.1 DP6/EP6 (GPU 0-5) + Quyet-1.0-Large x 2 replicas (GPU 6 route judge, GPU 7 judgments and form check; System One on vLLM) | one public model, `kairyu-verified-tool`: Quyet routes a turn whose next reply needs a tool call to the verified tool route (10-16 DeepSeek candidate calls, Quyet's judgments of each, a DeepSeek move that batches independent safe calls, and Quyet's form check with up to two DeepSeek fixes) and every other request to DeepSeek at the caller's effort; Open WebUI |
| [`glm-5.3-flash-6gpu`](glm-5.3-flash-6gpu/README.md) | GLM-5.3-Flash (official FP8) as one TP2 x DP3 / EP6 replica with MTP on GPUs 0-5 | text + image, OpenAI tools, efforts low/high/max (default max), 1,048,576 tokens; replica pool only |
| [`qwen3.8-27b-dp8-8gpu`](qwen3.8-27b-dp8-8gpu/README.md) | Qwen TP1 x 8 replicas, one per card | one public model with OpenAI tool calling; Kairyu L2 is the replica pool only (even, prefix-aware placement) |
| [`deepseek-v4.1-flash-8gpu`](deepseek-v4.1-flash-8gpu/README.md) | DeepSeek-V4.1-Flash, one TP8 replica (GPU 0-7) | text + image, OpenAI tools, default thinking high; V4 ReplicaPool/API/UI structure |
| [`deepseek-v4.1-flash-6gpu`](deepseek-v4.1-flash-6gpu/README.md) | DeepSeek-V4.1-Flash, one DP6/EP6 replica (GPU 0-5) | the `deepseek-v4.1-flash-8gpu` L2/L3 structure on six GPUs; GPUs 6-7 unused |
| [`deepseek-v4.1-flash-1017-6gpu`](deepseek-v4.1-flash-1017-6gpu/README.md) | DeepSeek-V4.1-Flash-1017, one DP6/EP6 replica (GPU 0-5) | the V4.1 L2/L3 structure; vLLM v0.32.0 + SM120 overlay with the 1017 encoder rule; L1 selection in progress |
| [`deepseek-v4.1-flash-1017-8gpu`](deepseek-v4.1-flash-1017-8gpu/README.md) | DeepSeek-V4.1-Flash-1017, one DP8/EP8 replica (GPU 0-7) | the V4.1 L2/L3 structure; replica shape (DP8/EP8 or TP8/EP8) selection in progress |

All of them use Kairyu as L3, vLLM as L1 and Open WebUI as the public chat
surface, except `quyet-1.0-large-1gpu`, a System One decision model with no
chat: Kairyu publishes only `/v1/systemone`, and an example-owned adapter reads
the answers from an internal vLLM.
Every Open WebUI talks to Kairyu over `/v1/responses` (`api_type: responses`,
m11 D7 amendment 2026-10-17) through an example-owned filter that leaves only
Responses fields, and Kairyu forwards each request to vLLM in the API its
caller used (`upstream_chat_models`, `upstream_responses_models`,
`upstream_messages_models`); the Quyet replicas keep their System One wire. The
example gates call `/v1/responses` and `verify.sh api-equivalence` checks that
L1 and L3 answer the same Responses requests alike. GPU verification of this
change is pending in every one of these environments.
`qwen3.8-27b-dp8-8gpu` adds no orchestration: Kairyu L2 only spreads requests
over identical L1 replicas, and its `verify.sh` proves the per-replica split
and the OpenAI tool-calling agent contract (`tool-calling`).
Each `run.sh` command prints its API and Chat UI URLs when the stack is ready.
Qwen3.8-27B and GLM-5.3-Flash use the digest-pinned official vLLM v0.31.0
image, unpatched. Qwen moved there from v0.23.0 on 2026-10-18: v0.23.0
prefills a request that carries `extra_args` about 3x slower, and its
Responses serving always sets `extra_args={}`, so every Responses request
Kairyu forwarded prefilled slowly (see `qwen3.8-27b-1gpu/MEASUREMENTS.md`).
Every environment keeps persistent model, UI, and cache state on bind-backed
storage below `/mnt/nvme` (`qwen3.8-27b-dp8-8gpu` reuses `qwen3.8-27b-1gpu`'s
attested checkpoint).

Start everything and print the local Chat UI URL:

```sh
./examples/qwen3.8-27b-1gpu/run.sh
./examples/quyet-1.0-large-1gpu/run.sh
./examples/qwen3.8-deepseek-v4.1-8gpu/run.sh
./examples/qwen3.8-27b-dp8-8gpu/run.sh
./examples/glm-5.3-flash-6gpu/run.sh
```

Run serving verification through the Kairyu L3 endpoint:

```sh
./examples/qwen3.8-27b-1gpu/verify.sh serving
./examples/quyet-1.0-large-1gpu/verify.sh all
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh serving-auto-max
./examples/qwen3.8-27b-dp8-8gpu/verify.sh serving
./examples/glm-5.3-flash-6gpu/verify.sh serving
```

List the supported operations with `verify.sh list`. Model and product
evaluations are separate and are invoked explicitly through `python -m evals`;
see [the benchmark documentation](../docs/benchmarks.md).
