# Kairyu L3 + vLLM L1 examples

`examples/` contains three complete environments:

| Environment | GPU layout | Model/context |
|---|---|---|
| [`qwen3.8-27b-1gpu`](qwen3.8-27b-1gpu/README.md) | one selected RTX PRO 6000 Blackwell | official FP8, 262,144 tokens |
| [`deepseek-v4-flash-0731-8gpu`](deepseek-v4-flash-0731-8gpu/README.md) | TP8 + EP8 on eight RTX PRO 6000 Blackwell cards | mixed FP4/FP8, 1,048,576 tokens |
| [`qwen3.8-deepseek-v4-8gpu`](qwen3.8-deepseek-v4-8gpu/README.md) | Qwen TP1 x 4 replicas + DeepSeek TP4/EP4 | verifier-gated Kairyu L2 role DAG |

All three use Kairyu as L3, vLLM as L1, and Open WebUI as the public chat surface.
Each `run.sh` command prints its API and Chat UI URLs when the stack is ready.
Qwen3.8 uses the digest-pinned official vLLM v0.23.0 image. Both DeepSeek
deployments share the same measured `aa0d513027` SM120 build, retaining
DSpark performance and checkpoint compatibility that v0.23.0 cannot provide.
Persistent model and UI state stays below `/mnt/nvme`.

Start everything and print the local Chat UI URL:

```sh
./examples/deepseek-v4-flash-0731-8gpu/run.sh
./examples/qwen3.8-27b-1gpu/run.sh
./examples/qwen3.8-deepseek-v4-8gpu/run.sh
```

Run serving verification through the Kairyu L3 endpoint:

```sh
./examples/deepseek-v4-flash-0731-8gpu/verify.sh serving
./examples/qwen3.8-27b-1gpu/verify.sh serving
./examples/qwen3.8-deepseek-v4-8gpu/verify.sh serving-auto-max
```

List the supported operations with `verify.sh list`. Model and product
evaluations are separate and are invoked explicitly through `python -m evals`;
see [the benchmark documentation](../docs/benchmarks.md).
