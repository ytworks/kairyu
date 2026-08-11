# Kairyu L3 + vLLM L1 examples

`examples/` contains two complete environments:

| Environment | GPU layout | Model/context |
|---|---|---|
| [`qwen3.6-27b-1gpu`](qwen3.6-27b-1gpu/README.md) | one selected RTX PRO 6000 Blackwell | official FP8, 262,144 tokens |
| [`deepseek-v4-flash-0731-8gpu`](deepseek-v4-flash-0731-8gpu/README.md) | TP8 + EP8 on eight RTX PRO 6000 Blackwell cards | mixed FP4/FP8, 1,048,576 tokens |

Both use Kairyu as L3, vLLM as L1, and Open WebUI as the public chat surface.
Each `run.sh` command prints its API and Chat UI URLs when the stack is ready.
The Qwen example keeps all model and UI state below `/mnt/nvme`.

Start everything and print the local Chat UI URL:

```sh
./examples/deepseek-v4-flash-0731-8gpu/run.sh
./examples/qwen3.6-27b-1gpu/run.sh
```

Run measured serving performance, the complete 1,055-problem LiveCodeBench, or
both:

```sh
./examples/deepseek-v4-flash-0731-8gpu/bench.sh serving
./examples/deepseek-v4-flash-0731-8gpu/bench.sh livecodebench
./examples/deepseek-v4-flash-0731-8gpu/bench.sh all
./examples/qwen3.6-27b-1gpu/bench.sh livecodebench  # exactly 20 items
```
