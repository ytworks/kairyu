# Kairyu + vLLM / DeepSeek-V4-Flash-0731

`examples/` contains exactly one complete environment:

- [`deepseek-v4-flash-0731-8gpu`](deepseek-v4-flash-0731-8gpu/README.md)
- 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB)
- L3: Kairyu; L1: vLLM; UI: Open WebUI
- exact `deepseek-ai/DeepSeek-V4-Flash-0731` revision and native 1,048,576-token context

Start everything and print the local Chat UI URL:

```sh
./examples/deepseek-v4-flash-0731-8gpu/run.sh
```

Run measured serving performance, the complete 1,055-problem LiveCodeBench, or
both:

```sh
./examples/deepseek-v4-flash-0731-8gpu/bench.sh serving
./examples/deepseek-v4-flash-0731-8gpu/bench.sh livecodebench
./examples/deepseek-v4-flash-0731-8gpu/bench.sh all
```
