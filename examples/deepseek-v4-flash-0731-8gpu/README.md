# DeepSeek-V4-Flash-0731 / 8 GPU

The default topology is two independent EP4 + Attention-DP replicas on GPUs 0–3 and 4–7. The pinned mixed FP4/FP8 checkpoint is never requantized and the native 1,048,576-token context is never reduced. DSpark is off until its parity, accuracy and 5% goodput gates pass.

```sh
./run.sh vllm
./run.sh kairyu
./bench.sh compare all
```

Configuration is inherited only from the invoking process environment; dotenv
files are not read. Export `HF_TOKEN` when Hugging Face authentication is
required and optionally set an absolute `MODEL_STORAGE_ROOT` for bind-backed
model volumes. The external benchmark client uses concurrency 1; replica and
Attention-DP parallelism remain an internal deployment concern.

An EP8 topology lock is intentionally absent. It may be generated only after
the EP4/EP8 quality, 1M-context, stability, and 2% SLO-goodput gates complete.

The committed native 1M gate starts the same EP4/Attention-DP example, runs the
exact tokenizer-attested 1,048,576-token NIAH row, and finalizes its ordinary
JSON/Markdown evidence even on failure:

```sh
../../scripts/gpu_gates/deepseek_v4_native_1m.sh
../../scripts/gpu_gates/deepseek_v4_native_1m.sh ep8
```

The packed-FP4 and two-rank NCCL smoke tests are
`tests/gpu/test_deepseek_v4_{fp4,ep}_gpu.py`. Full-checkpoint gate completion is
recorded only after the script above actually finishes on the pinned model.
