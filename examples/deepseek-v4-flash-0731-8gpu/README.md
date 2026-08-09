# DeepSeek-V4-Flash-0731 / 8 GPU

The default topology is two independent EP4 + Attention-DP replicas on GPUs 0–3 and 4–7. The pinned mixed FP4/FP8 checkpoint is never requantized and the native 1,048,576-token context is never reduced. DSpark is off until its parity, accuracy and 5% goodput gates pass.

```sh
cp .env.example .env
./run.sh vllm
./run.sh kairyu
./bench.sh compare all
```

An EP8 topology lock is intentionally absent. It may be generated only after
the EP4/EP8 quality, 1M-context, stability, and 2% SLO-goodput gates complete.
The current native DeepSeek distributed worker is still an open GPU gate; see
the [frontier native runtime design](../../docs/design/frontier-native-runtime.md).
