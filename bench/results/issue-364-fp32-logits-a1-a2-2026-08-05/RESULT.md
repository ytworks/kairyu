# Issue #364 FP32 final-logits experiment result

Issue #364 closes with a valid negative experiment. The paired artifact keeps
its formal `verdict=FAIL`; independent replay establishes
`evidence_valid=true`, while the unchanged A1/A2 readiness rules establish
`feature_ready=false`. The quality classification is `mixed`, the disposition
is `withdrawn`, and no public `logits_dtype` option is shipped.

- Measurement commit: `ac589fb67452173f45f23d9107af313c2b79cc17`
- Analysis commit: recorded separately in the paired artifact's `analysis_code`
- A1 reference self-agreement floor: 1010/1024 (98.6328125%)
- A2 reference self-agreement floor: 1004/1024 (98.046875%)
- Integrity: every cell retains 1024/1024 raw positions, zero substantive
  disagreements, zero missing logprob observations, and one shared reference
  per gate
- Safety: every direct A2 TP4/TP8-versus-TP2 comparison passes, but the
  per-degree HF-relative readiness floor still fails where shown below

| Gate | Arm | TP1 | TP2 | TP4 | TP8 | Formal verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A1 | `model` | 1017/1024 | 1013/1024 | — | — | PASS |
| A1 | `float32` | 1015/1024 | 1013/1024 | — | — | PASS |
| A2 | `model` | — | 1009/1024 | 1004/1024 | 999/1024 | FAIL |
| A2 | `float32` | — | 1008/1024 | 1003/1024 | 1002/1024 | FAIL |

The paired deltas (`float32 - model`) are -2 and 0 at A1 TP1/TP2, then
-1, -1, and +3 at A2 TP2/TP4/TP8. The treatment therefore moves different
cells in different directions and does not lift either failing A2 TP4/TP8 cell
to the shared floor. Thresholds and historical A1/A2 artifacts were not changed.

## Main control

An isolated checkout of `main@6cff10f9f39c5d114d9a21875ea0c6e460d4cf32`
ran the same A2 TP8 64 x 16 teacher-forced workload against the exact shared
reference (SHA-256
`2f0649085c7261573136f883012d847d38fb88f4a6b61fc21e33897eb52ed18a`).
It also measured 999/1024. Its 1024 engine tokens, engine logprobs, and reference
rows are element-for-element equal to the experimental `model` arm. This shows
that the model-arm TP8 failure is present on the pre-experiment baseline; it
does not turn either A2 verdict into a pass and does not rescue the `float32`
treatment.

## Retained artifacts

| Artifact | SHA-256 |
| --- | --- |
| `a1-model-formal.json` | `860213e5d1bf44d47b14a15b97e2ad28ea8794845503a30eed69fc53e94755b9` |
| `a1-float32-formal.json` | `1cc3a20be0e304eb99f2448a0f901cd70ddb96be818fe4077783d1c89f3979f8` |
| `a2-model-formal.json` | `febc8f62605e6744989115e4c00064ce241e0ef702440b4b7b85c685874c0195` |
| `a2-float32-formal.json` | `aeb9acc84fa0fa8b9ac8d945de2231c86efb5e35499c80f5ed2b59a85e7c126a` |
| `a2-main-6cff10f-tp8-control.json` | `16a3840d842dfdf237c9361388206280f37fe2ef9801660dd4519b94135c304d` |
| `issue-364-fp32-logits-a1-a2-2026-08-05.json` | `773aab8d70672c6e5b2fc0498e0c29c0ac6cfdcce596d852a8fbe5b407a20663` |

The four formal artifacts are self-contained: each embeds its complete raw
measurement inputs in addition to source-file digests. The main control is
diagnostic evidence and is not used to alter the formal gate result.

## Replay

From the repository root, replay evidence integrity without requiring the
withdrawn feature to pass its readiness gate:

```bash
uv run python bench/gate_logits_dtype.py \
  --a1-model bench/results/issue-364-fp32-logits-a1-a2-2026-08-05/a1-model-formal.json \
  --a1-float32 bench/results/issue-364-fp32-logits-a1-a2-2026-08-05/a1-float32-formal.json \
  --a2-model bench/results/issue-364-fp32-logits-a1-a2-2026-08-05/a2-model-formal.json \
  --a2-float32 bench/results/issue-364-fp32-logits-a1-a2-2026-08-05/a2-float32-formal.json \
  --out /tmp/issue-364-negative-replay.json
```

That command exits zero only when the retained inputs, provenance, raw rows,
stored formal verdicts, and independent recomputation agree. Adding
`--assert-feature-ready` exits nonzero for this result because both A2 formal
artifacts remain `FAIL`.
