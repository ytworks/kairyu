# DeepSeek V4.1 + Qwen3.8 critical ensemble

This new example configures DeepSeek V4.1 on GPUs 0–5 and two Qwen3.8-27B
FP8 TP1 replicas on GPUs 6–7. Four Qwen candidate roles share the two replicas.
The public model IDs are `kairyu-auto-max` and `embed-small`.

**Implementation candidate: full-model startup and live gates are pending.**
The existing V4 ensemble and V4.1 standalone examples remain unchanged.

## Configuration

- `compose.yaml`: DeepSeek TP2 / attention DP3 / EP6, CPU Engram offload,
  DSpark 5, native V4.1 parsers, and the two pinned Qwen replicas.
- `auto-max.yaml`: native conversation roles and the five routing profiles.
- `kairyu.yaml`: ReplicaPools, backend capabilities, public API and embedding.
- `example.json`: checkpoint/runtime identities, GPU allocation and gate settings.
- `vllm-tokenize.Dockerfile` and `tokenize-native-chat.patch`: a source-bound
  adaptation of the existing V4.1 runtime image. It materializes tokenize tool
  history with the same validators as native chat generation. No Python file
  or alternate inference runner is added to this example.

The parent runtime and checkpoint are reused by immutable identity. Qwen's
model/revision, template, medium effort mapping and sampling remain unchanged.
This example uses its own Compose project, configuration and evidence directory.

## Primary workflow

1. Non-thinking Qwen streams a bounded public opening where the caller permits it.
2. DeepSeek extracts Requirement JSON with R1-based IDs and an effort floor of
   high; an explicit max remains max. An independent DeepSeek candidate reads
   the original conversation directly.
3. DeepSeek writes four policies; four Qwen medium calls produce full candidates.
4. DeepSeek compares all five candidates and records concrete criticisms.
5. A separate DeepSeek call reconstructs the answer, followed by an independent
   DeepSeek audit. Each requested choice receives its own audit and up to two repairs.

Original role, tool, reasoning and image structure is retained. Public MAX is
separate from private generation limits and includes selected final reasoning
and its continuation. Required dependencies cannot silently disappear. The
configured step allowance is 19 plus 9 for every additional public choice.
The private 131072-token ceiling is provisional until live completion evidence.

The real serial MAX8/5-second Qwen judge selects among `primary`, `qwen_direct`,
`qwen_think_medium`, `deepseek_direct` and `deepseek_think`. Tool/format/multiple-
choice/logprob requests disable the public head as required by the shared contract.

## Operations and validation

Thin shell/Compose entrypoints are under implementation, reusing existing model
storage and shared serving/browser verification tools. Existing example helpers
are not modified. The runtime parent must match the recorded image ID before
building; a derived image ID must be recorded before deployment validation.

The 32-head quantized-arithmetic probe passes 16/16 at unchanged tolerance;
the original floating-reference failures remain recorded separately. Native
CPU rendering/processing equality passes 43 cases on each pinned renderer,
including tool history, effort, response formats, prefill and images.

Full-model startup, deployed-byte attestation, functional/long-input/choice/
cancellation/restart checks, agent turns and paired c1/c8/c16/c32 performance
matrices remain open. A primary row must retain the real judge and every required
stage, and use a fresh completed six-GPU native baseline. Lossless over-context
processing remains separate from per-dispatch MAX fitting.

See [MEASUREMENTS.md](MEASUREMENTS.md) and the
[implementation plan](../../docs/superpowers/plans/2026-09-13-v41-six-gpu-critical-ensemble.md).
