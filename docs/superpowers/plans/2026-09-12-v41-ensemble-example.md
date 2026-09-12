# V4.1 ensemble example implementation

Original user amendment: implement locally and defer GPU validation; create a
PR with a complete cross-machine handoff. On 2026-09-13 the user released all
eight GPUs and requested the remaining development/validation and fixes pushed
to that PR (#598). GPU work is now authorized and active; preserve unrelated
workloads and record exact evidence for the selected configuration.

## Contract

- Separate `qwen3.8-deepseek-v4.1-8gpu` example; DeepSeek uses GPUs 0–5 and
  Qwen3.8 27B FP8 has two TP1 replicas on GPUs 6 and 7.
- Keep the existing five-route judge and dual-track DAG. Reduce four policies
  to two, and synthesize two policy answers plus the independently refined draft.
- DeepSeek defaults to official high; normal thinking roles inherit caller effort.
  Requirements always uses DeepSeek high. Qwen's non-thinking and medium roles
  retain their existing settings. Audit remains Qwen with two refinement rounds.
- Port PR #595 (`31f1adc`) extraction, propagation and audit contracts, not its
  historical measurements or changes to the original example.
- Both models receive images directly. Remove the image-description stage and
  the old text-only DeepSeek scaffold/template path.
- Fixed-runtime source inspection selects a six-GPU candidate; startup,
  capacity, correctness and performance remain explicitly unverified until GPU runs.

## Tasks and ownership

1. Deployment: static runtime inspection, compose/deployment/metadata, CPU tests.
2. Orchestration: DAG, native role hooks, checklist/audit contracts, CPU tests.
3. Integration: lifecycle, verification commands, documentation, progress log.
4. Review all interfaces, run focused CPU/regression checks, create draft PR with
   remaining GPU gates and reproducible handoff instructions.
5. After hardware release, isolate startup/numerical/runtime defects with bounded
   experiments; review and pin corrections before native and composed replay.
6. Complete route/effort/image/tool/quality, context/capacity, cancellation and
   performance gates. Preserve failed trials, separate contract/model quality,
   update the PR with actual results and any remaining limitations; do not merge.

## Acceptance

CPU tests exercise actual DAG effort propagation, fixed Requirement effort,
multimodal forwarding, direct-route bypass, PASS/repair/exhaustion, strict role
hook matching, deployment inventory, lifecycle isolation and measurement
provenance. CPU checks alone do not establish model quality or six-GPU serving.
GPU conclusions must identify the exact runtime and workload. V41E-D5/D6 amend
the initial shared-framework scope with an opt-in public paragraph separator
and streaming-iterator ownership fixes required by observed GPU failures.
V41E-D7/D8 address native grammar escape and forced-termination defects; details
and preserved experiments are in the design record and MEASUREMENTS.

## GPU handoff

After hardware is free, attest the exact image/model/config, validate DeepSeek
alone without speculation, then Qwen co-residency. Exercise text/images/tools,
streaming/cancellation, explicit/default efforts, all five routes, Requirement
high, restart and concurrency; record TTFT/TPOT/token throughput and capacity.
Only investigate DSpark after the base configuration passes. Unsupported
topology or memory failures require a new evidence-backed L1 decision.
