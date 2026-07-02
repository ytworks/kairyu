# Kairyu (rLLM) — Agent Instructions

Kairyu is a vLLM-compatible LLM inference framework with native orchestration,
layered as L3 Interface / L2 Orchestration / L1 Engines. Package: `kairyu/`.

## Session start

ALWAYS read `PROGRESS.md` first — it is the cross-session memory of design changes,
milestone status, and blockers.

## Progress log rules

Read and follow `.claude/rules/progress-log.md`. It defines when and how to update
`PROGRESS.md` (Current Status snapshot + append-only Change Log). These rules are
shared with Claude Code via `CLAUDE.md` — do not duplicate or diverge from them here.

## Where things live

- Design decisions and rationale (D-IDs, review amendments): `docs/design/m1..m4-*.md`
- GPU-day execution plan: `docs/gpu-runbook.md`
- Implementation plans: `docs/superpowers/plans/`
- Dev commands: `uv sync --extra dev`, `uv run pytest`, `uv run ruff check .`
