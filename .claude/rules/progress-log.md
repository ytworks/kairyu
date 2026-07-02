# Progress Log Rules

Single source of truth for how `PROGRESS.md` (repo root) is maintained.
Both Claude Code (via `CLAUDE.md`) and Codex (via `AGENTS.md`) MUST follow these rules
so that every agent records design changes and progress identically.

## Purpose

`PROGRESS.md` is the cross-session memory of this project. A fresh agent session must be
able to read it and immediately know: where the project stands, what has been decided,
and what changed since the design docs were written.

## When to update PROGRESS.md

Update `PROGRESS.md` BEFORE committing whenever any of the following happens:

1. A design decision is added, changed, or dropped (anything touching the D-IDs in
   `docs/design/m*.md`, including review amendments).
2. A milestone (M1–M4) changes status (started, CPU-half done, GPU-validated, complete).
3. A significant implementation lands or an architecture-affecting change is made.
4. A blocker appears or is resolved (e.g., waiting on GPU hardware).

Routine refactors, typo fixes, and small doc edits do NOT require an entry.

## Structure of PROGRESS.md

Exactly two sections:

### `## Current Status`

A snapshot that is OVERWRITTEN in place to always reflect the present state:
per-milestone status, what currently works, and active blockers. Keep it short —
a table or bullet list, no history.

### `## Change Log`

APPEND-ONLY, newest entry first. Entry format:

```markdown
### YYYY-MM-DD — [design|progress|amendment] short headline
- What: what was decided / changed / completed
- Why: rationale (REQUIRED for design changes and amendments)
- Refs: D-IDs in docs/design/, commit hashes, related files
```

Entry types:
- `design` — a new design decision or a change to an existing one
- `progress` — milestone/implementation progress, blockers appearing or clearing
- `amendment` — changes resulting from a design review

## Hard rules

- NEVER rewrite or delete past Change Log entries. If an entry was wrong, append a
  correction entry that references it.
- Keep `Current Status` consistent with the `Status:` lines in `docs/design/m*.md`;
  if they diverge, fix both in the same commit.
- Write entries in English (matching the rest of `docs/`).
