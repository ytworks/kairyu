# Progress Log Rules

Single source of truth for how `PROGRESS.md` (repo root) is maintained.
Both Claude Code (via `CLAUDE.md`) and Codex (via `AGENTS.md`) MUST follow these rules
so that every agent records design changes and progress identically.

## Purpose

`PROGRESS.md` is the cross-session memory of this project. A fresh agent session must be
able to read it and immediately know: what the product is aiming to be, where the
project stands, and what changed recently. It is loaded at every session start, so it
must stay small — detail lives in `docs/design/`, `bench/results/`, and the archive
(`docs/progress/archive/`), not here.

## When to update PROGRESS.md

Update `PROGRESS.md` BEFORE committing whenever any of the following happens:

1. A design decision is added, changed, or dropped (anything touching the D-IDs in
   `docs/design/m*.md`, including review amendments).
2. A milestone (M1–M4) changes status (started, CPU-half done, GPU-validated, complete).
3. A significant implementation lands or an architecture-affecting change is made.
4. A blocker appears or is resolved (e.g., waiting on GPU hardware).

Routine refactors, typo fixes, and small doc edits do NOT require an entry.

## Structure of PROGRESS.md

Exactly three sections:

### `## Product`

What Kairyu is aiming to be: the product goal, target hardware profiles, target model
classes, and the layering contract. This is a compact digest of `docs/roadmap.md` and
`docs/goals/` — update it only when the roadmap or goals themselves change, and keep it
under ~25 lines.

### `## Current Status`

A snapshot that is OVERWRITTEN in place to always reflect the present state:
per-milestone status, formal-gate status, what currently works, and active blockers.
Tables and one-line bullets only — no history, no per-run evidence (agreement counts,
hashes, artifact paths); point to `bench/results/` and `docs/design/` instead.
Keep it under ~80 lines.

### `## Change Log`

Newest entry first. Entry format:

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

Keep each entry under ~15 lines: What/Why state the decision and rationale, not the
full evidence chain — Refs point to the design doc, issue, or artifact that has it.

Only the most recent entries stay in `PROGRESS.md` (see the size budget below);
older entries live in `docs/progress/archive/change-log.md`.

## Size budget and archiving

`PROGRESS.md` has a hard size budget so it never again grows into a context sink:

- Total file: ≤ 300 lines.
- `## Change Log`: ≤ 20 entries.

The harness enforces this: a SessionStart hook (`.claude/settings.json`) runs
`scripts/check_progress_size.py` at every session start and injects a warning into the
session when the budget is exceeded. When you see that warning (or the script fails
when run manually), run the archiving procedure below in the same session, as its own
commit, BEFORE other work updates `PROGRESS.md`.

### Archiving procedure

1. Cut the oldest Change Log entries from `PROGRESS.md` until at most 10 entries
   remain, keeping each removed entry byte-for-byte verbatim.
2. Insert the removed entries into `docs/progress/archive/change-log.md` directly
   below the `ARCHIVE-INSERT-POINT` marker (above the previously archived entries),
   preserving their order. The archive stays newest-first across the whole file.
3. If `## Current Status` or `## Product` is over budget, rewrite it more compactly
   in place (they are snapshots, not history — no archiving needed). If a verbose
   status is worth preserving, snapshot it to
   `docs/progress/archive/status-YYYY-MM-DD.md` first.
4. Verify with `python3 scripts/check_progress_size.py` and commit the trim as its
   own commit (message: `docs(progress): archive old change log entries`).

## Hard rules

- NEVER rewrite or delete past Change Log entries, in `PROGRESS.md` or in the archive.
  Moving entries verbatim to the archive per the procedure above is the ONLY allowed
  relocation. If an entry was wrong, append a correction entry that references it.
- NEVER edit archived files except to insert trimmed entries at the marker.
- Keep `Current Status` consistent with the `Status:` lines in `docs/design/m*.md`;
  if they diverge, fix both in the same commit.
- Write entries in English (matching the rest of `docs/`).
