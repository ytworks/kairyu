# Progress Log Rules

How `PROGRESS.md` (repo root) is maintained. Shared by Claude Code (`CLAUDE.md`)
and Codex (`AGENTS.md`). This file is loaded every session — keep it, and
`PROGRESS.md`, token-minimal: detail belongs in `docs/design/`, `bench/results/`,
and `docs/progress/archive/`.

## When to update PROGRESS.md (BEFORE committing)

1. A design decision is added, changed, or dropped (D-IDs in `docs/design/m*.md`,
   including review amendments).
2. A milestone changes status.
3. A significant implementation or architecture-affecting change lands.
4. A blocker appears or is resolved.

Routine refactors, typo fixes, and small doc edits do NOT require an entry.

## Structure — exactly three sections

- `## Product` — target product shape, digest of `docs/roadmap.md` + `docs/goals/`.
  Update only when roadmap/goals change. ≤25 lines.
- `## Current Status` — snapshot OVERWRITTEN in place: milestone/gate status, what
  works, blockers. No history, no per-run evidence (point to `bench/results/`,
  `docs/design/`). ≤80 lines.
- `## Change Log` — newest first. Entry format:

```markdown
### YYYY-MM-DD — [design|progress|amendment] short headline
- What: what was decided / changed / completed (compressed — Refs hold the detail)
- Why: rationale (REQUIRED for design changes and amendments)
- Refs: D-IDs, issues/PRs, files
```

Entries ≤10 lines each. Only recent entries stay here; older ones move to
`docs/progress/archive/change-log.md`.

## Size budget (harness-enforced)

`PROGRESS.md`: ≤200 lines total, ≤10 Change Log entries. A SessionStart hook
(`.claude/settings.json` → `scripts/check_progress_size.py`) injects a warning
when the budget is exceeded: then follow `docs/progress/archiving.md` — as its
own commit, before other work touches `PROGRESS.md`.

## Hard rules

- NEVER rewrite or delete past Change Log entries, here or in the archive.
  Verbatim relocation to the archive (per `docs/progress/archiving.md`) is the
  ONLY allowed move; corrections are new entries referencing the old one.
- Keep `Current Status` consistent with `Status:` lines in `docs/design/m*.md`;
  if they diverge, fix both in the same commit.
- Write entries in English.
