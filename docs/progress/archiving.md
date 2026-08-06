# PROGRESS.md archiving procedure

Run this when `scripts/check_progress_size.py` reports the size budget exceeded
(the SessionStart hook runs it automatically; see `.claude/rules/progress-log.md`
for the budget). Do it as its own commit, before other work updates `PROGRESS.md`.

1. Cut the oldest Change Log entries from `PROGRESS.md` until at most 5 entries
   remain, keeping each removed entry byte-for-byte verbatim.
2. Insert the removed entries into `docs/progress/archive/change-log.md` directly
   below the `ARCHIVE-INSERT-POINT` marker (above previously archived entries),
   preserving their order. The archive stays newest-first across the whole file.
3. If `## Current Status` or `## Product` is over budget, rewrite it more
   compactly in place (snapshots, not history — no archiving needed). If a
   verbose status is worth preserving, snapshot it to
   `docs/progress/archive/status-YYYY-MM-DD.md` first.
4. Verify with `python3 scripts/check_progress_size.py`, then commit as
   `docs(progress): archive old change log entries`.

Archive files are frozen history: never edit them except to insert trimmed
entries at the marker, and never delete or reword archived entries.
