# L4 chat workspace, PR B: the six other single-model chat examples

Status: **Part of the plan approved 2026-10-22 (owner),
`docs/superpowers/plans/2026-10-22-l4-chat-workspace.md` (PR B row and its
verification plan).** Stacked on PR A (#665).

## Scope

Every single-model chat example moves onto the L4 workspace and gains Japanese
OCR knowledge bases, Word / Excel / PowerPoint and download links:

| Example | Port / bind | Auth | Default params | Effort levels |
|---|---|---|---|---|
| `deepseek-v4.1-flash-1017-6gpu` | 3017 / 127.0.0.1 | none | max_tokens 32768, temperature 1.0, top_p 1.0 | default/low/high/max |
| `deepseek-v4.1-flash-6gpu` | 3008 / 127.0.0.1 | none | 32768, 1.0, top_p 0.95 | default/low/high/max |
| `deepseek-v4.1-flash-8gpu` | 3007 / 127.0.0.1 | none | 32768, 1.0, 1.0, reasoning_effort high | default/low/high/max |
| `glm-5.3-flash-6gpu` | 3016 / 0.0.0.0 | none | 131072, 1.0, 0.95 | default/low/high/max (default = max) |
| `qwen3.8-27b-1gpu` | 3000 / 127.0.0.1 | admin | 32768, 1.0, 0.95 | none |
| `qwen3.8-27b-dp8-8gpu` | 3000 / 127.0.0.1 | admin | 32768, 1.0, 0.95 | none |

Each keeps its current port, bind, auth and parameters (values in its new
`workspace` section), profile `ja-v1`, CPU caps Chat UI 8 / Docling 4.

## Changes

- Per example (the PR A pattern): `example.json` `webui` -> `workspace`;
  `compose.yaml` loses `chat-ui`; `control.py` loads the workspace at import,
  starts L4's compose file after its own, fetches the retrieval models and
  runs L4 setup instead of its own Chat UI helpers; its filter file goes;
  `verification.py` registers L4's gates with its own serving rows, and
  `api-equivalence` gains the `image-tools` case; README Chat UI sections.
- L4: the browser gates sign in for `auth: admin` workspaces (the admin
  credentials from `WEBUI_ADMIN_EMAIL` / `WEBUI_ADMIN_PASSWORD`);
  `knowledge-serving` measures its reference itself (the same rows unloaded,
  first, in the same run) when an example records none. The 1017 eight-GPU
  example keeps its recorded reference.
- Tests: the per-example filter cases and the "every example ships a filter"
  enumeration go (one L4 filter case with and without effort levels remains,
  plus the two PR C examples' filters until PR C); static Chat UI compose
  assertions in example tests go (the workspace loader enforces them).

## Verification plan

Stop at the first failing gate and bring the fix plan. GPU stage after review;
the running 1017 eight-GPU stack is stopped for it and restarted at the end.
Per example, one at a time:

| Gate | Pass | Budget |
|---|---|---|
| restart (`run.sh up`) | ready; L4 setup checks all match | 15-40 min (L1 start) |
| `knowledge` (kept) | key chunk in the top 5, query <= 10 s, scan <= 300 s | 5 min |
| `chat-ui` | 323, 324, Red, knowledge answer cited; `/v1/chat/completions` unchanged | 5 min |
| `office` | three edits verified from the reply's links; the image after the tool calls answered | 10 min |
| `file-download` | desktop, after reload (same SHA-256), phone | 15 min |
| `knowledge-serving` | Docling busy >= 95 % of the rows; c1/c8 >= 95 % of the same run's unloaded rows | 15 min |
| `api-equivalence` | every case incl. `image-tools` | 10 min |

About 1.5-2 h per example including startup, 10-12 h in all. `docling` is
model-independent and already passed on the profile (PR A).
