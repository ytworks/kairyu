# Word, Excel and PowerPoint in the 1017 eight-GPU Chat UI

Status: **Approved 2026-10-22 (owner).**

Accepted scope (owner, 2026-10-22):

- The Chat UI (Open WebUI v0.12.3) of `examples/deepseek-v4.1-flash-1017-8gpu/`
  reads attached Word, Excel and PowerPoint files and edits or creates them,
  returning a file the user downloads. Speech is out of scope.
- Formats: docx, xlsx, pptx. Legacy doc / xls / ppt and conversion to PDF
  are out of scope (no LibreOffice).
- Only this example changes. `kairyu/`, the other examples, kairyu-bench and
  the L1 / L2 settings are not changed.
- Code execution is on by default in every chat (owner decision).

## Facts

1. Reading: attached docx / xlsx / pptx go through the Docling service added
   by PR #661 (they are among its input formats). Only PDF has been verified.
2. Editing: the Chat UI's code interpreter runs Pyodide in the browser and
   ships python-docx 1.2.0, python-pptx 1.0.2, openpyxl 3.1.5, xlsxwriter and
   pandas, loaded from the bundle when the code imports them (no download).
3. When the model calls the `execute_code` tool, the browser fetches the
   chat's attached originals from the server and writes them to
   `/mnt/uploads` before running the code (`executePythonAsWorker` in
   `+layout.svelte`). Files written there can be downloaded from the Files tab
   of Chat Controls (`PyodideFileNav`).
4. With file persistence off (current), Pyodide runs in a sandboxed iframe
   (`allow-scripts` only) that writes the attachments before executing and
   reads files back for download. One execution may take 60 s.
5. The tool is offered only while the chat's code interpreter feature is on;
   `DEFAULT_MODEL_METADATA` `defaultFeatureIds` turns it on by default.

Code runs only in the user's browser, never on the server, so the publicly
reachable, unauthenticated Chat UI does not gain server-side code execution
(why Open WebUI's Open Terminal integration is not used).

## Changes

1. `compose.yaml` (chat-ui), explicit rather than defaulted, values from
   `example.json` `webui` through `control.py` (as in PR #661):
   `ENABLE_CODE_INTERPRETER=true`, `CODE_INTERPRETER_ENGINE=pyodide`,
   `ENABLE_PYODIDE_FILE_PERSISTENCE=false` (the more isolated sandboxed
   iframe), `DEFAULT_MODEL_METADATA={"defaultFeatureIds":["code_interpreter"]}`.
2. Startup check in `control.py`, like `_check_chat_ui_retrieval`:
   `_check_chat_ui_code_interpreter` requires `/api/config` to report the code
   interpreter on with the pyodide engine and `/api/models` to list
   `code_interpreter` in this model's `defaultFeatureIds`; otherwise
   `run.sh up` fails.
3. Gate `office` (`verification.py` plus the example's Playwright script
   `webui-browser-smoke.mjs`): Japanese docx / xlsx / pptx fixtures, each
   with its own key fact, generated offline in the Docling image with
   python-docx / python-pptx / openpyxl (no new repository dependency).
4. Docs: README section (use: attach and ask, download from the Files tab;
   limits: computed in the user's browser, 60 s per execution, working files
   vanish on page reload, Excel formulas are not recalculated on save, legacy
   formats out of scope), MEASUREMENTS, a short addendum to the 1017
   amendment, one PROGRESS entry.
5. Tests: no pytest added or removed (configuration guarded at startup,
   behavior by the gates); base and head collection counts reported.

## Verification

Behavior only. Stop at the first failure, fix all known risks together,
re-run only the affected gates.

| # | Gate | Pass | Budget |
|---|---|---|---|
| W0 | Local CPU | ruff on changed files; the example's 15 unit tests; pytest collection base = head | 2 min |
| W1 | Restart | `run.sh up` with the external URL settings passes with the startup check (only the Chat UI is recreated, about 1 min down) | 5 min |
| W2 | `office` read | each of docx / xlsx / pptx attached with the clip and asked about in Japanese; the answer carries that file's key fact | 3 min each |
| W3 | `office` edit | in the same chat, a concrete edit (docx: replace a word and add a paragraph; xlsx: add a total row; pptx: add a "まとめ" slide); the result is downloaded from the Files tab and checked with the libraries in the Docling image: the change is there and the original content (table rows, existing slide titles, source data) is kept | 5 min each |
| W4 | Regression | `knowledge` (kept base) then `chat-ui` (323, 324, Red, knowledge answer) pass with the code interpreter on by default; Kairyu sees `/v1/responses` only | 5 min |

No GPU load is added (code runs in the browser, extraction is the existing
Docling), so the serving gates are not re-run. Afterwards: MEASUREMENTS,
PR status, push, CI, report. Leaving draft and merging are the owner's
decisions.
