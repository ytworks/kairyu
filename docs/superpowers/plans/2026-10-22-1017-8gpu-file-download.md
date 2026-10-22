# Downloadable files from the 1017 eight-GPU Chat UI

Status: **Approved 2026-10-22 (owner).**

## Problem

The owner asked the Chat UI of `examples/deepseek-v4.1-flash-1017-8gpu/`
(on a phone, 2026-10-22 18:14 JST) for a one-year business plan template in
Excel. The file was made but could not be downloaded:

1. The model wrote a ChatGPT-style `sandbox:/mnt/uploads/...xlsx` link, which
   does nothing in Open WebUI.
2. Asked how to download, the model, which cannot see the screen, described
   buttons and folders that do not exist; the owner answered "all lies".
3. The only real exit is Chat Controls > Files (`PyodideFileNav`), hard to
   find on a phone, and the working files live only in the browser page: a
   reload loses them.
4. The model probed `importlib.import_module("openpyxl")`. Open WebUI loads
   a bundled package only when the code says `import openpyxl`, so the
   libraries looked missing and the model wrote the xlsx XML by hand.

Open WebUI v0.12.3 stores only PNG images found in code output and replaces
them with links; docx / xlsx / pptx have no such path.

Goal: a file the model creates or edits reaches the user as a real link (and
an attachment) in its reply, which works on a phone, after a reload and from
another device.

## Changes (example only; `kairyu/`, L1 / L2 and other examples unchanged)

1. Open WebUI Tool `file_download` (`webui-file-download-tool.py`),
   installed by `control.py` like the Reasoning Effort filter.
   `provide_download(path)`:
   - accepts only paths under `/mnt/uploads/`;
   - reads the file in the user's browser Pyodide through `__event_call__`
     (`execute:python`, the code interpreter's own workspace), in 4 MB
     base64 chunks within the 16 MB websocket buffer, never through the
     model's context;
   - stores it in Open WebUI's file store (`upload_file_handler`, Open
     WebUI's size limit applies);
   - attaches it to the reply (`files` event, kept in chat history);
   - returns `[name](/api/v1/files/{id}/content?attachment=true)` for the
     model to show as written.
2. The Tool's description, which the model reads, says: always deliver
   files through it; never write `sandbox:` links or paths as links, never
   describe UI steps; python-docx, python-pptx, openpyxl, xlsxwriter and
   pandas are available and load on a plain `import` at the top of the code
   (an `importlib` probe reports them missing).
3. On by default: `DEFAULT_MODEL_METADATA` `toolIds: [file_download]`, so
   new chats start with it selected; values in `example.json`.
4. Startup check in `control.py`: the Tool is installed with this file's
   content and listed in the model's default `toolIds`; otherwise
   `run.sh up` fails.
5. Chat Controls > Files keeps working.
6. Docs: README (download from the link in the reply), MEASUREMENTS, 1017
   design addendum, PROGRESS. No pytest added (configuration guarded at
   startup, behavior by the gates); base / head collection counts reported.

## Verification

Behavior only. Stop at the first failure, fix all known risks together,
re-run only the affected gates.

| # | Check | Pass | Budget |
|---|---|---|---|
| D0 | Local CPU | ruff; the Tool compiles; the example's 15 unit tests; pytest collection base = head | 2 min |
| D1 | Restart | `run.sh up` installs the Tool and passes the startup check (only the Chat UI is recreated, about 1 min down) | 5 min |
| D2 | Create, desktop | the owner's request ("一年の事業計画のフォーマットをエクセルで作って") gets a reply with a link (no `sandbox:`); clicking it saves an xlsx that openpyxl opens, with sheets and header cells | 3 min |
| D3 | Edit (`office`) | the docx / xlsx / pptx edits are saved from the link in the reply instead of Chat Controls and pass the same content checks | 5 min |
| D4 | Phone | the D2 request in Chromium emulating a phone screen and touch; tapping the link saves a valid xlsx | 3 min |
| D5 | After reload | after reloading and reopening the chat from history, the link saves the same file | 2 min |
| D6 | Regression | `chat-ui` (323, 324, Red, knowledge answer) and `knowledge` pass; Kairyu sees `/v1/responses` only | 5 min |

No GPU load is added, so the serving gates are not re-run. Real iPhone /
Android devices are not available here (D4 is Chromium emulation).
Afterwards: PR status, push, CI, report. Merging is the owner's decision.
