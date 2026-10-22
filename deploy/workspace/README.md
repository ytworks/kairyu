# L4 chat workspace

Every example's Chat UI: Open WebUI v0.12.3 (pinned by digest, not forked) in
front of the example's Kairyu, with Japanese knowledge bases (Docling OCR,
`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`), Word / Excel / PowerPoint through
the browser code interpreter, and download links for the files it writes.
L4 uses only Kairyu's public `/v1/responses`; design and rationale:
[`docs/design/l4-chat-workspace.md`](../../docs/design/l4-chat-workspace.md).

| Path | Content |
|---|---|
| `compose.yaml` | the `docling` and `chat-ui` services, started after an example's compose file (which defines `kairyu`) |
| `kairyu_workspace/` | standard library only: `load` checks a `workspace` section, `compose_env` fills `compose.yaml`, `fetch_models` fetches and attests the retrieval models, `setup` installs the plugins and checks the running Open WebUI |
| `profiles/ja-v2.json` | the pinned extraction, retrieval and code-interpreter settings and the gates' thresholds; immutable (a change is a new name) |
| `profiles/ja-v1.json` | `ja-v2` with 20 reranker candidates instead of 50; the 1017 eight-GPU example's PR A evidence |
| `plugins/responses_filter.py` | the one global filter: only Responses fields reach Kairyu; with effort levels, a Reasoning Effort dropdown |
| `plugins/file_download_tool.py` | `file_download`: stores a file the code interpreter wrote and returns a link that works on any device |

Gates: [`verification/workspace/gates.py`](../../verification/workspace/gates.py)
(`docling`, `knowledge`, `chat-ui`, `knowledge-serving`, `office`,
`file-download`), run by each example's `verify.sh`.

## An example's `workspace` section

```json
"workspace": {
  "profile": "ja-v2",
  "port": 3018,
  "bind": "127.0.0.1",
  "auth": "none",
  "models": ["deepseek-v4.1-flash-1017"],
  "default_params": {"max_tokens": 32768, "temperature": 1.0, "top_p": 1.0},
  "reasoning_effort": {"levels": ["default", "low", "high", "max"], "help": "..."},
  "cpus": {"chat_ui": 8, "docling": 4, "docling_workers": 2}
}
```

- `auth`: `none` (no sign-in; the one user is the admin) or `admin` (accounts;
  `WEBUI_ADMIN_EMAIL` / `WEBUI_ADMIN_PASSWORD` are required to set up).
- `models`: the public models the Chat UI offers; the first is the default.
- `reasoning_effort`: `null` for no dropdown, or levels starting with
  `default` (the model's own default applies).
- `cpus`: the Chat UI embeds and reranks on CPU, Docling OCRs on CPU; the caps
  keep both from slowing the example's L1. Each example's
  `knowledge-serving` gate proves its own caps.
- `CHAT_UI_PORT`, `CHAT_UI_BIND_ADDRESS` and `WEBUI_URL` override the port,
  the bind address and the URL Open WebUI prints, as before.

## Lifecycle (an example's `run.sh up`)

1. `load` the section at import (`control.py` refuses a wrong one).
2. `fetch_models` into `<storage root>/model-volumes/workspace-models/`: each
   model at its pinned revision, checked against its tree hash and attested
   (`VERIFY_MODEL=1` re-hashes). Every example shares these read-only copies.
3. `docker compose -f <example>/compose.yaml -f deploy/workspace/compose.yaml
   up --wait` with `compose_env` added to the environment.
4. `setup`: sign in as the admin, check Open WebUI is v0.12.3, keep Kairyu's
   connection on Responses, install the filter (removing the earlier
   per-example `reasoning_effort` / `responses_api` filters) and
   `file_download` (readable by every signed-in user, so a workspace with
   sign-in gives it to every account), then fail unless retrieval, the code
   interpreter and the models' default metadata equal the workspace.

The environment is the configuration on every start
(`RESET_CONFIG_ON_START`): Admin Panel changes do not survive a restart.
A knowledge base embedded with another model must be re-indexed after an
example moves to this workspace. A model's saved settings (Admin Panel >
Models > Advanced Params) must be Responses fields (m11 D7).

The CPU demo (`deploy/compose/docker-compose.webui.yaml`) uses only the setup,
with `"profile": null`: `python3 -m kairyu_workspace setup` in the Open WebUI
image installs the filter and checks the connection.
