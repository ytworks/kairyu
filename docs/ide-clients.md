# IDE coding agents

Kairyu can be configured as an OpenAI-compatible model provider for coding
agents embedded in an editor. The supported baseline is:

- `GET /v1/models`
- streamed and non-streamed `POST /v1/chat/completions`
- function tools with `tool_choice`
- `parallel_tool_calls`; `false` is enforced per generated choice
- optional `POST /v1/responses` (required by Codex CLI, below)

Tool-bearing Chat Completions streams are validated before any SSE bytes are
sent. This preserves a truthful `tool_choice` and `parallel_tool_calls`
contract, but it means the first structured tool-call chunk arrives only after
generation finishes.

## Cline

Cline is an active coding agent distributed as editor extensions, a CLI, an
SDK, and a Kanban surface. It is not a standalone IDE. For an editor connection
test, use the VS Code/Cursor extension and select:

```text
API Provider: OpenAI Compatible
Base URL:     http://<kairyu-host>:<port>/v1
API Key:      any non-empty value when Kairyu authentication is disabled
Model ID:     <an ID returned by /v1/models>
```

For a local or self-hosted model, open Cline's advanced model configuration
and set `Max Output Tokens` explicitly. Kairyu's omitted-request default is 16,
which is intentionally conservative but too small for most Cline XML tool
calls; the gpu02 compatibility check uses 4096.

Cline's VS Code extension follows VS Code's proxy handling. Its CLI and
JetBrains plugin document HTTP proxy support only, so a SOCKS-only validation
must use the VS Code/Cursor extension. Treat that as a client/environment gate,
not as guaranteed SOCKS support: in a 2026-07-30 macOS test with Cline 4.0.12,
the VS Code extension host detected the operating-system SOCKS5 proxy but
Cline's OpenAI SDK returned `Connection error` before
`POST /v1/chat/completions` reached Kairyu. The same tunnel returned 200 for
`curl --socks5-hostname ... /v1/models`. An HTTP proxy supported by the editor,
or an SSH local port forward for diagnosis, avoids relying on the extension
host's experimental SOCKS path.

A 2026-07-31 follow-up used
`ssh -L 127.0.0.1:18002:127.0.0.1:8002 ...` and configured Cline with
`http://127.0.0.1:18002/v1`. Cline reached `/v1/models` and issued three
`POST /v1/chat/completions` requests; Kairyu returned 200 for all three. The
Llama-3.1-8B deployment then emitted plain reasoning text instead of Cline's
required Plan-mode XML tool response, and Cline stopped at its three-mistake
limit. This proves the prior failure was in the editor's SOCKS transport path;
the remaining end-to-end blocker is model/agent-protocol readiness, not
firewall or API reachability.

A 2026-07-31 protocol follow-up resolved that blocker for the existing
Llama-3.1-8B deployment. Kairyu now auto-loads the Hugging Face tokenizer's
chat template and special-token metadata from the configured local model path,
and recognizes Llama 3.1's bare
`{"name": "...", "parameters": {...}}` function-call form. With the official
model template and Cline's output limit set to 4096, a Plan-mode response
completed in one request. An Act-mode task then emitted `read_file`, consumed
the file result, emitted `attempt_completion`, and reached `Task Completed`;
both chat-completions requests returned 200.

For an attested Llama-native tool template, Kairyu rejects the request-level
`builtin_tools`, `custom_tools`, and `tools_in_user_message` template variables.
Those variables select alternate tokenizer-template branches whose output
syntax is outside this parser contract; configure a separate reviewed model
template instead of changing that branch per request.

Cline 4.0.12's global `Native Tool Call` switch is not sufficient by itself:
its model-family matcher enables native calls only for selected model IDs, and
an unrecognized Llama ID continues to use Cline's XML protocol. Kairyu supports
both paths: Cline XML remains assistant content for Cline to parse, while
OpenAI `tools` requests receive structured `tool_calls`. Do not rename a model
to impersonate an allow-listed family. For higher agent quality, Cline's local
model guide recommends Qwen3 Coder 30B; validate the checkpoint and its own
tokenizer template before making it the production default.

That Qwen path was validated on gpu02 on 2026-08-04. The 16-shard
`Qwen3-Coder-30B-A3B-Instruct` checkpoint was downloaded directly to
`/models`, avoiding a second local copy or tar archive, and passed Kairyu's
checkpoint validation. The isolated TP1 replica uses
`deploy/ide-client/qwen3-coder-gpu-replica.yaml` on GPU 2 / host port 8003.
Qwen3-Coder emits its native
`<function=name><parameter=name>...</parameter></function>` XML inside
`<tool_call>` rather than JSON; Kairyu parses that form and converts parameter
values using the declared top-level property types, then enforces required
properties and strict additional-property boundaries. This is not a general
JSON Schema evaluator. A live required-tool API test completed the two-request
`read_file` → `attempt_completion` loop through an SSH local forward. After
setting a 65,536-token context window and 4,096
output tokens, a fresh Cline 4.1.3 Act-mode task also read `README.md` once and
reached `Task Completed` with the exact first line. The two corresponding
chat-completions requests returned 200, establishing the Qwen Cline UI loop as
well as the direct model/API protocol path.

For a native local model deployment, keep the checkpoint's tokenizer artifacts
beside the model and let Kairyu resolve them from `model_path` instead of
selecting the legacy role-prefix renderer:

```yaml
engines:
  llama-3.1-8b-instruct:
    backend: kairyu
    options:
      model_path: /models/llama-3.1-8b-instruct/current
```

## Continue

Continue is an agent available as a VS Code extension, JetBrains plugin, and
`cn` CLI; it is not a standalone IDE. Confirm the configuration schema against
the installed client version because the project remains under active
development.

An OpenAI-compatible model entry uses:

```yaml
name: Kairyu
version: 0.0.1
schema: v1

models:
  - name: Kairyu
    provider: openai
    model: <model-id>
    apiBase: http://<kairyu-host>:<port>/v1
    apiKey: unused
    useResponsesApi: false
    capabilities:
      - tool_use
    roles:
      - chat
      - edit
      - apply
```

Do not assign an embedding or autocomplete role until the selected Kairyu
deployment provides a production embedding backend and the required infill
contract. The built-in mock embedding backend is not suitable for semantic
code retrieval, and legacy completion `suffix` infill is not implemented.
The two IDE example deployments in this guide are text-only; Kairyu's separate
remote-VLM multimodal boundary does not by itself establish that an IDE's image
workflow is compatible with either example.

## Codex CLI

Codex speaks only the Responses API (`wire_api = "responses"`). The supported
setup is a custom provider plus a model catalog generated from the deployment;
`python -m scripts.codex_gate.run_matrix` gates this configuration against
real Codex binaries (pinned rust-v0.160.0; m20 D22). Server-side behavior and
the live gate are in `docs/deployment.md` (Responses API and Codex).

`~/.codex/config.toml`:

```toml
model = "qwen3-32b"                  # an ID returned by /v1/models
model_provider = "kairyu"
model_catalog_json = "/home/me/.codex/kairyu-models.json"
review_model = "qwen3-32b"           # auxiliary slugs, see below

[model_providers.kairyu]
name = "Kairyu"
base_url = "http://<kairyu-host>:<port>/v1"
env_key = "KAIRYU_API_KEY"           # any non-empty value without Kairyu auth
wire_api = "responses"
stream_idle_timeout_ms = 300000      # Codex's default; Kairyu sends a data
                                     # heartbeat after 15 s without events
```

Keep Codex's default `stream_max_retries` and `request_max_retries`.

**Model catalog.** Without a catalog Codex warns that model metadata was not
found and assumes a 272,000-token context window, image input and reasoning
summaries: a smaller model overflows before Codex compacts, and an attached
image gets a 400. Generate the catalog from the running deployment and
regenerate it when served models or `max_model_len` change:

```sh
python -m scripts.codex_model_catalog --base-url http://<kairyu-host>:<port>/v1 \
  --output ~/.codex/kairyu-models.json
```

Each `/v1/models` entry with a `max_model_len` becomes a Codex `ModelInfo`:
`context_window` is `max_model_len` and auto-compaction starts at 90 % of it.
Input stays text-only until `/v1/responses` accepts images, so Codex drops an
attachment instead of sending it; the freeform `apply_patch` tool and
reasoning summaries stay off until Kairyu executes them. Codex sends no
reasoning effort unless you list the levels the model honors:
`--reasoning-levels qwen3-32b=low,high --default-reasoning-level
qwen3-32b=high`. Other options: `--model ID` (restrict), `--context-window
ID=N` (models without `max_model_len`), `--vision ID`, `--static FILE` (a YAML
list instead of a server) and `--base-instructions FILE` (otherwise Codex's own
base instructions are fetched from its pinned tag). A catalog in use shows no
"Model metadata ... not found" warning.

**Auxiliary model slugs.** Some Codex features request their own OpenAI model
IDs, which Kairyu answers with 404 `model_not_found`: set `review_model`
(`/review`) and, with memories enabled, `[memories] extract_model` and
`consolidation_model` to a served ID, and keep the default
`approvals_reviewer` (the automatic reviewer uses `codex-auto-review` or
`gpt-5.6-luna`). Do not serve IDs that start with a Codex bundled slug such as
`gpt-5.5` or `gpt-6-`: without a catalog Codex would apply that slug's
responses-lite wire profile.

**Hosted web search.** Codex declares `web_search` on every request (live
under `--dangerously-bypass-approvals-and-sandbox` or `danger-full-access`).
Kairyu accepts every mode but runs no search; set `web_search = "disabled"` to
drop the declaration.

**Compaction.** With a custom provider Codex 0.160 compacts locally: an
ordinary request summarizes the history, and on an in-band
`context_length_exceeded` Codex drops the oldest items and retries. Remote
compaction (`compaction_trigger`, a sealed `compaction` item) is used only by
the built-in `openai` provider and Azure-named providers; the 0.162
pre-release adds `capabilities = { remote_compaction = "v2" }` (and
`external_web_access = false`) for custom providers. Gateways serving remote
compaction need `server.responses_compaction_secret_env`.

**Harbor/Terminal-Bench shape.** Repointing the built-in provider also works:
`openai_base_url = "http://<kairyu-host>:<port>/v1"` with the key in
`OPENAI_API_KEY` (plus `model` and `model_catalog_json`). Codex first tries a
WebSocket upgrade, gets 426 and continues over HTTPS.

**Local-provider shape.** `CODEX_OSS_BASE_URL=http://<kairyu-host>:<port>/v1
codex --oss --local-provider lmstudio -m <model-id>` also works: Codex lists
`/v1/models`, loads the model with a one-token request and shows raw
reasoning.

## SOCKS validation

When the Kairyu port is firewalled and an SSH endpoint is available, open a
local dynamic forward and configure the operating system or VS Code to use it:

```sh
ssh -ND 127.0.0.1:1080 <gpu-host>
curl --socks5-hostname localhost:1080 http://<kairyu-host>:<port>/v1/models
```

Keep the serving port closed to the public network. The IDE request should
traverse the SOCKS tunnel; a direct request from an untrusted network should
remain unreachable. Verify both the editor result and Kairyu access logs: a
successful `/v1/models` curl proves the tunnel, but it does not prove that an
editor extension's Node.js HTTP client uses that tunnel.

For a transport-independent editor check, forward the serving port directly:

```sh
ssh -N -L 127.0.0.1:18002:127.0.0.1:<kairyu-host-port> <gpu-host>
curl http://127.0.0.1:18002/v1/models
```

Then use `http://127.0.0.1:18002/v1` as the editor provider's Base URL and
confirm both the editor outcome and the server-side request log.

The example Docker commands publish Kairyu on the host loopback interface.
Publishing on a non-loopback address requires an authentication configuration
and a host firewall rule that limits access to trusted clients.
