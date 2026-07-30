# IDE coding agents

Kairyu can be configured as an OpenAI-compatible model provider for coding
agents embedded in an editor. The supported baseline is:

- `GET /v1/models`
- streamed and non-streamed `POST /v1/chat/completions`
- function tools with `tool_choice`
- `parallel_tool_calls`; `false` is enforced per generated choice
- optional `POST /v1/responses`

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

## Continue

Continue is an agent available as a VS Code extension, JetBrains plugin, and
`cn` CLI; it is not a standalone IDE. Upstream published 2.0.0 as the final
release and made the source repository read-only, so it is useful as a
compatibility target but should not be the primary long-term integration.

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
code retrieval, image input is currently rejected, and legacy completion
`suffix` infill is not implemented.

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
