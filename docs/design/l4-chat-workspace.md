# L4 Design: Chat Workspace

Status: **Accepted 2026-10-22 (owner)**; PR A (L4 and the 1017 eight-GPU
example, PR #665) GPU-verified 2026-10-22: every Chat UI gate passes on L4
(the example's `MEASUREMENTS.md`). Plan:
`docs/superpowers/plans/2026-10-22-l4-chat-workspace.md`.

## Context

The 1017 eight-GPU example's Chat UI gained Japanese knowledge bases (PR
#661), Word / Excel / PowerPoint through the browser code interpreter (PR
#662) and download links (PR #664) (frontier-native-runtime.md, 1017
addenda). The other chat examples had none of them, and each carried its own
copy of the same Open WebUI plumbing. The owner asked for every feature in
every example, shared as a framework layer.

## Decisions

- **WS-D1 Layer.** L4 Workspace is the Chat UI and its document services
  (OCR, embedding, reranking, the browser code interpreter, downloads). It
  runs outside the Kairyu server and uses only L3's public API
  (`/v1/responses`); it never reaches L1/L2 and works for any model behind L3.
  It lives in `deploy/workspace/` (compose services, the standard-library
  `kairyu_workspace` setup, plugins, profiles) and `verification/workspace/`
  (its gates). The `kairyu/` package (L1-L3) does not change: nothing L3
  exposes is missing, example launchers run on the host's standard-library
  Python and cannot import `kairyu`, and the plugins are Open WebUI code.
  Open WebUI is not forked: L4 uses its environment configuration and its
  plugin API only (m11 D7).
- **WS-D2 Ownership.** L4 owns the mechanism: the `docling` and `chat-ui`
  services, the setup, the one Responses filter (effort levels optional),
  the `file_download` tool, and immutable named profiles. An example's
  `workspace` section in `example.json` selects a profile and owns its model
  ids, default params, effort levels, port, bind, auth (`none` or `admin`)
  and CPU caps; `kairyu_workspace.load` checks it when `control.py` is
  imported. The exception to "each example owns its scripts" is the owner's
  decision of 2026-10-22; L1 scripts, overlays and other gates stay per
  example.
- **WS-D3 Profiles.** `profiles/ja-v1.json` holds the settings measured on
  the 1017 eight-GPU example: Docling (pinned docling-serve-cpu v1.37.0,
  RapidOCR `iso:ja`), `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` (pinned
  revision and tree hash), chunks 300/30, BM25 weight 0 (Open WebUI's BM25
  splits on whitespace and cannot tokenize Japanese), top_k 20/5, the code
  interpreter (Pyodide, no file persistence, on by default, `file_download`
  by default) and the gates' thresholds. A profile is never edited; a change
  is a new name. An example must name one, so L4 never selects a policy.
- **WS-D4 One Open WebUI.** L4 supports exactly Open WebUI v0.12.3 (pinned
  by digest): the plugins use its internals. Setup signs in as the admin,
  checks `/api/version`, keeps Kairyu's connection on Responses, installs the
  filter (`kairyu_responses`, removing the examples' and the demo's earlier
  filters) and the profile's tools, and fails closed on any retrieval,
  code-interpreter or model-metadata difference. The environment is the
  configuration on every start (`RESET_CONFIG_ON_START`); Admin Panel edits
  do not survive a restart.
- **WS-D5 Retrieval models.** One read-only, hash-attested copy per profile
  model in `model-volumes/workspace-models/` serves every example (4.4 GB
  instead of a copy each), fetched on the host side through the Open WebUI
  image on Docker's default network.
- **WS-D6 Gates.** `verification/workspace/gates.py` implements `docling`,
  `knowledge`, `chat-ui`, `knowledge-serving`, `office` and `file-download`
  once; each example registers them with its values, and `knowledge-serving`
  runs the example's own serving rows against its own reference while L4
  supplies the OCR load and Docling's CPU measurement.
- **WS-D7 CPU demo.** `deploy/compose/docker-compose.webui.yaml`'s
  `webui-setup` runs the L4 setup (setup only: that Open WebUI is not
  deployed by L4; its Kairyu-embeddings RAG, G6 P-C3, stays), so CI runs the
  L4 setup against a real Open WebUI on every PR. Its admin credentials are
  required.
- **WS-D8 Orchestrated examples.** `qwen3.8-deepseek-v4.1-8gpu` and
  `deepseek-v4.1-quyet-8gpu` join in a later PR together with an L2/L3
  change: an orchestrated public model still rejects an image beside tool
  history (only `upstream_chat_models` carry it, m9 D2 amendment
  2026-10-22), which the default-on code interpreter makes common.

## Scope

PR A: L4 and `deepseek-v4.1-flash-1017-8gpu` moved onto it without a
behaviour change. PR B: the six other single-model chat examples. PR C: the
two orchestrated examples (WS-D8). `quyet-1.0-large-1gpu` has no chat.
