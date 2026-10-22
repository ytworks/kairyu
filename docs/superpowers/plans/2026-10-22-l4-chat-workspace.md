# L4 chat workspace: the 1017 eight-GPU Chat UI in every example

Status: **Plan approved 2026-10-22 (owner); PR A (this PR) awaits approval of
the full plan below, including the verification plan, before implementation.**

## Problem

`examples/deepseek-v4.1-flash-1017-8gpu/` is the only example whose Chat UI
(Open WebUI v0.12.3) has the features GPU-verified on 2026-10-21/22:

- Japanese knowledge bases (PR #661): pinned docling-serve with RapidOCR
  `iso:ja` (scanned PDFs too), `BAAI/bge-m3` embeddings and
  `BAAI/bge-reranker-v2-m3` reranking (pinned revision and tree hash,
  offline), hybrid search at BM25 weight 0, 300-character chunks;
- Word / Excel / PowerPoint read and edit (PR #662) through the default-on
  browser code interpreter (Pyodide);
- download links for files the Chat UI makes (PR #664, `file_download`).

The other eight chat examples have none of them, and each carries its own
copy of the same Chat UI plumbing: a ~30-line `chat-ui` compose block, one of
six diverged Responses/effort filter files, ~200 lines of Open WebUI admin-API
provisioning in `control.py`, and (1017 eight-GPU only) ~1,200 lines of
Chat UI gates. The owner wants every feature in every example, shared as a
Kairyu framework layer.

## Owner decisions (2026-10-22)

1. A new layer **L4** (chat workspace) is created at repository level in
   `deploy/workspace/` and `verification/workspace/`; the `kairyu/` package
   (L1-L3) is unchanged.
2. **Exception to "each example owns its scripts"**: the Chat UI compose
   services, the Open WebUI setup and fail-closed checks, the plugins and the
   Chat UI GPU gates (chat-ui, office, file-download, knowledge-serving) exist
   once in L4. Examples keep only values.
3. The two orchestrated examples (`qwen3.8-deepseek-v4.1-8gpu`,
   `deepseek-v4.1-quyet-8gpu`) follow in a later PR (PR C) together with an
   L2/L3 change: today an image after a tool call is rejected for orchestrated
   public models (`chat_service.py:1063` renders the orchestration multimodal
   prompt without the tool transcript; `responses_service.py:267-270` exempts
   only `upstream_chat_models`), and the default-on code interpreter makes
   that happen in ordinary chats. PR C gets its own plan; it re-confirms the
   verified-tool route's latency per tool step (~75 s at c1) for Office edits.

## L4 definition

L4 runs outside the Kairyu server and uses only L3's public API
(`/v1/responses`). It holds the Chat UI and its document services: OCR,
embedding, reranking, the browser code interpreter and file downloads. L1-L3
all run inside the Kairyu server; L4 is a client of it, so it works for any
model behind L3 and never reaches L1/L2 directly. Open WebUI is not forked:
L4 uses its environment configuration and its plugin API (filters, tools)
only.

Framework admission (`.claude/rules/framework-boundary.md`):

1. Missing shared contract: no shared Chat UI deployment exists; nine
   examples copy it (six filter variants, per-example provisioning), and only
   one has knowledge, Office and downloads.
2. Existing extension point: `deploy/compose/` installs one filter for the
   CPU demo only; it has no retrieval, code interpreter, tool, version or
   drift checks, and examples cannot use it without copying.
3. Concrete use and observable regression independent of one example: the
   seven single-model chat examples plus the CPU demo; a drifted retrieval
   setting, a missing tool or a wrong Open WebUI version makes `run.sh up`
   fail closed with the differing key.
4. Smallest mechanism: one compose fragment, a standard-library setup module,
   two plugins and one versioned profile. Policy stays in each example: model
   ids, default parameters, effort levels, vision, ports, bind, auth, CPU caps,
   and which profile it selects.

Why not inside `kairyu/`: L4 needs nothing L3 does not already expose; the
example launchers run on the host's standard-library Python and cannot import
`kairyu`; the file tool is Open WebUI plugin code, not Kairyu wheel content.

## Design

### L4 files (new)

| File | Content |
|---|---|
| `deploy/workspace/README.md` | the L4 contract: spec keys, profiles, lifecycle, the one supported Open WebUI version |
| `deploy/workspace/compose.yaml` | shared services `workspace-models` (one-shot fetch and attestation, `service_completed_successfully`), `docling`, `chat-ui`; merged after each example's file with a second `-f` |
| `deploy/workspace/profiles/ja-v1.json` | the 1017 eight-GPU retrieval settings: Docling image and parameters, bge-m3 / reranker repo, revision and tree hash, chunk 300/30, BM25 0, top_k 20/5, code interpreter and default tool, model-independent gate fixtures; immutable (a change becomes `ja-v2`) |
| `deploy/workspace/plugins/responses_filter.py` | the one Responses filter; effort levels from the Chat UI's environment (no levels = no effort valve), replacing every example's filter |
| `deploy/workspace/plugins/file_download_tool.py` | moved unchanged from the 1017 eight-GPU example |
| `deploy/workspace/kairyu_workspace/` (`spec.py`, `models.py`, `openwebui.py`, `__main__.py`) | standard library only: load and check the example's `workspace` spec and profile at import time, compute compose values, fetch and attest retrieval models, sign in (auth on or off), check `/api/version`, install the filter and tool, and fail closed on any retrieval, code-interpreter or model-metadata drift |
| `verification/workspace/` (`client.py`, `correctness/docling.py`, `knowledge.py`, `browser.py`, `browser.mjs`) | the 1017 eight-GPU Chat UI gates (`verification.py` 1147-2349, `webui-browser-smoke.mjs`) moved here; examples load `client.py` by path |

### Example side

- `example.json`: `webui` becomes `workspace` (profile `ja-v1`, port, bind,
  auth, model ids, default params, vision, code interpreter default, effort
  levels, CPU caps, knowledge-serving reference).
- `compose.yaml`: the `chat-ui` (and the 1017 eight-GPU `docling`) service is
  deleted; `control.py` adds `-f ../../deploy/workspace/compose.yaml`.
- `control.py`: the Chat UI helpers (`_webui_api`, `_check_chat_ui_*`,
  `_provision_chat_ui_*`, `_ensure_retrieval_models`) and the filter / tool
  files are deleted for ~10 lines calling `kairyu_workspace`.
- `verification.py`: ~10 lines register the L4 gates with the example's
  values; for knowledge-serving L4 supplies the document load and Docling CPU
  sampling while the example runs its own `serving` rows.

### Details decided in planning

- Retrieval models live once in a shared read-only, hash-attested L4 store
  under the storage root instead of a 4.4 GB copy per example.
- Each example keeps its auth, port and bind; the Qwen pair stays auth-on.
  Configuration is environment-driven everywhere (`RESET_CONFIG_ON_START`),
  so Admin Panel edits reset on restart; missing admin credentials fail
  `run.sh up`.
- One filter id, `kairyu_responses`; setup deletes the old ids (users' saved
  effort choice resets once).
- Every example starts at Chat UI 8 / Docling 4 CPUs; its knowledge-serving
  gate decides.
- Existing knowledge bases need re-indexing after the embedding model changes
  (on this host only the quyet example's UI has one); documented in READMEs.
- The CPU demo's `webui-setup` (`deploy/compose/`) switches to the L4 setup
  and its two private files are deleted, so every PR runs L4 setup against a
  real Open WebUI v0.12.3 in CI; the demo's Kairyu-embeddings RAG (G6 P-C3)
  stays.

## PRs

| PR | Scope | Layers |
|---|---|---|
| A (this) | L4, `deepseek-v4.1-flash-1017-8gpu` moved onto it without behaviour change, CPU demo setup | L4 + example |
| B | `deepseek-v4.1-flash-1017-6gpu`, `deepseek-v4.1-flash-6gpu`, `deepseek-v4.1-flash-8gpu`, `glm-5.3-flash-6gpu`, `qwen3.8-27b-1gpu`, `qwen3.8-27b-dp8-8gpu` gain every feature | examples |
| C | `qwen3.8-deepseek-v4.1-8gpu`, `deepseek-v4.1-quyet-8gpu` plus the orchestrated images-beside-tools change (own plan) | L2/L3 + examples |

`quyet-1.0-large-1gpu` has no chat (System One only) and is out of scope.

## Tests (CPU)

- New: `tests/unit/test_workspace_setup.py`, two cases against a fake Open
  WebUI: an existing active global filter is updated and old ids removed; a
  configuration drift fails closed naming the key. The fresh-database CI demo
  cannot reach either branch.
- Removed or reduced: `tests/unit/test_webui_responses_filters.py` (every
  example filter becomes the one L4 filter, with and without effort levels),
  static compose / filter assertions in example tests that the spec loader now
  enforces (e.g. `test_frontier_examplectl.py:51-66`), duplicated filter tests.
- Base/head collection counts with the same pytest command are reported; the
  count must fall. Locally only ruff and the touched tests run; CI runs all.

## Verification plan

Stop at the first failing gate and bring the fix plan; a passed gate is not
re-run or questioned.

PR A:

| Gate | Checks | Pass | Budget |
|---|---|---|---|
| L4 `docling` (CPU) | image-only Japanese PDF OCR; text-layer PDF | key sentence on 3/3 pages, <= 30 s per page, no text-layer line lost | 10 min |
| L4 `knowledge` (CPU) | 20-page scan plus 20 distractors in a knowledge base | key chunk in the top 5, query <= 10 s, scan ingestion <= 300 s | 10 min |
| CI `webui-e2e` | demo with the L4 setup | green | CI |
| 1017-8gpu restart | `run.sh up` with L4 setup and checks | ready; every check matches | 15 min |
| `chat-ui` | 323 then 324, an image answered Red, `#` knowledge answer with citation | all answered; `/v1/chat/completions` unchanged | 10 min |
| `office` | docx / xlsx / pptx read and edit, then an image after the tool calls | edits verified in all three files; Red | 15 min |
| `file-download` | the owner's Excel request on desktop, after a reload, on an emulated iPhone 13 | three saved workbooks; same SHA-256 after the reload | 20 min |
| `knowledge-serving` | c1/c8 while Docling ingests | Docling busy >= 95 % of the rows, each row >= 95 % of the reference | 15 min |
| `api-equivalence` | every case incl. `image-tools` | L1 and L3 agree | 10 min |

PR B: per example, in turn: restart, `chat-ui`, `office`, `file-download`,
`knowledge-serving` (reference = the example's own unloaded `serving` c1/c8
in the same run, then recorded), `api-equivalence` with `image-tools` added;
about 1.5-2 h each including startup, 10-12 h in all. Qwen accepts one image
per request, so its gates attach one.

PR C: in its own plan.

## Records

- New `docs/design/l4-chat-workspace.md` (D-IDs, boundary, profiles, version
  pin); L4 added to the layer map (`README.md`, `CLAUDE.md`); one line in
  `.claude/rules/framework-boundary.md` (L4 owns the mechanism and versioned
  profiles; examples select a profile and own model, ports, CPUs and auth).
- `examples/README.md`, the examples' Chat UI sections, `MEASUREMENTS.md`,
  `PROGRESS.md`.

## Risks

- L4 depends on Open WebUI v0.12.3 internals (admin API, the tool's
  `upload_file_handler`); one supported version, checked at startup through
  `/api/version`; an upgrade is one PR.
- CPU caps are quotas, not pinning, and share cores with each example's L1;
  each example's knowledge-serving gate proves its own budget.
- Qwen3.8-27B may fall short of the office edit checks; reported as found.
- Examples published on 0.0.0.0 without auth (glm; quyet and the ensemble in
  PR C) let anonymous users run OCR (capped at 4 CPUs) and store files;
  uploads are already possible today, and code runs in the browser, never on
  the server.
- First startup downloads the retrieval models from Hugging Face.
