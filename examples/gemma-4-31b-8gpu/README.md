# Gemma 4 31B with EmbeddingGemma 2 on eight RTX PRO 6000 GPUs

`google/gemma-4-31B-it` chat replicas and `google/embeddinggemma-2` embedding
replicas on stock vLLM, behind Kairyu, with the Chat UI of the 1017 eight-GPU
example: Japanese OCR knowledge bases, Word / Excel / PowerPoint, download
links, images, tools and a thinking selector.

```text
Open WebUI (:3020, sign-in) -> Kairyu (:8020) -> pool gemma-4-31b       -> vLLM chat replicas (Gemma 4 31B)
   |                                  \-> pool embeddinggemma-2 -> vLLM pooling replicas (EmbeddingGemma 2)
   +-> Docling (OCR, CPU) for knowledge-base documents
   +-> knowledge-base embeddings: Kairyu /v1/embeddings (embeddinggemma-2)
```

> **Not GPU-verified yet.** The layout in `compose.yaml` / `kairyu.yaml`
> (seven TP1 chat replicas on GPUs 0-6, two embedding replicas sharing GPU 7)
> is provisional until the replica search below; `MEASUREMENTS.md` records
> the search and the gates.

## Start

```sh
export WEBUI_ADMIN_EMAIL=admin@example.com WEBUI_ADMIN_PASSWORD='<a strong password>'
./run.sh up          # preflight, image pull, checkpoint download + hash check, readiness, Chat UI checks
./run.sh status
./verify.sh list
./run.sh down
```

`run.sh up` refuses to start when the GPUs are busy, when the host has less
than 64 GiB of available memory, or without the admin credentials: the Chat
UI is published on the host's address (`0.0.0.0:3020`, the URL printed by
`run.sh up`; set `PUBLIC_HOST` if it cannot be discovered), every user signs
in, the first start creates this admin, and the admin adds other users. The
three checkpoints (≈66 GiB) go to
`/mnt/nvme/kairyu/model-volumes/gemma-4-31b-8gpu/models` and are hashed
against the pinned trees (`model-manifest.json`): Gemma 4 31B, its MTP
assistant (used only if the search selects it) and EmbeddingGemma 2.

API: `http://127.0.0.1:8020/v1`, chat model `gemma-4-31b`, embedding model
`embeddinggemma-2`.

## Replicas and load balancing (L2)

Kairyu holds two replica pools (`kairyu.yaml`):

- **`gemma-4-31b`**: one entry per chat replica. A request whose prompt
  prefix is warm on an idle replica goes there; otherwise it goes to the
  replica with the fewest requests in flight, so concurrent chats spread one
  per replica (`queue_depth_threshold: 0`, `prefix_index: true`, as in
  `qwen3.8-27b-dp8-8gpu`).
- **`embeddinggemma-2`**: one entry per EmbeddingGemma 2 replica
  (`backend: openai`, m11 D4 upstream-embeddings amendment). Each batch goes
  to the least-busy replica; a replica takes traffic only after its
  `/health` answers, one failure ejects it until `/health` answers again,
  and `/readyz` reports unready while no embedding replica is eligible.

Both pools write one JSONL row per placement under
`model-volumes/gemma-4-31b-8gpu/placement-log`; the `serving`, `tool-calling`,
`embeddings` and `knowledge` gates read it.

## Embeddings used automatically (L3)

- Kairyu publishes `embeddinggemma-2` on `/v1/embeddings` and `/v1/models`.
  Kairyu never forwards `dimensions`; vectors are EmbeddingGemma 2's 768
  dimensions, L2-normalized.
- The Chat UI's knowledge bases embed every document chunk and every query
  through Kairyu's `embeddinggemma-2`, never inside the Chat UI
  (`RAG_EMBEDDING_ENGINE=openai`, `RAG_OPENAI_API_BASE_URL=http://kairyu:8000/v1`).
  Batches of 32 chunks go out four at a time, so ingestion uses every
  embedding replica.
- EmbeddingGemma 2 expects task instructions: the Chat UI prepends
  `task: search result | query: ` to queries and `title: none | text: ` to
  document chunks (`RAG_EMBEDDING_QUERY_PREFIX` / `RAG_EMBEDDING_CONTENT_PREFIX`).
  API clients add them themselves: an OpenAI embeddings request does not say
  whether its input is a query or a document.
- The embedding model is not offered in the Chat UI's model picker
  (`OPENAI_API_CONFIGS` `model_ids`).

`run.sh up` fails unless Open WebUI reports exactly this configuration.

## Chat UI and API

Open WebUI talks to Kairyu over `/v1/responses` (m11 D7 amendment
2026-10-17). `run.sh up` installs the Thinking filter
(`webui-responses-filter.py`) as a global filter: Gemma 4 has no effort
levels, only thinking on or off. `default` keeps the L1 default (thinking
on, `--default-chat-template-kwargs '{"enable_thinking":true}'`); `none`
sends `reasoning.effort: none`, which vLLM renders with thinking off. The
filter also turns Open WebUI's chat body into a strict Responses body (as in
the other examples).

> **Open decision.** Kairyu's `/v1/responses` currently answers
> `reasoning.effort: none` with 400 (`responses_protocol.py`), so the `none`
> choice fails until that is decided; chat-completions callers turn thinking
> off with `chat_template_kwargs: {"enable_thinking": false}`.

Kairyu sends each request to vLLM in its caller's API (`upstream_chat_models`,
`upstream_responses_models`, `upstream_messages_models`). vLLM renders the
checkpoint's own chat template (Google's canonical Gemma 4 template,
2026-07-09) and parses tool calls and reasoning with its `gemma4` parsers.
Sampling follows the checkpoint's `generation_config.json` (temperature 1.0,
top_p 0.95, top_k 64).

## Japanese knowledge bases

As in the 1017 eight-GPU example, with the embedding replaced:

- **Extraction**: Docling (`docling-serve-cpu` v1.37.0, pinned by digest),
  text layers kept, bitmaps read by RapidOCR with `iso:ja`; Docling is
  limited to 4 CPUs and the Chat UI to 8.
- **Embedding**: EmbeddingGemma 2 through Kairyu (above).
- **Reranking**: `BAAI/bge-reranker-v2-m3` on the Chat UI's CPUs, downloaded
  by `run.sh up` at a pinned revision and hashed.
- **Search**: hybrid search with BM25 weight 0 (Open WebUI's BM25 cannot
  tokenize Japanese): 20 vector candidates, the reranker keeps 5.
- **Chunks** of 300 characters with 30 overlapping.

Create a knowledge base in Workspace > Knowledge with a name and a
description of what it holds, add files, then attach it to the model or pick
it in a chat with `#`.

## Word, Excel and PowerPoint

Attach a docx, xlsx or pptx and ask about it or for a change. Reading goes
through Docling; the Chat UI's browser code interpreter (Pyodide, on in every
chat) edits or creates files in `/mnt/uploads`, and the `file_download` tool
(`webui-file-download-tool.py`, on in every chat and readable by every
signed-in user) stores the file and puts a download link in the reply.
Limits as in the 1017 example: code runs in the viewer's browser, one run may
take 60 s, openpyxl formulas are computed when Excel opens the file, no
legacy doc / xls / ppt.

## Replica search

The layout is chosen by measurement (`tune.sh`, about 8 hours on an idle
host; the committed configuration is restored afterwards):

1. `./tune.sh layout tp1-facts tp2-facts`: KV capacity of one TP1 and one TP2
   chat replica and how many 262,144-token requests each holds. Every chat
   replica must hold at least one (the model's full context, as the 1017
   example serves its full context); a TP that cannot is excluded.
2. `./tune.sh l1 --layout <layout> baseline baseline-repeat mtp-4 mtp-8 ...`:
   L1 options one at a time; a change is kept only at 5 % or more.
3. `./tune.sh embedding --layout <layout>`: one embedding replica's
   throughput for Open WebUI's batches and its query latency, and the chat
   rows with and without embedding load where embeddings share a chat GPU.
4. `./tune.sh layout --load <layouts>`: the surviving full layouts (for
   example seven TP1 chat replicas with a dedicated embedding GPU, eight TP1
   with shared embeddings, four TP2) under three knowledge-base ingestions.
5. The layout with the highest c32 / c64 aggregate output throughput wins;
   within 5 %, the faster c1. At least two embedding replicas always, so one
   failure does not stop knowledge bases. The result goes into
   `example.json` (allocation), `compose.yaml` and `kairyu.yaml`; `run.sh up`
   refuses a compose file that disagrees with `example.json`.

## Verification gates

Every gate that goes through Kairyu uses `/v1/responses` or `/v1/embeddings`,
the Chat UI's APIs. Stop at the first failing gate.

- `l1`: every chat replica holds one 262,144-token request; the checkpoint's
  sampling defaults; thinking on (default) and off render as the template
  defines; 12 exact `17 * 19` probes per replica and mode with finite
  log-probabilities (thinking on must reason); every embedding replica's
  vectors against sentence-transformers 6.1.0 (cosine ≥ 0.999, unit norm);
  a tool call, an image and an embedding batch through Kairyu.
- `serving`: 8K-in rows capped at 256 output tokens at c1/8/16/32/64 (64
  requests each); every chat replica used and the largest at most 1.25 times
  the mean at c ≥ 8.
- `completed`, `tool-calling` (every chat replica places tool calls),
  `vision`, `reasoning` (default reasons, `none` does not; both answer 323),
  `cancellation`, `long-context` (32K / 128K / 250K retrieval).
- `embeddings`: Kairyu's vectors equal the replica's (cosine ≥ 0.9999);
  concurrent batches reach every embedding replica; with one replica stopped
  at most one request fails before it is ejected, `/readyz` stays ready and
  the rest succeed; after a restart the replica rejoins and batches spread
  again.
- `restart` (`run.sh up` after restarting every service), `api-equivalence`
  (vLLM and Kairyu agree, including `image-tools`).
- `docling`, `knowledge` (20-page scan: key chunk in the top 5, ingestion
  ≤ 300 s, query ≤ 10 s, the ingestion reached every embedding replica and
  the query the embedding pool), `chat-ui` (signed in: 323 then 324, Red for
  an image, a `#` knowledge answer citing the scan, only the chat model
  offered), `knowledge-serving` (c1/c8 while ingesting at least 95 % of the
  same run's unloaded rows), `office`, `file-download`.

## Primary references

- Gemma 4 model card and `chat_template.jinja`: <https://huggingface.co/google/gemma-4-31B-it>
- vLLM Gemma 4 recipe: <https://recipes.vllm.ai/Google/gemma-4-31B-it>
- EmbeddingGemma 2 model card: <https://huggingface.co/google/embeddinggemma-2>
- vLLM EmbeddingGemma 2 support: <https://github.com/vllm-project/vllm/pull/60254> (v0.32.0)
- Open WebUI v0.12.3 retrieval settings (embedding engine, prefixes, `model_ids`)
