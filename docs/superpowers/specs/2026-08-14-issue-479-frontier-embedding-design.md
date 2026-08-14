# Issue 479 Frontier Embedding Capability Design

**Status:** Approved

**Issues:** `ytworks/kairyu#479`; consumer-side selection is tracked separately
by `ytworks/kairyu-bench#5`

## Goal

Make the `qwen3.6-deepseek-v4-8gpu` deployment expose a truthful, production
offline embedding model through Kairyu's OpenAI-compatible surface.

## Deployment contract

The example publishes `embed-small`, backed by the already-pinned
`sentence-transformers/all-MiniLM-L6-v2` FastEmbed bundle. The Docker build
enables embeddings and vision together and supplies the immutable repository,
revision, model digest, and provenance digest already used by the production
WebUI image.

`/v1/models` contains the public chat model and public embedding model. The
`/routing` endpoint remains chat-only. Readiness performs a real two-input
`/v1/embeddings` request and fails closed unless the response identifies
`embed-small`, returns indices 0 and 1, contains two finite 384-dimensional
vectors, and reports positive token usage.

The example manifest records the embedding implementation and immutable pins.
No compatibility alias claims that MiniLM is OpenAI's
`text-embedding-3-large`. Selecting `embed-small` in tau2 is consumer-side work
tracked by kairyu-bench #5 and is outside this PR.

## Verification

Portable tests cover DeploymentSpec parsing, exact image build pins, the
public-model/routing split, manifest consistency, and response validation for
the two-input readiness smoke. Live image rebuild and tau warmup remain host
acceptance gates.
