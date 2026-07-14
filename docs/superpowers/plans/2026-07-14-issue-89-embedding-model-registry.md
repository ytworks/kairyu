# Issue 89 Embedding Model Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make embedding model discovery, routing, metrics, and metering use a bounded, explicitly configured model registry.

**Architecture:** Replace the anonymous embedding backend with a model-ID-to-backend mapping at the application boundary. Resolve the requested ID before validation or execution, share the existing `model_not_found` response, expose the configured IDs from `/v1/models`, and carry only the resolved key into metrics and usage accounting. Extend `DeploymentSpec` with named embedding sections and construct their backends in the serve builder.

**Tech Stack:** Python protocols and mappings, FastAPI, Pydantic deployment models, Prometheus metrics, usage ledger, pytest/HTTPX, Ruff.

## Global Constraints

- Unknown embedding IDs return the shared 404 `model_not_found` payload before backend work.
- `/v1/models` lists all configured embedding IDs alongside chat engines and orchestrators.
- Multiple embedding IDs route independently and echo the resolved registry key.
- Metrics and usage accounting never use an unverified caller-controlled model ID.
- Embedding IDs cannot collide with engine, pool, named orchestrator, or legacy `kairyu-auto` IDs.
- The legacy anonymous `embedding_backend` keyword is removed; all in-repo callers migrate in this PR.
- Base this sibling stacked PR on `codex/issue-86-batch-request-boundary`; its diff must contain only Issue #89.

---

### Task 1: Pin the HTTP registry contract

**Files:**
- Create: `tests/server/test_embeddings_models.py`
- Modify: `tests/server/test_m11_product.py`

- [x] **Step 1: Migrate existing fixtures to named mappings**

Replace anonymous embedding backend arguments with explicit mappings whose key matches each request model.

- [x] **Step 2: Add unknown-ID and discovery tests**

Assert an unknown embedding ID returns the shared 404 shape without invoking any backend, and `/v1/models` contains every configured embedding ID alongside chat and auto models.

- [x] **Step 3: Add multi-backend routing tests**

Configure two embedding IDs with different dimensions, assert each request reaches only its selected backend, returns the matching vector size, and echoes the resolved key.

- [x] **Step 4: Add bounded identity accounting tests**

Assert successful usage is recorded under the resolved ID, the limiter is charged once, successful metrics use that ID, and an unknown request records no ledger entry while its 404 metric collapses to `unknown`.

- [x] **Step 5: Verify RED**

Run: `uv run pytest tests/server/test_embeddings_models.py tests/server/test_m11_product.py -q`

Expected: the mapping keyword is unsupported and the anonymous route cannot resolve or discover model IDs.

### Task 2: Pin DeploymentSpec and builder behavior

**Files:**
- Modify: `tests/server/test_serve_builder.py`
- Modify: `tests/unit/test_deployment_spec.py`

- [x] **Step 1: Add schema and collision tests**

Parse two named embedding sections with backend and dimensions, and reject blank IDs plus collisions with engines, pools, named orchestrators, and the legacy `kairyu-auto` name.

- [x] **Step 2: Add builder integration coverage**

Build an app from YAML, discover both embedding IDs, and prove requests route to distinct configured vector dimensions.

- [x] **Step 3: Verify RED**

Run: `uv run pytest tests/server/test_serve_builder.py tests/unit/test_deployment_spec.py -q`

Expected: `DeploymentSpec` currently ignores the new section and the builder exposes no embedding route or models.

### Task 3: Implement explicit embedding model resolution

**Files:**
- Modify: `kairyu/entrypoints/server/app.py`
- Modify: `kairyu/entrypoints/server/extra_routes.py`

- [x] **Step 1: Replace the anonymous application argument**

Accept `Mapping[str, EmbeddingBackend]`, copy it at construction, reject cross-surface model collisions, pass it to extra routes, and include its keys in `/v1/models`.

- [x] **Step 2: Resolve before work**

Look up the requested embedding model before input validation or backend execution, return shared `model_not_found` on a miss, and set request metric state only after a successful lookup.

- [x] **Step 3: Use the resolved identity everywhere**

Execute the selected backend and use its registry key for the response and exact-once usage accounting.

### Task 4: Wire named embeddings through deployment configuration

**Files:**
- Modify: `kairyu/deploy/spec.py`
- Modify: `kairyu/deploy/builder.py`

- [x] **Step 1: Add the frozen embedding section**

Define backend name plus positive dimensions, add the named mapping, validate non-blank keys and every cross-surface collision.

- [x] **Step 2: Build the mapping before application construction**

Resolve the configured embedding backend factories, instantiate one backend per model ID, and pass the complete mapping to `create_app`.

- [x] **Step 3: Verify focused GREEN**

Run: `uv run pytest tests/server/test_embeddings_models.py tests/server/test_m11_product.py tests/server/test_serve_builder.py tests/unit/test_deployment_spec.py tests/server/test_health_metrics.py -q`

Expected: PASS.

### Task 5: Record the decision and document configuration

**Files:**
- Modify: `docs/design/m11-product.md`
- Modify: `docs/deployment.md`
- Modify: `PROGRESS.md`

- [x] **Step 1: Amend D4**

Record explicit model registries, truthful discovery, resolved routing, bounded labels, and unknown-ID rejection.

- [x] **Step 2: Document the YAML**

Add a concise named `embeddings:` example with backend and dimensions.

- [x] **Step 3: Update progress memory**

Update Current Status and prepend the required English amendment entry before committing.

### Task 6: Verify and publish the sibling stacked PR

**Files:**
- No additional source files.

- [x] **Step 1: Run complete verification**

Run: `uv run pytest`

Run: `uv run ruff check .`

- [x] **Step 2: Audit the parent-relative diff and commit**

Run: `git diff --check` and compare against `codex/issue-86-batch-request-boundary`.

Commit: `fix: resolve configured embedding models`

- [ ] **Step 3: Push and create the sibling stacked PR**

Push `codex/issue-89-embedding-model-registry`, create a ready PR against `codex/issue-86-batch-request-boundary`, link parent PR #94, list verification, and include `Fixes #89`.
