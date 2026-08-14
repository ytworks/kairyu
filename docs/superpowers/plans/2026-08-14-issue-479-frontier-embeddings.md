# Issue 479 Frontier Embeddings Implementation Plan

**Goal:** Add Kairyu's pinned offline embedding capability to the tiered 8-GPU
example and prove its public endpoint contract during readiness.

**Architecture:** Reuse the existing FastEmbed backend and Docker bundle pins.
The deployment advertises one chat model and one embedding model while routing
inventory remains chat-only.

## Task 1: Define failing deployment-contract tests

**File:** `tests/unit/test_tiered_frontier_examplectl.py`

Extend the example assertions to require the `embed-small` DeploymentSpec,
immutable image build args, manifest identity, and the public/routing split.
Add readiness tests with a complete valid two-input response plus malformed
identity, indices, shape, numeric, and usage cases. Run the file and observe the
new failures.

## Task 2: Wire the offline model into the image and spec

**Files:**
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/compose.yaml`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/kairyu.yaml`
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/example.json`

Enable combined vision/embedding dependencies, pass the four pinned bundle
arguments, configure FastEmbed with 384 dimensions and bounded concurrency,
and publish `embed-small` alongside `kairyu-auto-max`.

## Task 3: Make readiness verify behavior

**File:** `examples/qwen3.6-deepseek-v4-8gpu/control.py`

Keep `/routing` validation chat-only. Add a two-string embeddings request and
validate exact model identity, indices, vector count/dimensions, finite numeric
values, and positive usage before declaring the stack ready. Print both public
model IDs.

## Task 4: Document and record the deployment change

**Files:**
- Modify: `examples/qwen3.6-deepseek-v4-8gpu/README.md`
- Modify: `docs/design/example-layered-orchestration.md`
- Modify: `PROGRESS.md`

Document the public chat/embedding split, endpoint probe, and truthful
`embed-small` identity. State that consumer-side tau selection is tracked by
kairyu-bench #5. Prepend a #479 change-log entry without rewriting history.

## Task 5: Verify and publish

Run the example unit tests, related compose/spec tests, full portable suite,
ruff, the progress-size check, compose rendering, `git diff --check`, and
self-review. Commit, push `codex/issue-479-frontier-embeddings`, and open a PR
linked to #479 and kairyu-bench #5, clearly separating portable completion
from live GPU gates.
