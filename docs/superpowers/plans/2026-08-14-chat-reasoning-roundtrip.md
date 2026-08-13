# Chat Reasoning Round-Trip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Kairyu Chat Completions assistant response safe to append to a later request while retaining strict validation of unsupported message fields.

**Architecture:** Add `reasoning_content` to the typed inbound `ChatMessage` contract so key-sensitive HF templates and L2 orchestration receive the same field Kairyu emits. Treat only `provider_specific_fields: null` as ignorable client-library metadata at the shared message-preparation boundary; remove it before rendering and continue rejecting non-null values and every other unsupported extra before backend dispatch.

**Tech Stack:** Python 3.11+, Pydantic v2, FastAPI, Jinja chat templates, pytest, Ruff.

## Global Constraints

- `reasoning_content` is accepted only as `str | None` and remains present for downstream rendering only when it was present on the request wire.
- `provider_specific_fields` is compatibility-only metadata: ignore it only when its value is exactly `null`, never expose it through the typed OpenAPI schema, and never pass the ignored field into a prompt or L2 conversation.
- Any non-null `provider_specific_fields` value remains an HTTP 400 `invalid_request` and must not dispatch backend work.
- Existing strict rejection of every other message-level extra, including alternate prompt carriers, remains unchanged.
- HTTP, batch, ordinary chat-template, and L2 orchestration paths share the same `ChatMessage` and `_prepare_chat_messages` policy; do not add route-specific workarounds or a kairyu-bench adapter shim.
- Preserve the existing omitted-versus-explicit-null wire shape used by key-sensitive HF templates.
- The pinned mini-SWE-agent revision `a83fcae82d2a08f0ee0c688f9d137b3566c097f8` stores LiteLLM's normalized assistant `message.model_dump()` and removes only its own `extra` field before the next request; tests must exercise the resulting wire shape directly without adding LiteLLM as a core dependency.
- The portable macOS full-suite baseline cannot collect `tests/unit/test_kv_tier_policy.py` because `/proc/cpuinfo` is absent; focused server tests and Linux CI are the binding automated verification for this change.

---

### Task 1: Enforce the assistant-history compatibility contract

**Files:**
- Modify: `kairyu/entrypoints/server/protocol.py:53-60`
- Modify: `kairyu/entrypoints/server/chat_service.py:328-419`
- Test: `tests/server/test_chat_template_policy.py`
- Test: `tests/server/test_openai_api.py:1212-1275`
- Test: `tests/server/test_orchestration_usage_trace.py:142-191`
- Test: `tests/server/test_prompt_offload.py:346-368`

**Interfaces:**
- Consumes: `ChatCompletionRequest.messages: list[ChatMessage]`, `ChatMessage.model_fields_set`, `ChatMessage.model_extra`, `_prepare_chat_messages(request, validate_message_fields=True)`.
- Produces: `ChatMessage.reasoning_content: str | None`; `_is_ignored_message_extra(name: str, value: object) -> bool`; rendered message dictionaries that retain `reasoning_content` but omit nullable `provider_specific_fields`.

- [ ] **Step 1: Write the failing template-boundary test**

Add this behavior to `tests/server/test_chat_template_policy.py` using its existing imports:

```python
def test_litellm_assistant_metadata_preserves_reasoning_only():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "prior answer",
                    "reasoning_content": "visible work",
                    "provider_specific_fields": None,
                }
            ],
        }
    )
    template = ChatTemplate(
        "{{ messages[0].reasoning_content }}|"
        "{{ messages[0].content }}|"
        "{{ 'provider_specific_fields' in messages[0] }}"
    )

    validated = validate_chat_input(request, {"m": template})

    assert validated.prompt == "visible work|prior answer|False"
```

This catches either restoring blanket extra rejection, dropping `reasoning_content`, or leaking client metadata into the prompt.

- [ ] **Step 2: Run the template-boundary test and verify RED**

Run:

```bash
uv run pytest tests/server/test_chat_template_policy.py::test_litellm_assistant_metadata_preserves_reasoning_only -q
```

Expected: FAIL with `messages[0] has unsupported fields: provider_specific_fields, reasoning_content`.

- [ ] **Step 3: Write the remaining failing HTTP, orchestration, and schema tests**

Add an HTTP negative test to `tests/server/test_openai_api.py`:

```python
async def test_non_null_litellm_message_metadata_is_rejected_before_dispatch():
    backend = MockBackend()
    app = create_legacy_app(engines={"chat": backend})

    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "chat",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "prior answer",
                        "provider_specific_fields": {"reasoning_content": "work"},
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "messages[0] has unsupported fields: provider_specific_fields",
        "type": "invalid_request_error",
        "code": "invalid_request",
    }
    assert backend.prompts_seen == ()
```

Add a unary public-response round trip to `tests/server/test_orchestration_usage_trace.py`:

```python
def test_visible_reasoning_response_round_trips_through_litellm_history(tmp_path):
    backend = AccountingBackend()
    app = _app(tmp_path, backend, expose_intermediate_outputs=True)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": COMPLEX}],
            },
        )
        assistant = first.json()["choices"][0]["message"]
        assistant["provider_specific_fields"] = None
        second = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [
                    {"role": "user", "content": COMPLEX},
                    assistant,
                    {"role": "user", "content": "Continue."},
                ],
            },
        )

    assert first.status_code == 200
    assert assistant["reasoning_content"]
    assert second.status_code == 200
```

Extend `test_chat_and_route_openapi_keep_typed_request_contracts` in `tests/server/test_prompt_offload.py`:

```python
assert "reasoning_content" in schemas["ChatMessage"]["properties"]
assert "provider_specific_fields" not in schemas["ChatMessage"]["properties"]
```

- [ ] **Step 4: Run the new contract tests and verify RED**

Run:

```bash
uv run pytest \
  tests/server/test_chat_template_policy.py::test_litellm_assistant_metadata_preserves_reasoning_only \
  tests/server/test_openai_api.py::test_non_null_litellm_message_metadata_is_rejected_before_dispatch \
  tests/server/test_orchestration_usage_trace.py::test_visible_reasoning_response_round_trips_through_litellm_history \
  tests/server/test_prompt_offload.py::test_chat_and_route_openapi_keep_typed_request_contracts -q
```

Expected: the nullable metadata/template and orchestration tests fail at the unsupported-field boundary, and the schema assertion fails because `ChatMessage.reasoning_content` is not declared. The non-null negative may already pass and remains as the strict-policy pin.

- [ ] **Step 5: Add the typed input field and one shared nullable-extra predicate**

In `ChatMessage`, add:

```python
reasoning_content: str | None = None
```

Near `_message_wire_shape`, add one compatibility constant and predicate:

```python
_IGNORED_NULL_MESSAGE_EXTRAS = frozenset({"provider_specific_fields"})


def _is_ignored_message_extra(name: str, value: object) -> bool:
    return name in _IGNORED_NULL_MESSAGE_EXTRAS and value is None
```

Filter only ignored null extras from `_message_wire_shape`:

```python
wire.update(
    {
        name: value
        for name, value in (message.model_extra or {}).items()
        if not _is_ignored_message_extra(name, value)
    }
)
```

Replace blanket validation in `_prepare_chat_messages` with:

```python
unsupported_fields = {
    name
    for name, value in (message.model_extra or {}).items()
    if not _is_ignored_message_extra(name, value)
}
if unsupported_fields:
    raise ChatRequestError(
        f"messages[{index}] has unsupported fields: "
        + ", ".join(sorted(unsupported_fields))
    )
```

Do not type or advertise `provider_specific_fields`; it remains an extra whose only accepted value is null.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
uv run pytest \
  tests/server/test_chat_template_policy.py \
  tests/server/test_openai_api.py \
  tests/server/test_orchestration_usage_trace.py \
  tests/server/test_prompt_offload.py -q
```

Expected: PASS, including the existing alternate-prompt-carrier and tool-transcript strictness tests.

- [ ] **Step 7: Run the focused linter**

Run:

```bash
uv run ruff check \
  kairyu/entrypoints/server/protocol.py \
  kairyu/entrypoints/server/chat_service.py \
  tests/server/test_chat_template_policy.py \
  tests/server/test_openai_api.py \
  tests/server/test_orchestration_usage_trace.py \
  tests/server/test_prompt_offload.py
```

Expected: exit 0 with no diagnostics.

- [ ] **Step 8: Commit the API contract change**

```bash
git add \
  kairyu/entrypoints/server/protocol.py \
  kairyu/entrypoints/server/chat_service.py \
  tests/server/test_chat_template_policy.py \
  tests/server/test_openai_api.py \
  tests/server/test_orchestration_usage_trace.py \
  tests/server/test_prompt_offload.py
git commit -m "fix(api): round-trip assistant reasoning history"
```

### Task 2: Record and verify the public API amendment

**Files:**
- Modify: `docs/design/m11-product.md:100-107`
- Modify: `docs/design/example-layered-orchestration.md:74-85`
- Modify: `PROGRESS.md`
- Create: `docs/superpowers/plans/2026-08-14-chat-reasoning-roundtrip.md`

**Interfaces:**
- Consumes: Task 1's accepted `ChatMessage.reasoning_content` and null-only `provider_specific_fields` policy.
- Produces: a binding design statement, example acceptance criterion, and append-only progress record for issue #480.

- [ ] **Step 1: Amend the M11 message-history contract**

Immediately after the tiered-example transparency amendment in `docs/design/m11-product.md`, add an amendment dated 2026-08-14 stating all of the following explicitly:

```markdown
**Assistant-history compatibility amendment (2026-08-14).** A Chat
Completions assistant message may carry typed `reasoning_content`; Kairyu
preserves the field for key-sensitive model templates and L2 conversation
history so a response can be appended to the next request. Compatibility-only
`provider_specific_fields` metadata is discarded only when null. A non-null
value and every other unsupported message field still fail before backend
dispatch.
```

- [ ] **Step 2: Add the example acceptance criterion**

Add this bullet to `docs/design/example-layered-orchestration.md`:

```markdown
- An assistant response containing `reasoning_content` can be appended to the
  next Chat Completions request through the pinned LiteLLM message shape;
  nullable provider metadata is ignored without weakening non-null validation.
```

- [ ] **Step 3: Update PROGRESS.md before committing**

Change the snapshot date to `2026-08-14`. Add the round-trip behavior to the existing orchestration/API bullet in `What works today`. Prepend this Change Log entry, keeping the file within 200 lines and 10 entries:

```markdown
### 2026-08-14 — [amendment] Assistant reasoning history round-trips
- What: Chat Completions now accepts its typed `reasoning_content` response field in assistant history and drops only nullable LiteLLM `provider_specific_fields` metadata before rendering; non-null and unknown extras remain fail-closed.
- Why: The tiered product emitted visible intermediate work that normal LiteLLM serialization returned on the next agent turn, but the input schema rejected both its own field and nullable client metadata before dispatch.
- Refs: issue #480; `kairyu/entrypoints/server/{protocol,chat_service}.py`; `tests/server/test_{chat_template_policy,openai_api,orchestration_usage_trace,prompt_offload}.py`
```

Run:

```bash
uv run python scripts/check_progress_size.py
```

Expected: exit 0 and all progress-log budget checks pass.

- [ ] **Step 4: Run final portable verification**

Run:

```bash
uv run pytest \
  tests/server/test_chat_template_policy.py \
  tests/server/test_openai_api.py \
  tests/server/test_orchestration_usage_trace.py \
  tests/server/test_prompt_offload.py -q
uv run ruff check .
```

Expected: both commands exit 0. Also run `uv run pytest`; on this macOS worktree the only accepted non-green result is the unchanged collection failure from `tests/unit/test_kv_tier_policy.py` caused by absent `/proc/cpuinfo`. Linux CI must run the full selected CPU suite successfully before merge.

- [ ] **Step 5: Document the exact live acceptance smoke**

Record in the PR test plan that the GPU host should rebuild the tiered image and run:

```bash
./kairyu-bench run http://127.0.0.1:8003/v1 \
  --only swe-bench-verified \
  --limit 1 \
  --run-id issue-480-roundtrip
```

The smoke passes only when the pinned mini-SWE-agent completes at least two API turns for the same task without `unsupported fields`; this host cannot perform the eight-GPU run, so do not claim the live gate closed without its artifact.

- [ ] **Step 6: Commit documentation and plan**

```bash
git add \
  PROGRESS.md \
  docs/design/m11-product.md \
  docs/design/example-layered-orchestration.md \
  docs/superpowers/plans/2026-08-14-chat-reasoning-roundtrip.md
git commit -m "docs(api): define assistant history compatibility"
```
