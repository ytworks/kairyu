# Issue 482 LiteLLM Assistant-History Compatibility Design

**Status:** Approved

**Issue:** `ytworks/kairyu#482`

## Goal

Accept the exact assistant-history dictionary produced by LiteLLM 1.96.2 while
keeping compatibility-only metadata out of prompts and rejecting unrelated
message extras before backend dispatch.

## Policy

- Keep `reasoning_content` as a typed request field.
- Continue accepting and dropping `provider_specific_fields: null` on any role.
- Accept and drop object-valued `provider_specific_fields` only on assistant
  messages. Never inspect or render its contents.
- Accept and drop `function_call: null` only on assistant messages. Non-null
  legacy function calls remain unsupported; typed `tool_calls` remains the
  supported tool-history representation.
- Reject scalar or list provider metadata, provider objects on non-assistant
  roles, non-null `function_call`, and every other unknown field with HTTP 400
  before backend work.
- Do not add compatibility-only fields to `ChatMessage` or OpenAPI.

The validation and wire-copy paths use one predicate that receives the whole
message, ensuring that role-sensitive acceptance and prompt filtering cannot
diverge.

## Verification

Tests cover the exact LiteLLM dictionary, accepted provider metadata variants,
prompt non-leakage, all negative branches, the L2 response round trip, and the
public schema. A live SWE-bench acceptance run remains a deployment gate and is
not simulated in the portable suite.
