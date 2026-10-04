# M20 Design: Sealed Responses items v2 (D6) — Implemented (WP-20 part)

Status: **Implemented for compaction** (2026-10-05, WP-20). The reasoning
purpose is defined here and consumed by WP-21; `/compact` reuses it in WP-36.
Milestone: M20 (`docs/design/m20-responses-compat.md`, D6, C1/C22/C23)
Date: 2026-10-05
Closes: G-compaction-3 (key ring, kid, max age, per-token subkeys; `created_by`
is won't-do per C21), M-SR-3 and M-ST-3 (staged secret requirement).

## Context

Kairyu returns opaque `encrypted_content` that only it can read: today the
remote-compaction summary (Codex v2), and from WP-21 the reasoning items. The
first format (`kcp1.`, issue #530) sealed every token with one static AES-GCM
key derived from `server.responses_compaction_secret_env` and a random 96-bit
nonce:

- **Nonce budget.** A static key bounds how many tokens it may safely seal
  (random 96-bit nonces), and the count grows with every gateway sharing it.
- **No rotation.** A token carried no key id, so changing the secret stranded
  every outstanding compacted Codex session at once.
- **No bounds.** Tokens never expired, and any size was base64- and
  AES-decoded before refusal.
- **Optional secret.** Without the secret each process used an ephemeral key,
  so a restart, a rolling update or a hop to another gateway broke compacted
  sessions at any replica count, silently.

## Decision (D6, C1)

**Token.** `kst2.` + unpadded base64url of
`header ‖ AES-256-GCM(plaintext) ‖ tag` with a fixed 34-byte header
`ver(1)=2 ‖ purpose(1) ‖ kid(8) ‖ issued_at(8, big-endian Unix s) ‖ salt(16)`.

- **Per-token key.** HKDF-SHA256(IKM = the secret named by `kid`, salt =
  `salt`, info = `kairyu.sealed-item.v2\0` + purpose) yields the 32-byte key
  and the 12-byte nonce. Every token has its own key, so there is no shared
  nonce budget (FEA-9). AES-GCM-SIV was rejected to keep one primitive.
- **Binding.** AAD = domain ‖ header ‖ owner (tenant). The purpose, key id and
  issue time are authenticated, so a token never opens for another tenant or
  purpose.
- **Plaintext.** JSON `{"v": 2, "m": <served model or null>, **payload}`.
  Compaction seals `{summary}` with `m: null`; WP-21 seals reasoning with the
  served model so that replay to another model drops it.
- **Key id.** The first 8 bytes of HMAC-SHA256(secret, domain + `kid`); it
  reveals nothing about the secret.
- **Limits.** `sealed_item_max_bytes` (default 4 MiB, floor 64 KiB so the cap
  never refuses a summary this server issued) is checked on the encoded token
  before base64 or AES runs. `sealed_max_age_s` (default none) is checked
  after authentication.
- **Legacy.** `kcp1.` tokens stay decode-only (compaction purpose, every
  configured secret tried). They carry no issue time and are refused once
  `sealed_max_age_s` is set.

**Key ring.** The primary secret seals; it and every previous secret open.
Secrets are used byte-exact and need at least 32 bytes. Without a primary
secret, a process-local ephemeral key seals and nothing else opens.

**Policy** (callers own it; `SealedItemError.ignorable` marks the droppable
cases).

| Case | Compaction | Reasoning (WP-21) |
|---|---|---|
| Not a Kairyu token (foreign blob) | 400 | dropped |
| Unknown key id (rotated out, other deployment) | 400 | dropped |
| Expired (`sealed_max_age_s`) | 400 | dropped |
| Malformed, modified, other tenant or purpose | 400 | 400 |
| Larger than `sealed_item_max_bytes` | 400 | 400 |

A refusal is 400 `invalid_encrypted_content` with
`param: "input[i].encrypted_content"` and a message telling the client to start
a new session (Codex: `/new`). Codex does not retry it.

## Configuration (C23)

The primary secret keeps its existing field and name,
`server.responses_compaction_secret_env` (no alias); it now seals every
Kairyu-issued item. One frozen `SealingConfig`
(`kairyu/entrypoints/server/responses/sealing.py`) is the single definition
embedded in both `ServerSettings.sealing` and the deployment `server.sealing`
section. WP-08a nests it as `ResponsesConfig.sealing`.

| Field | Meaning |
|---|---|
| `previous_secrets_env` | Env var with comma-separated accept-only secrets; unset or empty means none. Requires a primary secret. |
| `sealed_max_age_s` | Optional maximum token age. |
| `sealed_item_max_bytes` | Encoded-size cap (default 4 MiB, minimum 64 KiB). |
| `ephemeral` | Deployment acknowledgement of a process-local key. Contradicts a configured secret. |

## Staged secret requirement (C22)

Restarts and rolling updates break compacted sessions even with one replica,
so the requirement does not depend on the replica count.

- **This release (N):** a DeploymentSpec without
  `server.responses_compaction_secret_env` and without
  `server.sealing.ephemeral: true` gets a warning from `kairyu validate`
  (`schema.sealing_secret_missing`; the report stays VALID) and at startup
  (`kairyu.deploy.responses_validation`).
- **Next release (N+1):** the same condition fails `kairyu validate` and
  startup. This is recorded here as the follow-up; nothing in N enforces it.
- Direct `create_app` callers (tests, embedding) keep the ephemeral key
  without a warning.
- **Helm** (`responsesSealing`): it injects the secret and the optional
  previous secrets from one Secret. Unless `existingSecret` names one, the
  chart generates `<release>-responses-sealing` (64 random characters),
  reuses it on upgrade via `lookup`, and keeps it on uninstall
  (`helm.sh/resource-policy: keep`). GitOps renders cannot `lookup`, so they
  use `existingSecret`. The kind f1c fixture shares one secret across its
  three gateways.

## Two-phase rotation

1. Add the new secret to `previous_secrets_env` on every gateway and finish
   the rollout. Every gateway now opens tokens sealed with either key.
2. Make the new secret the primary and move the old one to
   `previous_secrets_env`.
3. Remove the old secret once its tokens are gone: after the longest Codex
   session lifetime, or after `sealed_max_age_s` when it is set (stored
   responses: plus the store retention from WP-32).

A one-step swap fails during the rolling update: a gateway that is already
promoted issues tokens that a gateway not yet rolled cannot open (unknown key
id → 400).

## Tests

`tests/server/responses/test_compaction.py` covers the following:

- two-phase rotation, including the unknown-key refusal on both sides;
- `kcp1.` decode after the upgrade;
- refusal of an oversized token before decoding;
- the maximum age;
- tamper, forgery and cross-tenant refusal with the typed error.

`tests/unit/test_validate_cli.py` covers the staged warning in
`kairyu validate` and at startup, and the ephemeral opt-in. The Helm render
test asserts the kept, generated Secret and its env reference.
