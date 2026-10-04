"""Sealed Kairyu-issued Responses items: the ``kst2`` codec and its key ring (m20 D6).

A sealed item is an opaque ``encrypted_content`` string that only a gateway
holding the issuing secret can open. After the ``kst2.`` prefix a token is
unpadded base64url of ``header ‖ AES-256-GCM ciphertext ‖ tag`` where::

    header = ver(1) ‖ purpose(1) ‖ kid(8) ‖ issued_at(8, big-endian s) ‖ salt(16)

- The per-token key and nonce come from HKDF-SHA256 over the secret named by
  ``kid``, salted with the token's random ``salt``, with the purpose as info, so
  no AES-GCM key is ever reused across tokens.
- The associated data binds the whole header and the owning tenant.
- The plaintext is JSON ``{"v": 2, "m": <served model or null>, **payload}``.
- The size cap is checked on the encoded token before any decoding, and the
  optional maximum age after authentication.

The primary secret seals; it and every previous (accept-only) secret open,
which makes rotation two-phase (``docs/deployment.md``). Previous-release
``kcp1.`` compaction tokens (one static AES-GCM key per secret, no key id or
issue time) remain decode-only.

Callers own the policy. Compaction fails closed on every :class:`SealedItemError`.
The reasoning purpose (WP-21) may drop an ``ignorable`` item (a foreign blob,
an unknown key id, an expired token) and rejects a malformed, modified or
cross-tenant one.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import BaseModel, ConfigDict, Field

KST2_PREFIX = "kst2."
KCP1_PREFIX = "kcp1."
MIN_SECRET_BYTES = 32
DEFAULT_SEALED_ITEM_MAX_BYTES = 4 * 1024 * 1024
# A floor far above any compaction summary, so the cap never refuses a token
# this server issued itself.
MIN_SEALED_ITEM_MAX_BYTES = 64 * 1024

_VERSION = 2
_DOMAIN = b"kairyu.sealed-item.v2\0"
_HEADER = struct.Struct(">BB8sQ16s")
_KID_BYTES = 8
_SALT_BYTES = 16
_KEY_BYTES = 32
_NONCE_BYTES = 12
_TAG_BYTES = 16
_RESERVED_FIELDS = frozenset({"v", "m"})
_LEGACY_KEY_DOMAIN = b"kairyu.responses.compaction.key.v1\0"
_LEGACY_AAD_DOMAIN = b"kairyu.responses.compaction.v1\0"

_NOT_ISSUED = "was not issued by this server"
_NOT_OPENED = "was not issued by this server for this tenant, or was modified"


class SealPurpose(IntEnum):
    """What a token seals; the code is part of the authenticated header."""

    COMPACTION = 1
    REASONING = 2


class SealingConfig(BaseModel):
    """Key-ring sources and limits beside ``responses_compaction_secret_env``.

    One definition shared by ``ServerSettings`` and the deployment
    ``server.sealing`` section (m20 C23). The primary secret keeps its
    existing ``responses_compaction_secret_env`` field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    previous_secrets_env: str | None = Field(
        default=None,
        description=(
            "Env var holding comma-separated previous secrets that still open "
            "sealed items but never seal (two-phase rotation); unset or empty "
            "means none."
        ),
    )
    sealed_max_age_s: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Refuse sealed items issued longer ago than this; None keeps them "
            "valid while their key is accepted. Legacy kcp1 tokens carry no "
            "issue time and are refused once this is set."
        ),
    )
    sealed_item_max_bytes: int = Field(
        default=DEFAULT_SEALED_ITEM_MAX_BYTES,
        ge=MIN_SEALED_ITEM_MAX_BYTES,
        description="Largest encoded sealed item accepted; checked before decoding.",
    )
    ephemeral: bool = Field(
        default=False,
        description=(
            "Acknowledge a process-local key in a deployment without a secret "
            "(sealed items do not survive restarts or gateway hops)."
        ),
    )


def check_sealing_sources(secret_env: str | None, sealing: SealingConfig) -> None:
    """Reject key sources that contradict each other (settings and deploy spec)."""

    if sealing.ephemeral and secret_env is not None:
        raise ValueError(
            "sealing.ephemeral contradicts responses_compaction_secret_env"
        )
    if sealing.previous_secrets_env is not None and secret_env is None:
        raise ValueError(
            "sealing.previous_secrets_env requires responses_compaction_secret_env"
        )


class SealedItemError(Exception):
    """A token the ring cannot open; ``reason`` completes "encrypted_content …"."""

    def __init__(self, reason: str, *, ignorable: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        # True when the item may simply be dropped (reasoning purpose): it was
        # not ours, or it is ours but no longer openable. Never for tampering.
        self.ignorable = ignorable


@dataclass(frozen=True)
class SealingKey:
    """One secret: its public key id, HKDF input and legacy kcp1 AES key."""

    kid: bytes
    root: bytes = field(repr=False)
    legacy_key: bytes | None = field(default=None, repr=False)

    @classmethod
    def from_secret(cls, secret: bytes) -> SealingKey:
        if len(secret) < MIN_SECRET_BYTES:
            raise ValueError(
                f"sealing secrets must contain at least {MIN_SECRET_BYTES} bytes"
            )
        return cls(
            kid=_key_id(secret),
            root=secret,
            legacy_key=hashlib.sha256(_LEGACY_KEY_DOMAIN + secret).digest(),
        )

    @classmethod
    def ephemeral(cls) -> SealingKey:
        root = secrets.token_bytes(_KEY_BYTES)
        return cls(kid=_key_id(root), root=root)


@dataclass(frozen=True)
class UnsealedItem:
    """An authenticated payload and the served model it was sealed for."""

    model: str | None
    payload: Mapping[str, object]


@dataclass(frozen=True)
class SealingKeyRing:
    """Seals with ``primary``; opens with it and every ``previous`` key."""

    primary: SealingKey
    previous: tuple[SealingKey, ...] = ()
    max_age_s: float | None = None
    max_bytes: int = DEFAULT_SEALED_ITEM_MAX_BYTES
    clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)

    @classmethod
    def from_secrets(
        cls,
        primary: bytes | None,
        previous: Sequence[bytes],
        config: SealingConfig,
    ) -> SealingKeyRing:
        """``primary=None`` seals with a process-local ephemeral key."""

        primary_key = (
            SealingKey.ephemeral() if primary is None else SealingKey.from_secret(primary)
        )
        accepted = {key.kid: key for key in map(SealingKey.from_secret, previous)}
        accepted.pop(primary_key.kid, None)
        return cls(
            primary_key,
            tuple(accepted.values()),
            max_age_s=config.sealed_max_age_s,
            max_bytes=config.sealed_item_max_bytes,
        )

    def seal(
        self,
        purpose: SealPurpose,
        payload: Mapping[str, object],
        *,
        owner: str,
        model: str | None = None,
    ) -> str:
        if _RESERVED_FIELDS & payload.keys():
            raise ValueError("sealed payloads must not use the reserved fields v and m")
        salt = secrets.token_bytes(_SALT_BYTES)
        header = _HEADER.pack(_VERSION, purpose, self.primary.kid, int(self.clock()), salt)
        key, nonce = _token_key(self.primary.root, salt, purpose)
        plaintext = json.dumps(
            {"v": _VERSION, "m": model, **payload},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        sealed = AESGCM(key).encrypt(nonce, plaintext, _associated_data(header, owner))
        return KST2_PREFIX + _b64encode(header + sealed)

    def open(self, token: object, *, purpose: SealPurpose, owner: str) -> UnsealedItem:
        if not isinstance(token, str) or not token:
            raise SealedItemError("must be a non-empty string")
        if len(token) > self.max_bytes:
            raise SealedItemError(f"exceeds the {self.max_bytes}-byte sealed item limit")
        if token.startswith(KST2_PREFIX):
            return self._open_kst2(token.removeprefix(KST2_PREFIX), purpose, owner)
        if token.startswith(KCP1_PREFIX) and purpose is SealPurpose.COMPACTION:
            return self._open_kcp1(token.removeprefix(KCP1_PREFIX), owner)
        raise SealedItemError(_NOT_ISSUED, ignorable=True)

    def _open_kst2(self, encoded: str, purpose: SealPurpose, owner: str) -> UnsealedItem:
        raw = _b64decode(encoded)
        if len(raw) < _HEADER.size + _TAG_BYTES:
            raise SealedItemError(_NOT_OPENED)
        header = raw[: _HEADER.size]
        version, code, kid, issued_at, salt = _HEADER.unpack(header)
        if version != _VERSION or code != purpose:
            raise SealedItemError(_NOT_OPENED)
        key = next((key for key in (self.primary, *self.previous) if key.kid == kid), None)
        if key is None:
            raise SealedItemError(
                "was sealed with a key this server no longer accepts", ignorable=True
            )
        aead_key, nonce = _token_key(key.root, salt, purpose)
        try:
            plaintext = AESGCM(aead_key).decrypt(
                nonce, raw[_HEADER.size :], _associated_data(header, owner)
            )
        except InvalidTag:
            raise SealedItemError(_NOT_OPENED) from None
        if self.max_age_s is not None and self.clock() - issued_at > self.max_age_s:
            raise SealedItemError("has expired", ignorable=True)
        return _unsealed(plaintext)

    def _open_kcp1(self, encoded: str, owner: str) -> UnsealedItem:
        if self.max_age_s is not None:
            # No issue time to check: an operator who bounds token age
            # retires every previous-release token with it.
            raise SealedItemError("has expired", ignorable=True)
        raw = _b64decode(encoded)
        if len(raw) < _NONCE_BYTES + _TAG_BYTES:
            raise SealedItemError(_NOT_OPENED)
        associated = _LEGACY_AAD_DOMAIN + owner.encode("utf-8")
        for key in (self.primary, *self.previous):
            if key.legacy_key is None:
                continue
            try:
                plaintext = AESGCM(key.legacy_key).decrypt(
                    raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], associated
                )
            except InvalidTag:
                continue
            try:
                summary = plaintext.decode("utf-8")
            except UnicodeDecodeError:
                break
            # kcp1 sealed a bare compaction summary: map it to the v2 payload.
            return UnsealedItem(model=None, payload=MappingProxyType({"summary": summary}))
        raise SealedItemError(_NOT_OPENED)


def _key_id(secret: bytes) -> bytes:
    # A keyed digest, so the public id reveals nothing about the secret.
    return hmac.new(secret, _DOMAIN + b"kid", hashlib.sha256).digest()[:_KID_BYTES]


def _token_key(root: bytes, salt: bytes, purpose: SealPurpose) -> tuple[bytes, bytes]:
    material = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_BYTES + _NONCE_BYTES,
        salt=salt,
        info=_DOMAIN + purpose.name.lower().encode("ascii"),
    ).derive(root)
    return material[:_KEY_BYTES], material[_KEY_BYTES:]


def _associated_data(header: bytes, owner: str) -> bytes:
    # The header has a fixed size, so header ‖ owner is unambiguous.
    return _DOMAIN + header + owner.encode("utf-8")


def _unsealed(plaintext: bytes) -> UnsealedItem:
    # Authenticated, so a malformed body means a server bug: fail closed.
    try:
        body = json.loads(plaintext)
    except ValueError:
        raise SealedItemError(_NOT_OPENED) from None
    if not isinstance(body, dict) or body.get("v") != _VERSION:
        raise SealedItemError(_NOT_OPENED)
    model = body.get("m")
    if model is not None and not isinstance(model, str):
        raise SealedItemError(_NOT_OPENED)
    payload = {name: value for name, value in body.items() if name not in _RESERVED_FIELDS}
    return UnsealedItem(model=model, payload=MappingProxyType(payload))


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(encoded: str) -> bytes:
    try:
        return base64.b64decode(
            encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True
        )
    except (binascii.Error, ValueError):
        raise SealedItemError(_NOT_OPENED) from None
