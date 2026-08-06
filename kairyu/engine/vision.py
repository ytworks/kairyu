"""Fail-closed image-input policy for remote OpenAI-compatible VLMs.

The first production contract accepts only inline raster data URLs.  Kairyu
therefore performs no tenant-controlled network or local-file I/O: Open WebUI
already converts authorized uploads to data URLs, and stock vLLM receives the
same structured Chat Completions payload.  Remote URLs can be added later only
with an independently reviewed fetcher (redirect, DNS, byte, and timeout
ownership); they are not silently delegated today.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import threading
import warnings
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO

from kairyu.engine.prompt import MultimodalItem, MultimodalPrompt

_DATA_URL = re.compile(
    r"\Adata:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/]*={0,2})\Z"
)
_DATA_URL_PREFIXES = (
    "data:image/png;base64,",
    "data:image/jpeg;base64,",
    "data:image/webp;base64,",
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_START = b"\xff\xd8"
_WEBP_RIFF = b"RIFF"
_WEBP_SIGNATURE = b"WEBP"
_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
_VERIFIED_RASTER_CACHE_LIMIT = 512
_verified_raster_cache: OrderedDict[bytes, None] = OrderedDict()
_verified_raster_cache_lock = threading.Lock()
_prepared_image_cache_lock = threading.RLock()


@dataclass(frozen=True)
class _PreparedImageInput:
    data_urls: tuple[str, ...]
    decoded: tuple[tuple[str, bytes], ...]
    verified: bool = False


_prepared_image_cache: dict[
    int,
    tuple[
        weakref.ReferenceType[MultimodalPrompt],
        dict[ImageInputPolicy, _PreparedImageInput],
    ],
] = {}


def _raster_verification_key(mime: str, data: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(mime.encode("ascii"))
    digest.update(b"\x00")
    digest.update(data)
    return digest.digest()


def _raster_was_verified(key: bytes) -> bool:
    with _verified_raster_cache_lock:
        if key not in _verified_raster_cache:
            return False
        _verified_raster_cache.move_to_end(key)
        return True


def _remember_verified_raster(key: bytes) -> None:
    with _verified_raster_cache_lock:
        _verified_raster_cache[key] = None
        _verified_raster_cache.move_to_end(key)
        while len(_verified_raster_cache) > _VERIFIED_RASTER_CACHE_LIMIT:
            _verified_raster_cache.popitem(last=False)


def _clear_verified_raster_cache() -> None:
    """Test helper; production verification cache is otherwise process-lived."""

    with _verified_raster_cache_lock:
        _verified_raster_cache.clear()
    with _prepared_image_cache_lock:
        _prepared_image_cache.clear()


class InvalidImageInput(ValueError):
    """A tenant image cannot cross the VLM boundary."""

    code = "invalid_image"


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if (
        len(data) < 33
        or not data.startswith(_PNG_SIGNATURE)
        or data[12:16] != b"IHDR"
        or int.from_bytes(data[8:12], "big") != 13
    ):
        raise InvalidImageInput("image data is not a valid PNG header")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if not data.startswith(_JPEG_START):
        raise InvalidImageInput("image data is not a valid JPEG header")
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            raise InvalidImageInput("image data has a malformed JPEG segment")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if marker in {0xD9, 0xDA}:
            break
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            raise InvalidImageInput("image data has a malformed JPEG segment length")
        if marker in _JPEG_SOF_MARKERS:
            if length < 7:
                raise InvalidImageInput("image data has a truncated JPEG size segment")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            return width, height
        offset += length
    raise InvalidImageInput("image data does not contain JPEG dimensions")


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    if (
        len(data) < 30
        or data[:4] != _WEBP_RIFF
        or data[8:12] != _WEBP_SIGNATURE
        or int.from_bytes(data[4:8], "little") + 8 != len(data)
    ):
        raise InvalidImageInput("image data is not a valid WebP container")
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    if chunk == b"VP8 ":
        if len(data) < 30 or data[23:26] != b"\x9d\x01\x2a":
            raise InvalidImageInput("image data has a malformed VP8 frame header")
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8L":
        if len(data) < 25 or data[20] != 0x2F:
            raise InvalidImageInput("image data has a malformed VP8L frame header")
        b1, b2, b3, b4 = data[21:25]
        width = 1 + b1 + ((b2 & 0x3F) << 8)
        height = 1 + ((b2 >> 6) | (b3 << 2) | ((b4 & 0x0F) << 10))
        return width, height
    raise InvalidImageInput("image data uses an unsupported WebP frame encoding")


def _image_metadata(data: bytes) -> tuple[str, int, int]:
    if data.startswith(_PNG_SIGNATURE):
        width, height = _png_dimensions(data)
        return "image/png", width, height
    if data.startswith(_JPEG_START):
        width, height = _jpeg_dimensions(data)
        return "image/jpeg", width, height
    if data.startswith(_WEBP_RIFF):
        width, height = _webp_dimensions(data)
        return "image/webp", width, height
    raise InvalidImageInput("image data must be PNG, JPEG, or WebP")


@dataclass(frozen=True)
class ImageInputPolicy:
    """Immutable validation and admission contract shared by a replica pool."""

    max_images: int = 4
    max_image_bytes: int = 8 * 1024 * 1024
    max_total_image_bytes: int = 16 * 1024 * 1024
    max_image_pixels: int = 16_777_216
    max_total_image_pixels: int = 16_777_216
    max_image_dimension: int = 8192
    max_image_aspect_ratio: int = 200
    max_processed_prompt_tokens: int = 32_768

    def __post_init__(self) -> None:
        for field in (
            "max_images",
            "max_image_bytes",
            "max_total_image_bytes",
            "max_image_pixels",
            "max_total_image_pixels",
            "max_image_dimension",
            "max_image_aspect_ratio",
            "max_processed_prompt_tokens",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"image policy {field} must be a positive integer")
        if self.max_total_image_bytes < self.max_image_bytes:
            raise ValueError(
                "image policy max_total_image_bytes must be at least max_image_bytes"
            )

    @staticmethod
    def require_runtime() -> None:
        try:
            import PIL.Image  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "multimodal image validation requires the 'vision' extra "
                "(pip install 'kairyu[vision]')"
            ) from error

    @staticmethod
    def _verify_raster(data: bytes, *, mime: str, index: int) -> None:
        from PIL import Image, UnidentifiedImageError

        expected_format = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/webp": "WEBP",
        }[mime]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as image:
                    if image.format != expected_format:
                        raise InvalidImageInput(
                            f"image {index} decoder format does not match its media type"
                        )
                    if getattr(image, "n_frames", 1) != 1:
                        raise InvalidImageInput(
                            f"image {index} must contain exactly one frame"
                        )
                    image.verify()
                # ``verify`` does not decode pixels. Reopen and load once so
                # truncated compressed payloads fail before vLLM dispatch.
                with Image.open(BytesIO(data)) as image:
                    image.load()
        except InvalidImageInput:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
        ) as error:
            raise InvalidImageInput(
                f"image {index} is not a decodable {expected_format} image"
            ) from error

    def _decode_item(self, item: MultimodalItem, *, index: int) -> tuple[str, bytes]:
        if item.modality != "image":
            raise InvalidImageInput(
                f"multimodal item {index} has unsupported modality {item.modality!r}"
            )
        declared_mime: str | None = None
        if item.encoding == "uri":
            assert isinstance(item.data, str)
            match = _DATA_URL.fullmatch(item.data)
            if match is None:
                raise InvalidImageInput(
                    f"image {index} must use an inline PNG, JPEG, or WebP data URL; "
                    "remote and local image URLs are disabled"
                )
            declared_mime, encoded = match.groups()
        elif item.encoding == "base64":
            assert isinstance(item.data, str)
            encoded = item.data
        elif item.encoding == "bytes":
            assert isinstance(item.data, bytes)
            data = item.data
            mime, _width, _height = _image_metadata(data)
            return mime, data
        else:
            raise InvalidImageInput(
                f"image {index} uses unsupported encoding {item.encoding!r}"
            )

        # Reject an oversized payload before allocating its decoded copy.
        if len(encoded) > ((self.max_image_bytes + 2) // 3) * 4:
            raise InvalidImageInput(
                f"image {index} exceeds the {self.max_image_bytes}-byte limit"
            )
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise InvalidImageInput(
                f"image {index} contains invalid base64 data"
            ) from error
        mime, _width, _height = _image_metadata(data)
        if declared_mime is not None and declared_mime != mime:
            raise InvalidImageInput(
                f"image {index} media type does not match its encoded bytes"
            )
        return mime, data

    def _cached_preparation(
        self,
        prompt: MultimodalPrompt,
    ) -> _PreparedImageInput | None:
        with _prepared_image_cache_lock:
            entry = _prepared_image_cache.get(id(prompt))
            if entry is None or entry[0]() is not prompt:
                return None
            return entry[1].get(self)

    def _remember_preparation(
        self,
        prompt: MultimodalPrompt,
        prepared: _PreparedImageInput,
    ) -> None:
        prompt_id = id(prompt)

        def discard(reference: weakref.ReferenceType[MultimodalPrompt]) -> None:
            with _prepared_image_cache_lock:
                current = _prepared_image_cache.get(prompt_id)
                if current is not None and current[0] is reference:
                    _prepared_image_cache.pop(prompt_id, None)

        with _prepared_image_cache_lock:
            entry = _prepared_image_cache.get(prompt_id)
            if entry is None or entry[0]() is not prompt:
                reference = weakref.ref(prompt, discard)
                policies: dict[ImageInputPolicy, _PreparedImageInput] = {}
                _prepared_image_cache[prompt_id] = (reference, policies)
            else:
                policies = entry[1]
            policies[self] = prepared

    def _prepare_prompt(
        self,
        prompt: MultimodalPrompt,
    ) -> _PreparedImageInput:
        cached = self._cached_preparation(prompt)
        if cached is not None:
            return cached
        if not prompt.messages:
            raise InvalidImageInput(
                "OpenAI-compatible VLM input requires a role-preserving chat layout"
            )
        if len(prompt.items) > self.max_images:
            raise InvalidImageInput(
                f"image count {len(prompt.items)} exceeds the {self.max_images}-image limit"
            )

        total_bytes = 0
        total_pixels = 0
        data_urls: list[str] = []
        decoded: list[tuple[str, bytes]] = []
        for index, item in enumerate(prompt.items):
            mime, data = self._decode_item(item, index=index)
            if len(data) > self.max_image_bytes:
                raise InvalidImageInput(
                    f"image {index} exceeds the {self.max_image_bytes}-byte limit"
                )
            total_bytes += len(data)
            if total_bytes > self.max_total_image_bytes:
                raise InvalidImageInput(
                    "combined image data exceeds the "
                    f"{self.max_total_image_bytes}-byte limit"
                )
            _detected_mime, width, height = _image_metadata(data)
            if width < 1 or height < 1:
                raise InvalidImageInput(f"image {index} has invalid dimensions")
            if width > self.max_image_dimension or height > self.max_image_dimension:
                raise InvalidImageInput(
                    f"image {index} exceeds the {self.max_image_dimension}-pixel "
                    "dimension limit"
                )
            short_edge = min(width, height)
            long_edge = max(width, height)
            if long_edge > self.max_image_aspect_ratio * short_edge:
                raise InvalidImageInput(
                    f"image {index} exceeds the {self.max_image_aspect_ratio}:1 "
                    "aspect-ratio limit"
                )
            if width * height > self.max_image_pixels:
                raise InvalidImageInput(
                    f"image {index} exceeds the {self.max_image_pixels}-pixel limit"
                )
            total_pixels += width * height
            if total_pixels > self.max_total_image_pixels:
                raise InvalidImageInput(
                    "combined image pixels exceed the "
                    f"{self.max_total_image_pixels}-pixel limit"
                )
            decoded.append((mime, data))
            data_urls.append(
                f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            )
        prepared = _PreparedImageInput(
            data_urls=tuple(data_urls),
            decoded=tuple(decoded),
        )
        self._remember_preparation(prompt, prepared)
        return prepared

    def validate_prompt_headers(self, prompt: MultimodalPrompt) -> None:
        """Perform only constant-memory shape/encoded-size checks in-loop."""

        if not prompt.messages:
            raise InvalidImageInput(
                "OpenAI-compatible VLM input requires a role-preserving chat layout"
            )
        if len(prompt.items) > self.max_images:
            raise InvalidImageInput(
                f"image count {len(prompt.items)} exceeds the "
                f"{self.max_images}-image limit"
            )
        total_bytes = 0
        for index, item in enumerate(prompt.items):
            if item.modality != "image":
                raise InvalidImageInput(
                    f"multimodal item {index} has unsupported modality "
                    f"{item.modality!r}"
                )
            if item.encoding == "uri":
                assert isinstance(item.data, str)
                prefix = next(
                    (
                        candidate
                        for candidate in _DATA_URL_PREFIXES
                        if item.data.startswith(candidate)
                    ),
                    None,
                )
                if prefix is None:
                    raise InvalidImageInput(
                        f"image {index} must use an inline PNG, JPEG, or WebP "
                        "data URL; remote and local image URLs are disabled"
                    )
                encoded_length = len(item.data) - len(prefix)
                padding = int(item.data.endswith("=")) + int(
                    item.data.endswith("==")
                )
                decoded_bytes = max(
                    0,
                    ((encoded_length + 3) // 4) * 3 - padding,
                )
            elif item.encoding == "base64":
                assert isinstance(item.data, str)
                encoded_length = len(item.data)
                padding = int(item.data.endswith("=")) + int(
                    item.data.endswith("==")
                )
                decoded_bytes = max(
                    0,
                    ((encoded_length + 3) // 4) * 3 - padding,
                )
            elif item.encoding == "bytes":
                assert isinstance(item.data, bytes)
                decoded_bytes = len(item.data)
            else:
                raise InvalidImageInput(
                    f"image {index} uses unsupported encoding {item.encoding!r}"
                )
            if decoded_bytes > self.max_image_bytes:
                raise InvalidImageInput(
                    f"image {index} exceeds the {self.max_image_bytes}-byte limit"
                )
            total_bytes += decoded_bytes
            if total_bytes > self.max_total_image_bytes:
                raise InvalidImageInput(
                    "combined image data exceeds the "
                    f"{self.max_total_image_bytes}-byte limit"
                )

    def validate_prompt(self, prompt: MultimodalPrompt) -> tuple[str, ...]:
        """Fully decode and canonicalize images before an upstream dispatch."""

        prepared = self._prepare_prompt(prompt)
        if prepared.verified:
            return prepared.data_urls
        for index, (mime, data) in enumerate(prepared.decoded):
            verification_key = _raster_verification_key(mime, data)
            if not _raster_was_verified(verification_key):
                self._verify_raster(data, mime=mime, index=index)
                _remember_verified_raster(verification_key)
        prepared = _PreparedImageInput(
            data_urls=prepared.data_urls,
            decoded=prepared.decoded,
            verified=True,
        )
        self._remember_preparation(prompt, prepared)
        return prepared.data_urls

    def cached_validated_prompt(
        self,
        prompt: MultimodalPrompt,
    ) -> tuple[str, ...] | None:
        """Return a fully verified exact-prompt result without decoding work."""

        prepared = self._cached_preparation(prompt)
        if prepared is None or not prepared.verified:
            return None
        return prepared.data_urls


__all__ = ["ImageInputPolicy", "InvalidImageInput"]
