from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalMessage,
    MultimodalMessagePart,
    MultimodalPrompt,
)
from kairyu.engine.vision import ImageInputPolicy, InvalidImageInput


def _image_bytes(
    *,
    format: str = "PNG",
    size: tuple[int, int] = (2, 3),
    color: str = "red",
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format=format)
    return output.getvalue()


def _data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _prompt(*items: MultimodalItem) -> MultimodalPrompt:
    return MultimodalPrompt(
        base="display",
        items=items,
        messages=(
            MultimodalMessage(
                "user",
                tuple(
                    MultimodalMessagePart("item", item_index=index)
                    for index in range(len(items))
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("format", "mime"),
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("WEBP", "image/webp"),
    ],
)
def test_inline_raster_is_verified_and_canonicalized(format, mime):
    data = _image_bytes(format=format)
    policy = ImageInputPolicy(max_images=1)

    resolved = policy.validate_prompt(
        _prompt(MultimodalItem("image", "uri", _data_url(data, mime)))
    )

    assert resolved == (_data_url(data, mime),)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://example.test/image.png", "remote and local image URLs are disabled"),
        ("file:///etc/passwd", "remote and local image URLs are disabled"),
        ("data:image/svg+xml;base64,PHN2Zy8+", "inline PNG, JPEG, or WebP"),
        ("data:image/png;base64,%%%%", "inline PNG, JPEG, or WebP"),
    ],
)
def test_network_local_and_non_raster_image_references_fail_closed(url, message):
    with pytest.raises(InvalidImageInput, match=message):
        ImageInputPolicy(max_images=1).validate_prompt(
            _prompt(MultimodalItem("image", "uri", url))
        )


def test_declared_media_type_must_match_bytes():
    jpeg = _image_bytes(format="JPEG")

    with pytest.raises(InvalidImageInput, match="media type does not match"):
        ImageInputPolicy(max_images=1).validate_prompt(
            _prompt(
                MultimodalItem(
                    "image",
                    "uri",
                    _data_url(jpeg, "image/png"),
                )
            )
        )


def test_truncated_image_is_rejected_before_backend_dispatch():
    png = _image_bytes()

    with pytest.raises(InvalidImageInput, match="not a decodable PNG"):
        ImageInputPolicy(max_images=1).validate_prompt(
            _prompt(
                MultimodalItem(
                    "image",
                    "uri",
                    _data_url(png[:-8]),
                )
            )
        )


def test_image_count_individual_combined_byte_and_pixel_limits_are_independent():
    small = _image_bytes(size=(2, 2))
    second = _image_bytes(size=(3, 2), color="blue")
    two = _prompt(
        MultimodalItem("image", "bytes", small),
        MultimodalItem("image", "bytes", second),
    )

    with pytest.raises(InvalidImageInput, match="image count"):
        ImageInputPolicy(max_images=1).validate_prompt(two)
    with pytest.raises(InvalidImageInput, match="image 0 exceeds"):
        ImageInputPolicy(
            max_images=2,
            max_image_bytes=len(small) - 1,
            max_total_image_bytes=len(small) + len(second),
        ).validate_prompt(two)
    with pytest.raises(InvalidImageInput, match="combined image data"):
        ImageInputPolicy(
            max_images=2,
            max_image_bytes=max(len(small), len(second)),
            max_total_image_bytes=len(small) + len(second) - 1,
        ).validate_prompt(two)
    with pytest.raises(InvalidImageInput, match="pixel limit"):
        ImageInputPolicy(
            max_images=2,
            max_image_pixels=3,
        ).validate_prompt(two)
    with pytest.raises(InvalidImageInput, match="combined image pixels"):
        ImageInputPolicy(
            max_images=2,
            max_image_pixels=6,
            max_total_image_pixels=9,
        ).validate_prompt(two)


def test_image_aspect_ratio_matches_qwen_smart_resize_boundary():
    policy = ImageInputPolicy(max_images=1, max_image_aspect_ratio=200)

    accepted = _image_bytes(size=(200, 1))
    assert policy.validate_prompt(
        _prompt(MultimodalItem("image", "bytes", accepted))
    ) == (_data_url(accepted),)

    rejected = _image_bytes(size=(201, 1))
    with pytest.raises(InvalidImageInput, match="200:1 aspect-ratio limit"):
        policy.validate_prompt(
            _prompt(MultimodalItem("image", "bytes", rejected))
        )


def test_image_aspect_ratio_limit_must_be_a_positive_integer():
    for value in (0, -1, 200.0, True):
        with pytest.raises(ValueError, match="max_image_aspect_ratio"):
            ImageInputPolicy(max_image_aspect_ratio=value)  # type: ignore[arg-type]


def test_openai_vlm_policy_requires_role_preserving_layout():
    prompt = MultimodalPrompt(
        base="describe",
        items=(MultimodalItem("image", "bytes", _image_bytes()),),
    )

    with pytest.raises(InvalidImageInput, match="role-preserving"):
        ImageInputPolicy().validate_prompt(prompt)


def test_header_gate_does_not_treat_unbounded_padding_as_zero_bytes():
    prompt = _prompt(
        MultimodalItem(
            "image",
            "uri",
            "data:image/png;base64," + ("=" * 16),
        )
    )

    with pytest.raises(InvalidImageInput, match="byte limit"):
        ImageInputPolicy(
            max_images=1,
            max_image_bytes=8,
            max_total_image_bytes=8,
        ).validate_prompt_headers(prompt)
