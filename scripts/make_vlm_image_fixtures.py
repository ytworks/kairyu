#!/usr/bin/env python3
"""Create deterministic, dependency-free PNG fixtures for the real VLM gate."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import struct
import zlib
from pathlib import Path

WIDTH = 1024
HEIGHT = 1024

# A compact 5x7 block font is sufficient here and avoids making the GPU gate
# depend on host fonts, Pillow, ImageMagick, or network-fetched assets.
GLYPHS = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
}


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _render_word(word: str, background: tuple[int, int, int]) -> bytes:
    pixels = bytearray(bytes(background) * WIDTH * HEIGHT)
    columns = len(word) * 5 + max(0, len(word) - 1)
    scale = min((WIDTH - 128) // columns, (HEIGHT - 128) // 7)
    word_width = columns * scale
    word_height = 7 * scale
    left = (WIDTH - word_width) // 2
    top = (HEIGHT - word_height) // 2
    white = b"\xff\xff\xff" * scale

    cursor = left
    for letter in word:
        glyph = GLYPHS[letter]
        for glyph_y, row in enumerate(glyph):
            for glyph_x, enabled in enumerate(row):
                if enabled != "1":
                    continue
                x = cursor + glyph_x * scale
                y = top + glyph_y * scale
                for target_y in range(y, y + scale):
                    start = (target_y * WIDTH + x) * 3
                    pixels[start : start + len(white)] = white
        cursor += 6 * scale

    scanlines = bytearray()
    row_bytes = WIDTH * 3
    for y in range(HEIGHT):
        scanlines.append(0)  # PNG filter type: None
        start = y * row_bytes
        scanlines.extend(pixels[start : start + row_bytes])

    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _chunk(b"IEND", b"")
    )


def _write_fixture(output_dir: Path, name: str, word: str, color: tuple[int, int, int]) -> dict:
    payload = _render_word(word, color)
    path = output_dir / f"{name}.png"
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "width": WIDTH,
        "height": HEIGHT,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "red": _write_fixture(args.output_dir, "red", "RED", (220, 20, 60)),
        "blue": _write_fixture(args.output_dir, "blue", "BLUE", (25, 80, 220)),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
