"""The preview's reference image: one `seams` frame through `render_bloom`.

docs/images/preview/seams-render_bloom.png is what the browser renderer
(web/static/preview.js) is checked against by eye, side by side with its
own drawing of the same record (seams-browser.png beside it). This test
keeps the reference honest: it fails if `render_bloom` or `seams` drifts
from the committed image.

    DF2_UPDATE_REFERENCE=1 pytest test/test_preview_reference.py

rewrites the PNG after an intentional change to either.
"""

import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from df2_pi.animation import AnimationRegistry, default_animations_dir
from df2_pi.output.dev import render_bloom
from df2_pi.output.preview import encode_preview
from df2_pi.pixels import default_geometry

DOCS = Path(__file__).resolve().parents[3] / "docs"  # the repo's; absent where only src/pi is synced
REFERENCE = DOCS / "images" / "preview" / "seams-render_bloom.png"
FRAME = 45  # mid-pulse: every seam lit, the pulses partway along
SEED = 1
SCALE = 6  # pixels per cell: 816 x 816


def reference_frame():
    registry = AnimationRegistry.discover(default_animations_dir())
    run = registry["seams"].start(default_geometry(), seed=SEED, fps=30.0)
    rendered = None
    for _ in range(FRAME + 1):
        rendered = run.render()
    return rendered.frame


def reference_record() -> bytes:
    """The same frame as a `full` preview record: what the browser draws."""
    return encode_preview(reference_frame(), FRAME, "full")


# ---- a minimal PNG (8-bit RGB, no interlace), so the reference needs no imaging library


def write_png(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))  # filter 0 on every row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def read_png(path: Path) -> np.ndarray:
    """Reads what write_png writes (filter 0 rows only)."""
    data = path.read_bytes()
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            assert (depth, colour) == (8, 2), "8-bit RGB only"
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    rows = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, 1 + width * 3)
    assert not rows[:, 0].any(), "filtered rows: not written by write_png"
    return rows[:, 1:].reshape(height, width, 3)


@pytest.mark.skipif(not DOCS.is_dir(), reason="no repo docs/ beside the package (e.g. the Pi's synced tree)")
def test_render_bloom_of_the_reference_frame_matches_the_committed_image():
    image = render_bloom(reference_frame(), scale=SCALE)
    if os.environ.get("DF2_UPDATE_REFERENCE"):
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        write_png(REFERENCE, image)
    assert REFERENCE.exists(), f"no reference image; run with DF2_UPDATE_REFERENCE=1 to create {REFERENCE}"
    committed = read_png(REFERENCE)
    assert committed.shape == image.shape == (136 * SCALE, 136 * SCALE, 3)
    assert np.array_equal(committed, image)


def test_the_png_helpers_round_trip(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    write_png(tmp_path / "x.png", rgb)
    assert np.array_equal(read_png(tmp_path / "x.png"), rgb)
