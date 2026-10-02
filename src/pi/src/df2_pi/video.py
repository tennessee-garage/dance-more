"""Video clips for the floor: imported once on a laptop, played anywhere.

    df2-pi video import waves.mp4 --name waves --duration 30   # on a laptop with the [preview] extra
    df2-pi video list

Importing decodes the source (anything imageio reads: mp4, mov, gif ...),
crops it square, and averages each frame down to the floor's 136 x 136 cell
grid in linear light - so a bright line of foam keeps its brightness rather
than being muddied into the water around it - then keeps only the 3,840
cells that are LEDs. The result is a `<name>.clip.npz` in the media
directory: frames as perceptual bytes in `PixelFrame` layout, plus the rate
they were sampled at. Playback (`animations/video.py`) is then just blending
stored frames, which the Pi does in well under a millisecond; it never
decodes video.

Orientation: the top of the video is the top of the floor as displayed -
the side away from the Pi. The floor-rotation setting and the animation's
own Rotation param turn it from there.

The media directory is `$DF2_MEDIA`, else `media/` beside the package's
`src/` (`src/pi/media` in the repo, `~/dance-floor/media` on the Pi, where
`sync-to-pi.sh` puts it). Clips are not committed: they are big, and
someone else's footage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from df2_pi.gamma import DECODE_LUT, from_linear
from df2_pi.geometry import FloorGeometry

SUFFIX = ".clip.npz"
FORMAT_VERSION = 1
DEFAULT_FPS = 15.0


@dataclass(frozen=True)
class Clip:
    name: str
    frames: np.ndarray  # (n, tiles, leds_per_tile, 3) uint8, perceptual
    fps: float  # the rate the frames were sampled at
    source: str  # the file it came from

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps


def default_media_dir() -> Path:
    env = os.environ.get("DF2_MEDIA")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "media"


def clip_path(name: str, media_dir: Path | None = None) -> Path:
    return (media_dir or default_media_dir()) / f"{name}{SUFFIX}"


def list_clips(media_dir: Path | None = None) -> list[str]:
    folder = media_dir or default_media_dir()
    if not folder.is_dir():
        return []
    return sorted(p.name[: -len(SUFFIX)] for p in folder.glob(f"*{SUFFIX}"))


def load_clip(name: str, media_dir: Path | None = None) -> Clip | None:
    """The clip, or None if there is no such file. Cached, and reloaded if
    the file has been re-imported since."""
    path = clip_path(name, media_dir)
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    return _load(str(path), mtime)


@lru_cache(maxsize=4)
def _load(path: str, mtime: int) -> Clip:
    with np.load(path, allow_pickle=False) as data:
        version = int(data["version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"{path}: clip format {version}, this build reads {FORMAT_VERSION}; re-import it")
        return Clip(
            name=Path(path).name[: -len(SUFFIX)],
            frames=data["frames"],
            fps=float(data["fps"]),
            source=str(data["source"]),
        )


# ---- importing ----------------------------------------------------------------------------------


def to_leds(image: np.ndarray, geometry: FloorGeometry, crop: tuple[float, float, float] = (0.5, 0.5, 1.0)) -> np.ndarray:
    """One video frame, (h, w, 3 or 4) uint8, as LED data (tiles, leds, 3).
    `crop` is the square taken from it: centre x and y as fractions of the
    width and height, and size as a fraction of the shorter side."""
    from PIL import Image

    rgb = np.asarray(image)[..., :3]
    h, w = rgb.shape[:2]
    cx, cy, size = crop
    if not (0.0 < size <= 1.0 and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise ValueError(f"crop is centre x, centre y, size, each 0..1 (size > 0), got {crop}")
    side = max(1, int(round(min(h, w) * size)))
    left = int(round(min(max(cx * w - side / 2, 0), w - side)))
    top = int(round(min(max(cy * h - side / 2, 0), h - side)))
    square = rgb[top : top + side, left : left + side]

    light = DECODE_LUT[square]  # linear, so the average is an average of light
    cells = geometry.height
    channels = [
        np.asarray(Image.fromarray(np.ascontiguousarray(light[..., c]), mode="F").resize((cells, cells), Image.Resampling.BOX))
        for c in range(3)
    ]
    grid = from_linear(np.stack(channels, axis=-1))[::-1]  # image rows run down; canonical y runs up, away from the Pi
    yx = geometry.led_to_cell
    return grid[yx[..., 0], yx[..., 1]]


def import_clip(
    source: str | Path,
    name: str | None = None,
    *,
    media_dir: Path | None = None,
    geometry: FloorGeometry | None = None,
    fps: float = DEFAULT_FPS,
    start: float = 0.0,
    duration: float | None = None,
    crop: tuple[float, float, float] = (0.5, 0.5, 1.0),
    progress: Callable[[int], None] | None = None,
) -> Path:
    """Decode `source`, sample it at `fps` from `start` for `duration`
    seconds (to the end if None), and write the clip. Returns its path."""
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError('importing video needs imageio: pip install -e ".[preview]"') from exc
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    source = Path(source)
    name = name or source.name.split(".")[0]
    geometry = geometry or FloorGeometry()

    meta = iio.immeta(source)
    source_fps = meta.get("fps")
    if not source_fps and meta.get("duration"):  # a GIF: duration is ms per frame
        source_fps = 1000.0 / float(meta["duration"])
    source_fps = float(source_fps or fps)

    frames = []
    next_t = start
    end = None if duration is None else start + duration
    for i, image in enumerate(iio.imiter(source)):
        t = i / source_fps
        if end is not None and t >= end:
            break
        if t + 1e-9 < next_t:
            continue  # between samples: the nearest-at-or-after source frame is used
        frames.append(to_leds(image, geometry, crop))
        next_t += 1.0 / fps
        if progress is not None:
            progress(len(frames))
    if len(frames) < 2:
        raise ValueError(f"{source}: fewer than two frames between {start} s and {end if end is not None else 'the end'}")

    folder = media_dir or default_media_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = clip_path(name, folder)
    with open(path, "wb") as fh:
        np.savez_compressed(
            fh, frames=np.stack(frames), fps=np.float64(fps), version=np.int64(FORMAT_VERSION), source=np.str_(source.name)
        )
    return path
