"""Development sinks: see the floor with no floor attached.

Nobody should have to walk to the garage to find out a gradient is upside
down. These render frames on a laptop:

    TerminalSink    ANSI truecolour in the terminal, half-block characters
                    (two grid rows per text line) for the 136x136 grid, or
                    the 8x8 tile view. Self-limits to ~10 FPS so the
                    terminal is never the bottleneck; 256-colour fallback.
    WindowSink      A real window (pygame), for judging motion the terminal
                    cannot show: top-down orthographic, a fixed per-LED
                    bloom, additive, no controls. Pumped from the MAIN
                    thread (`pump()`), because macOS will not draw a
                    window from any other.
    FrameCollector  Keeps every frame in memory, synchronously - for
                    `--record`, where dropping frames would defeat the
                    point - and exports them as GIF / mp4 (imageio) or a
                    .df2rec recording.

Everything here draws through `FloorGeometry.to_display()`, so the floor
appears ORIGIN AT BOTTOM-LEFT: row 0 nearest the Pi along the bottom
edge, as you see it standing at the rack. A renderer that skips the flip
shows a vertically mirrored floor - which looks fine until something
directional runs across it.

pygame and imageio are optional (`pip install -e ".[preview]"`); the
terminal sink needs nothing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping, Sequence, TextIO

import numpy as np

from df2_pi.effects import Effect
from df2_pi.engine.clock import FrameInfo
from df2_pi.geometry import FloorGeometry
from df2_pi.output.preview import RECORDING_MAGIC, RECORD_PREFIX, encode_preview
from df2_pi.output.sink import Mailbox, ThreadedSink
from df2_pi.pixels import Frame, PixelFrame, TileFrame

# ---- rasterising ----------------------------------------------------------------------------


def display_grid(frame: Frame) -> np.ndarray:
    """`(height, width, 3)` uint8 image of the frame in display
    orientation (origin bottom-left, via `to_display`)."""
    pixels = frame.to_pixels() if isinstance(frame, TileFrame) else frame
    return frame.geometry.to_display(pixels.grid)


def display_tiles(frame: Frame) -> np.ndarray:
    """`(rows, cols, 3)` one colour per tile, display orientation."""
    tiles = frame.to_tiles() if isinstance(frame, PixelFrame) else frame
    return frame.geometry.to_display(tiles.data)


def rasterize(frame: Frame, scale: int = 4) -> np.ndarray:
    """The display grid scaled up `scale`x with sharp cells - for GIF and
    video export."""
    img = display_grid(frame)
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def render_bloom(frame: Frame, scale: int = 6, radius: float = 1.6, gain: float = 1.4) -> np.ndarray:
    """The preview renderer: each LED as a point of light with a fixed
    bloom, added on a black ground; top-down, orthographic, no controls.
    `radius` is the bloom's sigma in cells. Returns `(H*scale, W*scale, 3)`
    uint8."""
    img = display_grid(frame).astype(np.float32) / 255.0
    height, width = img.shape[:2]
    canvas = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
    # the LED itself: a small bright core at the cell centre
    core = max(1, scale // 3)
    off = (scale - core) // 2
    ys, xs = np.nonzero(img.any(axis=-1))
    for dy in range(core):
        for dx in range(core):
            canvas[ys * scale + off + dy, xs * scale + off + dx] += img[ys, xs]
    # the bloom: the same light spread by a gaussian, added
    sigma = radius * scale
    glow = np.zeros_like(canvas)
    glow[ys * scale + scale // 2, xs * scale + scale // 2] = img[ys, xs]
    glow = _gaussian_blur(glow, sigma) * (gain * sigma * sigma * 2.0)
    out = np.clip(canvas + glow, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian on an (H, W, C) float image, edges zero-padded.
    Each pass is one matrix product against a banded Toeplitz kernel, so
    the whole thing is three BLAS calls per axis rather than a Python
    loop over rows - this runs on the main thread beside the render
    thread and must not hold the GIL for long."""
    height, width, channels = img.shape
    out = np.tensordot(_blur_matrix(height, sigma), img, axes=([1], [0]))  # (H, W, C)
    out = np.tensordot(out, _blur_matrix(width, sigma), axes=([1], [1]))  # (H, C, W)
    return np.transpose(out, (0, 2, 1))


_BLUR_CACHE: dict[tuple[int, float], np.ndarray] = {}


def _blur_matrix(n: int, sigma: float) -> np.ndarray:
    """`(n, n)` matrix applying a 1-D gaussian of `sigma` to a length-n
    vector, zero-padded at the edges. Cached: a handful of sizes ever."""
    key = (n, float(sigma))
    m = _BLUR_CACHE.get(key)
    if m is None:
        i = np.arange(n, dtype=np.float32)
        d = i[:, None] - i[None, :]
        m = np.exp(-(d * d) / (2 * sigma * sigma)).astype(np.float32)
        m /= m[n // 2].sum()  # unit gain away from the edges
        _BLUR_CACHE[key] = m
    return m


# ---- the terminal ----------------------------------------------------------------------------

TERMINAL_MODES = ("grid", "tiles")


def truecolor_supported(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("COLORTERM", "").lower() in ("truecolor", "24bit")


def _fg(rgb, truecolor: bool) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"\x1b[38;2;{r};{g};{b}m" if truecolor else f"\x1b[38;5;{_ansi256(r, g, b)}m"


def _bg(rgb, truecolor: bool) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"\x1b[48;2;{r};{g};{b}m" if truecolor else f"\x1b[48;5;{_ansi256(r, g, b)}m"


def _ansi256(r: int, g: int, b: int) -> int:
    """Nearest entry of the 6x6x6 colour cube (or the grey ramp for greys)."""
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + (r - 8) * 24 // 240
    return 16 + 36 * (r * 6 // 256) + 6 * (g * 6 // 256) + (b * 6 // 256)


def render_terminal(frame: Frame, mode: str = "grid", truecolor: bool = True) -> str:
    """The frame as ANSI text, one string, newline-terminated lines. `grid`
    is the full cell grid with half-blocks: each character is two cells
    stacked, the upper as foreground on '▀', the lower as background.
    `tiles` is the 8x8 view, two characters per tile."""
    reset = "\x1b[0m"
    lines: list[str] = []
    if mode == "grid":
        img = _pair_aligned(display_grid(frame), frame.geometry)
        if img.shape[0] % 2:
            # An odd row count leaves the bottom edge as a glyph over an
            # empty background - a hairline of black under the floor.
            # Repeating it makes the last cell colour over colour.
            img = np.concatenate([img, img[-1:]])
        for y in range(0, img.shape[0], 2):
            upper, lower = img[y], img[y + 1]
            parts = [_fg(upper[x], truecolor) + _bg(lower[x], truecolor) + "▀" for x in range(img.shape[1])]
            lines.append("".join(parts) + reset)
    elif mode == "tiles":
        tiles = display_tiles(frame)
        for row in tiles:
            lines.append("".join(_fg(rgb, truecolor) + "██" for rgb in row) + reset)
    else:
        raise ValueError(f"mode must be one of {TERMINAL_MODES}, got {mode!r}")
    return "\n".join(lines) + "\n"


def _pair_aligned(img: np.ndarray, geometry: FloorGeometry) -> np.ndarray:
    """Pad a display-orientation grid so every horizontal tile seam lands
    on ONE half-block character.

    A character shows two grid rows: the upper as the `▀` glyph in the
    foreground colour, the lower as the background. A background fills
    the cell exactly; a glyph has hairline gaps at its edges that show
    whatever the background is. Tiles are `cell_size` rows tall, an odd
    pitch, so the pairing drifts by one row per tile row: at one seam the
    south edge above and the north edge below share a cell (glyph over
    background in the other edge's colour - seamless), at the next they
    fall in different cells and the north edge becomes a coloured glyph
    on a BLACK background, whose gaps show as thin black bands. Repeating
    each tile's middle row (dark interior, lit sides, so the vertical
    edges stay unbroken) makes the pitch even, and one blank row on top
    puts every seam pair in one cell. The only visible change is each
    tile being one cell taller."""
    cell = geometry.cell_size
    if cell % 2 == 0:
        return img
    rows, width, channels = img.shape
    out = [np.zeros((1, width, channels), dtype=img.dtype)]
    mid = cell // 2
    for start in range(0, rows, cell):
        block = img[start : start + cell]
        out.append(block[: mid + 1])
        out.append(block[mid : mid + 1])  # the middle row, twice
        out.append(block[mid + 1 :])
    return np.concatenate(out)


class TerminalSink(ThreadedSink):
    """Draw frames in the terminal, at most `max_fps` per second."""

    def __init__(
        self,
        mode: str = "grid",
        *,
        max_fps: float = 10.0,
        truecolor: bool | None = None,
        out: TextIO | None = None,
        name: str = "terminal",
    ) -> None:
        if mode not in TERMINAL_MODES:
            raise ValueError(f"mode must be one of {TERMINAL_MODES}, got {mode!r}")
        super().__init__(name)
        self.mode = mode
        self.min_interval = 1.0 / max_fps
        self.truecolor = truecolor_supported() if truecolor is None else truecolor
        self.out = out if out is not None else sys.stdout
        self._last_t: float | None = None
        self._began = False
        self.drawn = 0

    def handle(self, frame: Frame, info: FrameInfo) -> None:
        if self._last_t is not None and info.t - self._last_t < self.min_interval - 1e-9:
            return
        self._last_t = info.t
        if not self._began:
            self._began = True
            self.out.write("\x1b[2J\x1b[?25l")  # clear, hide cursor
        text = render_terminal(frame, self.mode, self.truecolor)
        self.out.write("\x1b[H" + text + f"\x1b[0mframe {info.n}  t={info.t:7.2f}s\x1b[K\n")
        self.out.flush()
        self.drawn += 1

    def on_close(self) -> None:
        if self._began:
            self.out.write("\x1b[0m\x1b[?25h\n")  # reset, show cursor
            self.out.flush()


# ---- the window ------------------------------------------------------------------------------


class WindowSink:
    """A pygame window showing the bloom preview. `submit()` only mailboxes
    the frame; the owner calls `pump()` from the main thread (pygame
    draws nowhere else on macOS), which handles events and draws the
    newest frame. `closed` goes true when the user closes the window."""

    name = "window"

    def __init__(self, scale: int = 6, title: str = "df2 preview") -> None:
        try:
            import pygame  # noqa: F401 - optional, checked here so the error is clear
        except ImportError as exc:
            raise ImportError('WindowSink needs pygame: pip install -e ".[preview]"') from exc
        self.scale = scale
        self.title = title
        self.mailbox = Mailbox()
        self.closed = False
        self.frames = 0
        self._screen = None

    def submit(self, frame: Frame, info: FrameInfo, effects: Mapping[int, Effect] | None = None) -> None:
        self.mailbox.put((frame, info))

    def pump(self, timeout: float = 0.05) -> bool:
        """Handle events and draw the newest frame, waiting up to
        `timeout` for one. Returns False once the window has been closed."""
        import pygame

        if self.closed:
            return False
        if self._screen is not None:  # no events until there is a window
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    self.close()
                    return False
        item = self.mailbox.take(timeout)
        if item is None:
            return True
        frame, info = item
        img = render_bloom(frame, self.scale)
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption(self.title)
            self._screen = pygame.display.set_mode((img.shape[1], img.shape[0]))
        pygame.surfarray.blit_array(self._screen, np.transpose(img, (1, 0, 2)))
        pygame.display.flip()
        self.frames += 1
        return True

    def close(self) -> None:
        """Stop accepting frames. Safe from any thread; the window itself
        is torn down by `shutdown()` on the main thread."""
        self.closed = True
        self.mailbox.close()

    def shutdown(self) -> None:
        """Tear pygame down. MAIN THREAD ONLY - SDL crashes otherwise."""
        self.close()
        if self._screen is not None:
            import pygame

            pygame.quit()
            self._screen = None


# ---- recording ---------------------------------------------------------------------------------


class FrameCollector:
    """Keeps every submitted frame, synchronously on the render thread -
    the one observer allowed to be slow, because it exists to capture a
    headless run exactly. Never attach it alongside real hardware."""

    name = "collector"

    def __init__(self, limit: int | None = None) -> None:
        self.frames: list[tuple[FrameInfo, Frame]] = []
        self.limit = limit

    def submit(self, frame: Frame, info: FrameInfo, effects: Mapping[int, Effect] | None = None) -> None:
        if self.limit is None or len(self.frames) < self.limit:
            self.frames.append((info, frame))

    def close(self) -> None:
        pass

    def export(self, path: Path | str, fps: float = 30.0, scale: int = 4, bloom: bool = False) -> Path:
        """Write the frames as `.gif` / `.mp4` (imageio) or `.df2rec`."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".df2rec":
            write_recording(path, self.frames)
            return path
        if suffix not in (".gif", ".mp4"):
            raise ValueError(f"cannot export {path.name}: use .gif, .mp4 or .df2rec")
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError('exporting video needs imageio: pip install -e ".[preview]"') from exc
        images = [render_bloom(f, scale) if bloom else rasterize(f, scale) for _, f in self.frames]
        if not images:
            raise ValueError("nothing to export: no frames were collected")
        if suffix == ".gif":
            iio.imwrite(path, images, duration=1000.0 / fps, loop=0)
        else:
            iio.imwrite(path, images, fps=fps)
        return path


def write_recording(path: Path | str, frames: Sequence[tuple[FrameInfo, Frame]], fmt: str = "full") -> None:
    """A `.df2rec` file, readable by `read_recording()`."""
    with open(path, "wb") as f:
        f.write(RECORDING_MAGIC)
        for info, frame in frames:
            record = encode_preview(frame, info.n, fmt)
            f.write(RECORD_PREFIX.pack(len(record), info.t))
            f.write(record)


def frame_count_of(path: Path | str) -> int:
    """How many frames an exported GIF / mp4 holds (needs imageio)."""
    import imageio.v3 as iio

    return len(iio.imread(path, index=None))


__all__ = [
    "FrameCollector",
    "TerminalSink",
    "WindowSink",
    "display_grid",
    "display_tiles",
    "frame_count_of",
    "rasterize",
    "render_bloom",
    "render_terminal",
    "truecolor_supported",
    "write_recording",
]

