"""The two LED data formats animations write into.

    TileFrame   (rows, cols, 3) uint8    - one flat colour per tile
    PixelFrame  (tiles, 60, 3)  uint8    - every LED, in WS2815 chain order

An animation declares which one it produces; the driver converts as needed.
`TileFrame` is what most floor-scale animations actually want, and it maps
straight onto Tile Bus SET_COLOR - 4 bytes per tile instead of 181, so a
per-row SEND_DATA is 32 bytes rather than 1,448 (see encode.py, #63).

`PixelFrame` stores chain order, not the 136x136 image, on purpose: that is
the wire order, so the encoder can slice a tile's 180 bytes with zero
reordering. The image view is a derived thing - `grid` scatters the LEDs
into a fresh (H, W, 3) array with every dark cell zero, and `from_grid()`
samples only the lit cells of an image and silently discards the other ~79%.
That is the naive authoring path: render a 136x136 picture any way you like
and the driver keeps the pixels that are real LEDs.

Colour space. Frame bytes are gamma-encoded (encode.py), so:

    gain()      works on the encoded bytes  - perceptually even falloff, which
                is what trails and decay want
    blend()     works in linear light        - a crossfade in encoded space dips
                in luminance mid-transition, visibly on a floor this size
    to_tiles()  averages in linear light     - a naive byte mean of a
                half-black / half-white tile is 128, i.e. 0.22 of full light

`gain()` is NOT the global brightness control; that is the encoder's LUT.
`gain()` is animation-level dimming, which is also what the encoder asks you
to prefer, since dropping global brightness costs bit depth.

Mutation and ownership. The runner hands each animation the PREVIOUS frame
and expects a NEW one back. Observer sinks may still hold a reference to
that previous frame while the next one renders, so it is passed frozen
(`freeze()` makes `data` read-only) and an animation that wants to evolve
it copies first:

    frame = previous.copy().gain(0.88)      # phosphor decay of the last frame
    frame.splat(x, y, WHITE, radius=3)      # this frame's new content on top
    return frame

Returning `previous` itself (or anything aliasing its buffer) is an error -
the runner calls `check_ownership()` and gets a clear message rather than a
silent aliasing bug. An 11,520-byte copy at 30 FPS is ~346 KB/s of memcpy,
irrelevant next to the render and worth it to keep the rule unambiguous.
`copy()`, `gain()` and `blend()` are all pure for the same reason: they
return new frames and never mutate their inputs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

import numpy as np

from df2_pi.encode import from_linear, to_linear
from df2_pi.geometry import FloorGeometry

F = TypeVar("F", bound="Frame")

CHANNELS = 3


@lru_cache(maxsize=1)
def default_geometry() -> FloorGeometry:
    """The real floor's geometry, built once. Frames made without an
    explicit geometry all share this instance, so their lookup tables are
    not rebuilt per frame."""
    return FloorGeometry.default()


class FrameOwnershipError(ValueError):
    """An animation returned the frame it was given instead of a new one."""


class Frame:
    """Common behaviour of TileFrame and PixelFrame. Not used directly.

    `data` is the uint8 array in the subclass's canonical shape; index the
    frame directly (`frame[row, col] = (r, g, b)`) or through `.data`.
    """

    data: np.ndarray
    geometry: FloorGeometry

    def __init__(self, data: np.ndarray, geometry: FloorGeometry | None = None) -> None:
        geometry = geometry if geometry is not None else default_geometry()
        data = np.asarray(data)
        if data.dtype != np.uint8:
            # Not converted silently: a float or int array wraps on the cast,
            # and 256 becoming 0 is exactly the bug a type check exists for.
            raise TypeError(f"{type(self).__name__} data must be uint8, got {data.dtype}")
        expected = self.shape_for(geometry)
        if data.shape != expected:
            raise ValueError(
                f"{type(self).__name__} data must have shape {expected}, got {data.shape}"
            )
        self.data = data
        self.geometry = geometry

    @classmethod
    def shape_for(cls, geometry: FloorGeometry) -> tuple[int, ...]:
        raise NotImplementedError

    # ---- construction ---------------------------------------------------------

    @classmethod
    def black(cls: type[F], geometry: FloorGeometry | None = None) -> F:
        """An all-off frame."""
        geometry = geometry if geometry is not None else default_geometry()
        return cls(np.zeros(cls.shape_for(geometry), dtype=np.uint8), geometry)

    @classmethod
    def like(cls, other: F) -> F:
        """A black frame of the same type and geometry as `other`. For
        generic code that produces a frame of whatever type it was handed -
        the runner seeding frame 0. An animator knows their own format and
        calls `PixelFrame.black()`."""
        return type(other).black(other.geometry)

    def copy(self: F) -> F:
        """A writable duplicate. This is how an animation evolves the
        read-only `previous` frame it is handed."""
        return type(self)(self.data.copy(), self.geometry)

    # ---- pure colour ops --------------------------------------------------------

    def gain(self: F, k: float) -> F:
        """Multiply every channel by `k`, saturating at 255: 0.5 halves,
        2.0 doubles and clips. Applied to the encoded bytes, so the falloff
        is perceptually even - what fades, decay and trails want. Returns a
        new frame."""
        if k < 0:
            raise ValueError(f"gain must be non-negative, got {k}")
        scaled = np.clip(np.rint(self.data * float(k)), 0, 255).astype(np.uint8)
        return type(self)(scaled, self.geometry)

    # ---- ownership ------------------------------------------------------------

    def freeze(self: F) -> F:
        """Make `data` read-only and return self. The runner does this to
        the previous frame before handing it to an animation."""
        self.data.flags.writeable = False
        return self

    @property
    def frozen(self) -> bool:
        return not self.data.flags.writeable

    # ---- array-like -------------------------------------------------------------

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value) -> None:
        self.data[key] = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Frame):
            return NotImplemented
        return type(self) is type(other) and np.array_equal(self.data, other.data)

    __hash__ = None  # type: ignore[assignment]  # mutable, like the array it wraps

    def __repr__(self) -> str:
        dims = "x".join(str(n) for n in self.data.shape[:-1])
        state = ", frozen" if self.frozen else ""
        return f"{type(self).__name__}({dims}{state})"


class TileFrame(Frame):
    """Tile-as-single-pixel: each tile is one flat colour.

        frame = TileFrame.black()
        frame[row, col] = (255, 0, 128)       # or frame.data[row, col] = ...
        frame.data                            # (8, 8, 3) uint8

    Indexed `[tile_row, tile_col]` in canonical floor orientation - row 0 is
    nearest the Pi, and `tile_index = row * 8 + col`.
    """

    @classmethod
    def shape_for(cls, geometry: FloorGeometry) -> tuple[int, ...]:
        return (geometry.tile_rows, geometry.tile_cols, CHANNELS)

    def to_pixels(self) -> PixelFrame:
        """Fill all of each tile's LEDs with the tile's colour."""
        geo = self.geometry
        flat = self.data.reshape(geo.tiles, 1, CHANNELS)
        return PixelFrame(np.repeat(flat, geo.leds_per_tile, axis=1), geo)


class PixelFrame(Frame):
    """Fully addressable: every LED, in chain order.

        frame = PixelFrame.black()
        frame.data                    # (64, 60, 3) uint8, chain order
        frame.tile(t)                 # (60, 3) writable view of one tile
        frame.grid                    # (136, 136, 3) image; dark cells (0, 0, 0)

    Chain order is `FloorGeometry.led_to_cell`'s order: LED 0 at the bottom
    of a tile's left side, climbing left, across the top, down the right,
    back along the bottom. Use the geometry's tables (`side_leds`,
    `led_positions`) rather than assuming that here.
    """

    def __init__(self, data: np.ndarray, geometry: FloorGeometry | None = None) -> None:
        super().__init__(data, geometry)
        self._scratch: np.ndarray | None = None

    @classmethod
    def shape_for(cls, geometry: FloorGeometry) -> tuple[int, ...]:
        return (geometry.tiles, geometry.leds_per_tile, CHANNELS)

    def tile(self, tile: int) -> np.ndarray:
        """`(leds_per_tile, 3)` view of one tile - writable unless the frame
        is frozen, in which case writes raise."""
        return self.data[tile]

    # ---- the grid view ----------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """A fresh `(height, width, 3)` uint8 image of the floor: every LED
        scattered into its cell, every dark cell `(0, 0, 0)`."""
        geo = self.geometry
        img = np.zeros((geo.height, geo.width, CHANNELS), dtype=np.uint8)
        img[geo.led_to_cell[..., 0], geo.led_to_cell[..., 1]] = self.data
        return img

    def grid_view(self) -> np.ndarray:
        """A mutable `(height, width, 3)` scratch image owned by this frame,
        for animations that build up a picture incrementally without paying
        for `grid`'s allocation every access.

        The buffer is created on first call, seeded from the frame, and then
        returned as-is on later calls - it is a working surface, not a live
        view, and nothing written to it reaches `data` until `commit()`.
        Dark cells may be drawn on freely; `commit()` ignores them.
        """
        if self._scratch is None:
            self._scratch = self.grid
        return self._scratch

    def commit(self) -> None:
        """Copy the lit cells of the `grid_view()` buffer back into `data`.
        Raises if `grid_view()` was never called."""
        if self._scratch is None:
            raise ValueError("commit() without a grid_view() buffer to commit")
        geo = self.geometry
        self.data[...] = self._scratch[geo.led_to_cell[..., 0], geo.led_to_cell[..., 1]]

    @classmethod
    def from_grid(
        cls,
        img: np.ndarray,
        *,
        strict: bool = False,
        geometry: FloorGeometry | None = None,
    ) -> PixelFrame:
        """Sample a `(height, width, 3)` uint8 image at the lit cells only.

        Whatever the image holds in dark cells is discarded. `strict=True`
        raises instead, for animations that believe they only write lit
        cells and want that checked.
        """
        geometry = geometry if geometry is not None else default_geometry()
        img = np.asarray(img)
        if img.dtype != np.uint8:
            raise TypeError(f"grid must be uint8, got {img.dtype}")
        expected = (geometry.height, geometry.width, CHANNELS)
        if img.shape != expected:
            raise ValueError(f"grid must have shape {expected}, got {img.shape}")
        if strict and img[~geometry.lit_mask].any():
            dark = np.argwhere(~geometry.lit_mask & img.any(axis=-1))
            raise ValueError(
                f"grid lights {len(dark)} dark cell(s), first at (y, x) = "
                f"{tuple(int(v) for v in dark[0])}"
            )
        # Fancy indexing copies, so the frame never aliases the caller's image.
        data = img[geometry.led_to_cell[..., 0], geometry.led_to_cell[..., 1]]
        return cls(data, geometry)

    # ---- conversion ---------------------------------------------------------

    def to_tiles(self) -> TileFrame:
        """Reduce each tile's LEDs to one colour: the per-channel mean in
        linear light, re-encoded.

        Lossy by construction - a downsampled tile carries no spatial
        information, and sparse content reduces to a dim tile (15 of 60
        LEDs lit averages to roughly a quarter brightness). This feeds the
        low-bandwidth preview path only; the full-fidelity preview sends
        every LED, and preview UIs should label this view as degraded.

        Channels average independently, so a tile red on one side and green
        on the other reduces to dark yellow.
        """
        geo = self.geometry
        lin = to_linear(self.data)  # (tiles, leds, 3) linear light
        avg = lin.mean(axis=1, dtype=np.float64)  # (tiles, 3)
        return TileFrame(from_linear(avg).reshape(geo.tile_rows, geo.tile_cols, CHANNELS), geo)


def blend(a: F, b: F, t: float) -> F:
    """Cross-fade two frames of the same type: `t=0` is `a`, `t=1` is `b`.
    Mixed in linear light, so a 50% blend of black and white is the
    photometric midpoint (186), not 128. Returns a new frame.

    For animators, compositing two computed layers; for the runner,
    crossfading between playlist entries.
    """
    if type(a) is not type(b) or a.data.shape != b.data.shape:
        raise TypeError(f"cannot blend {a!r} with {b!r}")
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"blend factor must be in 0..1, got {t}")
    lin = to_linear(a.data) * (1.0 - t) + to_linear(b.data) * t
    return type(a)(from_linear(lin), a.geometry)


def check_ownership(previous: Frame, result: Frame) -> None:
    """Raise `FrameOwnershipError` if `result` is the `previous` frame, or
    shares its buffer. The runner calls this on every frame an animation
    returns; see the module docstring for the rule it enforces."""
    if result is previous:
        raise FrameOwnershipError(
            "animation returned the previous frame itself; evolve a copy "
            "instead: frame = previous.copy()"
        )
    if np.may_share_memory(result.data, previous.data):
        raise FrameOwnershipError(
            "animation returned a frame that aliases the previous frame's "
            "buffer; evolve a copy instead: frame = previous.copy()"
        )
