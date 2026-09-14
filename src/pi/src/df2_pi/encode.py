"""Frame -> Row Bus bytes, and the one gamma definition everything shares.

The gamma curve itself (`GAMMA`, `DECODE_LUT`, `to_linear`, `from_linear`)
is defined in gamma.py and re-exported here: it is what `pixels.py` uses
for anything that averages or mixes colours and what the encoder decodes
through on the way out, so the preview and the floor agree about what a
byte means. `FrameEncoder` turns a rendered frame into the SEND_DATA
payloads that go to each row, the step between "an animation returned a
frame" and `Floor.send_data()`:

    enc = FrameEncoder(brightness=255, white_balance=(1.0, 0.85, 0.95))
    payloads = enc.encode(frame, effects)   # one bytes per row, index = row addr
    for row, payload in enumerate(payloads):
        floor.send_data(row, payload)
    floor.latch()

Per-tile command selection. A SEND_DATA payload is 8 tile entries and each
independently picks its Tile Bus command, so a tile whose LEDs are all one
colour is sent as SET_COLOR (4 bytes) instead of SET_LEDS (181). A
TileFrame is uniform by construction and skips the test; a PixelFrame gets
one vectorised comparison per frame. `uniform_tolerance` collapses
near-uniform tiles too, trading a little accuracy for a lot of wire time:

    frame                              per row     Row Bus phase, 2 chains x 4 rows
    all tiles uniform (any TileFrame)     32 B     ~0.5 ms
    half uniform                         740 B     ~9.6 ms
    all tiles fully addressed          1,448 B     ~18.6 ms of the 33 ms budget

That is the Row Bus phase only; the per-row Tile Bus tail adds ~15 ms after
it and the measured ceiling is ~25 FPS, limited by the row's receive path
(docs/row-bus-protocol.md section 1) - so fitting here is necessary, not
sufficient.

Colour pipeline. Frame bytes are gamma-ENCODED (perceptual); the WS2815 is
a linear PWM device and the tile firmware applies no correction of its
own, so the encoder decodes on the way out, with brightness and white
balance applied in linear light where they belong:

    linear = (value / 255) ** gamma
    out    = round(255 * linear * brightness / 255 * white_balance[channel])

collapsed into one (256, 3) uint8 LUT, rebuilt only when a setting changes
and applied as a single gather per frame. `gamma` defaults to the frame
encoding's GAMMA and should normally stay there; it is exposed as the one
display-calibration knob.

Two things that will otherwise be rediscovered painfully:

- LOW BRIGHTNESS COSTS BIT DEPTH. At brightness 64 there are only ~64
  distinct output levels per channel and gradients band visibly. Keep the
  global brightness high and dim inside the animation (`Frame.gain()`).
  Temporal dithering would fix it and is deliberately out of scope.
- WS2815 GREEN READS FAR BRIGHTER than red and blue at equal drive, so a
  nominal (255, 255, 255) looks green. That is what `white_balance` is for;
  the default of (1, 1, 1) needs measuring on real tiles.

Byte ordering. PixelFrame is stored in chain order so a tile's 180 bytes
are one contiguous slice, and the wire carries R G B per LED - the tile
firmware feeds those to its GRB strip driver, which does the reordering.
Nothing is swapped here.

Effects. `encode()` takes the effect writes collected during render
(`FrameContext.effects`) and emits a 6-byte SET_EFFECT entry for each, IN
PLACE OF that tile's pixel entry for the frame, so an effect change latches
with the frame it was made on. The tile keeps showing its buffer for that
one frame; picking a moment where that does not show is the animation's
job. Effects are registers and the encoder is stateless about them: it
emits exactly what it is handed, once.

Frame data is gamma-ENCODED bytes: 128 is not half as bright as 255. A
naive mean of encoded bytes comes out too dark - a tile half black and half
full white averages to 128, which is 0.22 of full light rather than 0.5.
Anything that is arithmetic on light rather than on perception goes through
`to_linear()` first and back through `from_linear()` after.

The curve is a plain power law, `linear = (byte / 255) ** GAMMA`, rather
than the piecewise sRGB transfer function: the LEDs are driven by PWM duty,
which is linear in light, and 2.2 is the conventional perceptual fit. The
value is a module constant rather than a parameter on every call so the
two directions can never be built from different exponents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from df2_pi.effects import Effect
from df2_pi.gamma import DECODE_LUT, GAMMA, from_linear, to_linear  # noqa: F401 - re-exported
from df2_pi.geometry import FloorGeometry
from df2_pi.pixels import CHANNELS, Frame, PixelFrame, TileFrame, default_geometry
from df2_pi.protocol.constants import FRAME_OVERHEAD, TILE_ENTRY_SIZE, TileCmd
from df2_pi.transport.row_bus import DEFAULT_BAUDRATE

# The gamma primitives live in gamma.py (pixels.py needs them and this
# module needs pixels.py); they are re-exported here so `from df2_pi.encode
# import GAMMA, to_linear, from_linear` keeps working.
__all__ = ["GAMMA", "DECODE_LUT", "to_linear", "from_linear", "FrameEncoder", "EncodeStats", "parse_send_data"]

# ---- the encoder ------------------------------------------------------------------


@dataclass
class EncodeStats:
    """What the last `encode()` produced, for telemetry: payload size per
    row, the total, and how many entries of each command."""

    row_bytes: tuple[int, ...] = ()
    entries: Counter = field(default_factory=Counter)

    @property
    def total_bytes(self) -> int:
        return sum(self.row_bytes)

    def wire_seconds(self, chains: int = 2, baudrate: int = DEFAULT_BAUDRATE) -> float:
        """Estimated Row Bus phase for this frame: rows dealt round-robin
        over `chains` driven concurrently (the floor's alternating
        wiring), 10 bits per byte plus the 8-byte frame overhead, and the
        slowest chain sets the time. An estimate for the admin page, not
        a measurement."""
        per_chain = [0] * max(1, chains)
        for row, n in enumerate(self.row_bytes):
            per_chain[row % len(per_chain)] += FRAME_OVERHEAD + n
        return max(per_chain) * 10 / baudrate


class FrameEncoder:
    """Frame -> one SEND_DATA payload per row. See the module docstring."""

    def __init__(
        self,
        geometry: FloorGeometry | None = None,
        *,
        brightness: int = 255,
        gamma: float = GAMMA,
        white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0),
        uniform_tolerance: int = 0,
    ) -> None:
        self.geometry = geometry if geometry is not None else default_geometry()
        self._brightness = 255
        self._gamma = GAMMA
        self._white_balance = (1.0, 1.0, 1.0)
        self.brightness = brightness
        self.gamma = gamma
        self.white_balance = white_balance
        self.uniform_tolerance = uniform_tolerance
        self.stats = EncodeStats()
        self._channel_index = np.arange(CHANNELS)

    # ---- settings (each rebuilds the LUT) -----------------------------------

    @property
    def brightness(self) -> int:
        return self._brightness

    @brightness.setter
    def brightness(self, value: int) -> None:
        if not 0 <= value <= 255:
            raise ValueError(f"brightness must be 0..255, got {value}")
        self._brightness = int(value)
        self._rebuild()

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        if not value > 0:
            raise ValueError(f"gamma must be positive, got {value}")
        self._gamma = float(value)
        self._rebuild()

    @property
    def white_balance(self) -> tuple[float, float, float]:
        return self._white_balance

    @white_balance.setter
    def white_balance(self, value: tuple[float, float, float]) -> None:
        wb = tuple(float(v) for v in value)
        if len(wb) != CHANNELS or any(v < 0 for v in wb):
            raise ValueError(f"white_balance must be three non-negative factors, got {value!r}")
        self._white_balance = wb  # type: ignore[assignment]
        self._rebuild()

    @property
    def uniform_tolerance(self) -> int:
        return self._uniform_tolerance

    @uniform_tolerance.setter
    def uniform_tolerance(self, value: int) -> None:
        if not 0 <= value <= 255:
            raise ValueError(f"uniform_tolerance must be 0..255, got {value}")
        self._uniform_tolerance = int(value)

    @property
    def lut(self) -> np.ndarray:
        """The `(256, 3)` uint8 table: frame byte -> wire byte, per channel."""
        return self._lut

    def _rebuild(self) -> None:
        linear = (np.arange(256, dtype=np.float64) / 255.0) ** self._gamma  # (256,)
        scale = (self._brightness / 255.0) * np.asarray(self._white_balance)  # (3,)
        out = np.rint(255.0 * linear[:, None] * scale[None, :])
        self._lut = np.clip(out, 0, 255).astype(np.uint8)

    # ---- encoding ------------------------------------------------------------

    def encode(self, frame: Frame, effects: Mapping[int, Effect] | None = None) -> list[bytes]:
        """SEND_DATA payloads for `frame`, one per row, index = row address.
        `effects` (tile -> Effect) replace those tiles' pixel entries."""
        geo = self.geometry
        if frame.geometry is not geo and frame.data.shape != frame.shape_for(geo):
            raise ValueError(f"{frame!r} does not match the encoder's geometry")
        effect_entries = self._effect_entries(effects)

        if isinstance(frame, TileFrame):
            colours = self._apply_lut(frame.data.reshape(geo.tiles, CHANNELS))
            uniform = np.ones(geo.tiles, dtype=bool)
            leds = None
        elif isinstance(frame, PixelFrame):
            data = frame.data
            uniform = self._uniform_tiles(data)
            colours = self._apply_lut(np.rint(data.mean(axis=1)).astype(np.uint8))
            leds = self._apply_lut(data)
        else:
            raise TypeError(f"cannot encode {frame!r}")

        entries = Counter()
        payloads: list[bytes] = []
        for row in range(geo.tile_rows):
            parts: list[bytes] = []
            for col in range(geo.tile_cols):
                tile = row * geo.tile_cols + col
                entry = effect_entries.get(tile)
                if entry is not None:
                    entries[TileCmd.SET_EFFECT] += 1
                elif uniform[tile]:
                    entry = bytes((TileCmd.SET_COLOR, *colours[tile]))
                    entries[TileCmd.SET_COLOR] += 1
                else:
                    entry = bytes((TileCmd.SET_LEDS,)) + leds[tile].tobytes()
                    entries[TileCmd.SET_LEDS] += 1
                parts.append(entry)
            payloads.append(b"".join(parts))

        self.stats = EncodeStats(tuple(len(p) for p in payloads), entries)
        return payloads

    def blackout_payload(self) -> bytes:
        """The payload for a known-black row: 8 x SET_COLOR (0, 0, 0). This
        only writes the pixel buffers - the BLACKOUT command additionally
        clears effect registers."""
        return bytes((TileCmd.SET_COLOR, 0, 0, 0)) * self.geometry.tile_cols

    def _apply_lut(self, data: np.ndarray) -> np.ndarray:
        return self._lut[data, self._channel_index]

    def _uniform_tiles(self, data: np.ndarray) -> np.ndarray:
        """(tiles,) bool: which tiles can go as one SET_COLOR."""
        spread = data.max(axis=1).astype(np.int16) - data.min(axis=1)  # (tiles, 3)
        return (spread <= self._uniform_tolerance).all(axis=1)

    def _effect_entries(self, effects: Mapping[int, Effect] | None) -> dict[int, bytes]:
        if not effects:
            return {}
        tiles = self.geometry.tiles
        out: dict[int, bytes] = {}
        for tile, effect in effects.items():
            if not isinstance(effect, Effect):
                raise TypeError(f"effect for tile {tile!r} must be an Effect, got {effect!r}")
            if not isinstance(tile, int) or not 0 <= tile < tiles:
                raise ValueError(f"effect tile must be 0..{tiles - 1}, got {tile!r}")
            out[tile] = bytes((TileCmd.SET_EFFECT,)) + bytes(effect)
        return out


def parse_send_data(payload: bytes) -> list[tuple[TileCmd, bytes]]:
    """Split a SEND_DATA payload back into `(tile_cmd, tile_data)` entries -
    the inverse of `encode()` for one row, for tests and diagnostics."""
    entries: list[tuple[TileCmd, bytes]] = []
    i = 0
    while i < len(payload):
        cmd = TileCmd(payload[i])
        size = TILE_ENTRY_SIZE[cmd]
        if i + size > len(payload):
            raise ValueError(f"truncated {cmd.name} entry at byte {i}")
        entries.append((cmd, bytes(payload[i + 1 : i + size])))
        i += size
    return entries
