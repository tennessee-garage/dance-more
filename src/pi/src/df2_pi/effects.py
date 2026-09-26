"""Tile effects: the host-side value object for the tile's effect register.

A tile holds two independent pieces of display state - the pixel buffer
(SET_COLOR / SET_LEDS) and the effect register (SET_EFFECT: an effect id
plus four parameter bytes). On every LATCH the tile drives its LEDs from
the buffer THROUGH the effect; with none set the buffer goes out untouched.
Both are registers: written once, they persist until written again, and
setting an effect never stops a tile accepting pixels.

The effects themselves - what each id does with its parameters - are
specified and implemented in the tile firmware (#72, docs/tile-effects.md).
This module only knows the wire shape, so the driver plumbing can carry
effect writes before any tile can act on them. The ids below follow
docs/tile-effects.md; only NONE and SHIMMER are implemented in tile
firmware so far, and a tile drops an id it doesn't implement.

An effect write rides inside SEND_DATA as a 6-byte tile entry and costs
that tile its pixel update for the frame - see animation/context.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

EFFECT_ID_MAX = 0x1F  # 5-bit id; bits 7:5 of the byte are reserved
EFFECT_PARAMS = 4

# Effect ids per docs/tile-effects.md §3. NONE is the reset state. Ids 1
# and 2 are deliberately unassigned (formerly SOLID and BREATHE); 5 is
# reserved for SPARKLE, not yet defined.
NONE = 0x00
SHIMMER = 0x03
CHASE = 0x04
HUE_SPLIT = 0x06
FADE = 0x07


@dataclass(frozen=True)
class Effect:
    """One effect register value: `id` 0-31 and four parameter bytes.

    `bytes(effect)` is the 5-byte SET_EFFECT payload (id then params),
    what the encoder puts after the tile_cmd byte.
    """

    id: int
    params: tuple[int, int, int, int] = (0, 0, 0, 0)

    NONE: ClassVar[Effect]  # assigned below the class

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or isinstance(self.id, bool):
            raise TypeError(f"effect id must be an int, got {self.id!r}")
        if not 0 <= self.id <= EFFECT_ID_MAX:
            raise ValueError(
                f"effect id must be 0..{EFFECT_ID_MAX} (bits 7:5 reserved), got {self.id:#x}"
            )
        params = tuple(self.params)
        if len(params) != EFFECT_PARAMS:
            raise ValueError(f"effect takes exactly {EFFECT_PARAMS} params, got {len(params)}")
        for p in params:
            if not isinstance(p, int) or isinstance(p, bool) or not 0 <= p <= 255:
                raise ValueError(f"effect params must be bytes 0..255, got {params!r}")
        object.__setattr__(self, "params", params)

    def __bytes__(self) -> bytes:
        return bytes((self.id, *self.params))


Effect.NONE = Effect(NONE)
