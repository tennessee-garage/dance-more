"""`FrameContext`: everything a render call gets besides the previous frame.

The previous frame alone is not enough to animate - you need time. `ctx`
carries it, plus the resolved parameters, the geometry, a scratch dict and
seeded random sources:

    ctx.frame      frames since this animation started; 0 on the first call
    ctx.t          seconds since it started
    ctx.dt, ctx.fps
    ctx.params     defaults <- playlist entry overrides <- live UI edits
    ctx.geometry   the FloorGeometry
    ctx.state      per-run scratch dict, {} on frame 0, persists across frames
    ctx.rng        random.Random, seeded per run so a recording reproduces
    ctx.np_rng     numpy Generator from the same seed, for the array-shaped
                   APIs (EdgeGraph.walk takes one)
    ctx.beat       reserved for the interfacing epic; None until then

`ctx.state` is how a stateful animation (particles, cellular automata)
keeps data across frames without anyone having to write a class.

There is deliberately NO countdown to the end of the entry. An animation is
never told how long it has left and never reasons about winding down: it
runs indefinitely, and the playlist runner owns stopping it and applying
whatever transition follows. Fading out is not the animation's job.

Effects. `ctx.send_effect(tile, effect)` and `ctx.send_effect_all(effect)`
write a tile's effect register. Writes are collected during render into
`ctx.effects` and the encoder merges them into that frame's SEND_DATA, so
an effect change latches with the frame it was made on. A write costs that
tile its pixel update for the one frame - SEND_DATA has one entry per
tile, so a tile receiving SET_EFFECT gets no SET_COLOR / SET_LEDS and keeps
showing its buffer. Harmless on frame 0 (nothing drawn yet); anywhere
else, picking a moment where a dropped frame on that tile does not show is
the animation's job. `effect=` in the metadata is the frame-0 case done
declaratively, and is applied before render() runs so render can still
override it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from df2_pi.effects import Effect
from df2_pi.geometry import FloorGeometry


@dataclass(frozen=True)
class BeatInfo:
    """Tempo information from external gear. Reserved for the interfacing
    epic; nothing produces one yet."""

    tempo: float  # beats per minute
    phase: float  # 0..1 within the current beat
    downbeat: bool  # True on the first beat of a bar


class FrameContext:
    """Per-frame view onto a running animation. Built by `AnimationRun`;
    animations only read it and call the two `send_effect` methods."""

    def __init__(
        self,
        *,
        frame: int,
        t: float,
        dt: float,
        fps: float,
        params: dict[str, Any],
        geometry: FloorGeometry,
        state: dict[str, Any],
        rng: random.Random,
        np_rng: np.random.Generator,
        beat: BeatInfo | None = None,
    ) -> None:
        self.frame = frame
        self.t = t
        self.dt = dt
        self.fps = fps
        self.params = params
        self.geometry = geometry
        self.state = state
        self.rng = rng
        self.np_rng = np_rng
        self.beat = beat
        self.effects: dict[int, Effect] = {}

    def send_effect(self, tile: int, effect: Effect) -> None:
        """Write `tile`'s effect register with this frame. `Effect.NONE`
        clears it. The tile gets no pixel update this frame."""
        if not isinstance(effect, Effect):
            raise TypeError(f"send_effect takes an Effect, got {effect!r}")
        if not isinstance(tile, int) or isinstance(tile, bool):
            raise TypeError(f"tile must be an int, got {tile!r}")
        if not 0 <= tile < self.geometry.tiles:
            raise ValueError(f"tile must be 0..{self.geometry.tiles - 1}, got {tile}")
        self.effects[tile] = effect

    def send_effect_all(self, effect: Effect) -> None:
        """`send_effect` for every tile on the floor."""
        for tile in range(self.geometry.tiles):
            self.send_effect(tile, effect)
