"""Waves: a lighting desk's favourite effect in one line - a waveform run
across the floor, each tile (or LED) offset in phase.

    offsets = phase.radial(ctx.geometry, spread=1.0)                  # df2_pi.phase
    level = lfo(ctx.t_beats, rate=0.5, shape="sine", offset=-offsets)  # df2_pi.tempo

That is the whole effect. The phase map decides the pattern - rows makes a
wave up the floor, radial rings out from the middle, angle a sweep like a
radar, checker an alternation, random a shimmer, perimeter a ring running
round every tile - and the waveform its character: sine swells, saw
sweeps, square snaps. Spread is how much of the cycle the floor covers (0
all together, 1 one whole cycle across it), Mirror folds it about the
centre so it fans in from the edges, Reverse runs it the other way.

On beat time (`ctx.t_beats`), so it locks to the music with a beat source
running and runs at the fallback tempo without. At tile resolution every
tile is one colour, which is cheap on the wire; LED resolution - which
the perimeter map always is - computes all 3,840.
"""

import numpy as np

from df2_pi import phase
from df2_pi.animation import Param, animation
from df2_pi.pixels import PixelFrame
from df2_pi.tempo import SHAPES, lfo


@animation(
    name="Waves",
    description="A waveform run across the floor in phase: waves, rings, sweeps, chases. On the beat.",
    author="df2",
    format="pixel",
    tags=["rhythm", "geometric"],
    sync="beat",
    params={
        "map": Param(str, default="radial", choices=list(phase.MAPS), label="Phase map", macro=1),
        "shape": Param(str, default="sine", choices=list(SHAPES), label="Waveform"),
        "rate": Param(float, default=0.5, min=0.0625, max=4.0, label="Cycles per beat", role="speed", curve="log"),
        "spread": Param(float, default=1.0, min=0.0, max=3.0, label="Spread", help="0 all together; 1 one cycle across the floor", role="scale"),
        "mirror": Param(bool, default=False, label="Mirror"),
        "reverse": Param(bool, default=False, label="Reverse"),
        "resolution": Param(str, default="tile", choices=["tile", "led"], label="Resolution"),
        "hue": Param(float, default=0.6, min=0.0, max=1.0, label="Hue"),
        "hue_spread": Param(float, default=0.15, min=0.0, max=0.5, label="Hue across the wave", role="variation"),
        "floor": Param(float, default=0.06, min=0.0, max=0.5, label="Floor", help="Brightness at the bottom of the wave"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    p = ctx.params
    resolution = "led" if p["map"] == "perimeter" else p["resolution"]
    offsets = phase.by_name(
        geo, p["map"], rng=ctx.np_rng, resolution=resolution, spread=p["spread"], mirror=p["mirror"], reverse=p["reverse"]
    )
    if resolution == "tile":
        offsets = offsets.reshape(-1, 1)  # one per tile, broadcast across its LEDs
    level = lfo(ctx.t_beats, rate=p["rate"], shape=p["shape"], offset=-offsets)

    rgb = _hue_to_rgb(p["hue"] + p["hue_spread"] * level)
    brightness = p["floor"] + (1.0 - p["floor"]) * level
    frame = PixelFrame.black(geo)
    frame.data[...] = np.broadcast_to(rgb * (brightness * 255.0)[..., None], frame.data.shape).astype(np.uint8)
    return frame


def _hue_to_rgb(hue: np.ndarray) -> np.ndarray:
    """Fully saturated colours for an array of hues, vectorised: (..., 3) in 0..1."""
    h6 = np.mod(hue, 1.0)[..., None] * 6.0
    return np.clip(np.abs(h6 - np.array([3.0, 2.0, 4.0])) * np.array([1.0, -1.0, -1.0]) + np.array([-1.0, 2.0, 2.0]), 0.0, 1.0)
