"""Waterline: the surface of the water, tile by tile, bobbing across the floor.

Each of the eight tile columns is a point on the surface with a height and a
velocity, moved by forces rather than drawn from a curve:

- A pull toward the water level, which itself rises and falls slowly - the
  tide - so the whole line breathes.
- A swell running across the floor, whose strength waxes and wanes on its
  own slow cycle, so it ebbs and flows instead of beating like a sine.
- Now and then a kick to one column, as if something broke the surface.
- Cohesion: each column is pulled toward its neighbours. The pull grows
  with the gap but levels off at a maximum (tanh), so small differences
  are held tight and the line stays smooth - but a column moving fast
  enough out-runs the pull and breaks away, and is then drawn back.
- Damping, so it settles between kicks.

The surface is drawn across the two rows it sits between, in proportion,
so it glides between tile rows rather than jumping. Below it the water is a
second colour from the same palette, dimmer, fading toward black at the
bottom row - which is still lit, just dimmest; above it is dark. Up and
down are as displayed (row 0 nearest the Pi is the bottom); the
floor-rotation setting turns it.
"""

import math

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.palette import choice, palette_param
from df2_pi.pixels import TileFrame

STEP_S = 1 / 120  # physics substep
LEVEL_PULL = 0.6  # 1/s^2: toward the water level
DAMPING = 0.9  # 1/s
COHESION_MAX = 7.0  # rows/s^2 per neighbour at full cohesion: the most the pull can be
COHESION_RANGE = 0.45  # rows: the gap over which the pull grows before it levels off
TIDE = (0.19, 0.6)  # rad/s and rows: the whole level rising and falling
SWELL_WAVES = 0.7  # wavelengths across the floor
SWELL_SPEED = 0.9  # rad/s
SWELL_FORCE = 3.2  # rows/s^2 at full swell
SWELL_CYCLE = 0.11  # rad/s: how fast the swell's strength waxes and wanes
KICKS_PER_S = 0.35  # at full swell
KICK = (2.0, 4.5)  # rows/s


@animation(
    name="Waterline",
    description="The water's surface across the floor, tile by tile, bobbing and settling, with the depths fading below.",
    author="df2",
    format="tile",
    tags=["ambient", "water", "palette"],
    params={
        "level": Param(float, default=3.6, min=0.5, max=6.5, label="Water level (rows)"),
        "swell": Param(float, default=0.5, min=0.0, max=1.0, label="Swell", help="How rough: the waves' strength and how often the surface is kicked", role="intensity", macro=1),
        "cohesion": Param(float, default=0.5, min=0.0, max=1.0, label="Cohesion", help="How hard neighbouring tiles hold together", role="variation"),
        "speed": Param(float, default=1.0, min=0.2, max=3.0, label="Speed", curve="log"),
        "palette": palette_param(default="ocean"),
        "surface": Param(float, default=0.75, min=0.0, max=1.0, label="Surface colour", help="Its place in the palette"),
        "depths": Param(float, default=0.25, min=0.0, max=1.0, label="Depths colour", help="Its place in the palette"),
        "depth_level": Param(float, default=0.8, min=0.0, max=1.0, label="Depths brightness"),
    },
)
def render(previous: TileFrame, ctx) -> TileFrame:
    geo = ctx.geometry
    rows, cols = geo.tile_rows, geo.tile_cols
    p = ctx.params
    state = ctx.state
    if not state:
        state["y"] = np.full(cols, p["level"], dtype=np.float64)  # surface height per column, in rows
        state["v"] = np.zeros(cols)
        state["time"] = 0.0
        state["phase"] = ctx.rng.uniform(0, 2 * math.pi)  # where the swell's cycle starts

    _simulate(state, ctx, ctx.dt * p["speed"], cols)

    pal = choice(ctx, p["palette"])
    surface = pal.at(p["surface"]).astype(np.float32)
    depths = pal.at(p["depths"]).astype(np.float32) * p["depth_level"]

    r = np.arange(rows, dtype=np.float64)[:, None]  # (rows, 1), row 0 the bottom
    y = state["y"][None, :]  # (1, cols)
    line = np.clip(1.0 - np.abs(r - y), 0.0, 1.0)  # the surface, shared between the two rows it sits on
    under = np.clip(y - r, 0.0, 1.0)  # how much of the row is below the surface
    # Brightest just under the surface, fading to black just below the floor - so the bottom row is the dimmest water, not none.
    fade = np.clip((r + 1.0) / (y + 1.0), 0.0, 1.0) ** 0.6
    water = (1.0 - line) * under * fade
    colour = line[..., None] * surface + water[..., None] * depths

    frame = TileFrame.black(geo)
    frame.data[...] = np.clip(colour, 0, 255).astype(np.uint8)
    return frame


def _simulate(state: dict, ctx, elapsed: float, cols: int) -> None:
    p = ctx.params
    y, v = state["y"], state["v"]
    steps = max(1, math.ceil(elapsed / STEP_S))
    h = elapsed / steps
    x = np.arange(cols) / cols
    pull = COHESION_MAX * p["cohesion"]
    for _ in range(steps):
        state["time"] += h
        t = state["time"]
        level = p["level"] + TIDE[1] * math.sin(TIDE[0] * t + state["phase"])
        strength = p["swell"] * (0.55 + 0.45 * math.sin(SWELL_CYCLE * t + 2.0 * state["phase"]))
        swell = SWELL_FORCE * strength * np.sin(2 * math.pi * SWELL_WAVES * x - SWELL_SPEED * t)

        gaps = np.diff(y)  # neighbour - this, for each pair
        hold = pull * np.tanh(gaps / COHESION_RANGE)  # grows with the gap, then levels off
        cohesion = np.zeros(cols)
        cohesion[:-1] += hold  # pulled up toward a higher right-hand neighbour
        cohesion[1:] -= hold  # and its neighbour pulled down toward it

        a = LEVEL_PULL * (level - y) + cohesion + swell - DAMPING * v
        v += a * h
        y += v * h

    if ctx.rng.random() < KICKS_PER_S * p["swell"] * elapsed:  # something breaks the surface
        v[ctx.rng.randrange(cols)] += ctx.rng.choice((-1, 1)) * ctx.rng.uniform(*KICK)
