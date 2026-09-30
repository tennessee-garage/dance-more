"""Twin Peaks: the red curtains of the Red Room, stirring in a draught.

Pleated drapes hanging from the top of the floor (as displayed) to the
bottom. Each LED's brightness is the shading of the fabric at its (x, y):
a cosine across x is the pleats - bright on the crests, deep red in the
folds, never black, which the floor shows poorly - and the pleats are
pushed sideways by a displacement that varies with height and time. That
displacement is the whole animation:

- a slow sway, the curtain never quite still;
- a ripple climbing from hem to rod;
- gusts: now and then a draught catches a stretch of curtain, and a
  burst of short ripples runs up through it, widening as it dies away.

All of it swings more at the hem than at the rod, which is what makes it
read as hanging cloth rather than a scrolling pattern. Horizontal edges
cut across the pleats and show the folds; vertical edges run down them
and shimmer as a fold sways past.

The lamp in the room isn't to be trusted either: now and then the whole
curtain dips, once or in a stutter, like a bulb on a bad circuit.
"""

import math

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.pixels import PixelFrame

# The fabric's shading, from the back of a fold to the lit crest of a pleat. Perceptual values.
SHADE_STOPS = np.array([0.0, 0.45, 0.8, 1.0])
SHADE_COLOURS = np.array([(70, 0, 4), (160, 0, 8), (225, 10, 12), (255, 60, 40)], dtype=np.float32)
HEM_SWING = 3.0  # how much more the hem moves than the rod
GUST_LIFE = 3.0  # seconds a gust takes to rise and settle


@animation(
    name="Twin Peaks",
    description="The Red Room's curtains: red pleated drapes stirring in a draught, under a lamp that flickers.",
    author="df2",
    format="pixel",
    tags=["ambient"],
    params={
        "flutter": Param(float, default=0.5, min=0.0, max=1.0, label="Flutter", help="Draught strength and how often it gusts", role="intensity", macro=1),
        "speed": Param(float, default=1.0, min=0.2, max=4.0, label="Speed", curve="log"),
        "pleats": Param(float, default=12.0, min=6.0, max=34.0, label="Pleat width (cells)", role="scale"),
        "flicker": Param(float, default=0.25, min=0.0, max=1.0, label="Lamp flicker", role="variation"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    if not ctx.state:
        ctx.state["time"] = 0.0  # the curtain's own clock, so Speed can change without a jump
        ctx.state["gusts"] = []  # (born, x, width, amplitude, wavelength)
        ctx.state["lamp"] = []  # remaining brightness dips, one per frame
    flutter, pleat = ctx.params["flutter"], ctx.params["pleats"]
    ctx.state["time"] += ctx.dt * ctx.params["speed"]
    t = ctx.state["time"]

    y, x = geo.led_positions[..., 0], geo.led_positions[..., 1]  # (64, 60) each, cell units
    hem = 1.0 + (HEM_SWING - 1.0) * (1.0 - y / geo.height)  # 1 at the rod, HEM_SWING at the hem
    reach = pleat * (0.04 + 0.16 * flutter)  # how far a fold can be pushed sideways, in cells

    shift = 0.6 * math.sin(2 * math.pi * 0.05 * t) + 0.4 * np.sin(2 * math.pi * (0.03 * t + x / 300))
    shift = shift + 0.5 * np.sin(2 * math.pi * (y / 70 - 0.18 * t) + x / 40)  # climbing ripple

    # Gusts: on average one every few seconds at full flutter, none when it is still.
    gusts = ctx.state["gusts"]
    if ctx.rng.random() < 0.35 * flutter * ctx.dt * ctx.params["speed"]:
        gusts.append((t, ctx.rng.uniform(0, geo.width), ctx.rng.uniform(12, 30), ctx.rng.uniform(0.6, 1.2), ctx.rng.uniform(25, 50)))
    gusts[:] = [g for g in gusts if t - g[0] < GUST_LIFE]
    for born, gx, width, amplitude, wavelength in gusts:
        age = (t - born) / GUST_LIFE  # 0..1
        spread = width * (1.0 + age)  # the disturbance widens as it settles
        envelope = amplitude * math.sin(math.pi * age) * np.exp(-(((x - gx) / spread) ** 2))
        shift = shift + envelope * np.sin(2 * math.pi * (y / wavelength - 0.9 * (t - born)))

    phase = 2 * math.pi * (x + reach * hem * shift) / pleat
    shade = 0.5 + 0.5 * np.cos(phase)
    # Where the cloth is pushed hardest it turns toward the light: a sheen riding the ripples.
    shade = np.clip(shade + 0.12 * np.tanh(shift) * np.sin(phase), 0.0, 1.0)

    colour = np.stack([np.interp(shade, SHADE_STOPS, SHADE_COLOURS[:, c]) for c in range(3)], axis=-1)
    frame = PixelFrame.black(geo)
    frame.data[...] = (colour * _lamp(ctx)).astype(np.uint8)
    return frame


def _lamp(ctx) -> float:
    """The room's light this frame: 1.0, or somewhere in a dip. A dip is a
    short queued sequence - a single blink, or a stutter of a few."""
    lamp = ctx.state["lamp"]
    if not lamp and ctx.rng.random() < 0.25 * ctx.params["flicker"] * ctx.dt:
        for _ in range(ctx.rng.choice((1, 1, 2, 3))):
            lamp.extend([ctx.rng.uniform(0.25, 0.6)] * ctx.rng.randint(1, 3))  # dark for a frame or three
            lamp.extend([ctx.rng.uniform(0.8, 1.0)] * ctx.rng.randint(1, 4))  # half-recovered between
    return lamp.pop(0) if lamp else 1.0
