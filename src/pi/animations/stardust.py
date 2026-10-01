"""Stardust: a slow night sky - nebula, twinkling stars, and stars that scatter.

Ambient and dim on purpose: it should sit behind a room, not compete with
it. Three layers, at two scales:

- Nebula, tile by tile. Each tile is one colour, so the clouds read at
  floor scale: two slow fields of drifting waves over the 8x8 grid, one
  picking the colour from a deep blue / violet / teal palette and one
  carving darker voids. It takes minutes to change much.
- Stars, LED by LED. A scatter of single LEDs swell and fade over several
  seconds, then turn up somewhere else.
- Novae, both. Now and then one tile slowly warms to a pale glow, and then
  comes apart: its sixty LEDs drift off its corners as dust, along the edges
  leading away from it (`edge_graph().edges_at`), slowing as they go and
  cooling into the nebula's colour before they fade. The ones nearest a
  corner leave first, so the tile crumbles from its corners in.

Everything runs on the animation's own clock times Speed, so the show's
Speed control and the param both slow it further.
"""

import colorsys
import math

import numpy as np

from df2_pi.animation import Param, animation
from df2_pi.geometry import Side
from df2_pi.pixels import PixelFrame

SIDES = (Side.NORTH, Side.EAST, Side.SOUTH, Side.WEST)
# The nebula, from the empty dark to its brightest wisps. Perceptual values, kept low.
PALETTE_STOPS = np.array([0.0, 0.35, 0.65, 1.0])
PALETTE = np.array([(10, 10, 38), (30, 18, 92), (78, 30, 122), (28, 96, 130)], dtype=np.float32)
STAR_COLOURS = np.array([(190, 205, 255), (255, 236, 214), (215, 220, 255)], dtype=np.float32)
STAR_PEAK = 200.0
GLOW = np.array([200, 215, 255], dtype=np.float32)  # a nova at its height
BLOOM_S = 3.0  # a nova warming up
DUST_LIFE_S = 6.0  # a speck of dust from leaving to gone
DUST_STAGGER_S = 0.12  # between specks leaving the same corner
DUST_DRIFT = (4.0, 11.0)  # starting speed, LEDs/s
DUST_DRAG_S = 1.6  # how quickly they slow: each travels about speed x this
DUST_EDGES = 3  # how far ahead a speck's route is planned


@animation(
    name="Stardust",
    description="A slow night sky: drifting nebula, twinkling stars, and stars that scatter into dust.",
    author="df2",
    format="pixel",
    tags=["ambient", "space"],
    params={
        "speed": Param(float, default=1.0, min=0.2, max=3.0, label="Speed", curve="log"),
        "stars": Param(int, default=50, min=0, max=200, label="Stars", role="density"),
        "novae": Param(float, default=2.0, min=0.0, max=12.0, label="Novae per minute", macro=1),
        "intensity": Param(float, default=0.7, min=0.2, max=1.0, label="Brightness"),
        "hue": Param(float, default=0.0, min=0.0, max=1.0, label="Hue shift"),
    },
)
def render(previous: PixelFrame, ctx) -> PixelFrame:
    geo = ctx.geometry
    state = ctx.state
    if not state:
        rng = ctx.rng
        state["time"] = 0.0
        state["colour_waves"] = _waves(rng, 4)
        state["void_waves"] = _waves(rng, 3)
        rows, cols = np.divmod(np.arange(geo.tiles), geo.tile_cols)
        state["rows"], state["cols"] = rows.astype(np.float32), cols.astype(np.float32)
        state["stars"] = []  # [led, born, life, colour]
        state["novae"] = []  # dicts: tile, born, and once scattered, the dust
    speed = ctx.params["speed"]
    step = ctx.dt * speed
    state["time"] += step
    t = state["time"]
    intensity = ctx.params["intensity"]

    # ---- nebula: one colour per tile ----
    shade = _field(state["colour_waves"], state["rows"], state["cols"], t)
    depth = 0.3 + 0.7 * _field(state["void_waves"], state["rows"], state["cols"], t)
    palette = _rotated(PALETTE, ctx.params["hue"])
    sky = np.stack([np.interp(shade, PALETTE_STOPS, palette[:, c]) for c in range(3)], axis=-1)
    sky *= depth[:, None]
    img = np.repeat(sky[:, None, :], geo.leds_per_tile, axis=1)  # (64, 60, 3) float

    # ---- novae: a tile warms, then scatters ----
    novae = state["novae"]
    if ctx.rng.random() < ctx.params["novae"] / 60.0 * step:
        busy = {n["tile"] for n in novae}
        free = [tile for tile in range(geo.tiles) if tile not in busy]
        if free:
            novae.append({"tile": free[ctx.rng.randrange(len(free))], "born": t, "dust": None})
    flat = img.reshape(-1, 3)
    for nova in novae:
        age = t - nova["born"]
        if age < BLOOM_S:
            warm = (age / BLOOM_S) ** 2
            tile = nova["tile"]
            img[tile] = img[tile] + (GLOW * 0.85 * intensity - img[tile]) * warm
        else:
            if nova["dust"] is None:
                nova["dust"] = _scatter(ctx, nova["tile"])
            _draw_dust(flat, nova["dust"], age - BLOOM_S, sky_tint=palette[3] * 1.4, intensity=intensity)
    novae[:] = [n for n in novae if t - n["born"] < BLOOM_S + (n["dust"]["last"] if n["dust"] else 0) + DUST_LIFE_S]

    # ---- stars: single LEDs swelling and fading ----
    stars = state["stars"]
    while len(stars) < ctx.params["stars"]:
        # new ones start partway through, so a fresh sky (or a raised count) is not in step
        life = ctx.rng.uniform(5.0, 11.0)
        stars.append([ctx.rng.randrange(geo.tiles * geo.leds_per_tile), t - ctx.rng.uniform(0, life), life, ctx.rng.randrange(len(STAR_COLOURS))])
    del stars[ctx.params["stars"]:]
    for star in stars:
        if t - star[1] >= star[2]:  # burnt out: rise again elsewhere
            star[0] = ctx.rng.randrange(geo.tiles * geo.leds_per_tile)
            star[1], star[2] = t, ctx.rng.uniform(5.0, 11.0)
    if stars:
        leds = np.array([s[0] for s in stars])
        u = np.array([(t - s[1]) / s[2] for s in stars])
        level = (np.sin(np.pi * np.clip(u, 0, 1)) ** 2 * STAR_PEAK * intensity)[:, None]
        colour = STAR_COLOURS[[s[3] for s in stars]] / 255.0
        np.maximum.at(flat, leds, level * colour)

    frame = PixelFrame.black(geo)
    frame.data[...] = np.clip(img, 0, 255).astype(np.uint8)
    return frame


def _waves(rng, count: int) -> np.ndarray:
    """Plane waves for a smooth field over the tile grid: direction, spatial
    frequency (per tile), drift (radians per second) and phase."""
    waves = []
    for _ in range(count):
        angle = rng.uniform(0, 2 * math.pi)
        waves.append((math.cos(angle), math.sin(angle), rng.uniform(0.25, 0.7), rng.uniform(0.02, 0.07) * rng.choice((-1, 1)), rng.uniform(0, 2 * math.pi)))
    return np.array(waves, dtype=np.float32)


def _field(waves: np.ndarray, rows: np.ndarray, cols: np.ndarray, t: float) -> np.ndarray:
    """0..1 per tile: the waves summed and normalised."""
    total = np.zeros_like(rows)
    for dx, dy, freq, drift, phase in waves:
        total += np.sin(freq * (cols * dx + rows * dy) + drift * t + phase)
    return 0.5 + 0.5 * total / len(waves)


def _rotated(palette: np.ndarray, turns: float) -> np.ndarray:
    if turns == 0:
        return palette
    out = []
    for r, g, b in palette / 255.0:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        out.append(colorsys.hsv_to_rgb((h + turns) % 1.0, s, v))
    return np.array(out, dtype=np.float32) * 255


def _scatter(ctx, tile: int) -> dict:
    """Turn a tile's 60 LEDs into dust: each runs to the nearer end of its
    side, then out from that corner along edges that lead away."""
    geo = ctx.geometry
    graph = geo.edge_graph()
    n = geo.leds_per_side
    row, col = divmod(tile, geo.tile_cols)
    centre = (row + 0.5, col + 0.5)  # on the junction lattice
    paths, release, drift = [], [], []
    for side in SIDES:
        edge = geo.edge(tile, side)
        for i in range(n):
            if i > n // 2 or (i == n // 2 and ctx.rng.random() < 0.5):
                along, corner, from_corner = edge.flat_leds[i:], edge.junctions[1], n - 1 - i
            else:
                along, corner, from_corner = edge.flat_leds[i::-1], edge.junctions[0], i
            paths.append(np.concatenate([along, *_away(graph, corner, centre, ctx.rng)]))
            release.append((from_corner + ctx.rng.random()) * DUST_STAGGER_S)
            drift.append(ctx.rng.uniform(*DUST_DRIFT))
    length = np.array([len(p) for p in paths])
    release = np.array(release)
    return {
        "paths": np.array([np.pad(p, (0, length.max() - len(p)), mode="edge") for p in paths]),
        "length": length,
        "release": release,
        "drift": np.array(drift),
        "last": float(release.max()),
    }


def _away(graph, junction, centre, rng) -> list[np.ndarray]:
    """Up to DUST_EDGES edges out from a tile corner, each ending further
    from `centre` than it began."""

    def distance(j):
        return (j[0] - centre[0]) ** 2 + (j[1] - centre[1]) ** 2

    route = []
    for _ in range(DUST_EDGES):
        onward = []
        for edge in graph.edges_at(junction):
            forward = edge.junctions[0] == junction
            far = edge.junctions[1] if forward else edge.junctions[0]
            if distance(far) > distance(junction):
                onward.append((edge, forward, far))
        if not onward:
            break
        edge, forward, junction = onward[rng.randrange(len(onward))]
        route.append(edge.flat_leds if forward else edge.flat_leds[::-1])
    return route


def _draw_dust(flat: np.ndarray, dust: dict, clock: float, sky_tint: np.ndarray, intensity: float) -> None:
    """Each speck at its place along its route: it drifts out and slows
    (exponential drag), cooling from the glow toward the sky's colour and
    fading. Straddles two LEDs by its fractional position."""
    age = clock - dust["release"]
    moving = np.clip(age, 0.0, None)
    position = np.minimum(dust["drift"] * DUST_DRAG_S * (1.0 - np.exp(-moving / DUST_DRAG_S)), dust["length"] - 1)
    u = np.clip(moving / DUST_LIFE_S, 0.0, 1.0)
    level = (1.0 - u) ** 1.5
    colour = (GLOW * 0.85 * intensity)[None, :] * (1.0 - u[:, None]) + np.clip(sky_tint, 0, 255)[None, :] * intensity * u[:, None]
    at = np.floor(position).astype(np.intp)
    frac = position - at
    rows = np.arange(len(at))
    nxt = np.minimum(at + 1, dust["length"] - 1)
    leds = np.concatenate([dust["paths"][rows, at], dust["paths"][rows, nxt]])
    values = np.concatenate([(level * (1 - frac))[:, None] * colour, (level * frac)[:, None] * colour])
    np.maximum.at(flat, leds, values)
