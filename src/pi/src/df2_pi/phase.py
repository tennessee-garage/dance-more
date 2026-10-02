"""Phase maps: console-style waves, fans and chases (#127).

Lighting desks build most effects from one idea: a waveform run across a
group of fixtures, each fixture offset in phase. A phase map is that set of
offsets for the floor, a float array in 0..1:

    from df2_pi import phase
    from df2_pi.tempo import lfo

    level = lfo(ctx.t_beats - phase.radial(ctx.geometry, spread=0.5), rate=0.5)   # rings out from the middle
    level = lfo(ctx.t_beats - phase.rows(ctx.geometry), rate=0.25)                 # a wave up the floor

At tile resolution (the default) a map is `(rows, cols)`, indexed like a
`TileFrame` - `[row, col]`, row 0 nearest the Pi. At LED resolution
(`resolution="led"`) it is `(tiles, leds_per_tile)`, aligned with a
`PixelFrame`'s `data`, so `level[..., None] * colour` is a frame.

    rows, cols  the tile's row / column; at LED resolution, y / x across the floor
    diagonal    row + column (y + x at LED resolution)
    radial      distance from `point` (x, y in cell units; default the floor centre)
    angle       angle around `point`, 0 due north (up as displayed), clockwise
    checker     alternating tiles: 0 and half a cycle
    random      a fixed shuffle, drawn from `rng` (pass `ctx.np_rng`) once per run
    perimeter   position round each tile's LED ring, in chain order (LED resolution only)

Index maps run 0 .. (n-1)/n, so a full-spread wave puts every row in a
different place in the cycle and none on the same as the first;
distance and angle maps run 0..1.

Options:

    mirror   symmetric about the floor's centre: rows, cols and diagonal
             fold about the centre lines (both ends 0, the middle largest -
             a fan in from the edges, or out with reverse); angle mirrors
             east-west about its point; radial measures to the nearest of
             the point's mirror images; random is made point-symmetric;
             checker already is; perimeter folds round its ring
    reverse  run the other way: the largest offset becomes 0
    spread   scale the offsets: 0 puts everything in phase, 1 spreads it
             over one full cycle, 2 over two

Base maps are cached per geometry; every call returns a fresh array, so it
is yours to modify. They are cheap enough to call every frame, though an
animation that uses the same map throughout can keep it in `ctx.state`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from df2_pi.geometry import FloorGeometry

RESOLUTIONS = ("tile", "led")
MAPS = ("rows", "cols", "diagonal", "radial", "angle", "checker", "random", "perimeter")
_RANDOM_CACHE_RUNS = 32  # random maps kept, most recent runs first


# ---- the maps -----------------------------------------------------------------------------------


def rows(geometry: FloorGeometry, *, resolution: str = "tile", spread: float = 1.0, reverse: bool = False, mirror: bool = False) -> np.ndarray:
    return _shaped(_base(geometry, "rows", resolution, None, mirror), spread, reverse)


def cols(geometry: FloorGeometry, *, resolution: str = "tile", spread: float = 1.0, reverse: bool = False, mirror: bool = False) -> np.ndarray:
    return _shaped(_base(geometry, "cols", resolution, None, mirror), spread, reverse)


def diagonal(geometry: FloorGeometry, *, resolution: str = "tile", spread: float = 1.0, reverse: bool = False, mirror: bool = False) -> np.ndarray:
    return _shaped(_base(geometry, "diagonal", resolution, None, mirror), spread, reverse)


def radial(
    geometry: FloorGeometry,
    point: tuple[float, float] | None = None,
    *,
    resolution: str = "tile",
    spread: float = 1.0,
    reverse: bool = False,
    mirror: bool = False,
) -> np.ndarray:
    return _shaped(_base(geometry, "radial", resolution, _point(geometry, point), mirror), spread, reverse)


def angle(
    geometry: FloorGeometry,
    point: tuple[float, float] | None = None,
    *,
    resolution: str = "tile",
    spread: float = 1.0,
    reverse: bool = False,
    mirror: bool = False,
) -> np.ndarray:
    return _shaped(_base(geometry, "angle", resolution, _point(geometry, point), mirror), spread, reverse)


def checker(geometry: FloorGeometry, *, resolution: str = "tile", spread: float = 1.0, reverse: bool = False, mirror: bool = False) -> np.ndarray:
    return _shaped(_base(geometry, "checker", resolution, None, mirror), spread, reverse)


def random(
    geometry: FloorGeometry,
    rng: np.random.Generator,
    *,
    resolution: str = "tile",
    spread: float = 1.0,
    reverse: bool = False,
    mirror: bool = False,
) -> np.ndarray:
    """A random offset per tile (or LED), drawn from `rng` the first time
    and the same for every later call with that generator - pass
    `ctx.np_rng`, and the map is stable for the run and reproducible from
    its seed."""
    _check_resolution(resolution)
    key = (id(rng), id(geometry), resolution)
    entry = _random_maps.get(key)
    if entry is None or entry[0] is not rng or entry[1] is not geometry:
        shape = _shape(geometry, resolution)
        entry = (rng, geometry, rng.random(shape))
        _random_maps[key] = entry
        while len(_random_maps) > _RANDOM_CACHE_RUNS:
            _random_maps.popitem(last=False)
    _random_maps.move_to_end(key)
    values = entry[2]
    if mirror:  # point-symmetric: each takes the value of whichever of it and its 180-degree partner comes first
        flat = values.reshape(-1)
        partner = _partner(geometry, resolution)
        values = np.where(np.arange(flat.size) <= partner, flat, flat[partner]).reshape(values.shape)
    return _shaped(values, spread, reverse)


def perimeter(geometry: FloorGeometry, *, spread: float = 1.0, reverse: bool = False, mirror: bool = False) -> np.ndarray:
    """Position round each tile's ring of LEDs, `(tiles, leds_per_tile)`.
    The LED chain already runs round the ring - up the west side, across
    the north, down the east, back along the south - so this is the chain
    index over the ring's length: what `df2-pi ledwalk` lights in order."""
    return _shaped(_base(geometry, "perimeter", "led", None, mirror), spread, reverse)


def by_name(geometry: FloorGeometry, name: str, *, rng: np.random.Generator | None = None, **options) -> np.ndarray:
    """Any map by its name in `MAPS` - for an animation that lets its user
    pick one. `random` needs `rng`; `perimeter` is LED resolution only."""
    if name not in MAPS:
        raise ValueError(f"phase map must be one of {MAPS}, got {name!r}")
    if name == "random":
        if rng is None:
            raise ValueError("the random map needs rng (pass ctx.np_rng)")
        return random(geometry, rng, **options)
    if name == "perimeter":
        if options.pop("resolution", "led") != "led":
            raise ValueError("the perimeter map is LED resolution only")
        return perimeter(geometry, **options)
    return globals()[name](geometry, **options)


# ---- shared ------------------------------------------------------------------------------------

_random_maps: OrderedDict[tuple[int, int, str], tuple[np.random.Generator, FloorGeometry, np.ndarray]] = OrderedDict()


def _shaped(base: np.ndarray, spread: float, reverse: bool) -> np.ndarray:
    out = base.astype(np.float64, copy=True)
    if reverse and out.size:
        out = out.max() - out
    return out * spread


def _check_resolution(resolution: str) -> None:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")


def _shape(geometry: FloorGeometry, resolution: str) -> tuple[int, int]:
    if resolution == "tile":
        return (geometry.tile_rows, geometry.tile_cols)
    return (geometry.tiles, geometry.leds_per_tile)


def _point(geometry: FloorGeometry, point: tuple[float, float] | None) -> tuple[float, float]:
    """(x, y) in cell units; the floor's centre by default."""
    return (geometry.width / 2, geometry.height / 2) if point is None else (float(point[0]), float(point[1]))


@lru_cache(maxsize=256)
def _base(geometry: FloorGeometry, name: str, resolution: str, point: tuple[float, float] | None = None, mirror: bool = False) -> np.ndarray:
    """The unshaped map, cached; never handed out (callers get a copy)."""
    _check_resolution(resolution)
    rows_n, cols_n = geometry.tile_rows, geometry.tile_cols
    height, width = geometry.height, geometry.width
    if name == "perimeter":
        n = geometry.leds_per_tile
        u = np.arange(n, dtype=np.float64) / n
        if mirror:
            u = 2.0 * np.minimum(u, 1.0 - u)  # 0 where the chain starts, 1 halfway round
        return np.tile(u, (geometry.tiles, 1))
    if resolution == "tile":
        r, c = np.indices((rows_n, cols_n), dtype=np.float64)
        tile_r, tile_c = r, c
        y = (r + 0.5) * geometry.cell_size  # tile centres, for distance and angle
        x = (c + 0.5) * geometry.cell_size
        if name == "rows":
            return np.minimum(r, rows_n - 1 - r) / (rows_n / 2) if mirror else r / rows_n
        if name == "cols":
            return np.minimum(c, cols_n - 1 - c) / (cols_n / 2) if mirror else c / cols_n
        if name == "diagonal":
            if mirror:
                return (np.minimum(r, rows_n - 1 - r) + np.minimum(c, cols_n - 1 - c)) / (rows_n / 2 + cols_n / 2 - 1)
            return (r + c) / (rows_n + cols_n - 1)
    else:
        y = geometry.led_positions[..., 0].astype(np.float64)
        x = geometry.led_positions[..., 1].astype(np.float64)
        tile_r = np.repeat((np.arange(geometry.tiles) // cols_n)[:, None], geometry.leds_per_tile, axis=1)
        tile_c = np.repeat((np.arange(geometry.tiles) % cols_n)[:, None], geometry.leds_per_tile, axis=1)
        if name == "rows":
            return np.minimum(y, height - y) / (height / 2) if mirror else y / height
        if name == "cols":
            return np.minimum(x, width - x) / (width / 2) if mirror else x / width
        if name == "diagonal":
            if mirror:
                return (np.minimum(y, height - y) + np.minimum(x, width - x)) / ((height + width) / 2)
            return (x + y) / (width + height)
    if name == "checker":
        return ((tile_r + tile_c) % 2) * 0.5  # point-symmetric already on an even floor: mirror changes nothing
    px, py = point
    if name == "radial":
        images = [(px, py)]
        if mirror:
            images += [(width - px, py), (px, height - py), (width - px, height - py)]
        distance = np.min([np.hypot(x - ix, y - iy) for ix, iy in images], axis=0)
        # normalised by the floor's farthest corner from the point, so the map
        # stays in 0..1 wherever the point is
        corners = [(0.0, 0.0), (width, 0.0), (0.0, height), (width, height)]
        far = max(math.hypot(cx - px, cy - py) for cx, cy in corners)
        return distance / far if far > 0 else np.zeros_like(distance)
    if name == "angle":
        # 0 due north (+y, up as displayed), increasing clockwise; mirrored,
        # east and west match: 0 north to 1 south either way round
        if mirror:
            return np.arctan2(np.abs(x - px), y - py) / np.pi
        return np.mod(np.arctan2(x - px, y - py) / (2 * np.pi), 1.0)
    raise ValueError(f"no phase map {name!r}")


@lru_cache(maxsize=8)
def _partner(geometry: FloorGeometry, resolution: str) -> np.ndarray:
    """Each tile's (or LED's) flat index -> its 180-degree partner's."""
    if resolution == "tile":
        index = np.arange(geometry.tile_rows * geometry.tile_cols).reshape(geometry.tile_rows, geometry.tile_cols)
        return index[::-1, ::-1].reshape(-1)
    cells = geometry.led_to_cell.reshape(-1, 2).astype(np.intp)
    grid = np.full((geometry.height, geometry.width), -1, dtype=np.intp)
    grid[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    return grid[geometry.height - 1 - cells[:, 0], geometry.width - 1 - cells[:, 1]]
