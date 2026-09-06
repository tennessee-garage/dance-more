"""FloorGeometry: the single source of truth for what is an LED and where it is.

Every other piece of the driver - the pixel frame, the edge API, the encoder,
the preview renderer - derives its mapping from a FloorGeometry instance
rather than hardcoding 136 or 60 anywhere. See docs/hardware-tile.md's "LED
layout and chain order" and the floor-orientation section of
docs/row-bus-protocol.md's driver work for the physical facts this encodes.

Four coordinate systems meet here:

    Tile  (tile_row, tile_col), 0-7 each - logical position on the floor
    Chain (tile_index, led_index), led 0-59 - WS2815 wire order (SET_LEDS)
    Cell  (y, x), 0-135 each - the 136x136 grid; the "image" view
    Bus   (row_addr, slot) - what transport.floor.Floor actually addresses

Bus is the identity of Tile: the Pi sits off to the side, all 8 row
controllers sit in one line along a single floor edge with row 0 nearest the
Pi, and each drives its 8 tiles outward from that edge with slot 0 nearest
the controller. So address_of(tile) is just divmod(tile, tile_cols).

Row indices ascend AWAY from the Pi, so the canonical (y, x) grid has y=0
nearest the Pi and the canonical view is drawn with origin at the
bottom-left. `to_display()` is the one place that flip happens; the array
itself, the encoder, and every animation stay in canonical orientation.

Each tile is a 17x17 cell block (leds_per_side=15, corners dark): the
64-cell perimeter ring holds the tile's 60 LEDs with the 4 corner cells
unpopulated, and the inner 15x15 is dark. Chain index 0 sits at the
lower-left of the tile (the bottom of its left side, since the corner
itself is dark) and the chain climbs the left side, crosses the top
left-to-right, descends the right side, and returns along the bottom -
stepping diagonally across each dark corner between sides.

One thing this model does NOT capture: the 15 LEDs on a side are 300/5m
strip at 16.7 mm pitch, so they span ~250 mm of a 381 mm (15") tile side -
the wooden frame's corner structures leave only a ~10" ledge to lay strip
on. The cells are therefore evenly spaced across the full side here while
the real LEDs sit inset, with ~65 mm dark at each end of every run. Cell
positions are exact in ORDER and correct in topology, but not to physical
scale near the corners; anything doing real-world distance work (or
matching the simulators' light propagation) needs that inset modelled
first. See docs/hardware-tile.md's open question on it.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Side(Enum):
    """A tile's four edges, named by compass direction in canonical floor
    orientation: NORTH is +y (away from the Pi), EAST is +x (away from the
    row controllers)."""

    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"


class FloorGeometry:
    """What is an LED and where it is, for one floor configuration.

    Physical orientation (which tile is row 0, which end of a row is slot 0)
    and LED-0 placement within a tile are fixed floor facts, not options -
    see the module docstring. The only real configuration surface is the
    tile's cell geometry, and even that has one supported configuration
    today: unpopulated corners. `corners_populated=True` is accepted as a
    constructor argument because the LED count formula generalizes, but the
    chain-order algorithm below is only specified for dark corners, so it is
    rejected rather than guessed at.

    Lookup tables are built once here and cached on the instance - never
    per-frame:

        lit_mask       (H, W) bool        - True where a real LED sits
        led_to_cell    (tiles, N, 2) int16 - chain index -> (y, x)
        cell_to_led    (H, W, 2) int32     - (y, x) -> (tile, led), (-1,-1) if dark
        led_positions  (tiles, N, 2) float32 - cell-centre coords, for distance fields

    The trailing `2` on each of these is not an index dimension - it is the
    width of the (y, x) pair stored at each slot. `led_to_cell[t, l]` is
    indexed by tile and led and returns a 2-vector; `cell_to_led[y, x]` is
    indexed by cell and returns a 2-vector. This is what lets the whole
    table be used as a single numpy fancy-index:

        ys, xs = geo.led_to_cell[..., 0], geo.led_to_cell[..., 1]
        grid[ys, xs] = frame.data          # scatter every LED in one op

    and why `cell_to_led[..., 0] >= 0` *is* `lit_mask` - both are built from
    the same source so they cannot disagree. `led_positions` holds cell
    *centres* (e.g. `[1.5, 0.5]`, not `[1, 0]`) rather than the exact
    integer index, so distance-field math (`splat`, `line`, ...) is not
    biased half a cell toward the origin; `axis=-1` on a `led_positions`
    difference reduces away that trailing `2` into a per-LED distance,
    leaving a `(tiles, N)` array that aligns element-for-element with a
    pixel frame's own leading axes.
    """

    def __init__(
        self,
        tile_grid: tuple[int, int] = (8, 8),
        leds_per_side: int = 15,
        corners_populated: bool = False,
    ) -> None:
        if corners_populated:
            raise NotImplementedError(
                "corners_populated=True has no specified LED chain order - "
                "the lower-left-origin, dark-corner walk this module implements "
                "assumes unpopulated corners"
            )
        tile_rows, tile_cols = tile_grid
        if tile_rows < 1 or tile_cols < 1:
            raise ValueError(f"tile_grid must be positive, got {tile_grid}")
        if leds_per_side < 1:
            raise ValueError(f"leds_per_side must be positive, got {leds_per_side}")

        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.tiles = tile_rows * tile_cols
        self.leds_per_side = leds_per_side
        self.corners_populated = corners_populated
        self.cell_size = leds_per_side + 2
        self.leds_per_tile = 4 * leds_per_side
        self.height = tile_rows * self.cell_size
        self.width = tile_cols * self.cell_size
        self.led_count = self.tiles * self.leds_per_tile

        local = _local_led_cell(leds_per_side)  # (leds_per_tile, 2), one tile's layout
        origins = np.stack(
            [
                (np.arange(self.tiles) // tile_cols) * self.cell_size,
                (np.arange(self.tiles) % tile_cols) * self.cell_size,
            ],
            axis=-1,
        )  # (tiles, 2)

        led_to_cell = origins[:, None, :] + local[None, :, :]  # (tiles, leds_per_tile, 2)
        self.led_to_cell = led_to_cell.astype(np.int16)
        self.led_positions = (led_to_cell.astype(np.float32)) + 0.5

        ys = led_to_cell[..., 0].ravel()
        xs = led_to_cell[..., 1].ravel()
        tile_ids = np.repeat(np.arange(self.tiles), self.leds_per_tile)
        led_ids = np.tile(np.arange(self.leds_per_tile), self.tiles)

        cell_to_led = np.full((self.height, self.width, 2), -1, dtype=np.int32)
        cell_to_led[ys, xs, 0] = tile_ids
        cell_to_led[ys, xs, 1] = led_ids
        self.cell_to_led = cell_to_led
        self.lit_mask = cell_to_led[..., 0] >= 0

        self._local_led_cell = local

    @classmethod
    def default(cls) -> FloorGeometry:
        """The real floor: 8x8 tiles, 10 LEDs/side, corners dark."""
        return cls()

    # ---- tile <-> cell ----------------------------------------------------

    def tile_origin(self, tile: int) -> tuple[int, int]:
        """(y0, x0) of `tile`'s cell_size x cell_size block."""
        row, col = divmod(tile, self.tile_cols)
        return row * self.cell_size, col * self.cell_size

    def tile_at(self, y: int, x: int) -> int | None:
        """The tile owning the LED at cell (y, x), or None if that cell is
        dark (a corner or the tile's interior)."""
        if not (0 <= y < self.height and 0 <= x < self.width):
            raise ValueError(f"cell ({y}, {x}) is outside the {self.height}x{self.width} grid")
        tile, _led = self.cell_to_led[y, x]
        return int(tile) if tile >= 0 else None

    # ---- tile <-> bus -------------------------------------------------------

    def address_of(self, tile: int) -> tuple[int, int]:
        """(row_addr, slot) for `tile` - the identity, by physical
        convention (see the module docstring)."""
        return divmod(tile, self.tile_cols)

    # ---- edges --------------------------------------------------------------

    def side_leds(self, tile: int, side: Side) -> np.ndarray:
        """The `leds_per_side` chain indices (0..leds_per_tile-1) running
        along one side of `tile`, ordered west->east (NORTH/SOUTH) or
        south->north (EAST/WEST) - i.e. left-to-right as displayed in
        canonical floor orientation, regardless of the WS2815 chain's own
        winding direction on that side."""
        n = self.leds_per_side
        cell_size = self.cell_size
        local = self._local_led_cell
        if side is Side.WEST:
            mask = local[:, 1] == 0
            axis = 0  # sort by ascending y (south -> north)
        elif side is Side.EAST:
            mask = local[:, 1] == cell_size - 1
            axis = 0
        elif side is Side.NORTH:
            mask = local[:, 0] == cell_size - 1
            axis = 1  # sort by ascending x (west -> east)
        elif side is Side.SOUTH:
            mask = local[:, 0] == 0
            axis = 1
        else:
            raise ValueError(f"unknown side: {side}")

        led_indices = np.nonzero(mask)[0]
        order = np.argsort(local[led_indices, axis])
        ordered = led_indices[order]
        assert len(ordered) == n
        return ordered

    # ---- display --------------------------------------------------------------

    def to_display(self, grid: np.ndarray) -> np.ndarray:
        """Flip a canonical (y, x, ...) grid for rendering: canonical y=0
        (nearest the Pi) is drawn at the BOTTOM, so this reverses the y
        axis and nothing else. Renderers should call this at the display
        boundary only - the array, the encoder, and every animation stay in
        canonical orientation. Applying this twice is the identity."""
        return grid[::-1, ...]


def _local_led_cell(n: int) -> np.ndarray:
    """One tile's (4n, 2) local (ly, lx) layout: chain index 0 at the
    bottom of the left side, climbing left, across the top, down the
    right, back along the bottom - see the module docstring."""
    cell_size = n + 2
    west = np.stack([np.arange(1, n + 1), np.zeros(n, dtype=int)], axis=-1)
    north = np.stack([np.full(n, cell_size - 1), np.arange(1, n + 1)], axis=-1)
    east = np.stack([np.arange(n, 0, -1), np.full(n, cell_size - 1)], axis=-1)
    south = np.stack([np.zeros(n, dtype=int), np.arange(n, 0, -1)], axis=-1)
    return np.concatenate([west, north, east, south], axis=0)
