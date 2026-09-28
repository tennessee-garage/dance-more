"""Every LED's position and raw-mode address: what a TouchDesigner (or any)
patch samples its image at to drive the floor in `raw` mode.

    GET /api/floor/leds[?format=csv]
    df2-pi leds > leds.csv

One row per LED in chain order - the order `raw` mode expects on the wire:

    index     0..3839, the LED's place in the raw stream
    tile, led the tile (row * 8 + col) and its LED 0..59
    x, y      cell centres in the canonical view: x to the right, y up
              away from the Pi, both 0..136
    u, v      the same, normalised 0..1 with the origin at the bottom-left
              - TouchDesigner's texture-coordinate convention
    universe  which universe carries it, counted from the first (0-based)
    channel   its red channel within that universe, 1-based as DMX software
              numbers them; green and blue follow

Positions are canonical: the floor-rotation setting turns the picture
after the fact, so a mapping built from this stays right at any rotation.
"""

from __future__ import annotations

import csv
import io

from df2_pi.geometry import FloorGeometry
from df2_pi.interfacing.packets import PIXELS_PER_UNIVERSE

FIELDS = ("index", "tile", "led", "x", "y", "u", "v", "universe", "channel")


def led_table(geometry: FloorGeometry) -> list[dict]:
    rows = []
    for tile in range(geometry.tiles):
        for led in range(geometry.leds_per_tile):
            index = tile * geometry.leds_per_tile + led
            y, x = (float(c) for c in geometry.led_positions[tile, led])
            rows.append(
                {
                    "index": index,
                    "tile": tile,
                    "led": led,
                    "x": x,
                    "y": y,
                    "u": round(x / geometry.width, 6),
                    "v": round(y / geometry.height, 6),
                    "universe": index // PIXELS_PER_UNIVERSE,
                    "channel": (index % PIXELS_PER_UNIVERSE) * 3 + 1,
                }
            )
    return rows


def led_csv(geometry: FloorGeometry) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(led_table(geometry))
    return out.getvalue()
