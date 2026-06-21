# System Architecture

## Overview

Dance Floor v2 is a portable light-up dance floor composed of an **8 × 8 grid
of tiles** (64 tiles total). Each tile is an open wood frame topped with a
frosted/translucent acrylic sheet, lit from beneath by WS2815 LEDs. A
Raspberry Pi generates animation frames and distributes them down a three-tier
control hierarchy so that every tile can be addressed independently.

## Control hierarchy

```
                         ┌──────────────────┐
                         │   Raspberry Pi   │   Frame generator / show controller
                         │ (primary host)   │
                         └────────┬─────────┘
                                  │  RS-485 "row" bus (single twisted pair,
                                  │  half-duplex, multidrop, daisy-chained)
        ┌──────────┬──────────┬───┴──────┬──────────┬─────────┐
        │          │          │          │          │   ...   │   (8 row controllers,
     ┌──┴──┐    ┌──┴──┐    ┌──┴──┐     ┌──┴──┐    ┌──┴──┐               one per row)
     │ Row │    │ Row │    │ Row │     │ Row │    │ Row │
     │ ctl │    │ ctl │    │ ctl │     │ ctl │    │ ctl │   Xiao RP2350
     │  0  │    │  1  │    │  2  │     │  3  │    │  7  │
     └──┬──┘    └─────┘    └─────┘     └─────┘    └─────┘
        │  RS-485 "tile" bus (2-pair: RS-485 data + GND/SENSE,
        │  half-duplex, multidrop, daisy-chained)
   ┌────┼────┬────┬────┬────┬────┬────┬────┐
 ┌─┴─┐┌─┴─┐┌─┴─┐ ...                    ┌─┴─┐    (8 tiles per row)
 │T0 ││T1 ││T2 │                        │T7 │    ATtiny3224 + WS2815
 └───┘└───┘└───┘                        └───┘
```

### Tier 1 — Raspberry Pi (host)

- Generates frames of animation as a stream of bytes.
- Drives a single RS-485 bus ("Row Bus") shared by all 8 row controllers.
- Works in a **fixed logical abstraction**: it assumes the floor is rows
  `0..7`, each with tiles `0..7`, and addresses tiles by that logical
  `(row, tile)` index. It does **not** know the discovered tile addresses —
  the per-row mapping is held entirely on the row controllers.

### Tier 2 — Row controllers (×8)

- MCU: **Seeed Xiao RP2350** (dual Cortex-M33 + PIO; no WiFi). See
  [hardware-row-controller.md](hardware-row-controller.md) for the rationale.
- One per physical row. **Row identity is statically assigned** — row
  controllers are always installed in the same orientation/position, so they do
  not use SENSE lines to discover their row.
- Bridge two RS-485 buses:
  - **Row Bus (upstream):** shared multidrop bus to the Raspberry Pi. Each row
    controller listens for frames addressed to it and ignores the rest.
  - **Tile Bus (downstream):** its own RS-485 bus to the 8 tiles in that row.
- Distribute per-tile data down the Tile Bus and run the tile SENSE auto-mapping
  procedure. The resulting **logical-slot (0..7) → discovered tile address**
  map is **held locally on the row controller**; it is not reported upstream.
  When the Pi sends data for logical tile `N`, the row controller translates
  `N` to the actual tile address it discovered and transmits to that tile.

### Tier 3 — Tile controllers (×64)

- MCU: **ATtiny3224**.
- One per tile; drives that tile's WS2815 LEDs.
- RS-485 transceiver: **THVD1420DR**, held in **RX by default** (always
  listening). Switches to **TX only** when a command requires a reply.
- Ingests frame data addressed to it; participates in SENSE-based position
  discovery at startup.

## Key quantities

| Item | Count | Notes |
| --- | --- | --- |
| Tiles total | 64 | 8 rows × 8 tiles |
| Rows | 8 | One row controller each |
| Tiles per row | 8 | One tile controller each |
| Row controllers | 8 | Xiao RP2350 |
| Tile controllers | 64 | ATtiny3224 |
| RS-485 buses | 9 | 1 host bus (A) + 8 row→tile buses (B) |
| LEDs per tile | 40 | WS2815; ≥10 per side around the square perimeter |
| LEDs total | 2,560 | 64 tiles × 40 |

## Physical floor layout

- The floor is an 8 × 8 arrangement of square tiles.
- Tile dimensions, overall floor footprint, and tile-to-tile alignment/edging:
  **TBD**.

## Open Questions

- **Coordinate convention:** how are rows and tiles numbered (origin corner,
  direction)? This affects the logical→physical map.
- **Frame rate / latency budget:** target FPS and the per-frame byte budget
  end-to-end, which constrains baud rates on buses A and B. With 40 LEDs/tile
  at 3 bytes/LED that is 120 B of pixel data per tile, 960 B per row.
