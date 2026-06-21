# Hardware — Tile

A **tile** is one square section of the floor. The floor has 64 tiles in an
8 × 8 grid.

## Mechanical structure

- **Frame:** an open wood frame forming a square perimeter, built from
  **1" × 2" boards**.
- **Bracing:** a single corner-to-corner board (diagonal) keeps the tile
  square and true.
- **Top surface:** a **3/8" acrylic sheet** that rests on the corners of the
  wood frame.
- **Lighting:** **WS2815 LEDs** arranged around the **square perimeter** of the
  tile beneath the acrylic — **at least 10 LEDs per side, 40 per tile** — to
  edge-light the sheet. (Frosted/translucent acrylic diffuses the LED light —
  see the simulators in `src/simulation/` and `src/native-simulator/` for
  light-propagation studies.)

## Electronics

- **MCU:** **Microchip ATtiny3224** (tile controller).
  - **Why ATtiny3224 (decision):** The tile's job is simple: listen on RS-485,
    decode addressed frames, drive WS2815 LEDs, and reply on SENSE discovery.
    The ATtiny3224 is cheap (~$2), small, low-power, and has sufficient I/O and
    memory. Early concern about **WS2815 timing conflicts with RS-485 RX** —
    driving 40 LEDs takes ~1.2 ms with interrupts disabled — is **solved by
    protocol design**, not by switching MCUs. Each tile only receives its own
    addressed packet once per frame (~33 ms at 30 FPS). The firmware pushes to
    LEDs immediately after receiving, leaving a ~31 ms quiet window before the
    row controller re-addresses this tile. Any broadcast or SENSE query is
    scheduled outside that window. Thus 1.2 ms of blocking is acceptable. **AVR
    EB was considered as an upgrade path**, but the EB14 has **no DMA
    controller**, so it offers no concurrency advantage over the ATtiny — the
    WS2815 bit-banging is serial either way. Staying with the ATtiny keeps cost
    and complexity low.
- **LED driver:** WS2815 (12 V addressable LEDs, with backup data line). The
  ATtiny3224 drives the WS2815 data line **directly at 5 V** (no level shifter);
  5 V is the data-high level the WS2815 expects.
- **RS-485 transceiver:** THVD1420DR, default **RX** (always listening),
  switched to **TX** only when replying to specific commands.
  - **No TVS protection on Tile Bus (decision):** TVS diodes are omitted on
    the Tile Bus at both the tile and row controller PCBs. The THVD1420DR's
    built-in ±16 kV HBM / ±12 kV IEC 61000-4-2 contact-discharge ESD
    protection is sufficient for this application: Tile Bus connectors are
    internal to the assembled floor and are never hot-plugged during operation,
    and the venue environment is non-industrial (no motor drives, welding, or
    lightning-exposed cable runs). If a tile transceiver were damaged by ESD
    during assembly or teardown, the failure affects one tile out of 64 — not a
    bus-level or system-level fault. By contrast, the Row Bus (Pi ↔ row
    controllers) does carry TVS protection because that connection is realistically
    hot-plugged. **Operating procedure:** power down before disconnecting any
    Tile Bus cables.
- **12 V → 5 V regulation:** **TI TLV76050DBZR** LDO (fixed 5 V, 100 mA,
  30 V max input, SOT-23-3). Powers the ATtiny3224 and the THVD1420DR only —
  the WS2815 LEDs run directly from the 12 V feed, **not** this rail.
  - **Budget:** ~15 mA typical, ~30 mA worst case (transceiver transmitting) —
    far under the 100 mA rating.
  - **Dissipation:** LDO drops 7 V → ~0.1 W typical (~0.2 W during the rare
    TX bursts). Acceptable given the small draw; an LDO is preferred here over
    a switcher for cost/size across 64 tiles and to avoid switching noise near
    the RS-485 receiver.

## Connectors & cabling

Each tile is wired into the row's daisy chain with two cables that run "in" and
"out":

- **Power:** a **12 V cable** that daisy-chains to each of the 8 tiles in the
  row.
- **Data:** a **2-pair, 4-conductor twisted-pair cable**. The incoming
  4-conductor cable and the outgoing 4-conductor cable **share a single 8P8C
  plug** at the tile:
  - **Pair 1 — RS-485** (UART data), electrically continuous in↔out.
  - **Pair 2 — GND + SENSE**, where the **in** SENSE and **out** SENSE are
    separate (used for startup position discovery — see
    [communication.md](communication.md)).

### Connector pinout (8P8C, data)

| Pin | Signal | Notes |
| --- | --- | --- |
| — | RS-485 A (in) | Pair 1 |
| — | RS-485 B (in) | Pair 1 |
| — | RS-485 A (out) | electrically common with "in" |
| — | RS-485 B (out) | electrically common with "in" |
| — | GND | Pair 2 |
| — | SENSE (in) | Pair 2 |
| — | GND | Pair 2 |
| — | SENSE (out) | Pair 2, separate from SENSE (in) |

> Pin numbers **TBD** — fill from the `pcb/tile/` KiCad project once finalized.

## Firmware

- Lives in `src/tile/DF2-Tile/` (PlatformIO, ATtiny3224 target).
- Responsibilities: listen on RS-485, decode addressed frames, drive WS2815
  pixels, participate in SENSE auto-mapping, and reply (TX) only when commanded.
- **Timing model:** The row controller addresses tiles in sequence within each
  frame period (~33 ms at 30 FPS). After receiving its data, a tile
  immediately pushes to WS2815 (blocking, ~1.2 ms, interrupts off). By the time
  the push is done, the row controller has moved on to the next tile. The tile
  will not be re-addressed for ~31 ms, so the brief blocking window is safe.
  Broadcast commands (like frame latch) must be scheduled outside individual
  tiles' push windows, e.g., at a fixed point after all tiles have been
  addressed.

## Open Questions

- **LED placement detail:** exact per-side spacing and whether the corners are
  populated (≥10/side × 4 = 40, but corner handling affects the exact count).
- **Acrylic finish:** frosting/diffusion spec and standoff height between LEDs
  and acrylic.
- **Mounting:** how the LEDs are fixed beneath the acrylic / to the frame.
- **8P8C pinout** — confirm assignment and document pin numbers.
