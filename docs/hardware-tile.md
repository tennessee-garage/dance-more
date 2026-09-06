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
  tile beneath the acrylic — **exactly 15 LEDs per side, 60 per tile**, with the
  **corners unpopulated** — to edge-light the sheet. (Frosted/translucent
  acrylic diffuses the LED light — see the simulators in `src/simulation/` and
  `src/native-simulator/` for light-propagation studies.) See
  [LED layout and chain order](#led-layout-and-chain-order) below.

## Electronics

- **MCU:** **Microchip ATtiny3224** (tile controller).
  - **Why ATtiny3224 (decision):** The tile's job is simple: listen on RS-485,
    decode addressed frames, drive WS2815 LEDs, and reply on SENSE discovery.
    The ATtiny3224 is cheap (~$2), small, low-power, and has sufficient I/O and
    memory. Early concern about **WS2815 timing conflicts with RS-485 RX** —
    driving 60 LEDs takes ~1.8 ms with interrupts disabled — is **solved by
    protocol design**, not by switching MCUs. Each tile only receives its own
    addressed packet once per frame (~33 ms at 30 FPS). The firmware pushes to
    LEDs immediately after receiving, leaving a ~31 ms quiet window before the
    row controller re-addresses this tile. That window narrows as the Tile Bus
    fills up — at 60 LEDs a full row of `SET_LEDS` occupies ~16 ms of the
    33 ms period (tile-bus-protocol.md §8) — but 1.8 ms of blocking still fits
    comfortably inside what remains. Any broadcast or SENSE query is
    scheduled outside that window. Thus 1.8 ms of blocking is acceptable. **AVR
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
    far under the 100 mA rating. Unaffected by the LED count: the WS2815s draw
    from 12 V directly, not this rail.
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

Pin numbers below are as fabricated, read from J6's net in `pcb/tile/tile.kicad_sch`.

| Pin | Signal | Notes |
| --- | --- | --- |
| 1 | RS-485 A (in) | Pair 1 |
| 2 | RS-485 B (in) | Pair 1 |
| 3 | RS-485 A (out) | electrically common with "in" |
| 4 | RS-485 B (out) | electrically common with "in" |
| 5 | SENSE (in) | Pair 2 |
| 6 | GND | Pair 2 |
| 7 | SENSE (out) | Pair 2, separate from SENSE (in) |
| 8 | GND | Pair 2 |

## Firmware

- Lives in `src/tile/DF2-Tile/` (PlatformIO, ATtiny3224 target).
- Responsibilities: listen on RS-485, decode addressed frames, drive WS2815
  pixels, participate in SENSE auto-mapping, and reply (TX) only when commanded.
- **Timing model:** The row controller addresses tiles in sequence within each
  frame period (~33 ms at 30 FPS). After receiving its data, a tile
  immediately pushes to WS2815 (blocking, ~1.8 ms at 60 LEDs, interrupts off).
  By the time the push is done, the row controller has moved on to the next
  tile — a `SET_LEDS` frame is 1.87 ms on the wire, so the push and the next
  tile's frame run concurrently rather than the tile falling behind. The tile
  will not be re-addressed for the rest of the frame period, so the brief
  blocking window is safe.
  Broadcast commands (like frame latch) must be scheduled outside individual
  tiles' push windows, e.g., at a fixed point after all tiles have been
  addressed.

## LED layout and chain order

Fixed 2026-08-22 at 10 LEDs/side; **revised 2026-08-24 to 15/side** when the
build switched to reusing the existing floor's frames (see "Why 15" below).
The host driver models the floor on these numbers, so they are
a build specification rather than a preference — see the driver's
[floor geometry issue](https://github.com/tennessee-garage/dance-more/issues/59)
and [epic](https://github.com/tennessee-garage/dance-more/issues/58).

**15 LEDs per side, corners unpopulated, 60 per tile.** Each side is an
independent run of 15; no LED sits on a corner. The driver models a tile as a
17 × 17 cell block whose 64-cell perimeter ring carries the 60 LEDs with the 4
corner cells dark, which is what makes each side a clean run of 15 — a 15 × 15
block would offer only 56 perimeter cells, because corners are shared between
adjacent sides.

**Why 15.** The floor reuses the existing wooden tile frames, replacing their
non-addressable strip with WS2815. That strip is **300 LED/5m** (16.7 mm
pitch), and square corner structures inside the 15" frame leave only a **~10"
ledge** per side to lay strip on — 15 LEDs is what fits. The alternative was
150 LED/5m at ~7 per side (28/tile); the denser strip was chosen for edge
resolution. The pipeline cost of 60 over 40 is documented in
[row-bus-protocol.md](row-bus-protocol.md) §1.

This is a build specification: firmware and host derive every buffer and
payload size from it. `LEDS_PER_SIDE` in
[protocol.h](../src/common/tile_bus_protocol/protocol.h) is the firmware's
single source of truth; `FloorGeometry` in
[geometry.py](../src/pi/src/df2_pi/geometry.py) is the host's.

**Chain index 0 is at the lower-left of the tile.** Since that corner is
unpopulated, LED 0 is the bottom-most LED of the **left** side, and the strip
runs **up** the left side from there:

```
              0     1     2    ...    15    16
            +--------------------------------------
        16  |  .    15    16    ...   29     .
        15  | 14                            30
        14  | 13                            31
         :  |  :                             :
         1  |  0                            44
         0  |  .    59    58    ...   45     .
               ^
               LED 0: bottom of the left side
```

| LEDs | Side | Direction |
| ---- | ---- | --------- |
| 0–14  | left   | bottom → top |
| 15–29 | top    | left → right |
| 30–44 | right  | top → bottom |
| 45–59 | bottom | right → left |

"Left" and "bottom" are as seen **looking down at the floor**, with the tile in
its installed orientation. The strip therefore passes each unpopulated corner as
a diagonal step: LED 14 → 15, 29 → 30, 44 → 45 and 59 → 0.

Nothing on the tile PCB constrains this — the board exposes only a `LED_Data`
net and a "LED Strip" connector — so it is an **assembly instruction**. Lay the
strip in starting at the lower-left, heading up. `df2-pi ledwalk` lights LEDs
0 → 59 one at a time to verify a built tile matches; a tile that walks any other
way has its strip in backwards.

## Open Questions

- **LED run inset vs. the cell model.** 15 LEDs at 16.7 mm pitch span ~250 mm
  of a 381 mm (15") side, so each run sits inset with **~65 mm dark at both
  ends of every side** — the corner structures the ~10" ledge works around.
  `FloorGeometry` currently spaces cells evenly across the full side, so its
  positions are exact in *order* and correct in *topology* but not to physical
  scale near the corners. Decide whether the model should carry the inset
  (it matters for corner-crossing effects, for edges joining across adjacent
  tiles, and for matching the simulators' light propagation), or whether even
  spacing is good enough given how much the acrylic diffuses. Applies equally
  at any per-side count — 7 LEDs of 150 LED/5m strip would span ~233 mm.
- **Acrylic finish:** frosting/diffusion spec and standoff height between LEDs
  and acrylic.
- **Mounting:** how the LEDs are fixed beneath the acrylic / to the frame.
- **Strip current through the tile PCB.** J1 carries +12V / LED_Data / GND on
  a 2.54 mm header, so all LED current crosses the board: ~0.75 A/tile at full
  white with 60 LEDs, up from ~0.5 A at 40. Fine for the header; confirm the
  12 V trace width in KiCad.
