# Power

The floor runs on **12 V**, supplied by two line-powered **Mean Well 12 V /
40 A** supplies (960 W combined) and distributed by daisy chain.

## Measured / estimated load

| Scope | LEDs | Current @ 12 V (full white) | Power |
| --- | --- | --- | --- |
| Per tile | 60 | **~0.75 A** | ~9 W |
| Per row (8 tiles) | 480 | ~6 A | ~72 W |
| Whole floor (64 tiles) | 3,840 | **~48 A** | **~576 W** |

- These are the **40-LED measurements scaled by 1.5** for the move to 60
  LEDs/tile. Per-LED draw doesn't change with strip density — 300 LED/5m and
  150 LED/5m use the same WS2815 emitters at a different pitch — so the scaling
  is straightforward, but the figures below are **derived, not measured at 60**.
  Re-run `tile_brightness_sweep.py` on a 60-LED tile to confirm.
- The underlying **0.5 A/tile at 40 LEDs** is a measured worst case (40 WS2815
  LEDs at full white). It is **LED current only** — it does not include the
  ATtiny3224, the Xiao row controllers, or transceivers.
- A later, instrumented remeasurement — `tile_brightness_sweep.py`, see
  [docs/measurements/](measurements/) — puts LED-only current at full white
  closer to **~418 mA** for 40 LEDs (**~627 mA** scaled to 60), with the row
  controller + tile's own quiescent draw (~108 mA) subtracted out explicitly
  rather than folded in. The provenance
  of the original 0.5 A figure isn't known (likely different driving
  hardware, possibly a different strip), so this isn't treated as a
  correction, just corroboration - same order of magnitude, and **0.5 A/tile
  at 40 LEDs, i.e. 0.75 A at 60, is kept as the figure used throughout this
  doc** (including the per-row/per-floor rollups below) for its clean rounding
  and built-in headroom.
- Full white is the worst case; typical animated content draws less, but the
  supply and wiring must be sized for the ~48 A peak (plus headroom).

## Distribution architecture

- **Supplies:** two **Mean Well 12 V / 40 A** PSUs, each line-powered.
- **Per-supply coverage:** each PSU feeds **every other row** via a **14 AWG
  zip cord**. So the two supplies interleave across the 8 rows (e.g. PSU 1 →
  rows 0, 2, 4, 6; PSU 2 → rows 1, 3, 5, 7), giving each PSU **4 rows**.
- **Into each row:** the 14 AWG zip cord arrives at the row's **row
  controller**. A **pigtail taps 12 V there to power the row controller**
  itself.
- **To the tiles:** from that tap, a **16 AWG cable** is spliced to each of the
  8 tiles in the row, supplying their 12 V.
- All power wiring is separate from the RS-485 data cabling.

### Load vs. capacity

| | Per row (LED) | Per supply (4 rows, LED) | Per supply rating |
| --- | --- | --- | --- |
| Current @ 12 V | ~6 A | ~24 A | **40 A** |

Each 40 A supply carries roughly **24 A of LED load** at full white (plus
controller/logic draw). That still fits, but the move from 40 to 60 LEDs/tile
took the margin from 2.5× down to **1.67×** — the supplies remain adequate,
and are no longer generously so.

## Why 12 V

- WS2815 LEDs are 12 V addressable LEDs, so the tile lighting runs natively at
  12 V with no per-tile boost/buck for the LEDs themselves.
- **Row controller:** a **Recom R-78E5.0-0.5** (5 V / 0.5 A switcher) derives
  the 5 V rail from the 12 V tap, powering the Xiao RP2350 and both RS-485
  transceivers. ~200–250 mA worst case vs 0.5 A rating. See
  [hardware-row-controller.md](hardware-row-controller.md).
- **Tile:** a **TI TLV76050DBZR** LDO (5 V / 100 mA, 30 V max input) derives
  5 V from the 12 V tile feed for the ATtiny3224 and THVD1420DR (~15 mA
  typical). The WS2815 LEDs run directly from 12 V. See
  [hardware-tile.md](hardware-tile.md).

## Open Questions

- **Confirm the 60-LED figures by measurement** rather than by scaling — see
  the note above. Everything in this doc is currently 1.5× the 40-LED numbers.
- **Controller/logic current:** add the ATtiny3224, Xiao, and transceiver draw
  on top of the ~24 A/supply LED figure (still under 40 A).
- **Voltage drop** along the 16 AWG per-tile splice run — confirm 12 V holds at
  the last tile in a row. **This got 50% worse at 60 LEDs** (~6 A per row
  rather than ~4 A) and is now the most pressing open question here, because
  the fix is a wiring-layout decision worth making once. The lever is **where
  the 14 AWG feeder enters the row** (i.e. row-controller placement), not the
  16 AWG. Extending 16 AWG to the middle does **not** help: that feeder segment
  still carries the full row current over its whole length. Only bringing the
  low-resistance 14 AWG feed to the row's center — so the 16 AWG never carries
  more than ~3 A over short half-runs — meaningfully reduces the worst-case
  drop.
- **Connectors & polarity protection** for the 12 V daisy chain at both levels.
- **Inrush / soft-start** for 64 tiles of WS2815 powering on together.
- **Grounding:** relationship between the 12 V return and the RS-485 GND/SENSE
  reference.
