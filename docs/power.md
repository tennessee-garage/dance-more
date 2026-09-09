# Power

The floor runs on **12 V**, supplied by two line-powered **Mean Well 12 V /
40 A** supplies (960 W combined) and distributed by daisy chain.

## Measured / estimated load

| Scope | LEDs | Current @ 12 V (full white) | Power |
| --- | --- | --- | --- |
| Per tile | 60 | **~0.55 A** | ~6.6 W |
| Per row (8 tiles) | 480 | ~4.4 A | ~53 W |
| Whole floor (64 tiles) | 3,840 | **~35 A** | **~420 W** |

- **Measured at 60 LEDs** on 2026-09-08, via a row controller's own INA226
  with two tiles attached. These replace the earlier figures, which were the
  40-LED measurements scaled by 1.5 and put per-tile draw at ~0.75 A.
- The scaled numbers were high because they assumed white costs the sum of
  three channels. It does not — see "LED current tracks the brightest
  channel" below, which is why the floor total drops from ~48 A to ~35 A.
- LED current only: it excludes the ATtiny3224, the Xiao row controllers and
  the transceivers. A row controller plus two idle tiles measured 0.30 A.
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

Measured on the bench (2026-09-08, one row controller and two tiles, via the
row's own INA226), rather than scaled from the 40-LED numbers:

| | Per tile (LED) | Per row (8 tiles) | Per supply (4 rows) | Per supply rating |
| --- | --- | --- | --- | --- |
| Full white @ 12 V | **0.55 A** | ~4.4 A | ~17.6 A | **40 A** |

That is a **2.3× margin**, not the 1.67× this section previously estimated by
scaling. Logic draw is on top: a row controller plus two idle tiles measured
0.30 A, so allow roughly 0.15 A per row of controller/transceiver/ATtiny
overhead.

### LED current tracks the brightest channel, not the sum

The measurement that moves the number. Tile current follows the sum over LEDs
of **max(R, G, B)** — not R+G+B:

| Commanded (all 60 LEDs) | Channel sum | Measured |
| --- | --- | --- |
| `(200, 0, 0)` | 200 | 427 mA |
| `(200, 200, 0)` | 400 | 427 mA |
| `(200, 200, 200)` | 600 | 429 mA |
| `(255, 255, 255)` | 765 | 549 mA |
| `(255, 0, 0)` | 255 | 548 mA |

So **full white costs the same as a single primary at the same level**, and a
three-channel estimate overstates white by ~3×. On the governing channel the
relationship is linear to better than 1%, and predictions from it matched
measurement within 2 mA of 427 across solid, gradient and rainbow frames.
Per-tile draw is additive to within 1 mA.

The mechanism is **not confirmed** — the WS2815 datasheet has not been checked
against this — so treat the relationship as an empirical result that holds
across the range tested, not as a datasheet claim.

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

- **Confirm the mechanism behind the max-channel result** against the WS2815
  datasheet. The measurement is solid and repeatable; the explanation is not
  established, and the whole supply margin now rests on it holding at scale.
- **Re-measure with more than two tiles.** Per-tile draw was additive to
  within 1 mA at two, but the full-row figure is still 8× a two-tile
  measurement rather than a measured row.
- **Voltage drop** along the 16 AWG per-tile splice run — confirm 12 V holds at
  the last tile in a row. At the measured 60-LED figure this is ~4.4 A per row
  rather than the ~6 A previously assumed — better than feared, but still up
  from ~4 A at 40 LEDs, and it remains the most pressing open question here
  because
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
