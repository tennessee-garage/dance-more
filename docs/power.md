# Power

The floor runs on **12 V**, supplied by two line-powered **Mean Well 12 V /
40 A** supplies (960 W combined) and distributed by daisy chain.

## Measured / estimated load

| Scope | LEDs | Current @ 12 V (full white) | Power |
| --- | --- | --- | --- |
| Per tile | 40 | **~0.5 A** (measured) | ~6 W |
| Per row (8 tiles) | 320 | ~4 A | ~48 W |
| Whole floor (64 tiles) | 2,560 | **~32 A** | **~384 W** |

- The **0.5 A/tile** figure is a measured worst case (40 WS2815 LEDs at full
  white). It is **LED current only** — it does not yet include the ATtiny3224,
  the Xiao row controllers, or transceivers.
- Full white is the worst case; typical animated content draws less, but the
  supply and wiring must be sized for the ~32 A peak (plus headroom).

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
| Current @ 12 V | ~4 A | ~16 A | **40 A** |

Each 40 A supply carries roughly **16 A of LED load** at full white (plus
controller/logic draw), leaving comfortable headroom.

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

- **Controller/logic current:** add the ATtiny3224, Xiao, and transceiver draw
  on top of the ~16 A/supply LED figure (still well under 40 A).
- **Voltage drop** along the 16 AWG per-tile splice run — confirm 12 V holds at
  the last tile in a row. Note the lever here is **where the 14 AWG feeder
  enters the row** (i.e. row-controller placement), not the 16 AWG. Extending
  16 AWG to the middle does **not** help: that feeder segment still carries the
  full ~4 A row current over its whole length. Only bringing the low-resistance
  14 AWG feed to the row's center — so the 16 AWG never carries more than ~2 A
  over short half-runs — meaningfully reduces the worst-case drop.
- **Connectors & polarity protection** for the 12 V daisy chain at both levels.
- **Inrush / soft-start** for 64 tiles of WS2815 powering on together.
- **Grounding:** relationship between the 12 V return and the RS-485 GND/SENSE
  reference.
