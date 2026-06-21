# Glossary

| Term | Meaning |
| --- | --- |
| **Tile** | One square floor section: open 1"×2" wood frame, diagonal brace, 3/8" acrylic top, WS2815 LEDs underneath, driven by an ATtiny3224. 64 total (8×8). |
| **Row** | One line of 8 tiles, driven by a single row controller. 8 rows total. |
| **Host / Raspberry Pi** | Top-tier controller; generates animation frames and drives the Row Bus. Owns the floor map. |
| **Row controller** | Tier-2 board (Xiao RP2350) bridging the Row Bus to a per-row Tile Bus. |
| **Tile controller** | Tier-3 MCU (ATtiny3224) on each tile; drives WS2815 LEDs and listens on the Tile Bus. |
| **Row Bus** | Single-pair, half-duplex, multidrop RS-485 bus between the Pi and all 8 row controllers. 4P4C daisy chain. |
| **Tile Bus** | Per-row 2-pair RS-485 bus (data pair + GND/SENSE pair) between a row controller and its 8 tiles. 8P8C daisy chain at tiles. |
| **SENSE** | A line in the Tile Bus's second pair, pulled low at startup so a tile can probe its adjacent neighbor; used to build the row controller's local logical-slot (0..7) → discovered tile-address map. "In" and "out" SENSE are separate. |
| **WS2815** | 12 V individually addressable LED (with backup data line) used to light each tile. |
| **ATtiny3224** | Microchip AVR MCU used as the tile controller. |
| **Xiao RP2350** | Seeed Studio compact module (RP2350: dual Cortex-M33 + PIO, no WiFi); the row controller MCU. |
| **THVD1420DR** | TI RS-485 transceiver on the tile; held in RX by default. |
| **4P4C** | 4-position 4-contact modular connector (single twisted pair). Used on the Row Bus and at the row controller end of the Tile Bus. |
| **8P8C** | 8-position 8-contact modular connector (two pairs). Used at tiles on the Tile Bus; carries both the "in" and "out" 4-conductor cables. |
| **Multidrop** | A bus where all nodes share one electrically-continuous pair; each node filters by address. |
| **Daisy chain** | Wiring where each node passes the bus through "in"→"out" to the next node. |
