# Dance Floor v2 — Design & Architecture

A portable, light-up dance floor built from an 8×8 grid of independently
addressable acrylic **tiles**, driven by a layered controller hierarchy.

This `docs/` folder is the design home for the project. It is written to grow
incrementally — sections marked **Open Question** or **TBD** are deliberate
placeholders to be resolved as the design firms up.

## Contents

| Document | Covers |
| --- | --- |
| [architecture.md](architecture.md) | System overview, control hierarchy, block diagram, key counts |
| [communication.md](communication.md) | Bus topologies, RS-485 layering, addressing, SENSE auto-mapping protocol |
| [row-bus-protocol.md](row-bus-protocol.md) | Row Bus (Pi ↔ Row Controller) command protocol: frame format, command set, timing |
| [tile-bus-protocol.md](tile-bus-protocol.md) | Tile Bus (Row Controller ↔ Tile) command protocol: frame format, command set, timing |
| [hardware-tile.md](hardware-tile.md) | Tile: wood frame, acrylic, WS2815 LEDs, ATtiny3224, connectors |
| [hardware-row-controller.md](hardware-row-controller.md) | Row controller: Xiao RP2350, dual RS-485 buses, connectors |
| [power.md](power.md) | 12 V distribution and daisy-chain power architecture |
| [glossary.md](glossary.md) | Shared terminology |

## Repository map (current)

| Path | Purpose |
| --- | --- |
| `pcb/tile/` | KiCad project for the tile controller PCB |
| `pcb/row-controller/` | KiCad project for the row controller PCB |
| `src/tile/DF2-Tile/` | PlatformIO firmware for the ATtiny3224 tile controller |
| `src/simulation/` | Three.js/TypeScript acrylic light-propagation simulator |
| `src/native-simulator/` | Python (pyglet/numpy) light-propagation simulator |
| `docs/` | This design documentation |
