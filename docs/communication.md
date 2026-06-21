# Communication

The floor uses two layers of **half-duplex RS-485** carrying UART data. Every
bus is a shared multidrop pair: nodes daisy-chain through their connectors so
the "in" and "out" wiring is electrically continuous.

## Row Bus — Host ↔ Row controllers

- **Members:** Raspberry Pi (master) + 8 row controllers.
- **Medium:** a single twisted pair carrying RS-485 UART data.
- **Topology:** multidrop. The "in" twisted pair shares the same **4P4C**
  plug/jack as the "out" twisted pair at each row controller, forming a
  daisy chain. The pair is electrically continuous through every controller.
- **Operation:** the Pi transmits frames tagged with a target row controller's
  address. Each controller captures frames addressed to it and ignores the
  rest. Row identity is **static** (by install position) — no SENSE on the Row Bus.
- **Connector:** 4P4C (single pair in + single pair out).

## Tile Bus — Row controller ↔ Tiles (one per row)

- **Members:** 1 row controller (master) + 8 tile controllers.
- **Medium:** a **2-pair (4-conductor) twisted-pair cable**:
  - **Pair 1 — RS-485 data:** UART, electrically continuous between the "in"
    and "out" cable so all tiles share one bus.
  - **Pair 2 — GND + SENSE:** ground plus a "sense" line used for startup
    position discovery (see below).
- **Topology:** multidrop daisy chain. At each tile the incoming 4-conductor
  cable and the outgoing 4-conductor cable **share a single 8P8C plug**.
- **Tile transceiver:** THVD1420DR, **RX by default** (always listening).
  A tile drives TX **only** when a command explicitly requires a reply.
- **Normal operation:** the row controller sends data to all 8 tiles on the bus
  with a specific tile's address; the addressed tile ingests its data.

### SENSE auto-mapping (startup position discovery)

The SENSE line lets the row controller learn which tile controller ID sits at
which physical position in the row, without hard-coding tile IDs to slots.

The **"in" SENSE** and **"out" SENSE** at each tile are **separate** signals.
The probe always walks the chain **outward from the row controller**, one tile
at a time:

1. The **row controller** lowers the SENSE line to its **first, immediately
   adjacent tile** (slot 0).
2. The row controller asks on the bus **"who sees their SENSE low?"** The
   adjacent tile responds with its **unique address**. The row controller now
   knows the address for **tile 0**.
3. The row controller commands **tile 0** to bring **its** (outgoing) SENSE
   low, then again asks on the bus **"who sees their SENSE low?"** The next
   adjacent tile responds with its address — the address for **tile 1**.
4. This pattern repeats — each newly-discovered tile is told to lower its
   outgoing SENSE so the controller can identify the next tile down the chain —
   until all 8 tiles have been discovered **in physical order**.

The row controller assembles the resulting **logical slot (0..7) → discovered
tile address** map and **keeps it locally**. It is **not** reported upstream:
the Raspberry Pi never learns the discovered tile addresses and simply
addresses logical slots `0..7`. The row controller translates each logical
slot to the discovered tile address at transmit time.

## Addressing model

Two address spaces meet at the row controller, which is the only place the
translation happens:

- **Logical (Pi side):** the Pi addresses a fixed `(row 0..7, tile 0..7)`
  grid. These logical slot numbers are stable regardless of which physical tile
  controller ended up where.
- **Physical (tile side):** each tile controller has its own discovered address
  on the Tile Bus.
- **Translation:** the row controller holds a local `slot 0..7 → discovered
  tile address` map (built by SENSE auto-mapping). On receiving data for
  logical tile `N`, it looks up `N`, then transmits to that physical address.
  The Pi never sees physical tile addresses.

## Tile Bus Command Protocol

The full command set, frame format, address space, retry policy, and timing
budget for Tile Bus are documented in [tile-bus-protocol.md](tile-bus-protocol.md).

## Open Questions

- **Termination & biasing:** RS-485 termination resistor placement (Pi end,
  last tile) and fail-safe bias for both buses.
- **SENSE electrical detail:** pull-up/pull-down values for the internal
  RP2350 / ATtiny3224 pull-ups, drive strength when a tile pulls SENSE low,
  and maximum cable capacitance on the SENSE pair.
- **Failure handling:** whether the SENSE map rebuilds at runtime (e.g. on
  tile hot-swap) or only at startup.
- **Row Bus protocol:** documented in [row-bus-protocol.md](row-bus-protocol.md).
