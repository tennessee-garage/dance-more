# Hardware — Row Controller

A **row controller** drives one physical row of 8 tiles and bridges the host
bus to that row's tile bus. There are 8 row controllers, one per row.

## MCU

**Seeed Xiao RP2350** (RP2350, dual Cortex-M33 + PIO).

### Why RP2350 (decision)

The row controller is a wired RS-485 **bridge** — receive on the Row Bus, forward to
the row's tiles on the Tile Bus. The job is I/O-bound, not compute-bound, and involves
**no wireless** in the data path.

- **No WiFi needed.** All communication is wired RS-485, so the ESP32's radio
  (and the dual-core argument for isolating its stack) is irrelevant here. Any
  WiFi capability for OTA/debug belongs on the Raspberry Pi, not on 8 row
  controllers.
- **DMA + PIO, not core count, is the real lever.** On a shared multidrop Row Bus
  the controller must *receive* the whole stream (~2.3 Mbps payload at 30 FPS)
  and discard the ~7/8 not addressed to its row. That is trivial with
  DMA-driven UARTs feeding a ring buffer. The RP2350's **PIO** can implement
  flexible, deeply-buffered UARTs (and offload framing / SENSE-pulse timing),
  which fits this problem better than a fixed UART block.
- **Dual core for free**, with no RTOS/radio baggage: Row Bus ingest on one core,
  Tile Bus egress on the other.
- **Same Xiao footprint** as the previously-considered ESP32 modules, so the
  existing `pcb/row-controller/` layout needs no rework to adopt it.
- **Cost/size** fit the ~$10, small-footprint criteria with headroom.

Considered and rejected: Xiao ESP32-C3 / ESP32-S3 (carry an unused radio; the
dual-core S3 was originally chosen to isolate WiFi, a need that disappears once
WiFi is dropped). The RP2040 is an equivalent fit; RP2350 was chosen for the
extra RAM/speed and longevity headroom.

## Buses

The row controller sits between two RS-485 buses:

### Row Bus — upstream to the Raspberry Pi

- A single twisted pair carrying RS-485 UART data, shared by the Pi and all 8
  row controllers (multidrop).
- The incoming pair and the outgoing pair **share the same 4P4C plug/jack**,
  forming a daisy chain that is electrically continuous through the controller.
- The controller captures frames addressed to its (static) row and ignores the
  rest.
- **RS-485 transceiver:** **THVD1420DR** (same as Tile Bus and the planned Pi
  hat). 12 Mbps maximum; 3–5.5 V supply; manual DE/RE direction control driven
  by the RP2350. Held in RX by default; RP2350 asserts DE to transmit
  responses.

### Tile Bus — downstream to the row's tiles

- The **start of the tile bus**: a **2-pair twisted-pair cable** (RS-485 data
  pair + GND/SENSE pair) terminated in a **second 4P4C plug/jack**.
- The row controller is the master of this bus and addresses each of its 8
  tiles individually. It also runs the SENSE auto-mapping procedure for the row
  (see [communication.md](communication.md)).

## Power

- **12 V** daisy-chained to each row controller (see [power.md](power.md)),
  tapped via a pigtail from the row's 14 AWG feed.
- **12 V → 5 V regulation:** **Recom R-78E5.0-0.5** switching regulator
  (5 V / 0.5 A, 7–28 V input). The 5 V rail feeds the Xiao RP2350 **and both
  RS-485 transceivers** (transceivers run at 5 V off this rail, not off the
  Xiao's onboard 3.3 V LDO).
- **Budget:** worst-case load ~200–250 mA (RP2350 ~60–100 mA, 2× transceiver
  quiescent ~10 mA, bus-B driver active ~30–40 mA into a ~60 Ω terminated bus,
  plus indicators) — roughly 2× headroom under the 0.5 A rating.
- **Bulk cap:** 100 µF on the 5 V rail, within the R-78E's max capacitive-load
  spec, so it starts cleanly.

## Row identity

- **Statically assigned.** Row controllers are always installed in the same
  physical orientation/position, so each one's row number is fixed and does not
  require SENSE discovery.

## Connector summary

| Connector | Type | Purpose |
| --- | --- | --- |
| Row Bus | 4P4C | Single-pair RS-485, in + out (daisy chain to Pi/other rows) |
| Tile Bus | 4P4C | 2-pair RS-485 data + GND/SENSE, start of tile bus |
| Power | TBD | 12 V in + out daisy chain |

> Connector pinouts **TBD** — fill from the `pcb/row-controller/` KiCad project.

## Open Questions

- **Tile Bus connector vs the tile's 8P8C:** Tile Bus starts on a **4P4C** at the row
  controller but tiles use an **8P8C** — document the transition/adapter for
  the first tile, and confirm which conductors carry through.
- **Termination/biasing** responsibilities for both buses at the controller.
- **Power connector** type and current rating for the 12 V daisy chain.
- **3.3 V loads** (if any beyond the Xiao's onboard rail) — confirm nothing
  else draws meaningfully from the Xiao's onboard 3.3 V LDO.
