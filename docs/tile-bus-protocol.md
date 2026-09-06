# Tile Bus — Row Controller ↔ Tile Protocol

This document specifies the command protocol on **Tile Bus**: the RS-485 link
between a row controller and its 8 tiles. For physical-layer wiring, topology,
and the SENSE auto-mapping overview see [communication.md](communication.md).

---

## 1. Physical Layer

| Parameter      | Value                          |
| -------------- | ------------------------------ |
| Medium         | RS-485, half-duplex, multidrop |
| Baud rate      | 1 Mbps                         |
| UART framing   | 8N1 (8 data bits, no parity, 1 stop bit) |
| Master         | Row controller (Xiao RP2350)   |
| Slaves         | Up to 8 tiles (ATtiny3224 + THVD1420DR) |
| Default state  | Tiles in RX; only transmit when commanded |

---

## 2. Frame Format

All traffic on Tile Bus uses the same frame layout, whether sent by the row
controller or a tile responding to a command.

```
 0        1        2        3        4        5 … 5+N-1   5+N      6+N
+--------+--------+--------+--------+--------+---------+--------+--------+
| SYNC1  | SYNC2  | ADDR   |  CMD   |  LEN   | PAYLOAD |  CRC_H | CRC_L  |
| 0xAA   | 0x55   | 1 byte | 1 byte | 1 byte | N bytes | CRC-16          |
+--------+--------+--------+--------+--------+---------+--------+--------+
```

| Field     | Size | Description |
| --------- | ---- | ----------- |
| `SYNC1`   | 1 B  | Always `0xAA`. Marks frame start. |
| `SYNC2`   | 1 B  | Always `0x55`. Two-byte preamble reduces false-sync probability. |
| `ADDR`    | 1 B  | Target tile address (`0x01`–`0x08`, see §3), or `0xFF` for broadcast. On response frames this is the responding tile's address. |
| `CMD`     | 1 B  | Command or response code (see §5 and §6). |
| `LEN`     | 1 B  | Number of payload bytes that follow (`0`–`180`). |
| `PAYLOAD` | N B  | Command-specific data; absent when `LEN = 0`. |
| `CRC`     | 2 B  | CRC-16/CCITT (polynomial `0x1021`, init `0xFFFF`). Computed over `ADDR`, `CMD`, `LEN`, and all `PAYLOAD` bytes. Transmitted big-endian (`CRC_H` first). |

**Minimum frame size:** 7 bytes (no payload).  
**Maximum frame size:** 187 bytes (`SET_LEDS`, 180-byte payload).

Both track `LEDS_PER_TILE`, which
[protocol.h](../src/common/tile_bus_protocol/protocol.h) defines once and
derives `MAX_PAYLOAD` / `MAX_FRAME_SIZE` from. `LEN` is a single byte and the
firmware's frame helpers take `uint8_t` lengths, so the **whole frame** must
stay under 256 — at 3 bytes/LED that caps a side at 20 LEDs. A `static_assert`
in that header enforces it; going past it means a 2-byte `LEN` here or fewer
bytes per LED.

### Receiver framing

1. Scan the incoming byte stream for the pattern `0xAA 0x55`.
2. Read the next 3 bytes (`ADDR`, `CMD`, `LEN`).
3. Read `LEN` payload bytes.
4. Read 2 CRC bytes; validate. Discard frame on CRC failure and re-enter step 1.

---

## 3. Address Space

| Range         | Meaning |
| ------------- | ------- |
| `0x00`        | Unassigned — a tile that has not yet been given an address |
| `0x01`–`0x08` | Tile at slot 0–7, assigned during the SENSE walk |
| `0x09`–`0xFE` | Unused |
| `0xFF`        | Broadcast — all tiles accept and process, none respond (except `DETECT_SENSE` and `SET_ADDRESS`) |

**Addresses are assigned by position, at every boot, and are not stored.** A
tile powers up as `0x00` and is named by its row controller during the SENSE
auto-mapping walk (§9): the tile found at slot *N* becomes address *N+1*.

Two facts make this work, and they are worth stating because the obvious
reading of "address" suggests otherwise:

- **Addresses need only be unique on one Tile Bus.** The floor's 8 rows are 8
  electrically separate buses that never join, so 8 addresses suffice for 64
  tiles. Nothing needs to be unique floor-wide.
- **Exactly one tile is selected at a time during the walk.** That is what
  lets a broadcast `SET_ADDRESS` reach precisely one tile, which is the only
  way to name a tile that has no address to unicast to yet.

The consequences are the point of the scheme: tiles are physically
interchangeable, a replacement needs no programming before it works, there is
no manufacturing step and no per-tile firmware build, and the slot→address map
can never disagree with the physical chain because position *is* the address.
The cost is that a tile is addressable only after discovery reaches it, and
that a row controller reset re-runs the walk (which it already did).

An unaddressed tile matches broadcasts only. That is sufficient: `DETECT_SENSE`
and `SET_ADDRESS`, the two commands it must see before it has a name, are both
broadcast.

---

## 4. Acknowledgement Rules

| Command category | ACK required? |
| ---------------- | ------------- |
| Admin (unicast)  | **Yes** — row controller waits for `ACK` response |
| Admin (broadcast, no response expected) | No |
| `DETECT_SENSE` (broadcast, one tile responds) | **Yes** — one tile sends `DETECT_RESP` |
| `SET_ADDRESS` (broadcast, one tile responds) | **Yes** — one tile sends `0x86`, from its new address |
| Display (all)    | **No** — fire and forget |

---

## 5. Commands

Command codes `0x00`–`0x7F` are row controller → tile.  
Response codes `0x80`–`0xFF` are tile → row controller (see §6).

### 5.1 Admin Commands

#### `0x01 ACTIVATE_SENSE` — unicast

Requests the addressed tile to pull its **outgoing** SENSE line low and hold
it until `CLEAR_SENSE` is received.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target tile address |
| `CMD`   | `0x01` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Yes** (`0x81 ACK`, status `0x00`) |

---

#### `0x02 DETECT_SENSE` — broadcast

Asks whichever tile currently has its **incoming** SENSE line low to identify
itself. During correct SENSE auto-mapping, exactly one tile will see its SENSE
low at any given moment.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | `0xFF` |
| `CMD`   | `0x02` |
| `LEN`   | `0` |
| Payload | none |
| Response | **Yes** — one tile sends `0x82 DETECT_RESP` |

If no tile responds within the timeout the row controller retries (see §8).
If multiple tiles respond simultaneously (wiring fault), the CRC check on both
responses will likely fail; the row controller will retry, then log an error.

---

#### `0x03 CLEAR_SENSE` — broadcast

Tells all tiles to release (stop driving low) their outgoing SENSE lines. Used
at the start of SENSE auto-mapping to ensure a clean baseline, and after
auto-mapping is complete.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | `0xFF` |
| `CMD`   | `0x03` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | No |

---

#### `0x04 TEST` — unicast

Requests the addressed tile to run its built-in self-test routine (LED check,
RS-485 loopback, etc.). The tile responds with a status byte indicating
pass/fail.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target tile address |
| `CMD`   | `0x04` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Yes** (`0x84 ACK`, status = test result) |

Test status byte:

| Value  | Meaning |
| ------ | ------- |
| `0x00` | All tests passed |
| `0x01` | LED driver failure |
| `0x02` | SENSE line fault |
| `0xFF` | Unspecified failure |

---

#### `0x05 VERSION` — unicast

Reports the addressed tile's own firmware identity. The row controller
queries this once per tile, at the end of the SENSE auto-mapping sequence
(and again on `RE_DISCOVER`), and caches the result to answer the Pi's Row
Bus `VERSION` (`0x06`, see
[row-bus-protocol.md](row-bus-protocol.md)) without a live Tile Bus round
trip. Broadcast is invalid — a `VERSION` sent to `0xFF` would have every
tile answer at once.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target tile address |
| `CMD`   | `0x05` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Yes** (`0x85 VERSION_RESP`) |

Response payload (7 bytes, big-endian; shared encoding, defined once in
`src/common/tile_bus_protocol/firmware_version.h`):

| Bytes | Field | Description |
| ----- | ----- | ----------- |
| 0–1   | `version` | `TILE_FW_VERSION`, hand-bumped per build |
| 2–5   | `git_sha` | First 4 bytes of the build's commit SHA |
| 6     | `flags`   | Bit 0 = built from a dirty tree; bits 1–7 reserved (must be 0) |

---

#### `0x06 SET_ADDRESS` — broadcast

Assigns the addressed-by-position address to the one tile whose incoming
SENSE line is currently asserted. Sent by the row controller during the SENSE
walk (§9), immediately after that slot's `DETECT_RESP`.

Broadcast is not a convenience here but a necessity: the target may hold
`0x00` and so cannot be reached by unicast. The SENSE walk supplies the
selectivity instead — exactly one tile has its incoming line asserted at any
point in the walk, and **a tile whose SENSE line is not asserted must ignore
this command entirely.** Without that rule every tile on the bus would take
the same address at once.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | `0xFF` |
| `CMD`   | `0x06` |
| `LEN`   | `1` |
| Payload | `new_addr` — the address the tile is to adopt |
| ACK     | **Yes** (`0x86`), **sent from the new address** |

`new_addr` of `0x00` or `0xFF` is invalid and must be ignored; neither is a
usable unicast address (§3).

The acknowledgement is sent from the newly adopted address, not the old one.
That is deliberate — it confirms the assignment actually took, where an ACK
from the previous address would only confirm the command was received. The row
controller matches on it before recording the slot.

Assignment is a plain overwrite with no notion of "already addressed": a row
controller that resets mid-show re-walks a chain of tiles that still hold
their previous names, and must be able to rename them.

---

### 5.2 Display Commands

Display commands are fire-and-forget (no ACK). The row controller does **not**
retry them.

All display commands cause the tile to **buffer** the incoming data. The LEDs
are not updated until the tile receives `LATCH (0x13)`. This gives every tile
on a row a shared trigger point so they all illuminate simultaneously, and
gives the tile a dedicated ~33 ms window after LATCH to drive the WS2815 data
line without risk of missing an incoming RS-485 frame during the push.

#### `0x10 SET_COLOR` — unicast

Sets all 60 LEDs on the addressed tile to a single RGB color.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target tile address |
| `CMD`   | `0x10` |
| `LEN`   | `3` |
| Payload | `R G B` (1 byte each, 0–255) |

---

#### `0x11 SET_PATTERN` — unicast

Applies a preset animation pattern to the addressed tile.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target tile address |
| `CMD`   | `0x11` |
| `LEN`   | `5` |
| Payload | see below |

Payload layout (5 bytes):

```
 Byte 0          Bytes 1–4
+---------------+-------------------------------+
| pattern_id    | params[0..3]                  |
| bits 4-0 used | pattern-defined               |
| bits 7-5 = 0  |                               |
+---------------+-------------------------------+
```

| Field        | Bits  | Description |
| ------------ | ----- | ----------- |
| `pattern_id` | 4:0   | Selects one of 32 preset patterns (0–31); values 0–31 |
| reserved     | 7:5   | Must be `0` |
| `params[0]`  | byte 1 | Parameter 0; meaning defined per pattern |
| `params[1]`  | byte 2 | Parameter 1; meaning defined per pattern |
| `params[2]`  | byte 3 | Parameter 2; meaning defined per pattern |
| `params[3]`  | byte 4 | Parameter 3; meaning defined per pattern |

Pattern ids and their parameter semantics are defined in
[tile-patterns.md](tile-patterns.md). Two points that affect this layer:

- A pattern is **staged**, not started, by `SET_PATTERN`. It begins on the next
  `LATCH`, so a row's tiles can be armed one at a time and started in step.
  A `LATCH` with nothing staged does not restart a running pattern.
- A running pattern **modulates the tile's existing buffer contents** rather
  than carrying its own colour, so `SET_COLOR`/`SET_LEDS` set the base first.
  Either of those commands also cancels a running pattern.

---

#### `0x12 SET_LEDS` — unicast

Loads explicit RGB values for all 60 LEDs on the addressed tile. LEDs are
ordered from LED 0 (start of WS2815 chain) to LED 59; see
[hardware-tile.md](hardware-tile.md)'s "LED layout and chain order" for where
each index sits physically.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target tile address |
| `CMD`   | `0x12` |
| `LEN`   | `180` |
| Payload | `R0 G0 B0  R1 G1 B1  … R59 G59 B59` (180 bytes) |

---

#### `0x13 LATCH` — broadcast

Signals all tiles to simultaneously push their buffered display data to their
WS2815 LED chain. This is the mechanism for tear-free frame updates across all
tiles in a row.

After LATCH the tile has approximately 33 ms (one frame period) to complete its
WS2815 push (~1.2 ms, interrupts disabled) before the next frame's display
commands will arrive. No RS-485 traffic is expected during this window, so
the interrupt-disabled push does not risk missing a frame.

If a tile has no buffered data (e.g. at boot, or if no display command has
been received since the last LATCH), LATCH is a no-op.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | `0xFF` |
| `CMD`   | `0x13` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | No |

---

## 6. Response Frames

Tiles use the same [frame format](#2-frame-format) for responses. Response
`CMD` codes have bit 7 set.

### `0x81 ACK` — tile → row controller

Generic acknowledgement for `ACTIVATE_SENSE (0x01)` and `TEST (0x04)`.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | responding tile's address |
| `CMD`   | `0x80 | original_cmd` (e.g. `0x81` for `ACTIVATE_SENSE`) |
| `LEN`   | `1` |
| Payload | status byte (0x00 = success; see per-command status table) |

### `0x82 DETECT_RESP` — tile → row controller

Sent by the tile that currently sees its incoming SENSE line low, in response
to `DETECT_SENSE`.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | responding tile's *current* address — see below |
| `CMD`   | `0x82` |
| `LEN`   | `0` |
| Payload | none |

`DETECT_RESP` means "a tile is present at this slot" and nothing more. The
address it carries is whatever the tile happens to hold — `0x00` on a freshly
booted tile, a stale assignment on one that survived a row controller reset —
and **the row controller must not record it.** The slot's address is decided
by the `SET_ADDRESS` that follows, and confirmed by the `0x86` sent from it.

### `0x86 ACK (SET_ADDRESS)` — tile → row controller

Sent in response to `SET_ADDRESS (0x06)`, **from the newly adopted address**,
with a 1-byte status payload (`0x00` = success). See §5.1's `SET_ADDRESS`
entry for why the source address is the meaningful part.

### `0x85 VERSION_RESP` — tile → row controller

Sent in response to `VERSION (0x05)`. Payload layout is in §5.1's `VERSION`
entry.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | responding tile's address |
| `CMD`   | `0x85` |
| `LEN`   | `7` |
| Payload | this tile's `FirmwareVersion`, see §5.1 |

---

## 7. Error Handling and Retry

### Retry policy (admin/unicast commands only)

1. The row controller transmits the command and starts a **5 ms** response
   timer.
2. If no valid response arrives before the timer expires, the controller
   retransmits the same frame. Up to **3 attempts** total.
3. If all 3 attempts fail, the controller:
   - Marks the tile as **non-responsive** in its local state.
   - Logs the error (tile address, command code, timestamp) to an in-memory
     error ring buffer.
   - Continues normal operation; the Raspberry Pi can query the error log via
     the Row Bus (see [row-bus-protocol.md](row-bus-protocol.md)).

### DETECT_SENSE collision handling

If two tiles both respond to `DETECT_SENSE` simultaneously (indicating a wiring
fault), the overlapping RS-485 signals will corrupt the response frame and the
CRC will fail. The row controller will retry up to 3 times before logging a
`SENSE_COLLISION` error and halting the auto-mapping procedure for that row.

---

## 8. Timing

### Frame transmission time (1 Mbps, 8N1)

Each UART byte is 10 bits (1 start + 8 data + 1 stop).

| Command      | Frame size | Transmission time |
| ------------ | ---------- | ----------------- |
| Admin (no payload) | 7 B | 70 µs |
| `SET_COLOR`  | 10 B       | 100 µs |
| `SET_PATTERN`| 12 B       | 120 µs |
| `SET_LEDS`   | 187 B      | 1.87 ms |
| ACK response | 8 B        | 80 µs |
| `DETECT_RESP`| 7 B        | 70 µs |

### Bus turnaround

After the row controller finishes transmitting a command frame, it must switch
its transceiver to RX mode before the tile begins its response. The row
controller must hold the bus idle for at least **100 µs** after the last stop
bit before the tile is expected to start transmitting. Tile firmware must not
begin its response frame until this guard time has elapsed.

### Full-frame display update budget (30 FPS, 33 ms period)

Sending `SET_LEDS` to all 8 tiles in a row (row controller → tiles):

| Item | Time |
| ---- | ---- |
| 8 × `SET_LEDS` frame (187 B each) | 15.0 ms |
| 8 × 100 µs inter-frame gap        | 0.8 ms |
| `LATCH` broadcast (7 B)           | < 0.1 ms |
| Total (RC → tiles)                | ~16 ms |

After LATCH all 8 tiles push to WS2815 simultaneously (~1.8 ms at 60 LEDs,
overlapping, not serial).

**This ~16 ms is now the largest single term in the floor's end-to-end frame
latency** — larger than the whole Row Bus phase, which two concurrent chains
bring down to ~18.6 ms for all 8 rows. Raising Tile Bus to 2 Mbps would halve
it and is the identified next lever on frame rate; see
[row-bus-protocol.md](row-bus-protocol.md) §8. The blocker is the ATtiny3224's
USART ceiling, which needs confirming against the datasheet — the THVD1420DR
transceiver is rated to 12 Mbps and is not the constraint.

At 60 LEDs the row controller's idle margin on Tile Bus within a 33 ms frame
is thin (~17 ms rather than the ~22 ms it had at 40), which matters for the
quiet-window assumption in [hardware-tile.md](hardware-tile.md)'s ATtiny
rationale: the tile's interrupt-disabled WS2815 push must still land inside
it.

---

## 9. SENSE Auto-Mapping Sequence

The full procedure is described in [communication.md](communication.md). The
commands used are:

1. Row controller asserts SENSE to tile 0 (hardware line, not a command).
2. RC → all: `DETECT_SENSE (0x02)` → tile 0 replies `DETECT_RESP (0x82)`.
   This says only that a tile is there; its reported address is ignored.
3. RC → all: `SET_ADDRESS (0x06)` carrying `0x01` → only tile 0 acts on it,
   because only tile 0 has its incoming SENSE asserted. It adopts `0x01` and
   replies `0x86` **from** `0x01`. RC records slot 0 → `0x01`.
4. RC → tile 0 (`0x01`): `ACTIVATE_SENSE (0x01)` → tile 0 pulls its outgoing
   SENSE low, selecting tile 1.
5. Repeat steps 2–4 for tiles 1–7, assigning `0x02`…`0x08`.
6. RC → all: `CLEAR_SENSE (0x03)` → all tiles release their SENSE lines.

After this sequence the row controller has a complete `slot 0..7 → tile
address` map, and it is trivially `slot + 1` — the map exists so the rest of
the firmware need not assume that, not because the assignment is interesting.

A tile that answers `DETECT_SENSE` but never acknowledges `SET_ADDRESS` is a
**fault**, not the end of the chain: it is demonstrably present. Discovery
fails rather than silently truncating the row. End of chain is signalled only
by silence at `DETECT_SENSE` (§7).

---

## 10. Open Questions

- **Baud rate confirmation:** 1 Mbps requires validation against the ATtiny3224
  UART tolerance and cable length/capacitance on the tile bus.
- **Response timeout value:** 5 ms is a placeholder. Tune after measuring
  actual tile firmware processing latency.
- **Display command errors:** currently display commands have no ACK and no
  retry. If reliable delivery for `SET_LEDS` is later needed, a lightweight
  per-frame CRC check or a heartbeat could be added.
- **`SET_PATTERN` param count:** resolved for the patterns defined so far —
  [tile-patterns.md](tile-patterns.md) takes the tile's existing buffer as the
  pattern's base image, so no parameter byte is spent on colour and 4 is
  enough. A pattern needing two *independent* colours would still not fit;
  revisit then.
- **LED ordering convention:** LED 0 = start of WS2815 chain; confirm physical
  position relative to the tile's corner/connector.
