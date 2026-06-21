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
| `ADDR`    | 1 B  | Target tile address (`0x01`–`0xFE`), or `0xFF` for broadcast. On response frames this is the responding tile's address. |
| `CMD`     | 1 B  | Command or response code (see §5 and §6). |
| `LEN`     | 1 B  | Number of payload bytes that follow (`0`–`120`). |
| `PAYLOAD` | N B  | Command-specific data; absent when `LEN = 0`. |
| `CRC`     | 2 B  | CRC-16/CCITT (polynomial `0x1021`, init `0xFFFF`). Computed over `ADDR`, `CMD`, `LEN`, and all `PAYLOAD` bytes. Transmitted big-endian (`CRC_H` first). |

**Minimum frame size:** 7 bytes (no payload).  
**Maximum frame size:** 127 bytes (`SET_LEDS`, 120-byte payload).

### Receiver framing

1. Scan the incoming byte stream for the pattern `0xAA 0x55`.
2. Read the next 3 bytes (`ADDR`, `CMD`, `LEN`).
3. Read `LEN` payload bytes.
4. Read 2 CRC bytes; validate. Discard frame on CRC failure and re-enter step 1.

---

## 3. Address Space

| Range         | Meaning |
| ------------- | ------- |
| `0x00`        | Reserved |
| `0x01`–`0xFE` | Valid tile unicast addresses |
| `0xFF`        | Broadcast — all tiles accept and process, none respond (except `DETECT_SENSE`) |

Each tile's address is **programmed into non-volatile storage at manufacturing
time** and is unique across all tiles in the system. Address assignment
procedure: TBD (see §10).

---

## 4. Acknowledgement Rules

| Command category | ACK required? |
| ---------------- | ------------- |
| Admin (unicast)  | **Yes** — row controller waits for `ACK` response |
| Admin (broadcast, no response expected) | No |
| `DETECT_SENSE` (broadcast, one tile responds) | **Yes** — one tile sends `DETECT_RESP` |
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

### 5.2 Display Commands

Display commands are fire-and-forget (no ACK). The row controller does **not**
retry them.

All display commands cause the tile to **buffer** the incoming data. The LEDs
are not updated until the tile receives `LATCH (0x13)`. This gives every tile
on a row a shared trigger point so they all illuminate simultaneously, and
gives the tile a dedicated ~33 ms window after LATCH to drive the WS2815 data
line without risk of missing an incoming RS-485 frame during the push.

#### `0x10 SET_COLOR` — unicast

Sets all 40 LEDs on the addressed tile to a single RGB color.

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

Pattern definitions and their parameter semantics are documented in
**tile-patterns.md** (TBD). Common parameters will include animation speed,
direction, and foreground/background colors.

---

#### `0x12 SET_LEDS` — unicast

Loads explicit RGB values for all 40 LEDs on the addressed tile. LEDs are
ordered from LED 0 (start of WS2815 chain) to LED 39.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target tile address |
| `CMD`   | `0x12` |
| `LEN`   | `120` |
| Payload | `R0 G0 B0  R1 G1 B1  … R39 G39 B39` (120 bytes) |

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
| `ADDR`  | responding tile's address ← **this is what the row controller captures** |
| `CMD`   | `0x82` |
| `LEN`   | `0` |
| Payload | none |

The tile address is carried in the `ADDR` field; no payload is needed.

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
| `SET_LEDS`   | 127 B      | 1.27 ms |
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
| 8 × `SET_LEDS` frame (127 B each) | 10.2 ms |
| 8 × 100 µs inter-frame gap        | 0.8 ms |
| `LATCH` broadcast (7 B)           | < 0.1 ms |
| Total (RC → tiles)                | ~11 ms |

After LATCH all 8 tiles push to WS2815 simultaneously (~1.2 ms, overlapping,
not serial). The row controller then has the remainder of the 33 ms frame
period idle on Tile Bus before the next `SEND_DATA` arrives from the Pi.

---

## 9. SENSE Auto-Mapping Sequence

The full procedure is described in [communication.md](communication.md). The
commands used are:

1. Row controller asserts SENSE to tile 0 (hardware line, not a command).
2. RC → all: `DETECT_SENSE (0x02)` → tile 0 replies `DETECT_RESP (0x82)`.
   RC records tile 0's address.
3. RC → tile 0: `ACTIVATE_SENSE (0x01)` → tile 0 pulls its outgoing SENSE low.
4. RC → all: `DETECT_SENSE (0x02)` → tile 1 replies `DETECT_RESP (0x82)`.
   RC records tile 1's address.
5. Repeat steps 3–4 for tiles 2–7.
6. RC → all: `CLEAR_SENSE (0x03)` → all tiles release their SENSE lines.

After this sequence the row controller has a complete `slot 0..7 → tile address`
map.

---

## 10. Open Questions

- **Tile address assignment:** how are unique addresses burned in at
  manufacture? Options: factory programming via UPDI, derived from ATtiny3224
  unique serial, or a one-time self-assignment command.
- **Baud rate confirmation:** 1 Mbps requires validation against the ATtiny3224
  UART tolerance and cable length/capacitance on the tile bus.
- **Response timeout value:** 5 ms is a placeholder. Tune after measuring
  actual tile firmware processing latency.
- **Pattern library:** pattern IDs and their `params[]` semantics are TBD;
  document in `tile-patterns.md` once patterns are designed.
- **Display command errors:** currently display commands have no ACK and no
  retry. If reliable delivery for `SET_LEDS` is later needed, a lightweight
  per-frame CRC check or a heartbeat could be added.
- **`SET_PATTERN` param count:** 4 parameter bytes covers most animation needs
  (speed, direction, 1 color), but may be insufficient for patterns needing two
  full RGB colors (6 bytes). Revisit when the pattern library is defined.
- **LED ordering convention:** LED 0 = start of WS2815 chain; confirm physical
  position relative to the tile's corner/connector.
