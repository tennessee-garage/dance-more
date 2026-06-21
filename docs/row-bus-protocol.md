# Row Bus — Raspberry Pi ↔ Row Controller Protocol

This document specifies the command protocol on **Row Bus**: the RS-485 link
between the Raspberry Pi (host) and the 8 row controllers. For physical-layer
wiring, topology, and the overall addressing model see
[communication.md](communication.md).

---

## 1. Physical Layer

| Parameter      | Value                                         |
| -------------- | --------------------------------------------- |
| Medium         | RS-485, half-duplex, multidrop                |
| Baud rate      | **4 Mbps minimum; 8 Mbps recommended**        |
| UART framing   | 8N1 (8 data bits, no parity, 1 stop bit)      |
| Master         | Raspberry Pi                                  |
| Slaves         | 8 row controllers (Xiao RP2350)               |
| Default state  | Row controllers in RX; only transmit on reply |

### Baud rate rationale

A full-row `SEND_DATA` frame with all tiles using `SET_LEDS` carries 968 bytes
of pixel data plus frame overhead (~976 bytes total). Updating all 8 rows
before a latch takes 8 such frames.

| Baud rate | Time per row | 8 rows total | Slack (33 ms frame) |
| --------- | ------------ | ------------ | ------------------- |
| 1 Mbps    | 7.8 ms       | 62 ms        | —  (over budget)    |
| 2 Mbps    | 3.9 ms       | 31 ms        | 2 ms (too tight)    |
| **4 Mbps**| **2.0 ms**   | **15.6 ms**  | **~17 ms**          |
| 8 Mbps    | 1.0 ms       | 7.8 ms       | ~25 ms              |

4 Mbps is the minimum for reliable 30 FPS operation. 8 Mbps is preferred for
headroom. Neither the Pi's UART nor the RP2350's PIO is a practical constraint
at these speeds; the transceiver choice and cable quality are the limiting
factors.

---

## 2. Frame Format

Row Bus uses a slightly extended variant of the Tile Bus frame format. The payload
length field is **2 bytes** to accommodate the large `SEND_DATA` payload
(up to 968 bytes).

```
 0        1        2        3        4        5   6        7 … 7+N-1   7+N      8+N
+--------+--------+--------+--------+--------+--------+--------+-----+--------+--------+
| SYNC1  | SYNC2  | ADDR   |  CMD   | LEN_H  | LEN_L  |   PAYLOAD   |  CRC_H | CRC_L  |
| 0xAA   | 0x55   | 1 byte | 1 byte |     2 bytes     |   N bytes   | CRC-16          |
+--------+--------+--------+--------+--------+--------+--------+-----+--------+--------+
```

| Field     | Size | Description |
| --------- | ---- | ----------- |
| `SYNC1`   | 1 B  | Always `0xAA` |
| `SYNC2`   | 1 B  | Always `0x55` |
| `ADDR`    | 1 B  | Target row address (`0x00`–`0x07`), or `0xFF` for broadcast. On response frames this is the responding row controller's address. |
| `CMD`     | 1 B  | Command or response code (see §5 and §6) |
| `LEN`     | 2 B  | Payload byte count, big-endian (`LEN_H` first). Range 0–968. |
| `PAYLOAD` | N B  | Command-specific data; absent when `LEN = 0` |
| `CRC`     | 2 B  | CRC-16/CCITT (polynomial `0x1021`, init `0xFFFF`), computed over `ADDR`, `CMD`, `LEN_H`, `LEN_L`, and all payload bytes. Transmitted big-endian. |

**Minimum frame size:** 8 bytes (no payload).
**Maximum frame size:** 976 bytes (`SEND_DATA`, 968-byte payload).

### Receiver framing

1. Scan for the pattern `0xAA 0x55`.
2. Read `ADDR`, `CMD`, `LEN_H`, `LEN_L`.
3. Read `(LEN_H << 8) | LEN_L` payload bytes.
4. Read 2 CRC bytes; validate. Discard frame on CRC failure, re-enter step 1.

---

## 3. Address Space

| Range       | Meaning |
| ----------- | ------- |
| `0x00`–`0x07` | Row controllers 0–7 (statically assigned by install position) |
| `0x08`–`0xFE` | Reserved |
| `0xFF`      | Broadcast — all row controllers accept; none respond |

Row addresses are static (no discovery mechanism on Row Bus).

---

## 4. Acknowledgement Rules

| Command         | ACK required? |
| --------------- | ------------- |
| Admin (unicast) | **Yes** — Pi waits for response |
| `RE_DISCOVER`   | **Immediate ACK** only ("started"); Pi polls `STATUS` for completion |
| `SEND_DATA`     | No |
| `LATCH`         | No (broadcast) |
| `BLACKOUT`      | No (broadcast) |

---

## 5. Commands

Command codes `0x00`–`0x7F` are Pi → row controller.
Response codes `0x80`–`0xFF` are row controller → Pi.

### 5.1 Admin Commands

---

#### `0x01 TEST` — unicast

Requests the row controller to run a self-test and report results. Tests cover
the Tile Bus transceiver, internal SRAM, and basic UART loopback. Does **not**
automatically test individual tiles (use `STATUS` to see tile health).

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target row address |
| `CMD`   | `0x01` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Yes** (`0x81 TEST_RESP`); allow up to 200 ms |

Response payload (2 bytes):

| Byte | Field        | Description |
| ---- | ------------ | ----------- |
| 0    | `result`     | `0x00` = all pass; non-zero = failure |
| 1    | `fault_flags`| Bit mask: bit 0 = Row Bus UART, bit 1 = Tile Bus transceiver, bit 2 = SRAM; bits 3–7 reserved |

---

#### `0x02 STATUS` — unicast

Returns the current operating state of the row controller and the health of each
tile slot in its row.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target row address |
| `CMD`   | `0x02` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Yes** (`0x82 STATUS_RESP`); allow up to 20 ms |

Response payload (10 bytes):

| Byte  | Field           | Description |
| ----- | --------------- | ----------- |
| 0     | `state`         | `0x00` = idle, `0x01` = discovering, `0x02` = running, `0x03` = error |
| 1     | `tiles_found`   | Number of tiles successfully discovered (0–8) |
| 2–9   | `tile_status[0..7]` | Per-slot status: `0x00` = not discovered, `0x01` = OK, `0x02` = non-responsive, `0x03` = test failed |

---

#### `0x03 POWER` — unicast

Requests the row controller to report electrical measurements for its row.
Measurements are sourced from the **INA220BIDGSR** power monitor IC on the row
controller PCB, which reads a **5 mΩ shunt resistor** on the 12 V supply via
I²C. The RP2350 reads the INA220 and packages the results into this response.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target row address |
| `CMD`   | `0x03` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Yes** (`0x83 POWER_RESP`); allow up to 20 ms |

Response payload (6 bytes), all values big-endian:

| Bytes | Field        | Units | Description |
| ----- | ------------ | ----- | ----------- |
| 0–1   | `voltage_mV` | mV    | 12 V rail voltage as measured by INA220 (uint16) |
| 2–3   | `current_mA` | mA    | Row current draw via 5 mΩ shunt (uint16) |
| 4–5   | `power_mW`   | mW    | Computed power from INA220 power register (uint16) |

---

#### `0x04 RE_DISCOVER` — unicast

Tells the row controller to discard its current slot→address map and re-run the
SENSE auto-mapping procedure for its row. Use after a tile hot-swap or when a
tile becomes permanently non-responsive.

The row controller responds immediately with an ACK (`0x84`) indicating the
procedure has started, then runs it asynchronously. Poll `STATUS` to detect
when `state` transitions from `discovering` to `running`.

During re-discovery the row controller will not process `SEND_DATA` or `LATCH`
commands. It will still respond to `STATUS`.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target row address |
| `CMD`   | `0x04` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Immediate** (`0x84 RE_DISCOVER_RESP`, status `0x00` = started) |

---

#### `0x05 ERROR_LOG` — unicast

Retrieves the row controller's buffered log of tile communication failures. The
log is a fixed-size ring buffer; oldest entries are overwritten when full.
Reading the log does **not** clear it.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target row address |
| `CMD`   | `0x05` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Yes** (`0x85 ERROR_LOG_RESP`); allow up to 20 ms |

Response payload:

| Bytes     | Field         | Description |
| --------- | ------------- | ----------- |
| 0         | `entry_count` | Number of log entries that follow (0–32) |
| 1 + 5×i   | `slot`        | Tile slot (0–7) that failed; see note for `LATCH_OVERRUN` |
| 2 + 5×i   | `tile_bus_cmd`   | Tile Bus command code involved; see note for `LATCH_OVERRUN` |
| 3 + 5×i   | `error_type`  | `0x01` = no ACK after 3 retries, `0x02` = CRC failure, `0x03` = sense collision, `0x04` = LATCH overrun |
| 4–5 + 5×i | `timestamp`   | Seconds since row controller boot (uint16, big-endian) |

For `LATCH_OVERRUN` entries (`error_type = 0x04`) the fields are repurposed:
- `slot` — number of tile slots that had been forwarded when `LATCH` arrived
  (0–7). Indicates how far behind the row was; a value of 7 means only the last
  tile was still in flight; a value of 0 means the row hadn't started at all.
- `tile_bus_cmd` — the tile command code that was in flight at the time (typically
  `0x12` SET_LEDS).

Maximum response payload: `1 + 32 × 5 = 161 bytes`.

---

### 5.2 Display Commands

Display commands are fire-and-forget. The Pi does **not** retry them.

---

#### `0x10 SEND_DATA` — unicast

Sends a full row's worth of display data to a single row controller. The row
controller immediately forwards each tile's data as individual Tile Bus commands
(SET_COLOR / SET_PATTERN / SET_LEDS); there is no frame-level buffering on
the row controller. Tiles buffer the received data and do not update their LEDs
until the row controller relays a Tile Bus `LATCH (0x13)` triggered by the Pi's
`LATCH (0x11)` broadcast.

Each of the 8 tile slots is encoded sequentially (slot 0 first, slot 7 last)
using the same command codes as the Tile Bus tile protocol. Slots can use
different commands — they may be freely mixed within a single `SEND_DATA` frame.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target row address |
| `CMD`   | `0x10` |
| `LEN`   | sum of all 8 tile entry sizes (32–968) |
| Payload | 8 tile entries, concatenated (see below) |

**Per-tile entry format:**

| Bytes  | Field       | Description |
| ------ | ----------- | ----------- |
| 0      | `tile_cmd`  | `0x10` = SET_COLOR, `0x11` = SET_PATTERN, `0x12` = SET_LEDS (same codes as [tile-bus-protocol.md](tile-bus-protocol.md)) |
| 1…     | `tile_data` | Payload for that command: 3 bytes for SET_COLOR, 5 bytes for SET_PATTERN, 120 bytes for SET_LEDS |

Entry sizes: SET_COLOR = 4 bytes, SET_PATTERN = 6 bytes, SET_LEDS = 121 bytes.

**Payload size examples:**

| Mix                      | Total payload |
| ------------------------ | ------------- |
| All 8 × SET_COLOR        | 32 bytes      |
| All 8 × SET_PATTERN      | 48 bytes      |
| All 8 × SET_LEDS         | 968 bytes     |
| 4 × SET_COLOR + 4 × SET_LEDS | 500 bytes |

---

#### `0x11 LATCH` — broadcast

Sent by the Pi at the end of each 33 ms frame period. Each row controller
immediately relays a Tile Bus `LATCH (0x13)` broadcast to its 8 tiles. Because
all row controllers receive the Pi's `LATCH` simultaneously, all tiles across
all rows illuminate at the same instant.

**Overrun handling:** if a row controller is still forwarding tile data on Tile Bus
when `LATCH` arrives (because Row Bus transfers consumed most of the frame period),
the row controller buffers the `LATCH`, finishes its Tile Bus forwarding, then
immediately broadcasts Tile Bus `LATCH`. This row will illuminate slightly later
than the others. The row controller logs a `LATCH_OVERRUN` error (type `0x04`)
in its error log (see `ERROR_LOG`). The Pi does not adjust its timing based on
Tile Bus completion — `LATCH` fires at a fixed point each frame. Overruns are a
diagnostic condition, not a fatal error.

If a row controller has no pending tile data (e.g. at boot, or after
`BLACKOUT`), its Tile Bus `LATCH` relay is a no-op.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | `0xFF` |
| `CMD`   | `0x11` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | No |

---

#### `0x12 BLACKOUT` — broadcast

Commands all row controllers to immediately black out their tiles. Each row
controller sends `SET_COLOR(0, 0, 0)` to each of its 8 tiles on Tile Bus and then
broadcasts Tile Bus `LATCH (0x13)`, causing tiles to go dark without waiting for
the Pi's next frame `LATCH`. Any data previously buffered in the tiles is
overwritten with black.

A subsequent Pi `LATCH` with no new `SEND_DATA` results in a no-op Tile Bus LATCH
relay (tiles have no pending data).

All 64 tiles go dark within ~2 ms of the row controllers completing their Tile Bus
`SET_COLOR` sweeps (8 × 10-byte SET_COLOR frames ≈ 0.8 ms + Tile Bus LATCH).

| Field   | Value |
| ------- | ----- |
| `ADDR`  | `0xFF` |
| `CMD`   | `0x12` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | No |

---

## 6. Response Codes

Response `CMD` codes have bit 7 set (`0x80 | original_cmd`).

| Response code | Name               | Triggered by |
| ------------- | ------------------ | ------------ |
| `0x81`        | `TEST_RESP`        | `TEST` |
| `0x82`        | `STATUS_RESP`      | `STATUS` |
| `0x83`        | `POWER_RESP`       | `POWER` |
| `0x84`        | `RE_DISCOVER_RESP` | `RE_DISCOVER` |
| `0x85`        | `ERROR_LOG_RESP`   | `ERROR_LOG` |

All response frames carry the responding row controller's address in the `ADDR`
field. See §5 for per-command payload layouts.

---

## 7. Error Handling and Retry

### Retry policy (admin/unicast commands only)

1. Pi transmits the command and starts a **per-command timeout** (see table).
2. On timeout, Pi retransmits. Up to **3 total attempts**.
3. On 3 failures: Pi logs the error (row address, command, timestamp) and marks
   the row controller as non-responsive in its floor state.

| Command       | Response timeout |
| ------------- | --------------- |
| `TEST`        | 200 ms          |
| `STATUS`      | 20 ms           |
| `POWER`       | 20 ms           |
| `RE_DISCOVER` | 20 ms (immediate ACK only) |
| `ERROR_LOG`   | 20 ms           |

`RE_DISCOVER` gets a short timeout because the ACK should arrive before the
mapping procedure begins. Use `STATUS` polls (with the 20 ms timeout and 3
retries each) to monitor discovery progress.

---

## 8. Timing and Frame Update Sequence

### Normal frame update (30 FPS, 33 ms period)

Frame updates use a **pipelined** model: each row controller begins forwarding
to its tiles on Tile Bus immediately upon receiving `SEND_DATA`, overlapping Tile Bus
transfers across rows. The Pi fires `LATCH` at t=33 ms regardless of Tile Bus
state.

**At 4 Mbps (Row Bus) / 1 Mbps (Tile Bus), worst case (all SET_LEDS):**

```
ms:   0    2    4    6    8   10   12   14   16              27   33
Row Bus: [RC0][RC1][RC2][RC3][RC4][RC5][RC6][RC7] ── idle ────── [LATCH]
RC0:       [────── Tile Bus 11ms ──────]
RC1:            [────── Tile Bus 11ms ──────]
RC2:                 [────── Tile Bus 11ms ──────]
 ⋮
RC7:                              [────── Tile Bus 11ms ──────]
                                                       ↑ done t≈27ms
```

RC7 finishes its Tile Bus update at ~t=27 ms, leaving **6 ms of slack** before
`LATCH` at t=33 ms. No overrun at 4 Mbps.

At 8 Mbps on Row Bus, the `SEND_DATA` phase completes at ~t=8 ms and RC7 finishes
Tile Bus at ~t=19 ms, providing ~14 ms of slack.

The overrun condition arises if RC7's Tile Bus update cannot complete before t=33 ms.
This cannot happen at the ≥ 4 Mbps minimum with all-`SET_LEDS` payloads, but
could occur during sustained admin traffic or if Row Bus runs below specification.

### Blackout sequence

```
Pi → all: BLACKOUT (0xFF)
```

Each row controller immediately issues `SET_COLOR(0,0,0)` to each of its 8
tiles on Tile Bus then broadcasts Tile Bus `LATCH`. All 8 rows do this in parallel.
All tiles go dark within ~2 ms of the `BLACKOUT` frame completing.

---

## 9. Open Questions

- **Pi hat PCB:** a dedicated PCB that mounts on the Raspberry Pi and translates
  its hardware UART to RS-485 for the Row Bus is planned but not yet designed.
  The planned transceiver is the **THVD1420DR** (same device used on the row
  controller and tiles); its 12 Mbps maximum data rate easily covers the ≥ 4 Mbps
  requirement.
- **Bus turnaround timing:** row controllers need a guard interval (≥ 100 µs,
  same as Tile Bus) after the Pi's last stop bit before they begin a response
  frame. Confirm this is sufficient at 4/8 Mbps cable lengths.
- **Error log ring buffer size:** 32 entries chosen arbitrarily. Tune to fit
  within RP2350 SRAM budget once firmware is written.
- **`RE_DISCOVER` during live show:** decide whether re-discovery is allowed
  during a running show or only during setup/maintenance. Rows go dark during
  re-mapping (~50–200 ms).
