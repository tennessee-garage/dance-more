# Row Bus — Raspberry Pi ↔ Row Controller Protocol

This document specifies the command protocol on **Row Bus**: the RS-485 link
between the Raspberry Pi (host) and the 8 row controllers. §1 below is
current on the physical layer and the two-chain topology;
[communication.md](communication.md) still describes a single bus on a 4P4C
connector and should not be used for wiring.

---

## 1. Physical Layer

| Parameter      | Value                                         |
| -------------- | --------------------------------------------- |
| Medium         | RS-485, half-duplex, multidrop                |
| Baud rate      | **3,125,000 (hardware maximum — see below)**  |
| UART framing   | 8N1 (8 data bits, no parity, 1 stop bit)      |
| Master         | Raspberry Pi                                  |
| Slaves         | 8 row controllers (Xiao RP2350)               |
| Chains         | **2, driven concurrently** — see below        |
| Default state  | Row controllers in RX; only transmit on reply |

### Two chains, driven concurrently

**The 8 rows are split across two independent RS-485 chains, and the Pi drives
both at the same time. This is a settled decision, not an available
optimization — the frame budget below assumes it and does not close without
it.**

The pi-hat carries two RS-485 transceivers landing in a single 8P8C jack, so
the parallelism is committed in copper. One Cat5 run passes every row
controller in physical order, each tapping the opposite pair from its
neighbour, giving an alternating split: **rows 0,2,4,6 on chain 0; rows 1,3,5,7
on chain 1**. Each chain therefore carries 4 rows, and a full-floor update is 4
frame times, not 8.

Row addresses stay global `0x00`–`0x07` — only the wiring is partitioned, and
row controller firmware is unaware there is more than one chain. Wiring
details, GPIO assignments and the host-side API are in
[src/pi/README.md](../src/pi/README.md).

### Baud rate rationale

**The Raspberry Pi's UART sets a hard ceiling of 3,125,000 baud.** The Pi 5's
RP1 UART is clocked at 50 MHz (`/sys/class/tty/ttyAMA0/uartclk`) and a PL011
derives its rate as `uartclk / (16 × divisor)` with `divisor >= 1`, so
`50e6 / 16 = 3,125,000` is the maximum. That rate divides exactly on both
ends — Pi `50e6/16`, RP2350 `150e6/48` — so neither side accumulates error.

Requesting a higher rate does **not** fail loudly. The Pi silently clamps to
its maximum while the row controller runs at whatever it was asked for; every
byte then arrives mis-timed and is rejected as a framing error by the
RP2350's UART IRQ handler, which discards bad characters without counting
them. The result is a completely mute bus, indistinguishable from a broken
wire. An earlier revision of this document specified 4 Mbps, which is
unreachable, and cost a full bring-up session to diagnose.

Because 3.125 Mbps is a ceiling rather than a choice, the second chain is the
only remaining lever on Row Bus wall-clock time.

### Worst-case frame budget

The worst case is every one of the 64 tiles sending `SET_LEDS` — 60 LEDs at
3 bytes each. One tile entry is 181 bytes, a full-row `SEND_DATA` payload is
`8 × 181 = 1,448` bytes, and the framed total is **1,456 bytes**. At 8N1 each
byte occupies 10 bit-times, so one row frame is 14,560 bits = **4.66 ms** at
3.125 Mbps.

End-to-end latency is the Row Bus phase plus one row's Tile Bus forwarding.
The two do not overlap within a row: `RowCommandHandler::handle_send_data()`
copies the whole frame before forwarding its first tile, so the last row
served cannot start on Tile Bus until its frame has fully arrived. Rows *do*
overlap each other — each has its own Tile Bus — so only the last row's tail
counts.

| | Row Bus phase | Tile Bus tail | End-to-end | Ceiling |
| --- | --- | --- | --- | --- |
| One chain, 8 rows serialized | 37.3 ms | 15.0 ms | 52.2 ms | 19.1 fps |
| **Two chains, 4 rows each** | **18.6 ms** | **15.0 ms** | **33.6 ms** | **29.8 fps** |
| Two chains + Tile Bus at 2 Mbps | 18.6 ms | 7.5 ms | 26.1 ms | 38.3 fps |

Two chains is what makes 30 FPS approximately reachable at all; without it the
floor runs at 19 fps worst case, with the last row a full frame behind the
first. Even with it the all-`SET_LEDS` case lands ~0.3 ms past the 33.3 ms
period — close enough that the overrun path in §8 is a normal occurrence at
full load rather than an exceptional one, and the last row illuminates
slightly late. Typical frames mixing `SET_COLOR`/`SET_PATTERN` are far smaller
and leave substantial slack.

**The identified next lever is the Tile Bus baud rate**, which at 1 Mbps now
contributes more to end-to-end latency than the whole Row Bus phase. The
THVD1420DR is rated to 12 Mbps; the binding constraint is the ATtiny3224's
USART, whose ceiling must be confirmed against the datasheet before this is
relied on. See [tile-bus-protocol.md](tile-bus-protocol.md) §1.

Going faster on Row Bus itself requires different host hardware — the Pi's
UART, not the transceiver or cable, is the constraint there.

---

## 2. Frame Format

Row Bus uses a slightly extended variant of the Tile Bus frame format. The payload
length field is **2 bytes** to accommodate the large `SEND_DATA` payload
(up to 1,448 bytes).

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
| `LEN`     | 2 B  | Payload byte count, big-endian (`LEN_H` first). Range 0–1,448. |
| `PAYLOAD` | N B  | Command-specific data; absent when `LEN = 0` |
| `CRC`     | 2 B  | CRC-16/CCITT (polynomial `0x1021`, init `0xFFFF`), computed over `ADDR`, `CMD`, `LEN_H`, `LEN_L`, and all payload bytes. Transmitted big-endian. |

**Minimum frame size:** 8 bytes (no payload).
**Maximum frame size:** 1,456 bytes (`SEND_DATA`, 1,448-byte payload).

Both figures track `LEDS_PER_TILE`; firmware derives them in
[row_bus_protocol.h](../src/row/lib/row_core/row_bus_protocol.h) rather than
restating them, and the host in
[protocol/constants.py](../src/pi/src/df2_pi/protocol/constants.py).

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

Response payload (14 bytes):

| Byte  | Field           | Description |
| ----- | --------------- | ----------- |
| 0     | `state`         | `0x00` = idle, `0x01` = discovering, `0x02` = running, `0x03` = error |
| 1     | `tiles_found`   | Number of tiles successfully discovered (0–8) |
| 2–9   | `tile_status[0..7]` | Per-slot status: `0x00` = not discovered, `0x01` = OK, `0x02` = non-responsive, `0x03` = test failed |
| 10–13 | `uptime_s`      | Seconds since the row controller booted (uint32, big-endian) |

`uptime_s` decreasing between two polls is a reboot, and is the only direct
way to detect one. The alternative — watching the newest `ERROR_LOG`
timestamp — only advances while the row is logging, which on a healthy row is
almost never, so a stale clock is indistinguishable from a steady one. It is
32 bits because a uint16 of seconds wraps at 18.2 hours, and a floor is
expected to run longer than that; a wrap would read as exactly the reboot the
field exists to rule out.

Hosts should treat a payload shorter than 14 bytes as firmware predating this
field rather than as a malformed reply.

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
| 3 + 5×i   | `error_type`  | `0x01` = no ACK after 3 retries, `0x02` = CRC failure, `0x03` = sense collision, `0x04` = LATCH overrun, `0x05` = Row Bus RX overflow, `0x06` = row boot, `0x08` = sense start, `0x09` = tile no version. `0x07` is **retired** — see below |
| 4–5 + 5×i | `timestamp`   | Seconds since row controller boot (uint16, big-endian) |

`ROW_BUS_RX_OVERFLOW` (`error_type = 0x05`) reports a receive overrun on the
**Pi-facing** link rather than a Tile Bus fault: the row controller's UART
dropped inbound bytes, so whichever frame was in flight failed CRC and was
discarded. `slot` and `tile_bus_cmd` describe Tile Bus faults and carry no
meaning here, so both are logged as `0`. Consecutive overruns collapse into a
single entry rather than flooding the 32-deep log.

For `LATCH_OVERRUN` entries (`error_type = 0x04`) the fields are repurposed:
- `slot` — number of tile slots that had been forwarded when `LATCH` arrived
  (0–7). Indicates how far behind the row was; a value of 7 means only the last
  tile was still in flight; a value of 0 means the row hadn't started at all.
- `tile_bus_cmd` — the tile command code that was in flight at the time (typically
  `0x12` SET_LEDS).

`CRC_FAILURE` (`0x02`) repurposes its fields as a **count**: `slot` and
`tile_bus_cmd` are the high and low bytes of how many frames failed the check
since the last such entry, saturating at 65,535, and entries are rate-limited
to one per 5 s. A count rather than one entry per failure because a bad link
produces them in floods; one entry each would evict everything else.

This is the only signal a corrupted link gives. A frame that fails CRC is
discarded, and every other diagnostic — `STATUS`, uptime, the rest of this
log — keeps reporting a perfectly healthy row. Worth knowing what that costs:
20 ft of unterminated Cat5 at 3.125 Mbps puts a round-trip reflection at
~60 ns against a 320 ns bit period, and until this entry existed the firmware
had no way to say so. Note also that Cat5 is a **100 Ω** differential cable —
the usual 120 Ω RS-485 figure is for dedicated RS-485 cable, and mismatched
termination is its own reflection source. See the open question in
[hardware-row-controller.md](hardware-row-controller.md).

Three further entry types are **diagnostics rather than faults**, and also
repurpose the fields:

- `ROW_BOOT` (`0x06`) — logged once per boot, before discovery runs. `slot` and
  `tile_bus_cmd` are the high and low bytes of `POWMAN_CHIP_RESET >> 16`, the
  RP2350's sticky reset causes: power-on, brownout, RUN low, the four watchdog
  variants, glitch detect and so on. More than one bit can be set. A plain
  "was it the watchdog" flag would not be enough: the distinction that matters
  is watchdog versus supply, and a POR with no watchdog or brownout bit set is
  what identified a row restarting mid-boot as a power problem rather than a
  firmware one.
- `SENSE_START` (`0x08`) — one per discovery sweep, `slot` a wrapping sweep
  counter, `tile_bus_cmd` `0`. The error log is cleared by a chip reset, so
  these also answer whether a restart *was* a reset: sweeps logged either side
  of a gap mean the chip kept running, whereas a log that starts over means it
  rebooted.
- `TILE_NO_VERSION` (`0x09`) — discovery mapped a tile that then would not
  answer `VERSION`. `slot` is the slot, `tile_bus_cmd` the address it was
  assigned. The post-discovery version sweep already detected this and moved
  on silently; now it says so. It catches a tile that took an address and then
  died, one unplugged mid-run, and — because `SET_ADDRESS` makes a tile
  abandon its previous address — a row that mis-walked one tile into several
  slots, which shows up here as *every slot but the last* going silent.

`0x07` (`SENSE_EXTRA_SLOT`) is **retired**. It flagged "discovery assigned a
tile to slot 1 or higher", which was a usable proxy for the mis-walk only
while a bench had a single tile. Once addresses are assigned by position, slot
1 holding `0x02` is simply correct, so it fired on every sweep of a healthy
two-tile row — twice a second through the boot retry window, filling the ring
in 16 seconds and evicting `ROW_BOOT`. The code is not reused, so an older
row's entries decode as unknown rather than being mislabelled.

`ROW_BOOT` is stored outside the ring buffer and always reported first, so a
busy row cannot evict the entry that says why it last restarted. The ring
holds the other 31 entries, keeping the response at the same maximum.

Maximum response payload: `1 + 32 × 5 = 161 bytes` — 31 ring entries plus
`ROW_BOOT`, unchanged from before it was moved out of the ring.

---

#### `0x06 VERSION` — unicast

Reports the row controller's own firmware identity plus the cached identity
of each of its tiles (see [tile-bus-protocol.md](tile-bus-protocol.md)'s
`VERSION`). The row answers entirely from a cache filled at the end of the
SENSE auto-mapping sequence (and again on `RE_DISCOVER`) — it does **not**
query tiles over Tile Bus in response to this command, so it never competes
with a display frame in flight and carries the same 20 ms timeout as
`STATUS` rather than `TEST`'s 200 ms.

Because the cache is filled at discovery time, a tile reflashed in place
without a subsequent `RE_DISCOVER` will keep reporting its previous
version here until one runs.

Broadcast (`0xFF`) is invalid — see §4/§3, same as every other admin command.

| Field   | Value |
| ------- | ----- |
| `ADDR`  | target row address |
| `CMD`   | `0x06` |
| `LEN`   | `0` |
| Payload | none |
| ACK     | **Yes** (`0x86 VERSION_RESP`); allow up to 20 ms |

Response payload (fixed 64 bytes):

| Bytes | Field | Description |
| ----- | ----- | ----------- |
| 0–6   | `row_version` | This row controller's own identity, 7-byte `FirmwareVersion` encoding (see below) |
| 7     | `tiles_valid` | Bitmask; bit N set = `tile_version[N]` holds a real value |
| 8–63  | `tile_version[0..7]` | 8 × 7-byte `FirmwareVersion` entries, slot 0 first. An entry whose `tiles_valid` bit is clear is transmitted as seven `0x00` bytes and **must** be ignored by the reader — it means the slot was never discovered, or was discovered but never answered `VERSION`. |

**`FirmwareVersion` wire encoding** (7 bytes, big-endian; shared with Tile
Bus's `VERSION_RESP` and defined once in
`src/common/tile_bus_protocol/firmware_version.h`):

| Bytes | Field | Description |
| ----- | ----- | ----------- |
| 0–1   | `version` | Hand-bumped `ROW_FW_VERSION`/`TILE_FW_VERSION` constant |
| 2–5   | `git_sha` | First 4 bytes of the build's commit SHA |
| 6     | `flags`   | Bit 0 = built from a dirty tree; bits 1–7 reserved (must be 0) |

At 3.125 Mbps a 64-byte response is well under a millisecond on the wire, so
this is safe to issue between display frames.

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
| `LEN`   | sum of all 8 tile entry sizes (32–1,448) |
| Payload | 8 tile entries, concatenated (see below) |

**Per-tile entry format:**

| Bytes  | Field       | Description |
| ------ | ----------- | ----------- |
| 0      | `tile_cmd`  | `0x10` = SET_COLOR, `0x11` = SET_PATTERN, `0x12` = SET_LEDS (same codes as [tile-bus-protocol.md](tile-bus-protocol.md)) |
| 1…     | `tile_data` | Payload for that command: 3 bytes for SET_COLOR, 5 bytes for SET_PATTERN, 180 bytes for SET_LEDS |

Entry sizes: SET_COLOR = 4 bytes, SET_PATTERN = 6 bytes, SET_LEDS = 181 bytes
(1 + 60 LEDs × 3).

**Payload size examples:**

| Mix                      | Total payload |
| ------------------------ | ------------- |
| All 8 × SET_COLOR        | 32 bytes      |
| All 8 × SET_PATTERN      | 48 bytes      |
| All 8 × SET_LEDS         | 1,448 bytes   |
| 4 × SET_COLOR + 4 × SET_LEDS | 740 bytes |

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
| `0x86`        | `VERSION_RESP`     | `VERSION` |

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
| `VERSION`     | 20 ms (answered from cache, no Tile Bus round trip) |

`RE_DISCOVER` gets a short timeout because the ACK should arrive before the
mapping procedure begins. Use `STATUS` polls (with the 20 ms timeout and 3
retries each) to monitor discovery progress.

---

## 8. Timing and Frame Update Sequence

### Normal frame update (30 FPS, 33 ms period)

Frame updates are **pipelined across rows and parallel across chains**. The Pi
writes a `SEND_DATA` to one row on each chain at the same time (see §1), and
each row controller begins forwarding to its tiles on Tile Bus as soon as its
own frame has fully arrived. Tile Bus transfers therefore overlap between rows.
The Pi fires `LATCH` at a fixed point each frame regardless of Tile Bus state.

**At 3.125 Mbps (Row Bus) / 1 Mbps (Tile Bus), worst case (all `SET_LEDS`,
1,456-byte frames):**

```
ms:      0     4.7   9.3  14.0  18.6                          33.3
chain 0: [RC0][RC2][RC4][RC6] ─────── idle ─────────────────── [LATCH]
chain 1: [RC1][RC3][RC5][RC7] ─────── idle ───────────────────
RC0:          [────── Tile Bus 15.0 ms ──────]
RC1:          [────── Tile Bus 15.0 ms ──────]
RC2:                [────── Tile Bus 15.0 ms ──────]
 ⋮
RC6/RC7:                        [────── Tile Bus 15.0 ms ──────]
                                                     ↑ done t≈33.6 ms
```

Both chains finish their Row Bus phase at t≈18.6 ms. The last pair of rows
(RC6 on chain 0, RC7 on chain 1) then take 15.0 ms on their own Tile Buses,
finishing at **t≈33.6 ms** — roughly 0.3 ms past the 33.3 ms `LATCH`.

So at *full* load the overrun path below is the normal case for the last row
in each chain, not an exception: those two rows illuminate a fraction of a
frame late and log `LATCH_OVERRUN`. This is a visible-only-under-worst-case
condition — any frame mixing `SET_COLOR`/`SET_PATTERN` into some slots shrinks
the Row Bus phase and clears it entirely.

Two things would remove the overrun outright, in order of leverage:

1. **Tile Bus at 2 Mbps** halves the 15.0 ms tail to 7.5 ms, finishing at
   t≈26.1 ms with ~7 ms of slack. Gated on the ATtiny3224's USART ceiling —
   confirm against the datasheet.
2. **Fewer bytes per LED.** At 2 bytes/LED, RGB565 puts 60 LEDs on the wire
   in the 120 bytes/tile that 40 LEDs at RGB888 used — restoring the old
   frame sizes exactly, at the cost of colour depth.

Serializing the two chains — driving them one after the other — pushes this to
t≈52 ms and 19 fps. That is why §1 states the concurrency as a requirement
rather than an optimization.

### Blackout sequence

```
Pi → all: BLACKOUT (0xFF)
```

Each row controller immediately issues `SET_COLOR(0,0,0)` to each of its 8
tiles on Tile Bus then broadcasts Tile Bus `LATCH`. All 8 rows do this in parallel.
All tiles go dark within ~2 ms of the `BLACKOUT` frame completing.

---

## 9. Open Questions

- **Bus turnaround timing:** row controllers need a guard interval (≥ 100 µs,
  same as Tile Bus) after the Pi's last stop bit before they begin a response
  frame. Confirm this is sufficient at 3.125 Mbps cable lengths.
- **Tile Bus baud ceiling:** §8's first-choice fix for the worst-case overrun
  is 2 Mbps on Tile Bus. Confirm the ATtiny3224's USART can reach it (and at
  what system clock) against the datasheet before planning around it.
- **Error log ring buffer size:** 32 entries chosen arbitrarily. Tune to fit
  within RP2350 SRAM budget once firmware is written.
- **`RE_DISCOVER` during live show:** decide whether re-discovery is allowed
  during a running show or only during setup/maintenance. Rows go dark during
  re-mapping (~50–200 ms).
