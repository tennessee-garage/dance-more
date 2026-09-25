# Eight-row Row Bus bring-up, 2026-09-15

First bench run with all eight row controllers on the two-chain pi-hat
(rows 0,2,4,6 on chain 0 / `ttyAMA0`, rows 1,3,5,7 on chain 1 / `ttyAMA2`),
no tiles attached. Row firmware v5 on every board. Every number below was
taken on the Pi 5 at `garth@testing-pi` with the tools named.

The headline: **the floor cannot run at 30 fps yet, and both sides of the
Row Bus were responsible.** The host took ~48 ms to put one floor update on
the wire against a 33 ms budget — fixed the same day, now 19.2 ms — and a
row controller can only ingest ~100–108 maximum-size frames per second
before it wedges and is watchdog-reset; 4 rows × 30 fps is 120. With the
host fixed, 20 fps is the highest rate all eight rows sustain. Both are
software; neither is the cabling, the transceivers or the supply.

## Protocol conformance — `test_row_bus_scan.py`

All eight rows found on their mapped chains. 20/20 checks per row, 10/10
broadcast checks. Idle draw per row 13–18 mA at 11.94–11.96 V (no tiles).
This is the quiescent baseline to subtract from tile-attached numbers.

The script needed updating first: it still expected the 10-byte `STATUS`
payload from before `uptime_s` was appended, and did not know the
`ROW_BOOT`/`SENSE_START`/`TILE_NO_VERSION` log entry types.

## Row-side ingest ceiling — `test_chain_saturation.py`

Sends a maximum-size `SEND_DATA` (1456 bytes on the wire, 4.66 ms) to every
row slot on one chain plus a `LATCH`, at a target rate, then asks the row
whether it survived. With four real rows on the chain:

| Load on the chain | Frames/s | Bytes/s | Result |
| --- | --- | --- | --- |
| 4 rows @ 10 fps | 40 | 58 kB/s | pass |
| 4 rows @ 15 fps | 60 | 87 kB/s | pass |
| 4 rows @ 20 fps | 80 | 116 kB/s | pass |
| 4 rows @ 25 fps | 100 | 146 kB/s | pass |
| 3 rows @ 30 fps | 90 | 131 kB/s | pass |
| 2 rows @ 30 fps | 60 | 87 kB/s | pass |
| **4 rows @ 27 fps** | **108** | **157 kB/s** | **fail — continuous reset** |
| 4 rows @ 28–30 fps | 112–120 | 163–175 kB/s | fail — continuous reset |

The ceiling is a byte rate, not a frame rate: three rows' worth at 30 fps
passes where four rows' worth at 27 fps fails. It sits between 146 and
157 kB/s, consistent with the ~6 µs/byte `_pumpFIFO()` cost measured on
2026-09-08 with one row and synthetic neighbours
(`docs/row-bus-protocol.md` §1, "Measured ceiling"); this run confirms that
number with four real rows on the chain, on all eight boards.

Past the ceiling the row does not merely drop frames. Sampling `STATUS`
once a second during a 10 s run at 30 fps showed **`uptime_s` = 0 at every
sample** — the row is reset faster than once a second for as long as the
traffic lasts, and only stays up once it stops. The 12 V rail at the row's
own INA226 never moved (11.95 V, 16–19 mA) throughout, and rows on the other
chain were unaffected. This is the RX-overrun wedge described in
`src/row/src/main.cpp`'s watchdog comment, now caught by the 500 ms
watchdog, and its threshold is exactly the "sustained 120 fps of
maximum-size frames" that comment quotes.

### The `ROW_BOOT` reset cause is blind to the watchdog

Every one of those resets was logged as `ROW_BOOT` with cause **POR only** —
no watchdog bit — which, per the comment on `ERROR_TYPE_ROW_BOOT`, is meant
to identify a supply dropout. It does not. `log_boot()` reads
`POWMAN_CHIP_RESET`, but the SDK's `watchdog_enable()` (which
`rp2040.wdt_begin()` wraps) routes the watchdog through PSM only
(`psm_hw->wdsel`), never POWMAN, so a watchdog reset leaves POWMAN's sticky
POR bit from the original power-up in place and sets none of the
`HAD_WATCHDOG_RESET_*` bits. arduino-pico's own `getResetReason()` checks
`WATCHDOG_REASON` *before* falling back to POWMAN for this reason.

Until `log_boot()` also reads `watchdog_hw->reason` (or
`watchdog_caused_reboot()`), a POR-only `ROW_BOOT` means "watchdog or
power", and the rail has to be measured to tell them apart. That is a
firmware change and a physical flash round trip on eight boards, so it is
recorded here rather than fixed.

## Host-side frame time — `Floor.send_rows()`

Timed on the Pi with a synthetic 136×136 frame, 100 iterations
(`phasetime.py`, not committed):

| Phase | Median | p95 |
| --- | --- | --- |
| `FrameEncoder.encode()` (numpy) | 0.53 ms | 0.74 ms |
| `Floor.send_rows()` — 8 × 1448-byte payloads | **47.8 ms** | 53.6 ms |

`docs/row-bus-protocol.md` §8 budgets `send_rows()` at ~19 ms: four rounds
of one frame per chain, chains concurrent, 4.66 ms each. Two things eat the
other ~29 ms, both in Python and both fixable without touching hardware:

1. **`Frame.encode()` costs 2.5 ms per 1448-byte frame — 20 ms per floor
   update.** `crc16_ccitt()` is a bit-at-a-time Python loop.
   `binascii.crc_hqx(data, 0xFFFF)` computes the same CRC-16/CCITT-FALSE
   in C (verified bit-identical on 200 random 1456-byte frames) and is a
   drop-in replacement.

2. **`RowBus.start_write()` blocks for 3.76 ms of a 4.66 ms frame**, so the
   two chains do not overlap: chain 1's write cannot begin until chain 0's
   is ~80 % on the wire, and each round costs ~8.4 ms instead of 4.66. The
   cause is pyserial, not the UART: `os.write()` on the same fd returns in
   64 µs for any size up to 2912 bytes, but `Serial.write()` with the
   default `write_timeout=None` follows every `os.write()` with a
   `select()` for writability (`serialposix.py:640`, pyserial 3.5), and
   the Linux tty layer only reports a tty writable once fewer than
   `WAKEUP_CHARS` (256) bytes remain in its transmit buffer. Frames under
   256 bytes — every admin command, `LATCH`, `BLACKOUT` — are unaffected,
   which is why this never showed up before `SEND_DATA` was timed.
   Writing with `os.write()` directly (looping on a short write) in
   `start_write()` gives the concurrency the method's docstring already
   promises.

### After the two host fixes (same day)

Both were applied (`crc.py` now wraps `binascii.crc_hqx`; `start_write()`
queues with `os.write()`) and re-timed on the same bench:

| Phase | Before | After |
| --- | --- | --- |
| `Frame.encode()`, 1448-byte payload | 2.5 ms | ~µs |
| `RowBus.start_write()`, 1456-byte frame | 3.77 ms | 0.08 ms |
| `Floor.send_rows()`, full floor | 47.8 ms (p95 53.6) | **19.19 ms (p95 19.20)** |

19.2 ms is the §8 budget — four rounds of 4.66 ms plus margins — so the
host side is now deterministic and the chains genuinely overlap. Through
the show path:

| Rate | Frames | Host dropped | Jitter p95 | Rows |
| --- | --- | --- | --- | --- |
| 30 fps | 300 (10 s) | 0 | 0.12 ms | **all 8 boot-loop** — host now delivers the byte rate that resets them |
| 25 fps | 1500 (60 s) | 0 | 0.12 ms | rows 3 and 4 each reset once |
| 20 fps | 2400 (120 s) | 0 | 0.13 ms | no resets, no new log entries on any row |

So 25 fps — 94 % of a row's receive budget by the §1 figures — is not a
safe operating point across eight boards over a minute, even though a
single row passed a 15 s saturation run at it. **20 fps is the highest
rate the floor sustains today**, and the remaining lever is entirely in
the row firmware's receive path.

### Real show path — `df2-pi play`

`df2-pi play --animation plasma` sends a full 1448-byte payload to every
row every frame regardless of content:

| Rate | Frames | Host dropped | Jitter p95 | Rows |
| --- | --- | --- | --- | --- |
| 30 fps | 900 | **866** | 0.15 ms | no resets (host never approached 30 fps) |
| 20 fps | 600 | 45 | 23.9 ms | no resets |
| 15 fps | 9000 (10 min) | 0 | 0.15 ms | no resets, no CRC failures |

At 30 fps the host is so far behind that the rows never see the byte rate
that resets them; the 900 frames took roughly twice the 30 s they should have.

## RE_DISCOVER hammer — `live_demo.py --rediscover-every 5`

75 s, RE_DISCOVER every 5 s across all eight rows with health polls
between. Every row stayed responsive, no resets, `POWER` steady on all.

## Link integrity — CRC soak at 15 fps

15 fps × 4 rows = 60 max-size frames/s per chain, under both the host and
row ceilings, so any `CRC_FAILURE` entry logged during the run is a real
link event on the full-length cable with all eight transceivers loading it.

`df2-pi play --animation plasma --fps 15 --frames 9000`, 10 minutes:
9000 frames, 0 dropped, jitter p95 0.15 ms. Afterwards every row's error
log was identical to the snapshot taken before the run, every uptime had
advanced monotonically by the soak length, and `POWER` was unchanged
(13–15 mA at ~11.95 V). **Zero `CRC_FAILURE` entries** on either chain.

The cabling and termination as installed are therefore clean at 3.125 Mbps
with the full eight-transceiver load — at least at 87 kB/s per chain; the
reflection note under `ERROR_LOG` in `docs/row-bus-protocol.md` §5.1 does not show at this
rate, and the same check should be repeated at 30 fps once the two
ceilings above are lifted.

Earlier `CRC_FAILURE` entries in the logs are all accounted for: the scan
test deliberately sends one corrupt frame per row (and every row on the
chain logs it, since a bad CRC means the address byte cannot be trusted
either), rows booting mid-stream during the saturation runs logged the
partial frame they woke up inside, and one batch of 15 on chain 0 came from
a write micro-benchmark that released XDIR before the frame had left the
wire.

Raw output from each run is in
[`2026-09-15-eight-row-bus-bringup/`](2026-09-15-eight-row-bus-bringup/).

## What this run could not test

With no tiles the rows do no Tile Bus forwarding, so the §8 budget (Row Bus
phase + one row's Tile Bus tail) and `LATCH_OVERRUN` behaviour are
untested. The row-side ceiling measured here is the *receive* path alone;
with forwarding work on core 1 it can only get lower.

## Follow-ups

- ~~**Host:** `binascii.crc_hqx` in `crc.py`; `os.write()` in
  `RowBus.start_write()`.~~ Done; `send_rows()` measured at 19.2 ms.
- ~~**Row firmware:** raise the ingest ceiling past 175 kB/s with
  headroom.~~ Done in v6 with a DMA receive ring; a row keeps up at 50 fps
  ([2026-09-24](2026-09-24-row-dma-receive.md)).
- **Row firmware:** `log_boot()` must consult `WATCHDOG_REASON` so a
  watchdog reset is not logged as POR.
- **Row firmware:** the boot-loop under overload is the watchdog doing its
  job, but a row that is reset every ~500 ms for as long as the host keeps
  sending is not much better than one that is wedged. v6 bounds core 0's
  frame loop so the watchdog is always fed (#100); no load the wire can
  deliver overloads a v6 row, so that bound is untested.
