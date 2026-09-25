# Row DMA receive, 2026-09-24

Row firmware v6 (`v6+a1c5ad80`) on row 0; rows 1–7 still on v5 as real
neighbours on the same two chains. No tiles discovered on any row. Tools and
bench as in [the 2026-09-15 bring-up](2026-09-15-eight-row-bus-bringup.md),
raw output in [`2026-09-24-row-dma-receive/`](2026-09-24-row-dma-receive/).

**Result: the row-side receive ceiling is gone.** Row 0 kept up with a
worst-case chain at 50 fps, 93 % of what the wire can carry, where v5 failed
at 27 fps.

## What changed in v6

- Row Bus receive moved off arduino-pico's `SerialUART` (~6 µs of CPU per
  byte) onto a DMA channel filling a 16 KB ring in RAM. Core 0 parses
  straight out of the ring.
- The parser drops other rows' frames itself, still CRC-checking them, so
  `CRC_FAILURE` keeps covering the whole chain.
- Core 0 handles at most 4 frames per `loop()` pass, so the watchdog is fed
  on every pass however busy the bus is (#100).

## Protocol conformance — `test_row_bus_scan.py`

20/20 on row 0, as on every v5 row; 10/10 broadcast.

## Saturation — `test_chain_saturation.py --row 0`

Full `SET_LEDS` frames to all four rows on chain 0 plus `LATCH`, 60 s per
rate:

| Target | Chain byte rate | Share of wire | Row 0 |
| --- | --- | --- | --- |
| 30 fps | 175 kB/s | 56 % | pass — no restart, no new log entries |
| 35 fps | 204 kB/s | 65 % | pass |
| 45 fps | 262 kB/s | 84 % | pass |
| 50 fps | 291 kB/s | 93 % | pass |

Uptime climbed monotonically from 97 s to 287 s across the four runs.

## Soak — `df2-pi play --animation plasma --fps 30 --frames 18000`

10 minutes through `Floor.send_rows()`, both chains: 18000 frames, 0
dropped, jitter p95 0.13 ms. Row 0's uptime went from 300 s to 903 s and
its error log was identical before and after — **zero CRC failures** with
every frame on the chain CRC-checked. The two `CRC_FAILURE` entries already
in it (counts 1 and 3) are the four corrupt frames the scan test sends on
chain 0.

The v5 rows boot-looped throughout, as expected at 30 fps; that has no
effect on row 0, which only listens.

## Not exercised

- **The bounded frame loop (#100).** It exists so an overloaded row keeps
  feeding its watchdog, but no load the wire can deliver overloads a v6
  row, so it could not be triggered here. It stays as a safeguard.
- **Tiles.** With none attached, the Tile Bus tail in
  [row-bus-protocol.md](../row-bus-protocol.md) §8 — ~15 ms per row after its
  frame arrives, putting the last row on each chain at ~33.6 ms against the
  33.3 ms `LATCH` — is untested. It is now the binding constraint for 30 fps
  with all `SET_LEDS`.

## Unexplained: row current

Row 0 read **~315 mA at 11.71 V** throughout this session, against 13–18 mA
at 11.95 V for every row on 2026-09-15. Row 1 read 88 mA; rows 2–7 read
14–18 mA as before. The reading did not change under load (315 mA before
and after the soak). Row 1 is still on v5 firmware, so the change is not
obviously the new firmware; something on the bench may have changed.
Not investigated.

## Pitfall hit on the first flash

A DMA channel triggered in ENDLESS mode with a transfer count of 0 halts
immediately (RP2350 datasheet §12.6.2.2.1). The first v6 build encoded the
mode with a zero count, and row 0 booted, ran both cores and never received a
byte. ENDLESS does not decrement the count, so any non-zero value runs
forever; the fix is commented in `pi_transport_rp2350.cpp`.
