# Row DMA receive, 2026-09-24

Row firmware v6 (`v6+a1c5ad80`) on row 0; rows 1–7 still on v5 as real
neighbours on the same two chains. Row 0 had four tiles on its Tile Bus, two
of them with LED strips, but **discovered none of them** — see below. No
other row discovered tiles either. Tools and
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
- **Tiles.** With none discovered, the Tile Bus tail in
  [row-bus-protocol.md](../row-bus-protocol.md) §8 — ~15 ms per row after its
  frame arrives, putting the last row on each chain at ~33.6 ms against the
  33.3 ms `LATCH` — is untested. It is now the binding constraint for 30 fps
  with all `SET_LEDS`.

## Open: row 0 discovers none of its four tiles

Row 0 had four tiles connected (tiles 0 and 1 with LED strips, 2 and 3
without), all showing their power-up test pattern, so each tile controller
was running. Row 0 still reported `tiles_found = 0` all session, and an
explicit `RE_DISCOVER` long after the tiles had booted finished in under
100 ms with nothing found — the timing of slot 0 never answering
`DETECT_SENSE` (20 ms settle plus three 5 ms attempts). Row 1, still on v5,
did the same. Consequently no `SEND_DATA` reached a tile and nothing lit
during any run above.

v6 changes only the Row Bus side (UART1, DMA, the Row Bus parser, core 0's
loop); discovery, SENSE and the Tile Bus on UART0 are untouched. That makes
v6 an unlikely cause but does not rule it out. Not yet investigated.

Row 0's draw — ~315 mA at 11.71 V, flat before, during and after the soak,
against 13–18 mA for a bare row — is consistent with the four tiles: the
2026-08-16 sweep measured ~108 mA for a row plus one tile with its strip
dark. Row 1 read 88 mA.

## Pitfall hit on the first flash

A DMA channel triggered in ENDLESS mode with a transfer count of 0 halts
immediately (RP2350 datasheet §12.6.2.2.1). The first v6 build encoded the
mode with a zero count, and row 0 booted, ran both cores and never received a
byte. ENDLESS does not decrement the count, so any non-zero value runs
forever; the fix is commented in `pi_transport_rp2350.cpp`.
