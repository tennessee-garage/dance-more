# Row DMA receive, 2026-09-24

Row firmware v6 (`v6+a1c5ad80`) on row 0; rows 1–7 still on v5 as real
neighbours on the same two chains. Row 0 had four tiles on its Tile Bus, two
of them with LED strips; the runs below the saturation table were first done
with none of them discovered (a plug wiring fault), then repeated with three
— see "With tiles forwarding". Tools and
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
- **The Tile Bus tail at full load.** [row-bus-protocol.md](../row-bus-protocol.md)
  §8 puts the last row on each chain, forwarding to eight tiles, at ~33.6 ms
  against the 33.3 ms `LATCH`. Only the first row with three tiles was
  tested here (see below). That tail is now the binding constraint for
  30 fps with all `SET_LEDS`.

## With tiles forwarding

Row 0's four tiles were not discovered during the runs above: the SENSE line
in the row's Cat6 plug was wired wrong. Row 0's draw of ~315 mA at 11.71 V
through those runs, against 13–18 mA for a bare row, was the four powered
tiles, and nothing lit because no tile had an address to send to.

After the plug was rewired, `RE_DISCOVER` found tiles in slots 0–2 (tiles 0
and 1 with strips); slot 3 still did not answer `DETECT_SENSE`, so discovery
ended there. Slots 0 and 2 run tile firmware v2; slot 1 runs v2 from a dirty
tree (`02df35e3-dirty`). Rerun with row 0 forwarding to those three tiles:

| Run | Result |
| --- | --- |
| Saturation, 30 fps, 60 s | pass — no restart, 0 `LATCH_OVERRUN`, 0 `ROW_BUS_RX_OVERFLOW` |
| Saturation, 35 fps, 60 s | pass — same |
| `df2-pi play --animation plasma --fps 30`, 10 min | 18000 frames, 0 dropped, jitter p95 0.12 ms; uptime 434 → 1037 s; no new log entries; still 3 tiles discovered afterwards |

Row 0 read 1,093 mA at 11.27 V after the soak against 319 mA before it, so
the strips were receiving data (the last frame stays latched).

This is not the §8 worst case. Row 0's frame is first on its chain, so it
arrives by 4.7 ms and forwarding to three tiles is done long before `LATCH`.
The binding case — the last row on a chain forwarding to eight tiles — needs
a fully populated row to test.

## Pitfall hit on the first flash

A DMA channel triggered in ENDLESS mode with a transfer count of 0 halts
immediately (RP2350 datasheet §12.6.2.2.1). The first v6 build encoded the
mode with a zero count, and row 0 booted, ran both cores and never received a
byte. ENDLESS does not decrement the count, so any non-zero value runs
forever; the fix is commented in `pi_transport_rp2350.cpp`.
