#!/usr/bin/env python3
"""
Integration test: can a row keep up with a fully loaded chain?

docs/row-bus-protocol.md §8 budgets a floor update as the Row Bus phase plus
one row's Tile Bus tail, and concludes 30 FPS is approximately reachable.
That model counts only *forwarding* work. It does not count what a row spends
merely receiving - and a row parses every byte on its chain, not just the
frames addressed to it, because address filtering happens after the frame is
assembled.

Measured on the bench (2026-09-08), a row ingested Row Bus bytes at ~6.0 us
each against a 3.2 us/byte arrival rate at 3.125 Mbps, so it finished parsing
a maximum-size frame roughly 4 ms after the last bit was already on the wire.
At that rate a chain's worth of worst-case frames - 4 rows x 1,456 bytes -
costs ~35 ms of core-0 time per 33.3 ms frame period, which would put 30 FPS
out of reach for reasons §8 does not model.

This test drives that case directly. It sends a full-size SEND_DATA to every
row slot on a chain plus a LATCH, at a target frame rate, for a set duration,
and then asks the row what happened. Frames addressed to absent rows are the
point, not a flaw: they load the receive path exactly as neighbours would.

What it watches for, in decreasing order of severity:

  ROW_BUS_RX_OVERFLOW (0x05)  the receive path fell behind far enough to drop
                              bytes. This is the failure the budget question
                              is really about.
  a restart                   uptime going backwards (STATUS_RESP), which on
                              this path historically meant a wedge that only
                              a power cycle cleared.
  silence                     stopped answering admin commands entirely.
  LATCH_OVERRUN (0x04)        expected at full load per §8; reported as a rate
                              rather than a failure.

Usage:
  python3 test/integration/test_chain_saturation.py                 # 30 fps, 15 s
  python3 test/integration/test_chain_saturation.py --fps 15 --duration 30
  python3 test/integration/test_chain_saturation.py --row 2 --chain /dev/ttyAMA0:23
"""

from __future__ import annotations

import argparse
import sys
import time

from df2_pi.protocol.constants import LEDS_PER_TILE, Cmd, TileCmd
from df2_pi.transport import ChainConfig, Floor, RowChainMap, RowNotResponding, default_chain_configs
from df2_pi.transport.row_bus import DEFAULT_BAUDRATE

ERROR_TYPE_LATCH_OVERRUN = 0x04
ERROR_TYPE_ROW_BUS_RX_OVERFLOW = 0x05

# Dim: this runs for a while and the point is bus load, not light output.
LEVEL = 30


def probe(floor: Floor, row: int, cmd: int) -> bytes | None:
    try:
        return floor.request(row, cmd).payload
    except RowNotResponding:
        return None


def uptime_s(floor: Floor, row: int) -> int | None:
    st = probe(floor, row, Cmd.STATUS)
    if st is None or len(st) < 14:
        return None
    return (st[10] << 24) | (st[11] << 16) | (st[12] << 8) | st[13]


def log_entries(floor: Floor, row: int) -> list[tuple] | None:
    e = probe(floor, row, Cmd.ERROR_LOG)
    if e is None:
        return None
    return [tuple(e[1 + i * 5 : 6 + i * 5]) for i in range(e[0])]


def count_new(previous: list[tuple] | None, current: list[tuple]) -> int:
    """Entries appended since `previous`. The log is an append-only ring
    emitted oldest-first, so a later read is the earlier one shifted left."""
    if previous is None:
        return len(current)
    for shift in range(len(previous) + 1):
        kept = len(previous) - shift
        if previous[shift:] == current[:kept]:
            return len(current) - kept
    return len(current)


def full_frame_payload() -> bytes:
    """One SET_LEDS entry per slot - the largest legal SEND_DATA payload."""
    entry = bytes([TileCmd.SET_LEDS]) + bytes((LEVEL, 0, 0)) * LEDS_PER_TILE
    return entry * 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=30.0, help="target frame rate (default: 30)")
    ap.add_argument("--duration", type=float, default=15.0, help="seconds to sustain (default: 15)")
    ap.add_argument("--row", type=int, default=0, help="row to interrogate (default: 0)")
    ap.add_argument("--latch-delay", type=float, default=0.0,
                    help="seconds between the last SEND_DATA and LATCH (default: 0)")
    ap.add_argument("--peers", type=int, default=0, metavar="N",
                    help="send only N frames per period instead of one per row slot "
                         "on the chain. Separates total byte rate from frame rate: "
                         "N=2 at 30 fps carries the same bytes/s as N=4 at 15 fps")
    ap.add_argument("--chain", action="append", metavar="DEV:XDIR")
    ap.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    args = ap.parse_args()

    chains = ([ChainConfig(*c.rsplit(":", 1)[:1], xdir_pin=int(c.rsplit(":", 1)[1]),
                           baudrate=args.baudrate) for c in args.chain]
              if args.chain else default_chain_configs(args.baudrate))
    chain_map = RowChainMap.single_chain() if len(chains) == 1 else RowChainMap.alternating(len(chains))

    with Floor(chains=chains, chain_map=chain_map) as floor:
        if args.row not in floor.scan():
            print(f"row 0x{args.row:02X} did not respond", file=sys.stderr)
            return 1

        chain = chain_map.chain_for(args.row)
        peers = [r for r in range(8) if chain_map.chain_for(r) == chain]
        if args.peers:
            peers = (peers * ((args.peers // len(peers)) + 1))[:args.peers]
        payload = full_frame_payload()
        wire_ms = (len(payload) + 8) * 10 / args.baudrate * 1000

        byte_rate = len(peers) * (len(payload) + 8) * args.fps / 1000.0
        print(f"row 0x{args.row:02X} on chain {chain}; {len(peers)} frames/frame-period "
              f"({', '.join(f'0x{r:02X}' for r in peers)}) = {byte_rate:.0f} kB/s on the wire")
        print(f"each frame {len(payload) + 8} bytes = {wire_ms:.2f} ms on the wire; "
              f"{len(peers) * wire_ms:.1f} ms of wire per period at {args.fps:g} fps "
              f"({1000 / args.fps:.1f} ms budget)")

        start_uptime = uptime_s(floor, args.row)
        before = log_entries(floor, args.row)
        print(f"start uptime {start_uptime}s\n" if start_uptime is not None
              else "start uptime unavailable (older firmware)\n")

        period = 1.0 / args.fps
        deadline = time.perf_counter() + args.duration
        frames = late = 0
        t0 = time.perf_counter()
        while time.perf_counter() < deadline:
            due = time.perf_counter() + period
            for r in peers:
                floor.send_data(r, payload)
            if args.latch_delay:
                time.sleep(args.latch_delay)
            floor.latch()
            frames += 1
            remaining = due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                late += 1
        elapsed = time.perf_counter() - t0

        achieved = frames / elapsed
        print(f"sent {frames} frame-periods in {elapsed:.1f}s = {achieved:.1f} fps "
              f"({'kept up' if late == 0 else f'{late} periods over budget on the host side'})")

        after = log_entries(floor, args.row)
        end_uptime = uptime_s(floor, args.row)
        failures = []

        if after is None:
            failures.append("row stopped answering ERROR_LOG")
        else:
            new = count_new(before, after)
            fresh = after[len(after) - new:] if new else []
            overflow = sum(1 for x in fresh if x[2] == ERROR_TYPE_ROW_BUS_RX_OVERFLOW)
            overrun = sum(1 for x in fresh if x[2] == ERROR_TYPE_LATCH_OVERRUN)
            print(f"new log entries: {new} ({overflow} rx_overflow, {overrun} latch_overrun)")
            if overflow:
                failures.append(f"{overflow} ROW_BUS_RX_OVERFLOW - the receive path dropped bytes")
            if overrun:
                print(f"  latch_overrun at {overrun / elapsed:.1f}/s - expected at full load per §8, "
                      f"not a failure in itself")

        if start_uptime is not None and end_uptime is not None:
            if end_uptime < start_uptime:
                failures.append(f"row RESTARTED (uptime {start_uptime}s -> {end_uptime}s)")
            else:
                print(f"end uptime {end_uptime}s (no restart)")
        elif end_uptime is None:
            failures.append("row stopped answering STATUS")

        floor.blackout()

        print()
        if failures:
            for f in failures:
                print(f"FAIL: {f}")
            return 1
        print(f"PASS: row kept up with a saturated chain at {achieved:.1f} fps")
        return 0


if __name__ == "__main__":
    sys.exit(main())
