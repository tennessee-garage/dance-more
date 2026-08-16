#!/usr/bin/env python3
"""
Stress and visual-verification test: drive one tile as hard as the chain allows.

Two halves:

  1. VISUAL - slow, watchable patterns whose correctness you judge by eye.
     Each announces what to look for. These check things telemetry cannot:
     colour mapping (a GRB/RGB swap looks fine to the protocol), per-pixel
     addressing, LED ordering and the actual LED count.

  2. STRESS - ramps SEND_DATA/LATCH throughput using maximum-size 968-byte
     SET_LEDS frames until the achieved rate stops tracking the requested
     one, then soaks at the ceiling. Checks the row controller is still
     responsive after each step, since a Row Bus RX overrun used to wedge
     it hard enough to need a power cycle (fixed by setFIFOSize() in
     pi_transport_rp2350.cpp - this is the regression test for that).

At full rate every frame's LATCH arrives while the row controller is still
forwarding, so LATCH_OVERRUN entries are expected here and are diagnostic,
not failures (docs/row-bus-protocol.md 5.2). The real show controller sends
all 8 rows and then latches once at the frame boundary.

Usage:
  python3 test/integration/test_tile_stress.py --chain /dev/ttyAMA0:23
  python3 test/integration/test_tile_stress.py --chain /dev/ttyAMA0:23 --skip-visual
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import time

from df2_pi.protocol.constants import Cmd, TileCmd
from df2_pi.transport import (
    ChainConfig,
    Floor,
    RowChainMap,
    RowNotResponding,
    default_chain_configs,
)
from df2_pi.transport.row_bus import DEFAULT_BAUDRATE

NUM_LEDS = 40
NUM_SLOTS = 8
# Full white on 40 LEDs approaches the bench supply's limit, which flattens
# any visual difference between bright colours. Cap so patterns stay honest.
MAX_LEVEL = 200


def set_leds_payload(pixels: list[tuple[int, int, int]]) -> bytes:
    """One SET_LEDS entry per slot: 8 x 121 = 968 bytes, the largest legal
    SEND_DATA payload."""
    if len(pixels) != NUM_LEDS:
        raise ValueError(f"need {NUM_LEDS} pixels, got {len(pixels)}")
    body = b"".join(bytes(p) for p in pixels)
    return (bytes([TileCmd.SET_LEDS]) + body) * NUM_SLOTS


def set_color_payload(r: int, g: int, b: int) -> bytes:
    return bytes([TileCmd.SET_COLOR, r, g, b]) * NUM_SLOTS


def wheel(pos: float, level: int = MAX_LEVEL) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(pos % 1.0, 1.0, 1.0)
    return (int(r * level), int(g * level), int(b * level))


def show(floor: Floor, row: int, pixels: list[tuple[int, int, int]]) -> None:
    floor.send_data(row, set_leds_payload(pixels))
    floor.latch()


# Printed in full BEFORE anything lights up: stdout is usually buffered
# (piped, or read over ssh), so per-step narration tends to arrive long after
# the LEDs have moved on. The operator needs the whole checklist in hand
# first, and each step announces itself again as it starts.
VISUAL_PLAN = """
  1. Solid RED, GREEN, BLUE, WHITE (2s each)
     -> Is each colour the one named? A red/green swap means the tile's RGB
        order is wrong - no protocol check can catch that, only your eyes.

  2. Single white pixel walking LED 0 -> 39 (~4s)
     -> Exactly one lit pixel, travelling end to end, no skips or gaps?
        It should reach the far end with none left over.

  3. LEDs 0-19 RED, LEDs 20-39 BLUE (3s)
     -> Clean split, boundary exactly at the midpoint? Confirms 40
        addressable LEDs - no more, no fewer.

  4. Static rainbow across the strip (3s)
     -> Smooth continuous hue sweep, no dead or repeated pixels? This drives
        all 120 per-pixel bytes independently.

  5. Rotating rainbow (~6s)
     -> Smooth motion, no stutter, tearing or flicker? Tearing would mean
        LATCH is not holding pixels until the frame is complete.
"""


def announce(title: str, hold: float = 0.0) -> None:
    print(f"\n  -> {title}", flush=True)
    time.sleep(hold)


def visual(floor: Floor, row: int) -> None:
    print("\n=== VISUAL VERIFICATION ===")
    print(VISUAL_PLAN)
    print("  Watch the tile. Starting in 5s...", flush=True)
    time.sleep(5)

    announce("1. Solid colours")
    for name, rgb in [("RED", (MAX_LEVEL, 0, 0)), ("GREEN", (0, MAX_LEVEL, 0)),
                      ("BLUE", (0, 0, MAX_LEVEL)),
                      ("WHITE", (MAX_LEVEL,) * 3)]:
        print(f"       {name}", flush=True)
        floor.send_data(row, set_color_payload(*rgb))
        floor.latch()
        time.sleep(2.0)

    announce("2. Walking pixel, LED 0 -> 39", 0.4)
    for i in range(NUM_LEDS):
        px = [(0, 0, 0)] * NUM_LEDS
        px[i] = (MAX_LEVEL,) * 3
        show(floor, row, px)
        time.sleep(0.09)
    time.sleep(0.4)

    announce("3. Split: 0-19 RED / 20-39 BLUE", 0.4)
    show(floor, row, [(MAX_LEVEL, 0, 0)] * 20 + [(0, 0, MAX_LEVEL)] * 20)
    time.sleep(3.0)

    announce("4. Static rainbow", 0.4)
    show(floor, row, [wheel(i / NUM_LEDS) for i in range(NUM_LEDS)])
    time.sleep(3.0)

    announce("5. Rotating rainbow", 0.4)
    t_end = time.time() + 6.0
    phase = 0.0
    while time.time() < t_end:
        show(floor, row, [wheel(i / NUM_LEDS + phase) for i in range(NUM_LEDS)])
        phase += 0.012
        time.sleep(0.02)

    floor.blackout()
    print("\n    -> blacked out. Visual section done.")


def responsive(floor: Floor, row: int, attempts: int = 3) -> bool:
    for _ in range(attempts):
        try:
            floor.request(row, Cmd.STATUS)
            return True
        except RowNotResponding:
            time.sleep(0.25)
    return False


def measure(floor: Floor, row: int, frames: int, target_fps: float | None) -> float:
    """Push `frames` max-size SET_LEDS frames, optionally paced. Returns fps."""
    period = 1.0 / target_fps if target_fps else 0.0
    payloads = [set_leds_payload([wheel(i / NUM_LEDS + f * 0.02) for i in range(NUM_LEDS)])
                for f in range(min(frames, 24))]
    start = time.perf_counter()
    for f in range(frames):
        due = start + f * period
        floor.send_data(row, payloads[f % len(payloads)])
        floor.latch()
        if period:
            while time.perf_counter() < due + period:
                pass
    return frames / (time.perf_counter() - start)


def stress(floor: Floor, row: int, soak_s: float) -> bool:
    print("\n=== THROUGHPUT: 968-byte SET_LEDS frames + LATCH ===")
    wire_ms = (968 + 8) * 10 / 3_125_000 * 1000
    print(f"    Each frame is 976 bytes on the wire = {wire_ms:.2f} ms at 3.125 Mbps,")
    print(f"    so the Row Bus alone caps this row near {1000 / wire_ms:.0f} fps.\n")
    print(f"    {'target':>8}  {'achieved':>9}  {'per frame':>10}  row")
    print("    " + "-" * 44)

    ok = True
    for target in [30, 60, 120, 200, None]:
        label = f"{target} fps" if target else "flat out"
        fps = measure(floor, row, frames=90, target_fps=target)
        alive = responsive(floor, row)
        ok = ok and alive
        print(f"    {label:>8}  {fps:>7.1f}fps  {1000 / fps:>8.2f}ms  "
              f"{'ok' if alive else 'NOT RESPONDING'}")
        if not alive:
            print("\n    Row controller stopped answering - stopping the ramp.")
            return False

    print(f"\n=== SOAK: flat out for {soak_s:.0f}s ===")
    start = time.perf_counter()
    sent = 0
    while time.perf_counter() - start < soak_s:
        sent += 1
        measure(floor, row, frames=1, target_fps=None)
    elapsed = time.perf_counter() - start
    print(f"    {sent} frames in {elapsed:.1f}s = {sent / elapsed:.1f} fps sustained")
    print(f"    ({sent * 976 / elapsed / 1000:.0f} kB/s on the Row Bus)")

    alive = responsive(floor, row)
    print(f"    row controller after soak: {'responsive' if alive else 'NOT RESPONDING'}")
    if not alive:
        return False

    frame = floor.request(row, Cmd.STATUS).payload
    print(f"    tiles still discovered: {frame[1]}")
    errs = floor.request(row, Cmd.ERROR_LOG).payload
    kinds: dict[int, int] = {}
    for i in range(errs[0]):
        kinds[errs[3 + i * 5]] = kinds.get(errs[3 + i * 5], 0) + 1
    names = {0x01: "no_ack_after_retries", 0x02: "crc_failure",
             0x03: "sense_collision", 0x04: "latch_overrun",
             0x05: "row_bus_rx_overflow"}
    summary = ", ".join(f"{names.get(k, hex(k))}={v}" for k, v in sorted(kinds.items()))
    print(f"    error log: {errs[0]} entries" + (f" ({summary})" if summary else ""))
    if kinds and set(kinds) - {0x04}:
        print("    NOTE: entries other than latch_overrun appeared - worth a look.")
    else:
        print("    (latch_overrun only: expected at this rate, see module docstring)")
    return ok and alive


def parse_chain(spec: str, baudrate: int) -> ChainConfig:
    port, _, xdir = spec.partition(":")
    if not xdir:
        raise argparse.ArgumentTypeError(f"--chain needs PORT:XDIR_GPIO, got {spec!r}")
    pin = None if xdir.lower() in ("none", "off") else int(xdir)
    return ChainConfig(port, pin, baudrate)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stress and visually verify one tile")
    ap.add_argument("--chain", action="append", metavar="PORT:XDIR_GPIO")
    ap.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--soak", type=float, default=20.0, help="soak seconds (default 20)")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--skip-stress", action="store_true")
    args = ap.parse_args()

    chains = ([parse_chain(c, args.baudrate) for c in args.chain]
              if args.chain else default_chain_configs(args.baudrate))
    cmap = RowChainMap.single_chain() if len(chains) == 1 else RowChainMap.alternating(len(chains))

    with Floor(chains=chains, chain_map=cmap) as floor:
        if not responsive(floor, args.row):
            print(f"row 0x{args.row:02X} is not answering - power cycle?", file=sys.stderr)
            return 1

        # Discovery only runs at boot, and tiles are deaf for their first few
        # seconds (startup animation), so a cold-booted row reports zero.
        floor.request(args.row, Cmd.RE_DISCOVER)
        time.sleep(0.4)
        tiles = floor.request(args.row, Cmd.STATUS).payload[1]
        print(f"row 0x{args.row:02X}: {tiles} tile(s) discovered")
        if tiles == 0:
            print("no tiles discovered - nothing to drive", file=sys.stderr)
            return 1

        if not args.skip_visual:
            visual(floor, args.row)
        ok = True
        if not args.skip_stress:
            ok = stress(floor, args.row, args.soak)
        floor.blackout()

    print("\n" + ("STRESS TEST PASSED" if ok else "STRESS TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
