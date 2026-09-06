#!/usr/bin/env python3
"""
Integration test: RS-485 physical-layer loopback on the two-chain pi-hat.

Bring-up test for a *new pi-hat board*, run with NO row controllers
attached. Wire a Cat5 pigtail into the hat's single RJ45 jack that bridges
the two chains' differential pairs directly at the plug - orange pair's A/B
tied to green pair's A/B. Since the hat lands both chains in one 8P8C jack
specifically so one Cat5 run can carry both (docs/row-bus-protocol.md §1),
that pigtail turns the Pi's own two RS-485 transceivers into a single loop:
whatever chain 0 transmits should arrive at chain 1's receiver, and vice
versa, with nothing else on the bus.

This exercises everything upstream of a row controller: both UARTs' TX/RX,
both transceivers' drivers/receivers, the GPIO XDIR direction control on
each chain, and the physical RJ45/pigtail continuity. It does NOT exercise
row controller firmware or the sense chain past the jack - for that, see
test_row_bus_scan.py, run with the pigtail removed and real hardware wired
in instead.

Steps:
  1. Silence check, both chains: with nothing transmitting, a chain's
     receiver should see nothing before the pigtail is exercised at all -
     catches a floating, shorted, or already-noisy line before it can be
     misread as a passing loopback.
  2. Framed round-trip, both directions: send a well-formed Row Bus frame
     out one chain, confirm the other chain's parser reconstructs it
     byte-for-byte (valid CRC, matching addr/cmd/payload). Run across a few
     payload sizes (0, a mid-size frame, and the maximum SEND_DATA-sized
     payload) and a few fixed bit patterns (all-zero, all-ones,
     incrementing) to catch data- or size-dependent faults a single short
     frame would miss.
  3. Payload-size sweep, both directions: ~10 evenly-spaced sizes up to the
     maximum, one round-trip each - if step 2's max-size case fails, this
     narrows down roughly where it starts.
  4. Random-payload burst, both directions: several more round-trips with
     random payloads, for statistical coverage beyond the fixed patterns
     above.

Every round-trip check flushes the receiving chain's buffered input and
resets its frame parser immediately beforehand (RowBus.flush_input()), so
one check's failure can't be misread as several: RowBus.read_frame() keeps
a single parser alive across calls, and a timeout used to leave it stuck
mid-frame - the next call would then feed a perfectly good new frame into
that stale state and misparse it too. Each check here is independent.

Because this bypasses row controllers entirely, it says nothing about the
sense chain, tile bus, or anything past the RJ45 - only that the hat's own
two transceivers and their host-side wiring are sound.

Usage:
  # two-chain hat defaults: ttyAMA0/GPIO17 (chain 0) + ttyAMA2/GPIO7 (chain 1)
  python3 test/integration/test_pi_hat_loopback.py

  # override either or both chains
  python3 test/integration/test_pi_hat_loopback.py --chain /dev/ttyAMA0:17 --chain /dev/ttyAMA2:7

  # cap every tested payload at 512 bytes - e.g. to check whether ever
  # attempting a large frame is what destabilizes the smaller ones that
  # follow it, by never attempting one in the first place
  python3 test/integration/test_pi_hat_loopback.py --max-payload-size 512
"""

from __future__ import annotations

import argparse
import os
import sys

from df2_pi.protocol.constants import MAX_PAYLOAD
from df2_pi.protocol.frame import Frame
from df2_pi.transport.floor import ChainConfig, default_chain_configs
from df2_pi.transport.row_bus import DEFAULT_BAUDRATE, RowBus

SILENCE_TIMEOUT_S = 0.2
ROUND_TRIP_TIMEOUT_S = 0.2
RANDOM_ROUND_TRIPS = 20
RANDOM_PAYLOAD_SIZE = 200


class Results:
    """Collects named PASS/FAIL checks, printing each as it's recorded."""

    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def record(self, name: str, passed: bool, note: str = "") -> bool:
        status = "PASS" if passed else "FAIL"
        suffix = f" - {note}" if note else ""
        print(f"  [{status}] {name}{suffix}")
        self.checks.append((name, passed, note))
        return passed

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def summary(self) -> str:
        passed = sum(1 for _, ok, _ in self.checks if ok)
        return f"{passed}/{len(self.checks)} checks passed"


def check_silence(bus: RowBus, label: str, results: Results) -> None:
    frame = bus.read_frame(timeout=SILENCE_TIMEOUT_S)
    ok = frame is None
    note = "" if ok else f"unexpected frame addr=0x{frame.addr:02X} cmd=0x{frame.cmd:02X} while idle"
    results.record(f"{label}: silent before any transmission", ok, note)


def check_frame_round_trip(
    tx: RowBus, rx: RowBus, label: str, results: Results, addr: int, cmd: int, payload: bytes
) -> None:
    # Isolate this check from any prior one: a previous timeout can otherwise
    # leave rx's parser (and the OS receive buffer) holding onto stale bytes
    # that would corrupt this attempt's read - see the module docstring.
    rx.flush_input()

    expected = Frame(addr, cmd, payload)
    tx.send(addr, cmd, payload)
    got = rx.read_frame(timeout=ROUND_TRIP_TIMEOUT_S)

    name = f"{label}: round-trips addr=0x{addr:02X} cmd=0x{cmd:02X} len={len(payload)}"
    if got is None:
        results.record(name, False, "no valid frame arrived on the other chain (bad wire, or CRC failed)")
        return
    ok = got == expected
    note = "" if ok else f"got addr=0x{got.addr:02X} cmd=0x{got.cmd:02X} payload={got.payload!r}"
    results.record(name, ok, note)


def check_fixed_patterns(tx: RowBus, rx: RowBus, label: str, results: Results, max_size: int) -> None:
    for size in sorted({0, min(64, max_size), max_size}):
        patterns = {"zero": bytes(size), "ones": bytes([0xFF] * size)}
        if size:
            patterns["incrementing"] = bytes(i & 0xFF for i in range(size))
        for name, payload in patterns.items():
            check_frame_round_trip(tx, rx, f"{label} [{name}]", results, addr=0x03, cmd=0x02, payload=payload)
            if size == 0:
                break  # zero-length payload is identical for every pattern


def sweep_sizes(max_size: int, steps: int = 10) -> list[int]:
    """~`steps` evenly-spaced sizes from max_size/steps up to max_size."""
    sizes = sorted({round(max_size * i / steps) for i in range(1, steps + 1)})
    return [s for s in sizes if s > 0]


def check_size_sweep(tx: RowBus, rx: RowBus, label: str, results: Results, max_size: int) -> None:
    for size in sweep_sizes(max_size):
        check_frame_round_trip(tx, rx, f"{label} sweep", results, addr=0x03, cmd=0x02, payload=os.urandom(size))


def check_random_burst(tx: RowBus, rx: RowBus, label: str, results: Results, max_size: int) -> None:
    size = min(RANDOM_PAYLOAD_SIZE, max_size)
    for i in range(RANDOM_ROUND_TRIPS):
        payload = os.urandom(size)
        addr = i % 0x08
        check_frame_round_trip(tx, rx, f"{label} random burst {i + 1}/{RANDOM_ROUND_TRIPS}",
                               results, addr=addr, cmd=0x10, payload=payload)


def parse_chain(spec: str, baudrate: int) -> ChainConfig:
    """PORT[:XDIR_GPIO] - e.g. /dev/ttyAMA0:17, or /dev/ttyAMA0:none"""
    port, _, xdir = spec.partition(":")
    if not xdir:
        raise argparse.ArgumentTypeError(f"--chain needs PORT:XDIR_GPIO, got {spec!r}")
    pin = None if xdir.lower() in ("none", "off") else int(xdir)
    return ChainConfig(port, pin, baudrate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a new pi-hat's two RS-485 chains via an RJ45 loopback pigtail"
    )
    parser.add_argument(
        "--chain", action="append", metavar="PORT:XDIR_GPIO",
        help="repeatable, exactly 2; defaults to the two-chain hat's ttyAMA0:17 and ttyAMA2:7",
    )
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help=f"default: {DEFAULT_BAUDRATE}")
    parser.add_argument(
        "--max-payload-size", type=int, default=MAX_PAYLOAD, metavar="N",
        help=f"cap every tested payload at N bytes (default: {MAX_PAYLOAD}, the real "
             "protocol max) - lower this to check whether ever attempting a large frame "
             "destabilizes the smaller ones tested after it",
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="wait for Enter right after claiming both XDIR GPIOs, before running any "
             "checks - gives you a window to inspect the pins from another shell (e.g. "
             "`pinctrl get 17` / `pinctrl get 7`) while gpiozero is actually holding them "
             "as outputs. Checking pin state with the script NOT running just shows "
             "gpiozero's idle default (floating input) either way and proves nothing.",
    )
    parser.add_argument(
        "--reverse", action="store_true",
        help="run chain 1->0 before chain 0->1 in every section, instead of the default "
             "0->1-first order - checks whether a receiving bus getting stuck after "
             "failures follows test order (whichever bus is tested second in each "
             "section) rather than that specific chain's hardware",
    )
    args = parser.parse_args()

    if not (0 <= args.max_payload_size <= MAX_PAYLOAD):
        parser.error(f"--max-payload-size must be 0-{MAX_PAYLOAD}, got {args.max_payload_size}")

    chains = (
        [parse_chain(spec, args.baudrate) for spec in args.chain]
        if args.chain
        else default_chain_configs(args.baudrate)
    )
    if len(chains) != 2:
        parser.error(f"need exactly 2 --chain values for a loopback test, got {len(chains)}")

    for i, c in enumerate(chains):
        print(f"Chain {i}: {c.port} @ {c.baudrate} baud, XDIR={c.xdir_pin}")
    print(
        "\nMake sure the RJ45 loopback pigtail is plugged in and NO row controllers "
        "are attached before continuing.\n"
    )

    with RowBus(port=chains[0].port, baudrate=chains[0].baudrate, xdir_pin=chains[0].xdir_pin) as bus_a, \
         RowBus(port=chains[1].port, baudrate=chains[1].baudrate, xdir_pin=chains[1].xdir_pin) as bus_b:

        if args.pause:
            print(
                f"Both XDIR GPIOs ({chains[0].xdir_pin}, {chains[1].xdir_pin}) are now "
                "claimed as outputs (idle low - RX mode). From another shell on this Pi, "
                "run:\n"
                f"  pinctrl get {chains[0].xdir_pin}\n"
                f"  pinctrl get {chains[1].xdir_pin}\n"
                "Both should read 'op dl' (output, driven low). If one instead shows an "
                "alt-function code (a0-a5) rather than 'op', that pin isn't actually under "
                "gpiozero's control - the dtoverlay claimed it first.\n"
            )
            input("Press Enter here to continue...")

        bus_a.flush_input()
        bus_b.flush_input()
        results = Results()

        # (label, tx, rx) per direction. Order matters for isolating whether a
        # receiving bus getting stuck after failures follows the specific
        # chain or just "whichever bus is tested second" - see --reverse.
        directions = [("0->1", bus_a, bus_b), ("1->0", bus_b, bus_a)]
        if args.reverse:
            directions.reverse()
            print("(--reverse: running 1->0 before 0->1 in every section below)\n")

        print("=== Silence check ===")
        check_silence(bus_a, "chain 0", results)
        check_silence(bus_b, "chain 1", results)

        print("\n=== Framed round-trip ===")
        for label, tx, rx in directions:
            print(f"--- {label} ---")
            check_fixed_patterns(tx, rx, label, results, args.max_payload_size)

        print("\n=== Payload-size sweep ===")
        for label, tx, rx in directions:
            print(f"--- {label} ---")
            check_size_sweep(tx, rx, label, results, args.max_payload_size)

        print("\n=== Random-payload burst ===")
        for label, tx, rx in directions:
            print(f"--- {label} ---")
            check_random_burst(tx, rx, label, results, args.max_payload_size)

    print(f"\n-- {results.summary()}")
    print("ALL CHECKS PASSED" if results.ok else "SOME CHECKS FAILED")
    return 0 if results.ok else 1


if __name__ == "__main__":
    sys.exit(main())
