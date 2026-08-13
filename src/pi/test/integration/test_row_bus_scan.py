#!/usr/bin/env python3
"""
Integration test: verify whatever is live on the Row Bus.

Run this on a Raspberry Pi wired to the Row Bus through the Pi hat. It drives
the Floor abstraction (src/df2_pi/transport/floor.py), so it exercises the
same row->chain routing the real show controller will use - see
docs/row-bus-protocol.md for the wire format and command set, and
src/row/lib/row_core/row_command_handler.cpp for the firmware side.

Steps:
  1. Scan rows 0-7, each on the chain that should carry it. A responder
     therefore also confirms the wiring matches the chain map.
  2. For each row controller found, exercise every admin command (TEST,
     STATUS, POWER, RE_DISCOVER, ERROR_LOG), the fire-and-forget display
     commands (SEND_DATA, LATCH, BLACKOUT), and a few protocol-robustness
     cases (unrecognized command, address filtering, corrupt CRC).
  3. Assert every admin command gets no response when sent as a broadcast
     (docs/row-bus-protocol.md §3: "all row controllers accept; none
     respond") - RowCommandHandler::handle() enforces this independently of
     the host, so this goes around Floor.broadcast()'s own guard against
     constructing such a frame to test the row controller's side of it.

With no tiles attached, a healthy row controller is expected to settle into
STATUS state=0x02 ("running", i.e. discovery finished) with tiles_found=0
shortly after boot or a RE_DISCOVER - see SenseMapper::finish_discovery()
in src/row/lib/row_core/sense_mapper.cpp for why "found nothing" is treated
as the expected end-of-chain result, not an error.

Usage:
  # two-chain hat (defaults: ttyAMA0/GPIO17 + ttyAMA2/GPIO7)
  python3 test/integration/test_row_bus_scan.py

  # single-chain bench board, XDIR still on GPIO23
  python3 test/integration/test_row_bus_scan.py --chain /dev/ttyAMA0:23
"""

from __future__ import annotations

import argparse
import sys
import time

from df2_pi.protocol.constants import Cmd, Resp, TileCmd
from df2_pi.protocol.frame import Frame
from df2_pi.transport import (
    ChainConfig,
    Floor,
    RowChainMap,
    RowNotResponding,
    default_chain_configs,
)
from df2_pi.transport.row_bus import DEFAULT_BAUDRATE

STATUS_STATE_NAMES = {0x00: "idle", 0x01: "discovering", 0x02: "running", 0x03: "error"}
TILE_STATUS_NAMES = {0x00: "not_discovered", 0x01: "ok", 0x02: "non_responsive", 0x03: "test_failed"}
ERROR_TYPE_NAMES = {0x01: "no_ack_after_retries", 0x02: "crc_failure", 0x03: "sense_collision", 0x04: "latch_overrun"}
TEST_FAULT_BIT_NAMES = ["row_bus_uart", "tile_bus_xcvr", "sram"]


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


def _try_request(floor: Floor, row: int, cmd: int) -> Frame | None:
    try:
        return floor.request(row, cmd)
    except RowNotResponding:
        return None


def check_test(floor: Floor, row: int, results: Results) -> None:
    frame = _try_request(floor, row, Cmd.TEST)
    if frame is None or len(frame.payload) != 2:
        results.record("TEST", False, "no valid TEST_RESP")
        return

    result, fault_flags = frame.payload[0], frame.payload[1]
    if result == 0:
        results.record("TEST", True, "self-test reports all-pass")
    else:
        bits = [name for bit, name in enumerate(TEST_FAULT_BIT_NAMES) if fault_flags & (1 << bit)]
        results.record("TEST", False, f"result=0x{result:02X} fault_flags={bits}")


def check_status(floor: Floor, row: int, results: Results, expect_no_tiles: bool = True) -> None:
    frame = _try_request(floor, row, Cmd.STATUS)
    if frame is None:
        results.record("STATUS", False, "no valid STATUS_RESP")
        return
    if len(frame.payload) != 10:
        results.record("STATUS", False, f"payload len {len(frame.payload)} != 10")
        return

    state, tiles_found = frame.payload[0], frame.payload[1]
    tile_status = list(frame.payload[2:10])
    print(f"      state={STATUS_STATE_NAMES.get(state, hex(state))} tiles_found={tiles_found} "
          f"tile_status={[TILE_STATUS_NAMES.get(s, hex(s)) for s in tile_status]}")
    results.record("STATUS", True)

    if expect_no_tiles:
        no_tiles = tiles_found == 0 and all(s == 0 for s in tile_status)
        results.record(
            "STATUS: no tiles discovered (expected - none attached)",
            no_tiles,
            "" if no_tiles else "expected 0 discovered tiles for this bench setup",
        )


def check_power(floor: Floor, row: int, results: Results) -> None:
    frame = _try_request(floor, row, Cmd.POWER)
    if frame is None or len(frame.payload) != 6:
        results.record("POWER", False, "no valid POWER_RESP")
        return

    p = frame.payload
    print(f"      voltage={(p[0] << 8) | p[1]} mV  current={(p[2] << 8) | p[3]} mA  "
          f"power={(p[4] << 8) | p[5]} mW")
    # No asserted range: whether the INA220's rail is even energised depends
    # on the bench setup. A well-formed response is what's verifiable; the
    # numbers are printed for a human to sanity-check.
    results.record("POWER", True)


def check_error_log(floor: Floor, row: int, results: Results, label: str = "ERROR_LOG") -> None:
    frame = _try_request(floor, row, Cmd.ERROR_LOG)
    if frame is None or len(frame.payload) < 1:
        results.record(label, False, "no valid ERROR_LOG_RESP")
        return

    count = frame.payload[0]
    expected_len = 1 + count * 5
    if len(frame.payload) != expected_len:
        results.record(label, False, f"payload len {len(frame.payload)} != expected {expected_len}")
        return

    print(f"      {count} entr{'y' if count == 1 else 'ies'}")
    for i in range(count):
        slot, tile_cmd, err_type, ts_hi, ts_lo = frame.payload[1 + i * 5 : 6 + i * 5]
        name = ERROR_TYPE_NAMES.get(err_type, f"unknown(0x{err_type:02X})")
        print(f"        slot={slot} tile_cmd=0x{tile_cmd:02X} type={name} t={(ts_hi << 8) | ts_lo}s")
    results.record(label, True)


def check_re_discover(floor: Floor, row: int, results: Results) -> None:
    frame = _try_request(floor, row, Cmd.RE_DISCOVER)
    if frame is None or len(frame.payload) != 1:
        results.record("RE_DISCOVER ack", False, "no valid RE_DISCOVER_RESP")
        return
    results.record("RE_DISCOVER ack", frame.payload[0] == 0x00, f"status byte=0x{frame.payload[0]:02X}")

    deadline = time.monotonic() + 2.0
    final_state = None
    while time.monotonic() < deadline:
        resp = _try_request(floor, row, Cmd.STATUS)
        if resp is not None and resp.payload[0] != 0x01:  # not still discovering
            final_state = resp.payload[0]
            break
        time.sleep(0.05)

    if final_state is None:
        results.record("RE_DISCOVER completes", False, "still discovering after 2s")
    elif final_state == 0x02:
        results.record("RE_DISCOVER completes", True, "state=running, 0 tiles as expected")
    else:
        results.record("RE_DISCOVER completes", False, f"unexpected final state 0x{final_state:02X}")


def all_set_color_payload(r: int, g: int, b: int) -> bytes:
    """8 identical SET_COLOR entries, one per slot - 32 bytes, the smallest
    valid SEND_DATA payload (docs/row-bus-protocol.md §5.2)."""
    return bytes([TileCmd.SET_COLOR, r, g, b]) * 8


def check_no_response(floor: Floor, row: int, results: Results, name: str,
                      cmd: int, payload: bytes = b"", broadcast: bool = False) -> None:
    bus = floor.bus_for(row)
    if broadcast:
        floor.broadcast(cmd, payload)
    else:
        floor.send(row, cmd, payload)
    frame = bus.read_frame(timeout=0.15)
    if frame is not None:
        results.record(name, False, f"unexpected response cmd=0x{frame.cmd:02X}")
    else:
        results.record(name, True, "no response, as expected")


def check_responsive(floor: Floor, row: int, results: Results, label: str) -> None:
    ok = _try_request(floor, row, Cmd.STATUS) is not None
    results.record(label, ok, "" if ok else "no STATUS_RESP")


def check_address_filtering(floor: Floor, row: int, present: set[int], results: Results) -> None:
    """Address a row that shares this row's chain but isn't present, and
    confirm nobody answers. Must be same-chain to prove filtering rather
    than merely proving the frame went somewhere else entirely."""
    chain = floor.chain_map.chain_for(row)
    candidates = [r for r in floor.chain_map.rows_on(chain) if r not in present]
    if not candidates:
        results.record("ignores frames addressed elsewhere", True, "skipped - every row on this chain is present")
        return

    other = candidates[0]
    bus = floor.bus_for(row)
    floor.send(other, Cmd.STATUS)
    frame = bus.read_frame(timeout=0.15)
    ok = frame is None
    note = "" if ok else f"unexpected response from 0x{frame.addr:02X}"
    results.record(f"ignores frames addressed to 0x{other:02X} (not us, same chain)", ok, note)


def check_corrupt_crc(floor: Floor, row: int, results: Results) -> None:
    bus = floor.bus_for(row)
    corrupt = bytearray(Frame(row, Cmd.STATUS).encode())
    corrupt[-1] ^= 0xFF  # flip CRC_L so the frame fails its checksum
    bus.send_raw(bytes(corrupt))

    frame = bus.read_frame(timeout=0.15)
    ok = frame is None
    note = "" if ok else f"unexpected response cmd=0x{frame.cmd:02X} to a corrupt-CRC frame"
    results.record("drops frame with corrupt CRC", ok, note)


def check_row_controller(floor: Floor, row: int, present: set[int]) -> Results:
    chain = floor.chain_map.chain_for(row)
    print(f"\n=== Row controller 0x{row:02X} (chain {chain}) ===")
    results = Results()

    check_test(floor, row, results)
    check_status(floor, row, results)
    check_power(floor, row, results)
    check_error_log(floor, row, results, label="ERROR_LOG (before)")

    check_no_response(floor, row, results, "SEND_DATA (no ack)",
                      Cmd.SEND_DATA, all_set_color_payload(0x11, 0x22, 0x33))
    check_no_response(floor, row, results, "LATCH (no ack)", Cmd.LATCH, broadcast=True)
    check_responsive(floor, row, results, "responsive after SEND_DATA + LATCH")

    check_no_response(floor, row, results, "BLACKOUT (no ack)", Cmd.BLACKOUT, broadcast=True)
    check_responsive(floor, row, results, "responsive after BLACKOUT")

    check_no_response(floor, row, results, "unrecognized command is ignored", 0x77)
    check_responsive(floor, row, results, "responsive after unrecognized command")

    check_address_filtering(floor, row, present, results)
    check_corrupt_crc(floor, row, results)
    check_responsive(floor, row, results, "responsive after corrupt-CRC frame")

    check_error_log(floor, row, results, label="ERROR_LOG (after)")

    # Last: RE_DISCOVER rebuilds the sense map, leaving the board freshly
    # discovered at the end of the run.
    check_re_discover(floor, row, results)
    check_status(floor, row, results)

    print(f"  -- {results.summary()}")
    return results


def check_broadcast_admin_rejected(floor: Floor) -> Results:
    """docs/row-bus-protocol.md §3: broadcast "all row controllers accept;
    none respond". Row controller firmware enforces this in
    RowCommandHandler::handle() - a real assertion, not a note, now that
    it's a documented invariant rather than an open question.

    Goes around Floor.broadcast()'s own guard (RowBus.broadcast() directly,
    same trick as check_corrupt_crc's send_raw()) specifically to prove the
    row controller enforces this independently of the host being polite -
    Floor's guard alone wouldn't catch a bug here, since it would stop the
    frame from ever being sent.
    """
    print("\n=== Broadcast admin commands must get no response ===")
    results = Results()
    admin_cmds = [
        ("TEST", Cmd.TEST), ("STATUS", Cmd.STATUS), ("POWER", Cmd.POWER),
        ("RE_DISCOVER", Cmd.RE_DISCOVER), ("ERROR_LOG", Cmd.ERROR_LOG),
    ]
    for name, cmd in admin_cmds:
        for chain in range(floor.chain_count):
            bus = floor.bus_for_chain(chain)
            bus.broadcast(cmd)
            frame = bus.read_frame(timeout=0.15)
            ok = frame is None
            note = "" if ok else f"unexpected response from 0x{frame.addr:02X} on chain {chain}"
            results.record(f"broadcast {name} on chain {chain}: no response", ok, note)
    print(f"  -- {results.summary()}")
    return results


def parse_chain(spec: str, baudrate: int) -> ChainConfig:
    """PORT[:XDIR_GPIO] - e.g. /dev/ttyAMA0:17, or /dev/ttyAMA0:none"""
    port, _, xdir = spec.partition(":")
    if not xdir:
        raise argparse.ArgumentTypeError(f"--chain needs PORT:XDIR_GPIO, got {spec!r}")
    pin = None if xdir.lower() in ("none", "off") else int(xdir)
    return ChainConfig(port, pin, baudrate)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify whatever is live on the Row Bus")
    parser.add_argument("--chain", action="append", metavar="PORT:XDIR_GPIO",
                        help="repeatable; defaults to the two-chain hat's ttyAMA0:17 and ttyAMA2:7")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE,
                        help=f"default: {DEFAULT_BAUDRATE}")
    args = parser.parse_args()

    if args.chain:
        chains = [parse_chain(spec, args.baudrate) for spec in args.chain]
    else:
        chains = default_chain_configs(args.baudrate)

    chain_map = (
        RowChainMap.single_chain() if len(chains) == 1 else RowChainMap.alternating(len(chains))
    )

    for i, c in enumerate(chains):
        print(f"Chain {i}: {c.port} @ {c.baudrate} baud, XDIR={c.xdir_pin}, "
              f"rows {chain_map.rows_on(i)}")

    with Floor(chains=chains, chain_map=chain_map) as floor:
        print("\nScanning rows 0-7, each on its mapped chain...")
        found = floor.scan()
        if not found:
            print("FAIL: no row controller responded on any row/chain")
            return 1
        print(f"Found {len(found)} row controller(s): "
              + ", ".join(f"0x{r:02X}(chain {c})" for r, c in sorted(found.items())))

        present = set(found)
        all_results = [check_row_controller(floor, row, present) for row in sorted(found)]
        all_results.append(check_broadcast_admin_rejected(floor))

    overall_ok = all(r.ok for r in all_results)
    print("\n" + ("ALL CHECKS PASSED" if overall_ok else "SOME CHECKS FAILED"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
