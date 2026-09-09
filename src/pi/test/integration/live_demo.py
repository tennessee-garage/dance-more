#!/usr/bin/env python3
"""
Bring-up demo: discover whatever is live on the Row Bus, drive any tiles
found with SET_COLOR then SET_LEDS in a loop, and report health as it goes.

Unlike test_row_bus_scan.py (protocol conformance) and test_tile_stress.py
(throughput ceiling), this is a long-running bench tool: point it at
whatever hardware is plugged in and watch it. Runs until Ctrl+C, then
blacks out and exits.

Every health poll asks STATUS first, then POWER, then ERROR_LOG, and prints
whatever each one gives - deliberately, because "POWER did not answer" on
its own can't distinguish a row that has stopped executing from a row that
is alive but whose INA226 read is failing (the I2C side is known to be
noise-sensitive; see power_monitor_rp2350.cpp on the calibration register
resetting under electrical noise). STATUS needs no I2C and no Tile Bus, so
it answers that question directly.

When a row does go silent the loop stops driving it and watches for it to
come back, feeding the bus filler frames while it waits - see
await_recovery(), which used to go quiet instead and thereby prevented the
commonest recovery from happening at all. A row that stays silent through
that is *not* self-diagnosing: it means either something no watchdog can
recover, or that main.cpp's watchdog isn't resetting this part at all, and
nothing here separates those two. Read the under-load POWER samples for
that, not this timeout.

Steps:
  1. Scan rows 0-7, each on the chain that should carry it (Floor.scan()).
  2. STATUS each responder to see which tile slots it found, and take a
     baseline health poll with the LEDs still dark.
  3. Loop forever: cycle SET_COLOR through a few solid colors, then run a
     rotating-rainbow SET_LEDS animation, on every row found. Health poll
     every --stats-interval seconds (default 5).

Usage:
  # two-chain hat (defaults: ttyAMA0/GPIO17 + ttyAMA2/GPIO7)
  python3 test/integration/live_demo.py

  # single-chain bench board, XDIR still on GPIO23
  python3 test/integration/live_demo.py --chain /dev/ttyAMA0:23
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import time

from df2_pi.protocol.constants import (
    FRAME_OVERHEAD,
    LEDS_PER_TILE,
    MAX_PAYLOAD,
    Cmd,
    TileCmd,
)
from df2_pi.transport import ChainConfig, Floor, RowChainMap, RowNotResponding, default_chain_configs
from df2_pi.transport.row_bus import DEFAULT_BAUDRATE

NUM_LEDS = LEDS_PER_TILE
STATUS_STATE_NAMES = {0x00: "idle", 0x01: "discovering", 0x02: "running", 0x03: "error"}
TILE_STATUS_NAMES = {0x00: "not_discovered", 0x01: "ok", 0x02: "non_responsive", 0x03: "test_failed"}
ERROR_TYPE_NAMES = {0x01: "no_ack_after_retries", 0x02: "crc_failure", 0x03: "sense_collision",
                    0x04: "latch_overrun", 0x05: "row_bus_rx_overflow",
                    0x06: "row_boot", 0x08: "sense_start", 0x09: "tile_no_version"}

# Diagnostic entries whose slot/tile_cmd fields mean something other than the
# usual "tile slot / Tile Bus command", so the generic line would misread them.
#
# row_boot carries POWMAN_CHIP_RESET >> 16, where the RP2350 keeps its sticky
# reset causes. More than one can be set at once, so these are decoded as
# flags rather than a single reason.
RESET_CAUSES = [
    (0x0001, "power-on"),
    (0x0002, "brownout"),
    (0x0004, "RUN pin low"),
    (0x0008, "debug port reset"),
    (0x0020, "rescue"),
    (0x0040, "watchdog(powman async)"),
    (0x0080, "watchdog(powman)"),
    (0x0100, "watchdog(swcore)"),
    (0x0200, "switched-core powerdown"),
    (0x0400, "glitch detect"),
    (0x0800, "hazard sys reset"),
    (0x1000, "watchdog(rsm)"),
]


def status_uptime_s(status: bytes) -> int | None:
    """Row uptime in seconds from a STATUS_RESP, or None on older firmware.

    Appended after the 8 tile-status bytes, so a payload shorter than 14 is a
    row that predates the field rather than a malformed reply.
    """
    if len(status) < 14:
        return None
    return (status[10] << 24) | (status[11] << 16) | (status[12] << 8) | status[13]


def fmt_uptime(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def reset_causes(bits: int) -> str:
    names = [name for bit, name in RESET_CAUSES if bits & bit]
    return ", ".join(names) if names else f"none set (0x{bits:04X})"

# Fractions of the commanded level (--brightness), not absolute values, so
# one flag scales the solid colors and the rainbow together. White is held
# lower because it drives all three channels at once.
COLOR_CYCLE = [("RED", (1.0, 0.0, 0.0)), ("GREEN", (0.0, 1.0, 0.0)), ("BLUE", (0.0, 0.0, 1.0)),
               ("WHITE", (0.75, 0.75, 0.75)), ("OFF", (0.0, 0.0, 0.0))]
COLOR_HOLD_S = 1.0
LEDS_PHASE_S = 6.0
DEFAULT_LEDS_FPS = 30.0

# How long to keep watching a silent row before giving up on it. The row
# controller's watchdog fires at 500 ms (plus a 250 ms core-1 stall window),
# and a reboot re-runs discovery, so anything that is coming back is back
# well inside this.
RECOVERY_LIMIT_S = 8.0

# Filler frames sent per recovery attempt, sized to cover the largest payload
# a row's parser can be left owing after a truncated frame. See
# await_recovery(): the row cannot resync until it has been fed the balance,
# so the count has to beat the worst case rather than merely be generous.
FLUSH_FRAMES = -(-(MAX_PAYLOAD + 2) // FRAME_OVERHEAD)  # ceil, 182 at 60 LEDs

# Rail drop from the dark baseline worth calling out. The row controller
# runs off the same 12 V it measures, so a sag this size is the thing to
# suspect when it stops answering the moment LEDs light.
SAG_THRESHOLD_MV = 300

# Let the LED current settle before sampling it, matching the settle used in
# tile_brightness_sweep.py.
LOAD_SETTLE_S = 0.25


# ---- Power-based tile verification ----
#
# Display commands carry no ACK by design (docs/tile-bus-protocol.md), so a
# tile that has stopped rendering is invisible to every other channel here:
# STATUS reports the row's cached tile map, which is frozen at discovery, and
# the row never re-checks a tile afterwards. Measured supply current is the
# only signal that comes from the tiles themselves rather than from the row
# talking about them.
#
# Measured on the bench 2026-09-08, one row + two tiles: LED current tracks
# the sum over LEDs of max(r,g,b) - NOT the sum of the channels. (200,200,200)
# draws the same as (200,0,0); (255,255,255) the same as (255,0,0). On that
# governing channel it is linear to better than 1%: predictions matched
# measurement within 2 mA of 427 across solid, gradient and rainbow frames.
# Mechanism unconfirmed - the WS2815 datasheet has not been checked - but the
# relationship is solid enough to predict from.
#
# Consequence worth knowing: a full-saturation rotating rainbow has max=level
# on every LED, so its predicted draw is constant and a frozen tile looks
# identical to a live one. The SET_COLOR phase is what makes a freeze visible,
# because its commanded draw actually moves.
POWER_VERIFY_TOLERANCE_MA = 60   # ~14% of one tile at level 200; noise is ~2 mA
POWER_CAL_LEVEL = 150            # calibration brightness, well clear of any limit
POWER_VERIFY_CONFIRM_S = 0.30    # re-read gap before believing a mismatch


def per_tile_max_sum(pixels: list[tuple[int, int, int]]) -> int:
    """Sum of max(r,g,b) over one tile's LEDs - the quantity current tracks."""
    return sum(max(px) for px in pixels)


# Row uptime at the previous health poll, for reboot detection.
LAST_UPTIME: dict[int, int] = {}

# Filled by calibrate_power(); None until then, which disables the check.
POWER_CAL: dict[str, float | None] = {"dark_ma": None, "ma_per_unit": None}
LAST_FRAME_PER_TILE_MAX = 0


def set_color_payload(fractions: tuple[float, float, float], level: int) -> bytes:
    r, g, b = (int(f * level) for f in fractions)
    return bytes([TileCmd.SET_COLOR, r, g, b]) * 8


def wheel(pos: float, level: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(pos % 1.0, 1.0, 1.0)
    return (int(r * level), int(g * level), int(b * level))


def set_leds_payload(pixels: list[tuple[int, int, int]]) -> bytes:
    body = b"".join(bytes(p) for p in pixels)
    return (bytes([TileCmd.SET_LEDS]) + body) * 8


def probe(floor: Floor, row: int, cmd: int) -> bytes | None:
    """One admin command; its payload, or None if the row didn't answer."""
    try:
        return floor.request(row, cmd).payload
    except RowNotResponding:
        return None


def show_frame(floor: Floor, rows: list[int], payload: bytes, latch_delay_s: float,
               per_tile_max: int = 0) -> None:
    """SEND_DATA to every row, then LATCH once.

    The gap before LATCH lets each row finish forwarding to its tiles first.
    Without it every single frame logs a LATCH_OVERRUN (expected per
    docs/row-bus-protocol.md §5.2, and harmless in itself) - but at these
    frame rates that buries every other entry in the row's 32-deep error log
    within a second, which is exactly the log this script exists to read.

    The default was raised from 5 ms to 15 ms after measuring the boundary on
    the bench with one tile: at 5 ms every SET_LEDS frame overran while every
    SET_COLOR frame was clean, and at 10 ms both were clean. That the two
    commands differ is the useful part - a SET_LEDS tile frame is 187 bytes
    (~1.9 ms at 1 Mbps) against SET_COLOR's 9, so what a row needs here
    tracks the Tile Bus write, not the fixed cost of walking 8 slots. Which
    means this figure does *not* generalise: a row with 8 discovered tiles
    writes 8 such frames, so budget accordingly rather than assuming 15 ms
    stays sufficient.
    """
    for row in rows:
        floor.send_data(row, payload)
    if latch_delay_s:
        time.sleep(latch_delay_s)
    floor.latch()
    # Remember what is on the tiles so verify_power() can predict the draw it
    # should be causing. Set after the LATCH, so a frame that was sent but
    # never latched is not what we check against.
    global LAST_FRAME_PER_TILE_MAX
    LAST_FRAME_PER_TILE_MAX = per_tile_max


def read_power(floor: Floor, row: int) -> tuple[int, int, int] | None:
    """(voltage_mV, current_mA, power_mW) from the row's INA226, or None."""
    p = probe(floor, row, Cmd.POWER)
    if p is None:
        return None
    return ((p[0] << 8) | p[1], (p[2] << 8) | p[3], (p[4] << 8) | p[5])


def sample_under_load(floor: Floor, rows: list[int], label: str,
                      baseline_mV: dict[int, int]) -> None:
    """Read the rail while a colour is actually lit.

    The scheduled health poll is on a wall clock and keeps landing during
    OFF, which is how two runs of this went by without ever measuring the
    supply under load. The row controller taps 12 V at the tile's own
    injection point (docs/power.md), so its INA226 sees the sag the LED
    current causes on the rail it is itself running from - which makes this
    reading, not the dark one, the interesting one.
    """
    for row in rows:
        reading = read_power(floor, row)
        if reading is None:
            print(f"      row 0x{row:02X}: POWER did not answer under {label}")
            continue
        v, c, _ = reading
        base = baseline_mV.get(row)
        sag = f"  (sag {base - v} mV from dark)" if base is not None and base - v > SAG_THRESHOLD_MV else ""
        print(f"      row 0x{row:02X}: {v} mV  {c} mA under {label}{sag}")


def count_new_entries(previous: list[tuple] | None, current: list[tuple]) -> int | None:
    """How many of `current` were appended since `previous` was read.

    The row's log is an append-only 32-deep ring emitted oldest-first, so
    once it is full a later read is the earlier one shifted left by however
    many entries were added. Recovering that shift is what separates "the
    row logged nothing" from "the row logged the same thing again", which
    reading raw timestamps cannot do: entries persist across runs of this
    script, and a full ring of stale entries looks exactly like a full ring
    of fresh ones. Returns None on the first poll, where there is no
    baseline and every entry predates us.
    """
    if previous is None:
        return None
    for shift in range(len(previous) + 1):
        kept = len(previous) - shift
        if previous[shift:] == current[:kept]:
            return len(current) - kept
    return len(current)  # no overlap at all - the whole log turned over


def print_error_log(row: int, payload: bytes, last_log: dict[int, list[tuple]]) -> None:
    count = payload[0]
    entries = [tuple(payload[1 + i * 5 : 6 + i * 5]) for i in range(count)]

    # Timestamps are seconds since the row controller booted, so the newest
    # one is a lower bound on its uptime. If that goes backwards between
    # polls, the row rebooted while we weren't looking - the fingerprint of
    # a watchdog reset or a brownout, neither of which announces itself.
    newest = max((e[3] << 8) | e[4] for e in entries) if entries else None
    previous = last_log.get(row)
    prev_newest = max((e[3] << 8) | e[4] for e in previous) if previous else None
    if newest is not None and prev_newest is not None and newest < prev_newest:
        # Fallback for firmware without STATUS uptime; health_poll reports the
        # reboot directly when the field is present.
        if row not in LAST_UPTIME:
            print(f"    row 0x{row:02X} REBOOTED since the last poll "
                  f"(error-log clock went {prev_newest}s -> {newest}s)")
        previous = None  # the ring restarted; nothing carries across

    n_new = count_new_entries(previous, entries)
    last_log[row] = entries

    if count == 0:
        print(f"    error log: empty")
        return

    kinds: dict[int, int] = {}
    for e in entries:
        kinds[e[2]] = kinds.get(e[2], 0) + 1
    summary = ", ".join(f"{ERROR_TYPE_NAMES.get(k, hex(k))}={v}" for k, v in sorted(kinds.items()))

    # This clock is a lower bound that stops advancing the moment the row stops
    # logging, so it is kept only to date entries against each other. Real
    # uptime now comes from STATUS_RESP and is printed by health_poll; the
    # explicit new-entry count is what stops a full ring of stale entries
    # reading as a fresh failure.
    if n_new is None:
        freshness = "all predate this run"
    elif n_new == 0:
        freshness = "0 new since last poll - all stale"
    else:
        freshness = f"{n_new} new since last poll"
    print(f"    error log: {count} entr{'y' if count == 1 else 'ies'} "
          f"({summary}); row clock >= {newest}s; {freshness}")

    for i, (slot, tile_cmd, err_type, ts_hi, ts_lo) in enumerate(entries):
        name = ERROR_TYPE_NAMES.get(err_type, f"unknown(0x{err_type:02X})")
        t = (ts_hi << 8) | ts_lo
        mark = "NEW " if n_new is not None and i >= count - n_new else "    "
        if err_type == 0x06:
            print(f"      {mark}{name}: {reset_causes((slot << 8) | tile_cmd)} t={t}s")
        elif err_type == 0x08:
            print(f"      {mark}{name}: sweep #{slot} t={t}s")
        elif err_type == 0x09:
            print(f"      {mark}{name}: slot {slot} (addr 0x{tile_cmd:02X}) "
                  f"mapped but silent t={t}s")
        else:
            print(f"      {mark}slot={slot} tile_cmd=0x{tile_cmd:02X} type={name} t={t}s")


def calibrate_power(floor: Floor, rows: list[int], tiles: int, latch_delay_s: float) -> bool:
    """Learn this rig's dark draw and mA per unit of commanded max-channel.

    Measured rather than hardcoded because the coefficient is a property of
    the strips actually attached - LED count, and whatever the WS2815 is
    really doing across channels. Two points are enough: the relationship is
    linear through the origin to better than 1%.
    """
    if not tiles:
        return False
    dark = [(0, 0, 0)] * NUM_LEDS
    lit = [(POWER_CAL_LEVEL, 0, 0)] * NUM_LEDS

    show_frame(floor, rows, set_leds_payload(dark), latch_delay_s, per_tile_max_sum(dark))
    time.sleep(LOAD_SETTLE_S)
    dark_reading = read_power(floor, rows[0])

    show_frame(floor, rows, set_leds_payload(lit), latch_delay_s, per_tile_max_sum(lit))
    time.sleep(LOAD_SETTLE_S)
    lit_reading = read_power(floor, rows[0])

    show_frame(floor, rows, set_leds_payload(dark), latch_delay_s, per_tile_max_sum(dark))
    time.sleep(LOAD_SETTLE_S)   # don't hand the caller a rail still settling

    if dark_reading is None or lit_reading is None:
        print("    power check: calibration failed - POWER did not answer")
        return False

    units = per_tile_max_sum(lit) * tiles
    delta = lit_reading[1] - dark_reading[1]
    if delta <= 0 or not units:
        print(f"    power check: calibration failed - lighting {tiles} tile(s) "
              f"changed current by {delta} mA; nothing is rendering")
        return False

    POWER_CAL["dark_ma"] = dark_reading[1]
    POWER_CAL["ma_per_unit"] = delta / units
    per_tile_full = POWER_CAL["ma_per_unit"] * NUM_LEDS * 255
    print(f"    power check: calibrated - dark {dark_reading[1]} mA, "
          f"{delta} mA for {tiles} tile(s) at level {POWER_CAL_LEVEL} "
          f"(~{per_tile_full:.0f} mA per tile at full brightness)")
    return True


def predicted_ma(tiles: int) -> float | None:
    """What the rail should be drawing for the frame currently latched."""
    if POWER_CAL["ma_per_unit"] is None:
        return None
    return POWER_CAL["dark_ma"] + POWER_CAL["ma_per_unit"] * LAST_FRAME_PER_TILE_MAX * tiles


def verify_power(floor: Floor, row: int, tiles: int) -> bool:
    """Compare measured draw against the frame we commanded.

    True if it matches (or the check is not calibrated). A mismatch is the
    only evidence available that a tile stopped rendering - see the note on
    POWER_VERIFY_TOLERANCE_MA.
    """
    expected = predicted_ma(tiles)
    if expected is None:
        return True
    reading = read_power(floor, row)
    if reading is None:
        return True  # a silent row is health_poll's business, not ours
    measured = reading[1]
    err = measured - expected
    if abs(err) <= POWER_VERIFY_TOLERANCE_MA:
        return True

    # Confirm before believing it. The first bench run of this check reported
    # both tiles dark on the very first colour, reading exactly the previous
    # frame's draw, and both were fine a second later - a stale sample, not a
    # fault. Frame delivery itself was then measured at 0 losses in 125
    # trials across inter-frame gaps from 0 to 33 ms, so a single disagreeing
    # sample is far likelier to be the reading than the tiles. Isolating
    # tiles blanks the floor, so it must not run on one sample.
    time.sleep(POWER_VERIFY_CONFIRM_S)
    reading = read_power(floor, row)
    if reading is None:
        return True
    measured = reading[1]
    err = measured - expected
    if abs(err) <= POWER_VERIFY_TOLERANCE_MA:
        return True
    share = POWER_CAL["ma_per_unit"] * LAST_FRAME_PER_TILE_MAX or 1
    print(f"    row 0x{row:02X}: POWER MISMATCH - expected ~{expected:.0f} mA for the "
          f"commanded frame, measured {measured} mA ({err:+.0f}, "
          f"~{abs(err) / share:.1f} tile-equivalents)")
    return False


def identify_dark_tiles(floor: Floor, row: int, tiles: int, latch_delay_s: float) -> None:
    """Light each tile alone and name the ones that don't draw their share.

    Only worth the disruption once verify_power() has already failed. Per-tile
    draw was measured additive to within 1 mA on the bench, so attribution is
    unambiguous.
    """
    if POWER_CAL["ma_per_unit"] is None:
        return
    dark = [(0, 0, 0)] * NUM_LEDS
    lit = [(POWER_CAL_LEVEL, 0, 0)] * NUM_LEDS
    expect = POWER_CAL["ma_per_unit"] * per_tile_max_sum(lit)
    print(f"    isolating tiles (expect ~{expect:.0f} mA each)...")

    for slot in range(tiles):
        body = b"".join(
            bytes([TileCmd.SET_LEDS]) + b"".join(bytes(p) for p in (lit if i == slot else dark))
            for i in range(8)
        )
        show_frame(floor, [row], body, latch_delay_s, 0)
        time.sleep(LOAD_SETTLE_S + POWER_VERIFY_CONFIRM_S)
        reading = read_power(floor, row)
        if reading is None:
            print(f"      slot {slot}: POWER did not answer")
            continue
        delta = reading[1] - POWER_CAL["dark_ma"]
        verdict = "ok" if delta > expect * 0.5 else "NOT RENDERING"
        print(f"      slot {slot}: {delta:+.0f} mA  {verdict}")

    show_frame(floor, [row], set_leds_payload(dark), latch_delay_s, 0)


def health_poll(floor: Floor, row: int, last_log: dict[int, list[tuple]]) -> bool:
    """STATUS, POWER and ERROR_LOG for one row. False if the row is silent.

    STATUS goes first and its failure is what defines "silent": it is
    answered from the sense map with no I2C and no Tile Bus round trip, so
    a row that can't answer it isn't executing.
    """
    status = probe(floor, row, Cmd.STATUS)
    if status is None:
        print(f"    row 0x{row:02X}: SILENT - no STATUS_RESP")
        return False

    state = STATUS_STATE_NAMES.get(status[0], hex(status[0]))
    uptime = status_uptime_s(status)
    up = f" up {fmt_uptime(uptime)}" if uptime is not None else ""
    print(f"    row 0x{row:02X}: state={state} tiles_found={status[1]}{up}")

    # Uptime going backwards is a reboot, full stop - no inference from log
    # timestamps needed. Those only move while the row is logging, which on a
    # healthy row is almost never, so they could never answer this.
    previous = LAST_UPTIME.get(row)
    if uptime is not None:
        if previous is not None and uptime < previous:
            print(f"    row 0x{row:02X}: REBOOTED since the last poll "
                  f"(uptime {fmt_uptime(previous)} -> {fmt_uptime(uptime)}) - "
                  f"read the row_boot entry below for the cause")
        LAST_UPTIME[row] = uptime

    reading = read_power(floor, row)
    if reading is None:
        # Alive (STATUS answered) but POWER didn't: the fault is in the
        # power monitor path, not the row.
        print(f"    row 0x{row:02X}: POWER did not answer, though STATUS did "
              f"- INA226/I2C side, row is alive")
    else:
        v, c, w = reading
        print(f"    row 0x{row:02X}: {v} mV  {c} mA  {w} mW")

    errors = probe(floor, row, Cmd.ERROR_LOG)
    if errors is None:
        print(f"    row 0x{row:02X}: ERROR_LOG did not answer")
    else:
        print_error_log(row, errors, last_log)
    return True


def await_recovery(floor: Floor, row: int, limit_s: float = RECOVERY_LIMIT_S) -> float | None:
    """Seconds until the row answers STATUS again, or None if it never does.

    This used to poll STATUS on an otherwise quiet bus, on the reasoning that
    silence isolates the row from traffic it might be losing replies to. That
    was exactly backwards for the most common cause of a silent row: a frame
    truncated on the wire leaves the row's parser owing the rest of a payload,
    and only bytes can pay it off, so a quiet bus is the one condition under
    which it can never recover. Eight seconds of politeness guaranteed the
    failure it was trying to observe.

    So the probes are now padded with filler. BLACKOUT is the vehicle: it is
    the largest thing here that is safe to send to a row in an unknown state -
    it drives no pixel data, and a row that does receive one just goes dark,
    which it already is. Enough of them to cover a full SEND_DATA payload
    clears any stuck parser within the first second.
    """
    start = time.perf_counter()
    while time.perf_counter() - start < limit_s:
        if probe(floor, row, Cmd.STATUS) is not None:
            return time.perf_counter() - start
        for _ in range(FLUSH_FRAMES):
            floor.blackout()  # broadcast: writes only, never waits for a reply
        time.sleep(0.1)
    return None


def discover(floor: Floor) -> dict[int, int]:
    print("Scanning rows 0-7, each on its mapped chain...")
    found = floor.scan()
    if not found:
        print("no row controller responded on any row/chain")
        return found

    print(f"found {len(found)} row controller(s): "
          + ", ".join(f"0x{r:02X}(chain {c})" for r, c in sorted(found.items())))

    for row in sorted(found):
        status = probe(floor, row, Cmd.STATUS)
        if status is None:
            print(f"  row 0x{row:02X}: STATUS did not answer")
            continue
        tile_status = [TILE_STATUS_NAMES.get(s, hex(s)) for s in status[2:10]]
        print(f"  row 0x{row:02X}: state={STATUS_STATE_NAMES.get(status[0], hex(status[0]))} "
              f"tiles_found={status[1]} tile_status={tile_status}")

    return found


def drive(floor: Floor, rows: list[int], stats_interval: float, latch_delay_s: float,
          level: int, leds_fps: float, rediscover_every: float) -> None:
    last_log: dict[int, list[tuple]] = {}
    active = list(rows)

    print("\nBaseline health, LEDs still dark:")
    baseline_mV: dict[int, int] = {}
    for row in active:
        health_poll(floor, row, last_log)
        reading = read_power(floor, row)
        if reading is not None:
            baseline_mV[row] = reading[0]

    print(f"\nDriving {len(active)} row(s): {', '.join(f'0x{r:02X}' for r in active)}")

    # Total tiles across every driven row: the predicted draw scales with it,
    # and it comes from STATUS rather than the payload because a payload
    # always carries all 8 slots while only discovered ones are forwarded.
    tiles = 0
    for row in active:
        st = probe(floor, row, Cmd.STATUS)
        if st:
            tiles += st[1]
    if tiles:
        calibrate_power(floor, active, tiles, latch_delay_s)
    else:
        print("    power check: no tiles discovered - nothing to verify")
    print("Ctrl+C to stop.\n")

    next_stats = time.perf_counter() + stats_interval
    next_rediscover = time.perf_counter() + rediscover_every if rediscover_every else None

    def maybe_rediscover() -> None:
        """Re-run discovery while the LEDs are being driven.

        The row only mis-discovers when a sweep runs while the tile is
        pushing its strip - on a normal boot that is the 3.6 s start-up
        animation, which is pure luck to catch. Driving frames keeps the tile
        pushing indefinitely, so a sweep forced in here reproduces the same
        condition on demand. Watch the error log for sense_extra_slot."""
        nonlocal next_rediscover
        if next_rediscover is None or time.perf_counter() < next_rediscover:
            return
        next_rediscover = time.perf_counter() + rediscover_every
        for row in active:
            probe(floor, row, Cmd.RE_DISCOVER)
        print(f"   [RE_DISCOVER sent while driving]", flush=True)

    def maybe_stats() -> None:
        """Health poll on schedule. Drops any row that goes silent and
        doesn't come back, so the run continues on whatever still works."""
        nonlocal next_stats
        if time.perf_counter() < next_stats:
            return
        print(f"\n  -- health @ {time.strftime('%H:%M:%S')} --")
        for row in list(active):
            if health_poll(floor, row, last_log):
                continue
            # Silent. Stop driving it and let the bus go quiet before
            # deciding whether it is stalled (watchdog will reboot it) or
            # gone (needs a power cycle).
            active.remove(row)
            print(f"    row 0x{row:02X}: display data stopped, watching for recovery...")
            recovery = await_recovery(floor, row)
            if recovery is None:
                print(f"    row 0x{row:02X}: still silent after {RECOVERY_LIMIT_S:.0f}s "
                      f"with the bus quiet - it stopped executing and did not come back.")
                print(f"    row 0x{row:02X}: cause not established from this alone. Either "
                      f"something the watchdog cannot recover (electrical - rail sag, "
                      f"noise, latch-up), or a stall plus a watchdog that isn't resetting "
                      f"this part. The under-load readings above are the evidence to "
                      f"read: a rail that held its voltage rules out the first.")
            else:
                print(f"    row 0x{row:02X}: came back after {recovery:.2f}s "
                      f"(watchdog reboot) - resuming")
                active.append(row)
                health_poll(floor, row, last_log)
        next_stats = time.perf_counter() + stats_interval

    while True:
        if not active:
            print("\nno rows left answering - stopping. Power-cycle the bench and re-run.")
            return

        print("-- SET_COLOR phase --")
        for name, fractions in COLOR_CYCLE:
            print(f"   {name}", flush=True)
            colour = tuple(int(f * level) for f in fractions)
            show_frame(floor, active, set_color_payload(fractions, level), latch_delay_s,
                       max(colour) * NUM_LEDS)
            time.sleep(LOAD_SETTLE_S)
            sample_under_load(floor, active, name, baseline_mV)
            # Checked here, not in the rainbow: a full-saturation rainbow has
            # max = level on every LED, so its predicted draw never moves and
            # a frozen tile is indistinguishable from a live one. The colour
            # steps are what make a freeze visible.
            for row in list(active):
                if not verify_power(floor, row, tiles):
                    identify_dark_tiles(floor, row, tiles, latch_delay_s)
            time.sleep(max(0.0, COLOR_HOLD_S - LOAD_SETTLE_S))
            maybe_rediscover()
            maybe_stats()
            if not active:
                break

        if not active:
            continue

        print(f"-- SET_LEDS phase (rotating rainbow, {leds_fps:g} fps) --")
        phase_end = time.perf_counter() + LEDS_PHASE_S
        period = 1.0 / leds_fps
        phase = 0.0
        while time.perf_counter() < phase_end and active:
            due = time.perf_counter() + period
            pixels = [wheel(i / NUM_LEDS + phase, level) for i in range(NUM_LEDS)]
            show_frame(floor, active, set_leds_payload(pixels), latch_delay_s,
                       per_tile_max_sum(pixels))
            phase += 0.015
            maybe_stats()
            remaining = due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)


def parse_chain(spec: str, baudrate: int) -> ChainConfig:
    port, _, xdir = spec.partition(":")
    if not xdir:
        raise argparse.ArgumentTypeError(f"--chain needs PORT:XDIR_GPIO, got {spec!r}")
    pin = None if xdir.lower() in ("none", "off") else int(xdir)
    return ChainConfig(port, pin, baudrate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover the Row Bus, drive whatever tiles are found, and monitor")
    parser.add_argument("--chain", action="append", metavar="PORT:XDIR_GPIO",
                        help="repeatable; defaults to the two-chain hat's ttyAMA0:17 and ttyAMA2:7")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE,
                        help=f"default: {DEFAULT_BAUDRATE}")
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        help="seconds between STATUS/POWER/ERROR_LOG polls (default: 5)")
    parser.add_argument("--latch-delay", type=float, default=0.015,
                        help="seconds between SEND_DATA and LATCH, so rows finish forwarding "
                             "before latching and the error log isn't buried in "
                             "LATCH_OVERRUN entries (default: 0.015; 0 to latch immediately)")
    parser.add_argument("--rediscover-every", type=float, default=0.0, metavar="SECONDS",
                        help="re-run discovery this often while driving LEDs (default: off). "
                             "Reproduces the mis-discovery on demand: the row only claims "
                             "phantom tiles when a sweep overlaps the tile driving its strip")
    parser.add_argument("--leds-fps", type=float, default=DEFAULT_LEDS_FPS, metavar="FPS",
                        help="rate for the SET_LEDS rainbow (default: 30). A SET_LEDS frame is "
                             "187 bytes on Tile Bus against a 9-byte SET_COLOR, so if the "
                             "rainbow is missing, run this at 1 to separate a per-frame "
                             "failure (still missing) from a rate one (comes back)")
    parser.add_argument("--brightness", type=int, default=200, metavar="0-255",
                        help="caps every color this drives; turn it down to test whether a "
                             "dropout follows LED current (default: 200)")
    args = parser.parse_args()

    if not 0 <= args.brightness <= 255:
        parser.error("--brightness must be 0-255")

    chains = ([parse_chain(c, args.baudrate) for c in args.chain]
              if args.chain else default_chain_configs(args.baudrate))
    chain_map = RowChainMap.single_chain() if len(chains) == 1 else RowChainMap.alternating(len(chains))

    for i, c in enumerate(chains):
        print(f"Chain {i}: {c.port} @ {c.baudrate} baud, XDIR={c.xdir_pin}, "
              f"rows {chain_map.rows_on(i)}")

    with Floor(chains=chains, chain_map=chain_map) as floor:
        found = discover(floor)
        if not found:
            return 1

        try:
            drive(floor, sorted(found), args.stats_interval, args.latch_delay,
                  args.brightness, args.leds_fps, args.rediscover_every)
        except KeyboardInterrupt:
            print("\nstopping...")
        finally:
            floor.blackout()

    return 0


if __name__ == "__main__":
    sys.exit(main())
