"""#106 verification on row 0 (row v9, tiles v5; strips on slots 0 and 1)."""
import random, statistics, sys, time
from df2_pi.effects import Effect, SHIMMER, CHASE, NONE
from df2_pi.protocol.constants import Cmd, TileCmd, LEDS_PER_TILE
from df2_pi.transport import Floor, RowChainMap, default_chain_configs

random.seed(106)
def amps(floor):
    p = floor.request(0, Cmd.POWER).payload
    return (p[2] << 8) | p[3]
def amps_med(floor, n=3):
    v = []
    for _ in range(n): v.append(amps(floor)); time.sleep(0.03)
    return statistics.median(v)
def frame(floor, payload, settle=0.25):
    floor.send_data(0, payload); time.sleep(0.02); floor.latch(); time.sleep(settle)
color  = lambda v: bytes([TileCmd.SET_COLOR, v, v, v]) * 8
leds   = lambda v: (bytes([TileCmd.SET_LEDS]) + bytes([v]) * (3 * LEDS_PER_TILE)) * 8
effect = lambda e: (bytes([TileCmd.SET_EFFECT]) + bytes(e)) * 8
CHASE_FX   = Effect(CHASE, (0, 255, 0, 5))        # red, 40 ms/step
SHIMMER_FX = Effect(SHIMMER, (255, 255, 0, 0))    # 4.25 Hz, full depth, in unison
def clean(floor):
    for _ in range(3):
        frame(floor, effect(Effect(NONE)), 0.05); frame(floor, leds(0), 0.05)
    floor.blackout(); time.sleep(0.3)
def out(msg): print(msg, flush=True)

def stream(floor, payload_of, n):
    """n frames at 30 fps, alternating 120/40, pipelined like FrameClock.
    Returns the colour of the last frame latched."""
    period = 1 / 30; deadline = time.perf_counter() + period
    floor.send_data(0, payload_of(120)); shown = 120
    for f in range(1, n + 1):
        while time.perf_counter() < deadline: pass
        floor.latch()
        shown = 120 if (f - 1) % 2 == 0 else 40
        if f < n: floor.send_data(0, payload_of(120 if f % 2 == 0 else 40))
        deadline += period
    return shown

with Floor(chains=default_chain_configs(), chain_map=RowChainMap.alternating(2)) as floor:
    clean(floor); idle = amps_med(floor); out(f"idle {idle} mA")
    which = sys.argv[1:] or ["blackout", "stream", "slow", "control"]

    if "blackout" in which:
        # 1a. CHASE: a miss leaves the chase overlay lit.
        lit = 0
        for _ in range(100):
            clean(floor)
            frame(floor, color(0), 0.05); frame(floor, effect(CHASE_FX), random.uniform(0.3, 0.7))
            floor.blackout(); time.sleep(0.3)
            lit += amps_med(floor) > idle + 40
        out(f"1a. BLACKOUT over CHASE, random phase: tiles left lit {lit}/100")
        # 1b. SHIMMER over grey: a miss leaves the tile lit, or leaves SHIMMER
        # running - seen as the current swinging once we paint grey again.
        lit = survived = 0
        for _ in range(100):
            clean(floor)
            frame(floor, color(60), 0.05); frame(floor, effect(SHIMMER_FX), random.uniform(0.3, 0.7))
            floor.blackout(); time.sleep(0.3)
            lit += amps_med(floor) > idle + 40
            frame(floor, color(60), 0.05)
            s = []
            for _ in range(12): s.append(amps(floor)); time.sleep(0.04)
            survived += (max(s) - min(s)) > 60
        out(f"1b. BLACKOUT over SHIMMER, random phase: tiles left lit {lit}/100, SHIMMER still running {survived}/100")

    if "stream" in which:
        # 2. 30 fps streams with CHASE running; check the final frame.
        for label, payload_of in (("SET_COLOR", color), ("SET_LEDS", leds)):
            clean(floor)
            frame(floor, payload_of(40)); frame(floor, effect(CHASE_FX), 0.3)
            his, los = [], []
            for _ in range(3):
                frame(floor, payload_of(120)); his.append(amps_med(floor))
                frame(floor, payload_of(40));  los.append(amps_med(floor))
            hi, lo = statistics.median(his), statistics.median(los); tol = (hi - lo) / 4
            wrong = 0
            for trial in range(100):
                shown = stream(floor, payload_of, 30 + trial % 2)
                time.sleep(0.25)
                a = amps_med(floor, 2)
                wrong += abs(a - (hi if shown == 120 else lo)) > tol
            out(f"2. 30 fps {label} stream, CHASE running: refs {lo:.0f}/{hi:.0f} mA, wrong final frame {wrong}/100")

    if "slow" in which:
        # 3. Isolated frames (~3 fps) with CHASE running: not streaming, so the
        # tile free-runs - the residual loss the fix does not cover.
        for label, payload_of in (("SET_COLOR", color), ("SET_LEDS", leds)):
            clean(floor)
            frame(floor, payload_of(40)); frame(floor, effect(CHASE_FX), 0.3)
            his, los = [], []
            for _ in range(3):
                frame(floor, payload_of(120)); his.append(amps_med(floor))
                frame(floor, payload_of(40));  los.append(amps_med(floor))
            hi, lo = statistics.median(his), statistics.median(los); tol = (hi - lo) / 4
            lost = 0
            for i in range(60):
                v = 120 if i % 2 == 0 else 40
                frame(floor, payload_of(v), 0.15)
                lost += abs(amps_med(floor, 2) - (hi if v == 120 else lo)) > tol
            out(f"3. slow {label} frames, CHASE running: frames not shown {lost}/60")

    if "control" in which:
        # 4. No effect: 30 fps streams and slow frames must be perfect.
        for label, payload_of in (("SET_COLOR", color), ("SET_LEDS", leds)):
            clean(floor)
            frame(floor, payload_of(120)); hi = amps_med(floor)
            frame(floor, payload_of(40));  lo = amps_med(floor)
            tol = (hi - lo) / 4; wrong = 0
            for trial in range(30):
                shown = stream(floor, payload_of, 30 + trial % 2); time.sleep(0.25)
                wrong += abs(amps_med(floor, 2) - (hi if shown == 120 else lo)) > tol
            out(f"4. control, no effect, 30 fps {label}: wrong final frame {wrong}/30")
    clean(floor); out(f"final {amps_med(floor)} mA (idle {idle})")
