"""Deaf window after LATCH, per effect, with the host streaming at 30 fps so a
v5 tile makes no own-clock pushes. SHIMMER at depth 0 does its full render but
leaves the buffer untouched, so current still reads the colour."""
import statistics, time
from df2_pi.effects import Effect, SHIMMER, HUE_SPLIT, NONE
from df2_pi.protocol.constants import Cmd, TileCmd
from df2_pi.transport import Floor, RowChainMap, default_chain_configs
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
effect = lambda e: (bytes([TileCmd.SET_EFFECT]) + bytes(e)) * 8
def busy_until(t):
    while time.perf_counter() < t: pass
def clean(floor):
    for _ in range(3):
        frame(floor, effect(Effect(NONE)), 0.05); frame(floor, color(0), 0.05)
    floor.blackout(); time.sleep(0.3)

def trial(floor, delay_s, final_v):
    """8 frames at 30 fps, each frame's data sent delay_s after the previous
    LATCH; returns after latching the last one."""
    t = time.perf_counter()
    for i in range(8):
        busy_until(t); floor.latch()
        busy_until(t + delay_s)
        v = final_v if i == 7 else (120 if (i + final_v) % 2 else 40)
        floor.send_data(0, color(v))
        t += 1 / 30
    busy_until(t); floor.latch()

with Floor(chains=default_chain_configs(), chain_map=RowChainMap.alternating(2)) as floor:
    cases = (("SHIMMER speed 255 depth 0", Effect(SHIMMER, (255, 0, 0, 0))),
             ("SHIMMER speed 255 depth 0 spread 255", Effect(SHIMMER, (255, 0, 255, 0))))
    for label, fx in cases:
        clean(floor)
        frame(floor, color(40))
        if fx: frame(floor, effect(fx), 0.2)
        hi, lo = 834, 495        # depth 0 leaves the buffer untouched: no-effect references
        tol = (hi - lo) / 4
        print(f"{label}: refs {lo:.0f}/{hi:.0f} mA", flush=True)
        for delay_ms in (1.0, 4.5, 5.0, 5.5, 6.5):
            lost = 0
            for i in range(30):
                v = 120 if i % 2 == 0 else 40
                trial(floor, delay_ms / 1000, v)
                time.sleep(0.15)
                lost += abs(amps_med(floor, 2) - (hi if v == 120 else lo)) > tol
            note = " (row holds it to the quiet window's end, 3-4 ms)" if delay_ms < 3 else ""
            print(f"  data {delay_ms:.1f} ms after LATCH{note}: final frame lost {lost}/30", flush=True)
    clean(floor)
