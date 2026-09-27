"""How long after a LATCH is a tile deaf? Send LATCH, then after a controlled
delay a small frame (alternating two greys), and check it was shown.
HUE_SPLIT renders on LATCH only (no own-clock pushes) and leaves greys
untouched, so the only deaf window is the LATCH-triggered render + push."""
import statistics, sys, time
from df2_pi.effects import Effect, HUE_SPLIT, NONE
from df2_pi.protocol.constants import Cmd, TileCmd, LEDS_PER_TILE
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
def busy(s):
    t = time.perf_counter() + s
    while time.perf_counter() < t: pass
def clean(floor):
    for _ in range(3):
        frame(floor, effect(Effect(NONE)), 0.05); frame(floor, color(0), 0.05)
    floor.blackout(); time.sleep(0.3)

with Floor(chains=default_chain_configs(), chain_map=RowChainMap.alternating(2)) as floor:
    for label, fx in (("no effect", None), ("HUE_SPLIT", Effect(HUE_SPLIT, (20, 0, 0, 0)))):
        clean(floor)
        frame(floor, color(40))
        if fx: frame(floor, effect(fx), 0.2)
        frame(floor, color(120)); hi = amps_med(floor)
        frame(floor, color(40));  lo = amps_med(floor)
        tol = (hi - lo) / 4
        print(f"{label}: refs {lo:.0f}/{hi:.0f} mA", flush=True)
        for delay_ms in (2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0):
            lost = 0
            for i in range(20):
                v = 120 if i % 2 == 0 else 40
                floor.latch()                      # tiles render + push now
                busy(delay_ms / 1000)
                floor.send_data(0, color(v))       # 32-byte frame, forwarded on arrival
                busy(0.03); floor.latch()          # show it
                time.sleep(0.12)
                lost += abs(amps_med(floor, 2) - (hi if v == 120 else lo)) > tol
            print(f"  data {delay_ms:.1f} ms after LATCH: frames lost {lost}/20", flush=True)
    clean(floor)
