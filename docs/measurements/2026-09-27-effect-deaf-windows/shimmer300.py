"""BLACKOUT over SHIMMER, 300 random-phase trials, with per-failure detail."""
import random, statistics, time
from df2_pi.effects import Effect, SHIMMER, NONE
from df2_pi.protocol.constants import Cmd, TileCmd, LEDS_PER_TILE
from df2_pi.transport import Floor, RowChainMap, default_chain_configs
random.seed(1060)
def amps(floor):
    p = floor.request(0, Cmd.POWER).payload
    return (p[2] << 8) | p[3]
def amps_med(floor, n=3):
    v = []
    for _ in range(n): v.append(amps(floor)); time.sleep(0.03)
    return statistics.median(v)
def swing(floor):
    s = []
    for _ in range(12): s.append(amps(floor)); time.sleep(0.04)
    return max(s) - min(s), statistics.median(s)
def frame(floor, payload, settle):
    floor.send_data(0, payload); time.sleep(0.02); floor.latch(); time.sleep(settle)
color  = lambda v: bytes([TileCmd.SET_COLOR, v, v, v]) * 8
leds   = lambda v: (bytes([TileCmd.SET_LEDS]) + bytes([v]) * (3 * LEDS_PER_TILE)) * 8
effect = lambda e: (bytes([TileCmd.SET_EFFECT]) + bytes(e)) * 8
def clean(floor):
    for _ in range(3):
        frame(floor, effect(Effect(NONE)), 0.05); frame(floor, leds(0), 0.05)
    floor.blackout(); time.sleep(0.3)
with Floor(chains=default_chain_configs(), chain_map=RowChainMap.alternating(2)) as floor:
    clean(floor); idle = amps_med(floor); print(f"idle {idle} mA", flush=True)
    fails = 0
    for trial in range(300):
        clean(floor)
        frame(floor, color(60), 0.05)
        wait = random.uniform(0.3, 0.7)
        frame(floor, effect(Effect(SHIMMER, (255, 255, 0, 0))), wait)
        floor.blackout(); time.sleep(0.3)
        after = amps_med(floor)
        if after > idle + 40:
            fails += 1
            sw, med = swing(floor)                       # effect still modulating the buffer?
            floor.blackout(); time.sleep(1.0)            # does a second BLACKOUT recover it?
            again = amps_med(floor)
            print(f"  FAIL trial {trial}: wait {wait:.3f}s, after BLACKOUT {after} mA "
                  f"(swing {sw} mA, median {med}), after 2nd BLACKOUT {again} mA", flush=True)
        if trial % 50 == 49: print(f"  ...{trial + 1} trials, {fails} failed", flush=True)
    print(f"BLACKOUT over SHIMMER, random phase: tiles left lit {fails}/300", flush=True)
    clean(floor); print(f"final {amps_med(floor)} mA", flush=True)
