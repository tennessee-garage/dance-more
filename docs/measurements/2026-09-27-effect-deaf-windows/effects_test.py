"""Effect tests on row 0 (tiles in slots 0-3, strips on 0 and 1)."""
import statistics, time
from df2_pi.effects import Effect, SHIMMER, CHASE, NONE
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
def color(v):   return bytes([TileCmd.SET_COLOR, v, v, v]) * 8
def leds(v):    return (bytes([TileCmd.SET_LEDS]) + bytes([v]) * (3 * LEDS_PER_TILE)) * 8
def effect(e):  return (bytes([TileCmd.SET_EFFECT]) + bytes(e)) * 8
BLACK_LEDS = leds(0)
def clean(floor):
    for _ in range(2):
        frame(floor, effect(Effect(NONE)), 0.05); frame(floor, BLACK_LEDS, 0.05)
    floor.blackout(); time.sleep(0.3)

with Floor(chains=default_chain_configs(), chain_map=RowChainMap.alternating(2)) as floor:
    clean(floor); idle = amps_med(floor)
    print(f"idle {idle} mA")

    # 1. Does an effect run? SHIMMER in unison (spread 0), 1 Hz, full depth over grey.
    frame(floor, color(100)); frame(floor, effect(Effect(SHIMMER, (60, 255, 0, 0))), 0.1)
    samples = []
    for _ in range(25): samples.append(amps(floor)); time.sleep(0.08)
    print(f"1. SHIMMER 1 Hz breathe over grey 100: current {min(samples)}..{max(samples)} mA over 2 s")
    clean(floor)

    # 2. Does BLACKOUT stop a running CHASE? 20 trials, black buffer, then grey buffer.
    chase = Effect(CHASE, (0, 255, 0, 5))          # red, full, 40 ms/step, 10 lit per tile
    for base in (0, 60):
        left = 0
        for _ in range(20):
            clean(floor)
            frame(floor, color(base), 0.05); frame(floor, effect(chase), 0.5)
            floor.blackout(); time.sleep(0.3)
            if amps_med(floor) > idle + 40: left += 1
        print(f"2. BLACKOUT over CHASE, buffer {base}: tiles left lit {left}/20")
    clean(floor)

    # 3. Frame loss: alternate two colours one frame at a time; a lost SET_COLOR or
    # LATCH leaves the previous colour. With no effect (control), then with CHASE.
    for label, payload_of in (("SET_COLOR", color), ("SET_LEDS", leds)):
        for eff in (None, chase):
            clean(floor)
            frame(floor, payload_of(40))
            if eff: frame(floor, effect(eff), 0.3)
            frame(floor, payload_of(120)); hi = amps_med(floor)
            frame(floor, payload_of(40));  lo = amps_med(floor)
            tol = (hi - lo) / 4; lost = 0; n = 40
            for i in range(n):
                v = 120 if i % 2 == 0 else 40
                frame(floor, payload_of(v), 0.15)
                a = amps_med(floor, 2)
                if abs(a - (hi if v == 120 else lo)) > tol: lost += 1
            print(f"3. {label} frames, {'CHASE running' if eff else 'no effect  '}: "
                  f"refs {lo:.0f}/{hi:.0f} mA, frames not shown {lost}/{n}", flush=True)
    clean(floor)
    print(f"final {amps_med(floor)} mA (idle {idle})")
