"""Are pipelined small frames actually displayed? Alternate two colours each frame,
stop on a known one, and compare the current with static references."""
import time
from df2_pi.protocol.constants import Cmd, TileCmd, LEDS_PER_TILE
from df2_pi.transport import Floor, RowChainMap, default_chain_configs
def amps(floor):
    p = floor.request(0, Cmd.POWER).payload
    return (p[2] << 8) | p[3]
def color(v): return bytes([TileCmd.SET_COLOR, v, v, v]) * 8
BLACK_LEDS = (bytes([TileCmd.SET_LEDS]) + bytes(3 * LEDS_PER_TILE)) * 8
def show(floor, v):
    floor.send_data(0, color(v)); time.sleep(0.05); floor.latch(); time.sleep(0.3)
with Floor(chains=default_chain_configs(), chain_map=RowChainMap.alternating(2)) as floor:
    show(floor, 120); hi = amps(floor)
    show(floor, 40);  lo = amps(floor)
    print(f"static references: colour 120 = {hi} mA, colour 40 = {lo} mA")
    wrong = 0
    for trial in range(10):
        frames = 60 + trial            # end on alternating colours across trials
        period = 1 / 30; deadline = time.perf_counter() + period
        floor.send_data(0, color(120))
        shown = 120
        for f in range(1, frames + 1):
            while time.perf_counter() < deadline: pass
            floor.latch()                                  # frame f-1 becomes visible
            shown = 120 if (f - 1) % 2 == 0 else 40
            if f < frames:
                floor.send_data(0, color(120 if f % 2 == 0 else 40))
            deadline += period
        time.sleep(0.3)
        a = amps(floor)
        got = 120 if abs(a - hi) < abs(a - lo) else 40
        ok = got == shown; wrong += not ok
        print(f"  trial {trial}: expected colour {shown}, current {a} mA -> {'ok' if ok else 'WRONG'}", flush=True)
    print(f"wrong final frame: {wrong}/10")
    floor.send_data(0, BLACK_LEDS); floor.latch(); time.sleep(0.2); floor.blackout()
