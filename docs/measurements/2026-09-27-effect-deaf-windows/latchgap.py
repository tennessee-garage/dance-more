"""Does a BLACKOUT sent right after a LATCH leave tiles stuck? Row 0."""
import time
from df2_pi.protocol.constants import Cmd, TileCmd, LEDS_PER_TILE
from df2_pi.transport import Floor, RowChainMap, default_chain_configs
def amps(floor):
    p = floor.request(0, Cmd.POWER).payload
    return (p[2] << 8) | p[3]
def bright(level):
    return bytes([TileCmd.SET_LEDS]) + bytes(200 + ((i * 7 + level) % 56) for i in range(3 * LEDS_PER_TILE))
BLACK_LEDS = (bytes([TileCmd.SET_LEDS]) + bytes(3 * LEDS_PER_TILE)) * 8
def reset(floor):
    floor.send_data(0, BLACK_LEDS); floor.latch(); time.sleep(0.2); floor.blackout(); time.sleep(0.2)
def trial(floor, idle, gap_s):
    reset(floor)
    floor.send_data(0, b"".join(bright(3) for _ in range(8))); time.sleep(0.2)   # forwarded, not yet shown
    floor.latch()                                   # tiles start their LED push
    if gap_s: time.sleep(gap_s)
    floor.blackout(); floor.latch()                 # what df2-pi play does on exit
    time.sleep(0.3)
    return amps(floor) > idle + 40
with Floor(chains=default_chain_configs(), chain_map=RowChainMap.alternating(2)) as floor:
    reset(floor); idle = amps(floor)
    for gap in (0, 0.005, 0):
        r = [trial(floor, idle, gap) for _ in range(10)]
        print(f"gap {gap*1000:.0f} ms between LATCH and BLACKOUT: stuck {sum(r)}/10")
    reset(floor)
