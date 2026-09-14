# LED strip tester

Hand-held checker for WS2815 strips as they come off the soldering bench.
Seeed XIAO ESP32-C3, one button, one strip connector.

## What the button does

| Action | Strip |
| --- | --- |
| Any press, however short | solid **red → green → blue**, 1 s each, always runs to completion |
| Released before the blue finishes | goes dark |
| Still held after the blue | **moving rainbow** until released, then dark |

## Wiring

| XIAO pin | GPIO | Goes to |
| --- | --- | --- |
| D2 | GPIO4 | strip DIN |
| D1 | GPIO3 | button, other side to GND (internal pull-up) |
| 5V | — | R-78E5.0-1.0 output |

Strips are chained DOUT → DIN as on a tile; up to four sides (60 LEDs) at once.

Power is two 9 V batteries in series into an R-78E5.0-1.0. `BRIGHTNESS` in
[src/main.cpp](src/main.cpp) is capped so a full tile's worth of strip stays
within what a pair of PP3s will give - see the comment there before raising it.

## Build

```bash
pio run -t upload
```
