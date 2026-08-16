# Measurements

Hardware characterization data — not design docs, not test results. Each
measurement here is real numbers pulled off real hardware with a specific
tool, kept alongside that tool so the measurement can be reproduced or
repeated on different hardware later.

## Brightness sweep (current vs. commanded intensity)

**Tool:** [`src/pi/test/integration/tile_brightness_sweep.py`](../../src/pi/test/integration/tile_brightness_sweep.py)

Steps `SET_COLOR` from 0-100% brightness in 5% increments for red, green,
blue, and white, and records the row controller's INA226 current at each
step (mean of several reads, first discarded as a settling sample). The
blacked-out baseline — row controller + tile quiescent draw, LEDs off — is
measured once up front and subtracted from every point, so the output
isolates what the commanded color/level actually costs rather than the
whole board's draw.

Requires a row controller running firmware built after
[`a3c283a`](https://github.com/tennessee-garage/dance-more/commit/a3c283a)
(the INA226 driver fix — see
[power_monitor_rp2350.cpp](../../src/row/src/rp2350/power_monitor_rp2350.cpp));
before that fix, current/power readings were inflated 8x and voltage read
0.4x true, so any sweep run against older firmware is not comparable to
what's here.

### Files

| File | What |
| --- | --- |
| `2026-08-16-brightness-sweep-df2-tile.csv` | Raw data: 85 rows (baseline + 21 steps × 4 colors) |
| `2026-08-16-brightness-sweep-df2-tile.html` | Self-contained interactive chart — open directly in a browser, no server needed |

### 2026-08-16 — `df2-tile` (this project's WS2815 tile, v1 PCB)

Row 0x00, single tile, bench supply at 12V. **Current draw tracks commanded
brightness, not how many color channels are lit** — red, green, blue, and
white sweeps land on nearly the same curve at every step, not just at full
white. Roughly linear from ~10% to 100% (consistent with PWM duty-cycle
dimming), with a softer knee below ~10%. Full-scale range: ~2 mA (level 0)
to ~418 mA (level 255) above the ~108 mA quiescent baseline.

Cross-check: total current at full white (baseline + delta) is ~526 mA,
close to the "~0.5 A measured" figure already in
[../power.md](../power.md) — that existing figure appears to be the
*total* reading (LEDs + row controller + tile), not LED current in
isolation as its own text suggests. Worth reconciling next time power.md is
touched.

## Measuring a new strip or tile

```bash
# on the Pi, from src/pi/
python3 test/integration/tile_brightness_sweep.py --chain /dev/ttyAMA0:23 --label <name>

# two-chain hat (production wiring)
python3 test/integration/tile_brightness_sweep.py --row <N> --label <name>
```

`--label` becomes part of the output filename
(`{date}-brightness-sweep-{label}.csv`) so runs against different hardware
never collide. The script only writes the CSV — to get a chart, either open
the CSV in a spreadsheet, or ask for one to be built from it the way this
one was (see the HTML file here for the format: a single self-contained
page, series colored by literal RGB + a neutral for "white", baseline
subtracted, no external dependencies).
