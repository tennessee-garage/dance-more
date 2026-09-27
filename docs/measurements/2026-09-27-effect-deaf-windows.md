# Effect deaf windows, 2026-09-26/27 (#106)

Row 0 with four tiles (strips on slots 0 and 1). Measured by row 0's current:
two lit strips read well above the ~325 mA idle, so a tile that failed to go
dark, or showed the wrong colour, is visible from `POWER`. Raw output and the
bench scripts are in [`2026-09-27-effect-deaf-windows/`](2026-09-27-effect-deaf-windows/).

**Result:** a tile running an effect was deaf to the Tile Bus for ~1.8 ms of
every render on its own clock. Row firmware v9 and tile v5 remove that while
the host streams frames and make `BLACKOUT` robust to it, except for a ~1%
residual over a free-running `SHIMMER` that is not yet explained (#107).

## Before: row v8, tile v4

Effects render and push on the tile's own clock — every 40 ms for `CHASE` at
full speed, every 20 ms for `SHIMMER` and `FADE` — and each push is a WS2815
write with interrupts off, during which the tile cannot receive
([tile-bus-protocol.md](../tile-bus-protocol.md), `LATCH`).

| Traffic to tiles running `CHASE` | Lost |
| --- | --- |
| Small frames (`SET_COLOR` + `LATCH`), one at a time | 2/40 (5%) |
| Full `SET_LEDS` frames | 8/40 (20%) |
| `BLACKOUT`, random phase | 7/80 trials left a tile lit |
| Controls with no effect | 0/80 |

A `BLACKOUT` sent a fixed 0.5 s after starting the chase never failed (0/40):
it always landed between pushes. Random delays are needed to see the real rate.

## After: row v9, tile v5

- **Tile:** once two `LATCH`es arrive within 100 ms, the tile pushes only on
  `LATCH` — inside the row's quiet window — and not on its own clock; the
  effect's clock keeps running ([tile-effects.md](../tile-effects.md) §4).
- **Row:** `BLACKOUT`'s sweep goes out three times, one quiet window apart
  ([row-bus-protocol.md](../row-bus-protocol.md), `BLACKOUT`).

| Test | Result |
| --- | --- |
| `BLACKOUT` over `CHASE`, random phase | **0/100** |
| `BLACKOUT` over `SHIMMER` (4.25 Hz, full depth), random phase | **4/400 (1%)** — #107 |
| 30 fps stream of small frames, `CHASE` running | **0/100** wrong final frame |
| 30 fps stream of full `SET_LEDS` frames, `CHASE` running | **0/100** |
| 30 fps stream, data at the quiet window's end or 4.5–6.5 ms after `LATCH`: no effect / `HUE_SPLIT` / `SHIMMER` | **0/150, 0/150, 0/300** |
| Slow frames (~3 fps, not streaming), `CHASE` running | 6/60 small, 11/60 full lost — not covered by the fix |
| Controls with no effect, 30 fps | 0/60 |

The slow-frame row is the fix's known limit: below 10 fps the tile renders
on its own clock again and the row cannot see when.

## Open: the `SHIMMER` residual

All four failures had the same shape: one tile missed all three `BLACKOUT`
sweeps, kept shimmering over its grey buffer, and cleared on a second
`BLACKOUT` a second later. The obvious mechanisms are ruled out — see #107,
which proposes a logic-analyser capture to find the real one.
