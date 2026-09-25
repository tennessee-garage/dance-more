# Tile Effects

Definitions for the `SET_EFFECT (0x11)` payload on Tile Bus. The wire format —
5 bytes, `effect_id` in bits 4:0 with bits 7:5 reserved — is specified in
[tile-bus-protocol.md](tile-bus-protocol.md) §5.2; this document defines what
the ids and the four parameter bytes mean.

An **effect** is not a preset animation the tile plays instead of host data. It
is a transform applied *over* host data: the host owns the pixels, the effect
changes how they reach the LEDs.

Implementation: [src/tile/lib/df2_core/pattern.h](../src/tile/lib/df2_core/pattern.h)
(still named for the old `SET_PATTERN` model; renamed in the rest of #72).

---

## 1. Why effects exist

**Bandwidth, not compute.** The Pi has ample horsepower to compute a shimmer;
what it does not have is a way to deliver one. A full-pixel floor frame is 8
rows × 1,456 bytes = **11.6 kB**, and the Row Bus ceiling is a hard 3.125 Mbps
([row-bus-protocol.md](row-bus-protocol.md) §1). At 30 FPS that is ~2.8 Mbps —
**about 90% of the bus spent doing nothing but pushing pixels**, with no lever
left to raise the ceiling.

An effect moves the per-frame cost of the *animation* to zero:

| Content | Row Bus per floor frame | Tile Bus per row |
| --- | --- | --- |
| `SET_LEDS` everywhere | 11,648 B, **every frame** (~18.6 ms) | 15.0 ms |
| `SET_EFFECT` everywhere | 448 B, **once** (~0.7 ms) | ~1.8 ms, once |

After the arming frame the host only has to send pixels when the *content*
changes — a colour change is a 4-byte `SET_COLOR` entry, not 181 bytes — and the
tiles keep animating over whatever they hold. That is the capability the
per-tile controllers buy that eight or sixteen dumb WS2815 chains driven
straight off the row controllers could not.

Second effect: **the animation rate stops being the frame rate.** A tile
renders at 50 Hz locally regardless of what the host is doing, so effects are
smooth even while the host is busy, idle, or sending nothing at all.

---

## 2. Model

A tile holds two independent pieces of display state:

| State | Written by | Holds |
| --- | --- | --- |
| Pixel buffer | `SET_COLOR` / `SET_LEDS` | 60 LED colours from the host |
| Effect register | `SET_EFFECT` | `effect_id` + `params[0..3]` |

The LEDs always show **the buffer passed through the effect**:

```
pixel buffer ──► effect(params, t) ──► output ──► WS2815
```

With no effect set (`OFF`), the buffer goes out untouched. The two are
orthogonal: pixel commands never touch the effect register, and `SET_EFFECT`
never touches the buffer.

**The effect never writes the buffer.** It renders into a separate output
buffer, so the buffer is always exactly what the host last sent. This is what
lets the host change the content under a running effect — `SET_LEDS` a new
image and the shimmer carries on over it, no re-arm — and it removes any risk
of an effect feeding back into itself frame after frame.

### Lifecycle

Effects are *staged* by `SET_EFFECT` and *started* by the next `LATCH`, the
same way pixel data is:

```
RC → tile 0..7 : SET_EFFECT(...)      ← arms each tile, 12 B per frame
RC → broadcast : LATCH                ← all 8 tiles start in step
                                      ← bus idle from here; tiles self-render
```

This reuses the existing latch point rather than adding one, and it means a
row's tiles start phase-aligned even though they were armed one at a time.

| Event | Effect |
| --- | --- |
| `SET_EFFECT`, valid | Staged. Nothing visible changes yet. |
| `SET_EFFECT`, invalid | **Ignored.** The current effect keeps running. |
| `LATCH` with an effect staged | Staged effect replaces the register; its clock starts from zero. |
| `LATCH` with nothing staged | The buffer's current contents go out through the current effect. The effect's clock is **not** restarted. |
| `SET_COLOR` / `SET_LEDS` | Buffer only. The effect register is untouched. |
| `SET_EFFECT(OFF)` + `LATCH` | Register cleared. The buffer is shown as-is — the LEDs are **not** blanked. |
| `BLACKOUT` | Clears the register, the buffer **and** any effect's private state. Otherwise a blackout cannot reliably blacken. |
| Power-up / re-discovery | Register and private state cleared. |

Display commands carry no ACK ([tile-bus-protocol.md](tile-bus-protocol.md)
§5.2), so an invalid `SET_EFFECT` is silently dropped — there is no channel to
report it on. The tile keeps whatever it was already showing rather than going
dark. Invalid means: payload shorter than 5 bytes, any of bits 7:5 set, an id
not listed as implemented below, or a parameter outside the range an effect
defines as valid.

Re-sending the *same* effect restarts its clock on the next `LATCH`. That is
how a host re-synchronises tiles that were armed at different times; it also
means a host must not re-send `SET_EFFECT` every frame, or a continuous effect
never gets past its first frame.

### The buffer as input

Because an effect reads the buffer rather than carrying its own image, the
host sets the content and the effect independently, in either order:

```
SET_COLOR(0, 0, 200)        →  LATCH   # tile is blue
SET_EFFECT(SHIMMER, ...)    →  LATCH   # tile shimmers blue
SET_LEDS(<gradient>)        →  LATCH   # the gradient shimmers
SET_EFFECT(OFF)             →  LATCH   # static gradient
```

This is also why four parameter bytes are enough for most effects — none of
them have to spend three bytes on a colour, which was the concern left open in
[tile-bus-protocol.md](tile-bus-protocol.md) §10. An effect that paints its own
light (`CHASE`) is the exception and spends one byte on a hue instead.

Effects fall into four kinds, and each definition below says which:

- **Modulating** — output is a darkened copy of the buffer (`SHIMMER`). The
  buffer is the ceiling, so these can never clip a channel.
- **Recolouring** — output is the buffer with its colour shifted, brightness
  kept (`HUE_SPLIT`). Stateless: the output depends only on the buffer.
- **Overlaying** — some LEDs show the effect's own light instead of the buffer
  (`CHASE`). The rest pass the buffer through.
- **History** — output depends on what was shown before, not just on the
  buffer now (`FADE`). The only kind with per-LED private state.

---

## 3. Effect ids

| id | Name | Status | Kind |
| --- | --- | --- | --- |
| 0 | `OFF` | **implemented** (semantics changing — see below) | none |
| 1 | — | unassigned (was `SOLID`, removed) | — |
| 2 | — | unassigned (was `BREATHE`, dropped) | — |
| 3 | `SHIMMER` | **implemented** | modulating |
| 4 | `CHASE` | specified, not implemented | overlaying |
| 5 | `SPARKLE` | reserved — see §5 | — |
| 6 | `HUE_SPLIT` | specified, not implemented | recolouring |
| 7 | `FADE` | specified, not implemented | history |
| 8–31 | — | reserved | — |

Reserved ids are rejected, so a host that sends one gets a no-op rather than an
undefined effect. Names are recorded here so the numbering doesn't get
re-litigated when they are implemented; nothing depends on them yet.

`SOLID` was removed because it was `SET_COLOR` under another name: it painted a
colour and stopped, which is not a transform of anything. `BREATHE` was dropped
because it was `SHIMMER` with `spread = 0`. Ids 1 and 2 stay unassigned rather
than being reused, so a stale host that still sends one gets a no-op rather
than a different effect.

### `0 OFF` — no effect

All four parameters ignored. Clears the effect register on the next `LATCH`;
from then on the buffer goes out untouched. The LEDs keep showing whatever the
buffer holds — `OFF` removes an effect, it does not darken the tile. To go dark,
send `SET_COLOR(0, 0, 0)`.

> The current firmware still implements the old meaning — paint black once —
> and changes with the rest of #72.

### `3 SHIMMER` — per-LED brightness modulation

**Modulating.** Each LED's brightness is modulated by a raised cosine. The LEDs
share one cycle rate but sit at different points in it, so the tile glitters
rather than pulsing as a block.

| Param | Name | Range | Meaning |
| --- | --- | --- | --- |
| 0 | `speed` | 0–255 | **Cycles per minute.** `0` freezes the effect at its start phase; `60` is 1 Hz; `255` is 4.25 Hz. |
| 1 | `depth` | 0–255 | Modulation depth. `0` leaves the buffer untouched; `255` dips to black at the bottom of the cycle. |
| 2 | `spread` | 0–255 | Per-LED phase scatter. `0` puts the whole tile in unison (a breathe); `255` fully scatters (a fine twinkle). |
| 3 | `seed` | 0–255 | Seeds the scatter. |

`seed` is the parameter that makes 64 tiles not look like 64 copies of the same
loop. The scatter is deterministic in the seed, so the host can pick: **same
seed across a row** for tiles that glitter in step, **different seeds** to
de-correlate them. It is a lookup, not randomness — a tile power-cycled
mid-show comes back to the same scatter it had.

Suggested starting points:

| Look | speed | depth | spread |
| --- | --- | --- | --- |
| Slow breathe, whole tile | 12 | 120 | 0 |
| Candle-ish flicker | 90 | 90 | 200 |
| Hard glitter | 180 | 255 | 255 |
| Barely-there sheen | 30 | 40 | 255 |

### `4 CHASE` — marquee around the tile edge

**Overlaying.** Evenly spaced lit LEDs step around the tile's perimeter, like
marquee lights. Lit LEDs show the chase colour; every other LED shows the
buffer, so a chase runs over whatever the host has painted.

| Param | Name | Range | Meaning |
| --- | --- | --- | --- |
| 0 | `hue` | 0–255 | Chase colour, once round the colour wheel at full saturation. `0` red, `85` green, `170` blue. |
| 1 | `brightness` | 0–255 | Chase LED brightness. `0` makes the chase LEDs black — a dark gap travelling through the buffer. |
| 2 | `speed` | 0–255 | Step period, `40 + 10 × speed` ms. `0` is the fastest at 40 ms/step; `255` is 2.59 s/step. |
| 3 | `spacing` | 1–59 | Unlit LEDs between chase LEDs. `1` lights every other LED (30 lit); `59` leaves a single LED going round. `0` and `≥ 60` are **invalid** — the `SET_EFFECT` is dropped. |

Chase LEDs repeat every `spacing + 1` LEDs along the chain, and each step moves
them one LED in the direction of increasing LED index. An LED `i` is lit when

```
(i + 60 − step) mod (spacing + 1) == 0         step counts 0, 1, 2, ... mod 60
```

**The seam.** When `spacing + 1` doesn't divide 60, the last gap before LED 0
comes out short. It sits at the corner where the WS2815 chain starts, which
hides it about as well as anywhere can. Seamless spacings — those where
`spacing + 1` divides 60 — are **1, 2, 3, 4, 5, 9, 11, 14, 19, 29, 59**.

**Timing.** The tile renders in 20 ms frames (§4), so step times land on
frame boundaries. The step clock accumulates (`next_step += period`, not
`now + period`), so a 50 ms period comes out as alternating 40/60 ms steps
with the correct average rather than drifting slow to 60.

The hue is converted to RGB once, when the effect starts — not per LED, not per
frame — so the colour wheel costs nothing at render time.

### `6 HUE_SPLIT` — opposite hue shifts on two halves

**Recolouring.** One half of the tile's perimeter has its hue nudged one way
round the colour wheel, the other half the other way. A flat colour becomes two
neighbouring colours meeting at two points on the edge; an image keeps its
shape and gains a split tint.

| Param | Name | Range | Meaning |
| --- | --- | --- | --- |
| 0 | `shift` | 0–255 | Hue shift applied to each half, `+shift` on one and `−shift` on the other, in units where 256 is a full turn. `0` is the identity. Small values are the intended use; large ones stay valid but stop reading as a tint. |
| 1 | `offset` | 0–59 | Where the split falls. The `+` half is LEDs `offset … offset+29` (mod 60), the `−` half the other 30. `0` splits at the corner where the chain starts; `≥ 60` is **invalid**. |
| 2–3 | — | 0 | Reserved, **must be 0** — non-zero makes the `SET_EFFECT` invalid, so they can be given meaning later. |

`offset` is in LEDs rather than as a named axis so the doc doesn't have to know
how the chain is laid round the tile: with 15 LEDs per side, `0` and `15` split
corner-to-corner, and `7`/`8` split through the middle of two opposite sides.

Stateless and not time-varying: the tile renders it only on `LATCH`, and
repeated latches with an unchanged buffer produce an unchanged output. Black
and greys have no hue and pass through untouched.

**Not a colour-space conversion.** An RGB→HSV→RGB round trip on an AVR with no
hardware divide is not something to run 60 times per latch. The implementation
rotates in RGB space — a table-driven mix of each channel with its neighbour —
which approximates a hue shift closely for the small `shift` values this is
meant for. The exact curve is the implementation's; what this document fixes
is `0` = identity, the sign convention, and brightness being preserved to
within rounding.

### `7 FADE` — decay on release

**History.** Leaves a decaying trail behind moving content while the host sends
only the leading edge. The host lights a pixel; when it stops lighting it
(sends it black), the tile fades it out instead of cutting it.

Per LED, every render:

```
if buffer[i] == (0,0,0) and out[i] != (0,0,0):
    out[i] = decay(out[i])        # host released it - fade it out
else:
    out[i] = buffer[i]            # host is driving it - follow exactly
```

It **never dims an actively-lit pixel**. Only pixels the host has released
decay, so a bright moving dot stays full brightness and drags a tail. A released
pixel that the host lights again jumps straight to the new value.

| Param | Name | Range | Meaning |
| --- | --- | --- | --- |
| 0 | `decay` | 0–255 | Per-frame retention, `out = out × decay / 256` per channel, per 20 ms render frame. `0` is instant off — indistinguishable from no effect. |
| 1–3 | — | 0 | Reserved, **must be 0** — non-zero makes the `SET_EFFECT` invalid. |

From full brightness (255), with the floor scale described below:

| `decay` | Down to 10% | Fully black |
| --- | --- | --- |
| 128 | 80 ms | 160 ms |
| 200 | 200 ms | 380 ms |
| 230 | 420 ms | 720 ms |
| 245 | 0.9 s | 1.4 s |
| 252 | 2.2 s | 2.7 s |
| 255 | 4.6 s | 5.1 s |

**Decay runs on the tile's clock, not per `LATCH`.** #72 first proposed one
decay step per latch. That makes the tail length depend on the host's frame
rate, and freezes a half-faded tail on the floor whenever the host goes quiet —
exactly when the tile is supposed to keep animating on its own. Per 20 ms frame,
a tail is the same length at any host frame rate and always finishes.

**Integer-math trap.** A released pixel must reach exactly `(0,0,0)`, not
stall one step above it. A *rounded* scale stalls: `(1 × 230 + 128) >> 8` is
`1`, so the pixel sits dim forever. A plain floor, `v × k >> 8`, cannot — with
`k ≤ 255` it is always strictly less than `v`. Keep the floor, and don't
"improve" it with rounding. The figures above use it.

The history is the output buffer itself, so `FADE` costs no RAM beyond what
every effect already uses. When `FADE` starts, the history is whatever the tile
was showing, so arming it causes no visible jump. `BLACKOUT` clears it with
everything else.

---

## 4. Implementation notes

**Render rate is 50 Hz** (`PatternEngine::FRAME_MS = 20`). The ceiling is the
WS2815 push itself, which runs with interrupts disabled for ~1.8 ms and leaves
the tile deaf to Tile Bus for that window. At 20 ms that is ~9% of the period,
so a tile running an effect stays comfortably responsive to a `SET_LEDS` or a
broadcast that takes it back. The rate is a constant, not a parameter — a tile
that rendered faster would trade bus responsiveness for smoothness nobody asked
for.

**When the tile renders.** On every `LATCH` (new buffer contents must go out
through the effect immediately), and every `FRAME_MS` while a time-varying
effect is set. With `OFF`, `HUE_SPLIT`, or `SHIMMER` at `speed = 0`, the tile
renders only on `LATCH`. `FADE` renders every frame while any pixel is still
decaying, and only on `LATCH` once they have all reached black.

**Effect math runs with interrupts enabled.** Only the WS2815 push must not.
The two costs are budgeted separately: effect render time adds to
`LATCH`→lit latency, but it does not add to the window in which the tile is
deaf to Tile Bus.

**Fixed point throughout.** A 256-byte raised-cosine table lives in flash
(`PROGMEM` on AVR); brightness scaling is a multiply and a `>>8` per channel,
which costs one part in 256 of darkening at full scale and avoids a divide the
AVR doesn't have. 180 multiplies per frame at 50 Hz is nothing on a 20 MHz core.

**Phase** is a `uint16_t` that wraps at one cycle. `speed` converts at 22 phase
units per frame per cycle-per-minute — 65536/(60×50) = 21.85 rounded up, so the
actual rate runs ~0.7% fast. Irrelevant for a shimmer; noted so nobody
re-derives it.

**RAM.** The old pattern engine snapshotted the buffer into a 180-byte base
image at start and rendered back into the buffer. The effect model inverts
that: the buffer stays the input and the effect renders into a 180-byte output
buffer, so the cost is the same — 180 for the output, 60 for `SHIMMER`'s per-LED
phase offsets, the rest state. The old build measured **1,179 of 3,072 bytes
(38%)** with patterns in; re-measure after the change. Effect-private state
(`SHIMMER`'s phase offsets, and anything a later effect needs) should share one scratch region
rather than each effect reserving its own, since only one effect is ever
active.

---

## 5. Open questions

- **`SPARKLE` (5).** `SHIMMER` is continuous and periodic: every
  LED is always somewhere in its cycle, and it only darkens. A distinct
  `SPARKLE` would be the opposite on both counts — *sparse* and *random*
  (individual LEDs fire at unpredictable times, most of the tile untouched at
  any instant) and *overlaying* (each fired LED flashes toward a sparkle colour
  or white and decays back to the buffer). Candidate params: `density`
  (sparkles per second), `decay` (flash length), `hue`/`brightness` or a
  white-mix amount, `seed`. Not yet decided.
- **`CHASE` direction.** Steps always run in increasing LED index. Reversing
  needs a bit the four params don't have; bits 7:5 of byte 0 are reserved and
  could carry per-effect flags, but that is a wire-format change to
  [tile-bus-protocol.md](tile-bus-protocol.md) §5.2.
- **No host-side API yet.** `df2-pi` can encode the command
  (`TileCmd.SET_PATTERN`, 6 bytes per tile in a `SEND_DATA`) but has no
  `Floor` method for arming effects. That is the next piece, and it is where
  the bandwidth claim in §1 gets measured rather than calculated.
- **Sub-tile addressing.** Every effect here applies to all 60 LEDs. An effect
  that treats the four sides differently needs a side index the params don't
  currently carry. (`CHASE` goes round corners, but uniformly.)
- **Start-up diagnostic interaction.** The boot diagnostic in
  [src/tile/src/main.cpp](../src/tile/src/main.cpp) is cancelled by the first
  real display data. Untested against a `SET_EFFECT` that arrives
  mid-diagnostic.
- **Not yet run on hardware.** Everything above is verified by unit test and by
  the `ATtiny3224` build fitting; the 1.8 ms push figure and the 50 Hz
  responsiveness claim want a bench check.
