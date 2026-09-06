# Tile Patterns

Definitions for the `SET_PATTERN (0x11)` payload on Tile Bus. The wire format —
5 bytes, `pattern_id` in bits 4:0 with bits 7:5 reserved — is specified in
[tile-bus-protocol.md](tile-bus-protocol.md) §5.2; this document defines what
the ids and the four parameter bytes mean.

Implementation: [src/tile/lib/df2_core/pattern.h](../src/tile/lib/df2_core/pattern.h).

---

## 1. Why patterns exist

**Bandwidth, not compute.** The Pi has ample horsepower to compute a shimmer;
what it does not have is a way to deliver one. A full-pixel floor frame is 8
rows × 1,456 bytes = **11.6 kB**, and the Row Bus ceiling is a hard 3.125 Mbps
([row-bus-protocol.md](row-bus-protocol.md) §1). At 30 FPS that is ~2.8 Mbps —
**about 90% of the bus spent doing nothing but pushing pixels**, with no lever
left to raise the ceiling.

A pattern moves the per-frame cost to zero:

| Content | Row Bus per floor frame | Tile Bus per row |
| --- | --- | --- |
| `SET_LEDS` everywhere | 11,648 B, **every frame** (~18.6 ms) | 15.0 ms |
| `SET_PATTERN` everywhere | 448 B, **once** (~0.7 ms) | ~1.8 ms, once |

After the arming frame the bus goes quiet and the tiles keep animating. That is
the capability the per-tile controllers buy that eight or sixteen dumb WS2815
chains driven straight off the row controllers could not.

Second effect: **the animation rate stops being the frame rate.** A tile
renders at 50 Hz locally regardless of what the host is doing, so patterns are
smooth even while the host is busy, idle, or sending nothing at all.

---

## 2. Lifecycle

Patterns are *staged* by `SET_PATTERN` and *started* by the next `LATCH`:

```
RC → tile 0..7 : SET_PATTERN(...)     ← arms each tile, 12 B per frame
RC → broadcast : LATCH                ← all 8 tiles start in step
                                      ← bus idle from here; tiles self-render
```

This reuses the existing latch point rather than adding one, and it means a
row's tiles start phase-aligned even though they were armed one at a time.

| Event | Effect |
| --- | --- |
| `SET_PATTERN`, valid | Staged. Nothing visible changes yet. |
| `SET_PATTERN`, invalid | **Ignored.** Any running pattern keeps running. |
| `LATCH` with a pattern staged | Pattern starts; the buffer's current contents become its **base image**. |
| `LATCH` with nothing staged | Ordinary latch. A running pattern is *not* restarted. |
| `SET_COLOR` / `SET_LEDS` | Cancels the pattern. Explicit pixel data always wins. |

Display commands carry no ACK ([tile-bus-protocol.md](tile-bus-protocol.md)
§5.2), so an invalid `SET_PATTERN` is silently dropped — there is no channel to
report it on. The tile keeps whatever it was already showing rather than going
dark. Invalid means: payload shorter than 5 bytes, any of bits 7:5 set, or an
id not listed as implemented below.

That `LATCH` with nothing staged is a no-op matters more than it looks: the
host's ordinary 30 FPS `LATCH` must not re-snapshot the base from an
already-modulated buffer, or a pattern would ratchet itself to black.

### The base image

A pattern **modulates whatever is already in the pixel buffer** — it does not
carry its own colour. So the host sets the tile up first, then arms:

```
SET_COLOR(0, 0, 200)  →  LATCH        # tile is blue
SET_PATTERN(SHIMMER, ...)  →  LATCH   # tile shimmers blue
```

Because the base is the buffer and not a parameter, the same applies to an
arbitrary image: `SET_LEDS` a gradient, then arm SHIMMER, and the gradient
shimmers. This is also why the four parameter bytes are enough — none of them
have to spend three bytes on a colour, which was the concern left open in
[tile-bus-protocol.md](tile-bus-protocol.md) §10.

**Patterns only ever darken.** The base image is the ceiling, so a pattern can
never clip a channel; there is no headroom question and no gamma surprise at
the top of the range.

---

## 3. Pattern ids

| id | Name | Status | Continuous? |
| --- | --- | --- | --- |
| 0 | `OFF` | **implemented** | no — paints black once |
| 1 | `SOLID` | **implemented** | no — paints once |
| 2 | `BREATHE` | reserved | — |
| 3 | `SHIMMER` | **implemented** | yes |
| 4 | `CHASE` | reserved | — |
| 5 | `SPARKLE` | reserved | — |
| 6 | `RAINBOW` | reserved | — |
| 7 | `FADE_TO` | reserved | — |
| 8–31 | — | reserved | — |

Reserved ids are rejected by `arm()`, so a host that sends one gets a no-op
rather than an undefined pattern. Names are recorded here so the numbering
doesn't get re-litigated when they are implemented; nothing depends on them yet.

### `0 OFF` — all LEDs black

All four parameters ignored. Paints black on the starting `LATCH` and stops
rendering. `SET_COLOR(0,0,0)` is the cheaper way to do this from the host; `OFF`
exists so a pattern can be cleared through the same channel that set it.

### `1 SOLID` — static colour

| Param | Meaning |
| --- | --- |
| 0 | R |
| 1 | G |
| 2 | B |
| 3 | ignored |

Paints on the starting `LATCH` and stops. Equivalent to `SET_COLOR`, which is
2 bytes cheaper on the wire; prefer `SET_COLOR` unless you are already in the
pattern channel.

### `3 SHIMMER` — per-LED brightness modulation

Each LED's brightness is modulated by a raised cosine. The LEDs share one
cycle rate but sit at different points in it, so the tile glitters rather than
pulsing as a block.

| Param | Name | Range | Meaning |
| --- | --- | --- | --- |
| 0 | `speed` | 0–255 | **Cycles per minute.** `0` freezes the pattern at its start phase; `60` is 1 Hz; `255` is 4.25 Hz. |
| 1 | `depth` | 0–255 | Modulation depth. `0` leaves the base untouched; `255` dips to black at the bottom of the cycle. |
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

---

## 4. Implementation notes

**Render rate is 50 Hz** (`PatternEngine::FRAME_MS = 20`). The ceiling is the
WS2815 push itself, which runs with interrupts disabled for ~1.8 ms and leaves
the tile deaf to Tile Bus for that window. At 20 ms that is ~9% of the period,
so an armed tile stays comfortably responsive to a `SET_LEDS` or a broadcast
that takes it back. The rate is a constant, not a parameter — a tile that
rendered faster would trade bus responsiveness for smoothness nobody asked for.

**Fixed point throughout.** A 256-byte raised-cosine table lives in flash
(`PROGMEM` on AVR); brightness scaling is a multiply and a `>>8` per channel,
which costs one part in 256 of darkening at full scale and avoids a divide the
AVR doesn't have. 180 multiplies per frame at 50 Hz is nothing on a 20 MHz core.

**Phase** is a `uint16_t` that wraps at one cycle. `speed` converts at 22 phase
units per frame per cycle-per-minute — 65536/(60×50) = 21.85 rounded up, so the
actual rate runs ~0.7% fast. Irrelevant for a shimmer; noted so nobody
re-derives it.

**RAM cost is 255 bytes** — 180 for the base-image snapshot, 60 for the per-LED
phase offsets, the rest state. Measured on the `ATtiny3224` build, which sits
at **1,179 of 3,072 bytes (38%)** with patterns in. The snapshot is what buys
shimmer-over-an-arbitrary-image; a colour-only pattern would fit in 3 bytes,
and the trade was made deliberately.

---

## 5. Open questions

- **No host-side API yet.** `df2-pi` can encode `SET_PATTERN`
  (`TileCmd.SET_PATTERN`, 6 bytes per tile in a `SEND_DATA`) but has no
  `Floor` method for arming patterns. That is the next piece, and it is where
  the bandwidth claim in §1 gets measured rather than calculated.
- **Sub-tile addressing.** Every pattern here applies to all 60 LEDs. A pattern
  that treats the four sides differently (a chase that turns corners) needs a
  side index the params don't currently carry.
- **`BREATHE` vs `SHIMMER` with `spread=0`.** They are the same thing today.
  Either drop id 2 or give it a distinct waveform before implementing it.
- **Start-up pattern interaction.** The boot diagnostic in
  [src/tile/src/main.cpp](../src/tile/src/main.cpp) is cancelled by the first
  real display data, patterns included. Untested against a `SET_PATTERN` that
  arrives mid-diagnostic.
- **Not yet run on hardware.** Everything above is verified by unit test and by
  the `ATtiny3224` build fitting; the 1.8 ms push figure and the 50 Hz
  responsiveness claim want a bench check.
