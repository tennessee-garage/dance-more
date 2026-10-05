# Writing animations

An animation is one Python file in `src/pi/animations/`. Copy
[`solid.py`](../src/pi/animations/solid.py), edit it, and run it — that is the
whole workflow, and it works on a laptop with no floor attached:

```bash
cd src/pi
pip install -e ".[dev]"                                    # once
cp animations/solid.py animations/mine.py
df2-pi play --no-hardware --terminal --animation mine     # Ctrl-C to stop
```

The filename stem (`mine`) is the animation's id: what playlists reference and
what `--animation` takes. A syntax error shows up as a warning and the other
animations keep loading; fix the file and run again.

The starter pack is the tutorial. Each file is short, commented, and exercises
one part of the API:

| File | Format | Shows |
| --- | --- | --- |
| [`solid.py`](../src/pi/animations/solid.py) | tile | The minimum: one colour, one param. Copy this one. |
| [`rainbow_sweep.py`](../src/pi/animations/rainbow_sweep.py) | tile | Position and beat time: `ctx.t_beats`, a `pulse()` on every beat |
| [`checkerboard.py`](../src/pi/animations/checkerboard.py) | tile | `ctx.state` — keeping data between frames without a class |
| [`plasma.py`](../src/pi/animations/plasma.py) | pixel | `PixelFrame.from_grid()` — "just hand me a 136×136 image" |
| [`ripple.py`](../src/pi/animations/ripple.py) | pixel | Continuous coordinates: `led_positions`, `splat()`, fading trails; `ctx.triggers` |
| [`lightning.py`](../src/pi/animations/lightning.py) | pixel | **The edge graph**: bolts stepped down the floor corner by corner along tile edges |
| [`chase.py`](../src/pi/animations/chase.py) | pixel | `floor_ring` — the outer boundary as one 480-LED loop; the floor palette |
| [`seams.py`](../src/pi/animations/seams.py) | pixel | `seams` — the facing pairs of edges between tiles |
| [`comet_squares.py`](../src/pi/animations/comet_squares.py) | pixel | `rails` and `edges_at` together; objects with phases in `ctx.state` |
| [`vortex.py`](../src/pi/animations/vortex.py) | pixel | Building your own rings from `geo.edge`; a little physics carried in `ctx.state` |
| [`twin_peaks.py`](../src/pi/animations/twin_peaks.py) | pixel | A whole-floor texture from `led_positions`: shading a surface, not drawing lines |
| [`spiral.py`](../src/pi/animations/spiral.py) | tile | A v1 processor ported: per-frame state in `ctx.state`, steps clocked by `ctx.t` |
| [`stardust.py`](../src/pi/animations/stardust.py) | pixel | Two scales in one frame: a per-tile field broadcast to its 60 LEDs, with single-LED stars and drifting dust on top |
| [`stripes.py`](../src/pi/animations/stripes.py) | tile | A v1 port that keeps v1's look: a fade built in linear light and encoded with `gamma.from_linear()` |
| [`waves.py`](../src/pi/animations/waves.py) | pixel | Phase maps: a `tempo` waveform run across `df2_pi.phase` offsets - the console-style effect in one line |
| [`video.py`](../src/pi/animations/video.py) | pixel | Real footage: clips from `df2-pi video import`, blended frame to frame (below) |
| [`waterline.py`](../src/pi/animations/waterline.py) | tile | A little physics per tile column: springs, a saturating pull between neighbours, kicks - motion that is never quite periodic |
| [`comet_train.py`](../src/pi/animations/comet_train.py) | pixel | Every edge one way as `rails` lanes, stepped on the beat; the floor as a window onto an endless lattice, so a 90° turn is a lane-to-lane map with comets entering and leaving at the sides |
| [`switchyard.py`](../src/pi/animations/switchyard.py) | pixel | Comet Train on a lattice of corners and links: a flow per link and a route per corner, so turns can be thrown one corner at a time and the floor re-routes round them. Experimental |
| [`_template.py`](../src/pi/animations/_template.py) | — | A commented skeleton. Underscore-prefixed, so the loader skips it. |

## 1. The contract

```python
from df2_pi.animation import Param, animation
from df2_pi.pixels import TileFrame

@animation(name="Solid", format="tile", params={"hue": Param(float, default=0.6, min=0, max=1)})
def render(previous, ctx):
    frame = TileFrame.black(ctx.geometry)
    ...
    return frame
```

`render(previous, ctx)` is called once per frame and returns a **new** frame of
the declared format. Three rules:

- **`previous` is read-only.** The driver may still be handing it to the
  preview or a recorder while you render the next one. To evolve it — trails,
  decay, anything that builds on the last frame — copy it first:
  `frame = previous.copy().gain(0.9)`. Returning `previous` itself is caught
  with an error naming your file.
- **An animation runs forever.** It declares no duration and is never told how
  long it has left; the playlist decides when to stop it and the runner applies
  the cut or crossfade to the output. Loop, evolve, or idle — never run out.
- **Raising does not take the floor down.** The last good frame is held, the
  error is logged with your file name, and the playlist moves on. Three
  consecutive failures disable that entry for the session.

On frame 0, `previous` is black. Anything else you need lives in `ctx` (§4).

## 2. Two formats — which one?

| | `TileFrame` | `PixelFrame` |
| --- | --- | --- |
| Shape | `(8, 8, 3)` — one colour per tile | `(64, 60, 3)` — every LED, in chain order |
| `format=` | `"tile"` | `"pixel"` |
| Index | `frame[row, col] = (r, g, b)` | `frame.tile(t)[led]`, `frame.flat[i]`, `frame.grid` |
| Wire cost per row | **32 bytes** | up to **1,448 bytes** |

**Pick `TileFrame` for anything floor-scale** — colour fields, chases at tile
resolution, anything a person sees as "the tiles doing something". It is what
most floor animations actually want, and it costs 45× less on the wire, which
is the difference between comfortable headroom and dropped frames.

**Pick `PixelFrame` when the effect lives on the edges** — a bolt running along
the tile boundaries, a pulse around the perimeter, anything the eye reads as a
line rather than a tile. The encoder still sends any tile whose 60 LEDs happen
to be one colour as 4 bytes, so a mostly-flat `PixelFrame` is cheap too.

Both are numpy `uint8` arrays underneath (`frame.data`); write into them freely.
Colour values are **perceptual** (gamma-encoded): 128 looks half as bright as
255, and the driver converts to what the LEDs need on the way out.

## 3. The cell model

Each tile is a **17×17 block of cells**. The 64-cell ring holds the tile's 60
LEDs — 15 per side — with the **4 corner cells dark**, and the 15×15 interior
is dark too. Eight tiles a side makes the floor a **136×136** grid of which
3,840 cells (21%) are real LEDs.

```
tile block, 17 x 17 cells         .  L L L L L L L L L L L L L L L  .
  .  = dark corner (unpopulated)  L  . . . . . . . . . . . . . . .  L
  L  = one WS2815 LED             L  . . . . . . . . . . . . . . .  L
  .  = dark interior (15 x 15)    L        ( 15 x 15 dark )        L
                                  L  . . . . . . . . . . . . . . .  L
  64 ring cells - 4 corners = 60  .  L L L L L L L L L L L L L L L  .
```

Why 17 and not 15: a 15×15 block has only 56 perimeter cells because corners
are shared between sides. Leaving the corners dark is what makes each side an
independent run of exactly 15.

Coordinates are `(y, x)`, canonical orientation: **cell (0, 0) is the floor
corner nearest the Pi**, row 0 is the tile row nearest the Pi, `y` increases
away from it. Every renderer (terminal, window, the web preview) draws row 0
along the **bottom**, matching the floor as you stand at the rack — you never
flip anything yourself.

Write for that orientation and nothing else. The floor's **rotation** setting
(0/90/180/270°, set in the web UI or with `--rotation`) turns the finished
picture clockwise so its "up" can face the audience wherever the venue puts
the Pi. The runner applies it after `render()`, including to your effect
writes, so an animation never sees it. Renderers show the rotated picture, as
the floor does.

`PixelFrame.grid` gives you the 136×136×3 image (dark cells zero) and
`PixelFrame.from_grid(img)` samples one back, keeping only the lit cells. That
is the image path `plasma.py` uses.

## 4. `FrameContext`

| Field | Type | Meaning |
| --- | --- | --- |
| `ctx.frame` | `int` | Frames since this animation started; 0 on the first call |
| `ctx.t` | `float` | Seconds since it started, on the animation's own clock. **Use this for motion**, not wall time or `ctx.frame`: it advances exactly `1/fps` per frame however late a frame renders, so motion never stutters, and the show's Speed control scales it |
| `ctx.dt`, `ctx.fps` | `float` | Seconds per frame on that clock (nominal `1/fps` times the show speed), target rate |
| `ctx.params` | `dict` | Your `Param` defaults ← playlist overrides ← live UI edits, already validated |
| `ctx.geometry` | `FloorGeometry` | The floor: sizes, lookup tables, edges, rings (§5) |
| `ctx.state` | `dict` | `{}` on frame 0, then yours until the animation is stopped. Particle lists, phase, anything |
| `ctx.rng` | `random.Random` | Seeded per run: a recording reproduces exactly |
| `ctx.np_rng` | `numpy.random.Generator` | Same seed, for numpy-shaped APIs (`EdgeGraph.walk` takes it) |
| `ctx.beat` | `BeatInfo \| None` | Where the music is, when a beat source is running (below); else `None` |
| `ctx.t_beats` | `float` | Beat time: the music's position, or `ctx.t` at the fallback tempo. **Use this for anything on the beat** (below) |
| `ctx.triggers` | `tuple[Trigger, ...]` | Hits since the last frame - pads, notes, the web UI's trigger buttons. Usually `()` (below) |
| `ctx.palette` | `Palette` | The floor's active palette; follow it with `palette_param()` (below) |
| `ctx.send_effect(tile, effect)` | | Write a tile's effect register (see below) |

There is deliberately no "time remaining". Fading out is the runner's job.

**Show speed.** An operator can run everything faster or slower (0–4×) from
the web UI or, later, a MIDI knob. It works by scaling `ctx.t` and `ctx.dt`,
so an animation that moves by either follows it for free — including one
that spawns things at `rate * ctx.dt`. Motion counted in `ctx.frame` does not
follow it. At speed 0 time stops: `ctx.dt` is 0 and `ctx.t` holds, but
`render()` is still called every frame. The show's other controls — freeze,
strobe, bump, tint, hue and saturation — act on your finished frame, so you
never see them.

**Effects** are registers on each tile — an id and four parameter bytes — that
transform the tile's pixels on their way to the LEDs, and persist until
rewritten. `effect=Effect(FADE, (230, 0, 0, 0))` in the decorator writes one to
every tile on frame 0; `ctx.send_effect()` writes one at any time. A write costs
that tile its pixel update for one frame. The effects themselves are still
being defined in the tile firmware, so treat this as plumbing for now.

**Layers.** Any animation can also run as a *layer* over whatever is playing
(the Layer button in the Animations tab), composited with add, max, multiply
or mix. Nothing changes for the author, with one exception: tile effects
belong to the base animation, so a layer's `send_effect` writes are dropped.

**The beat.** With a beat source running (Ableton Link, tap tempo; see
[external-input.md](external-input.md#beat-sync)) `ctx.beat` says where the
music is at the moment your frame is *seen*, latency included:

| Field | Meaning |
| --- | --- |
| `tempo` | BPM, after the operator's ½× / 1× / 2× |
| `beat`, `phase` | a beat count and 0..1 within the beat; `beat + phase` is a continuous position |
| `bar_phase`, `beats_per_bar` | 0..1 within the bar, and its length |
| `downbeat` | `True` on exactly one frame per bar: the one that crossed the bar line |

It is `None` whenever there is no tempo - no source, no Link peers, the
MIDI clock stopped.

**Beat time.** Most of the time you want `ctx.t_beats` instead: the music's
position (`beat + phase`) when there is a beat, and your own `ctx.t` at the
fallback tempo (a setting, default 120 BPM) when there is not. Anything
written against it locks to the music when there is music and runs
sensibly on its own when there is not - no `None` check. Shape it with the
waveforms in `df2_pi.tempo`, which all return 0..1 and take arrays as well
as numbers, so one call can light every tile at its own offset:

```python
from df2_pi.tempo import lfo, pulse

level = lfo(ctx.t_beats, rate=0.5)              # a swell every two beats
pump = pulse(ctx.t_beats, decay=0.3)            # 1 on each beat, dying away
wave = lfo(ctx.t_beats - offsets, rate=0.25)    # offsets: an array, one per tile
```

`rate` is cycles per beat (0.25 is once a bar in 4/4); `lfo` shapes are
`sine`, `tri`, `saw`, `ramp_down` and `square`.
[`rainbow_sweep.py`](../src/pi/animations/rainbow_sweep.py) sweeps once per
two bars of beat time and pumps with `pulse()`;
[`checkerboard.py`](../src/pi/animations/checkerboard.py) reads `ctx.beat`
directly. Declare `sync="beat"` when you follow the beat.

**Phase maps.** Lighting desks build most effects from one idea: a waveform
run across a group of fixtures, each offset in phase. `df2_pi.phase` gives
you the offsets for the floor - an array in 0..1, `(rows, cols)` like a
`TileFrame` or, with `resolution="led"`, `(tiles, 60)` like a `PixelFrame` -
and the waveforms take them as an `offset`:

```python
from df2_pi import phase
from df2_pi.tempo import lfo

offsets = phase.radial(ctx.geometry, spread=1.0)                   # rings out from the middle
level = lfo(ctx.t_beats, rate=0.5, shape="sine", offset=-offsets)  # one level per tile
```

| Map | Offset by |
| --- | --- |
| `rows`, `cols`, `diagonal` | row, column, row + column (0 at row 0, nearest the Pi, and the west edge) |
| `radial` | distance from a point, `(x, y)` in cell units; the floor's centre by default |
| `angle` | angle round a point: 0 due north (up as displayed), clockwise |
| `checker` | alternating tiles, 0 and half a cycle |
| `random` | a fixed shuffle; pass `ctx.np_rng` and it is the same all run |
| `perimeter` | position round each tile's LED ring, in chain order (LED resolution only) |

`spread` scales the offsets (0 in phase, 1 one cycle across the floor),
`reverse` runs them the other way, and `mirror` makes them symmetric about
the floor's centre - rows fold to fan in from both edges, angle matches
east to west. `phase.by_name()` takes a map's name, for an animation that
lets its user choose one, as [`waves.py`](../src/pi/animations/waves.py) does.

**Palette.** The floor has an active palette, chosen in the web UI's Palettes
tab and transport bar or by a desk (DMX control channel 18), so a lighting
designer can set the room's colours and every animation that follows it
falls in line. To follow it, declare a `palette` param with
`palette_param()` - its choices are `"floor"` (the default: whatever the
floor is set to) and the built-in library - and resolve it with `choice()`:

```python
from df2_pi.palette import choice, palette_param

params={"palette": palette_param()}
...
pal = choice(ctx, ctx.params["palette"])   # ctx.palette for "floor", else the named one
colour = pal.at(0.25)                      # (3,) uint8
colours = pal.at(offsets)                  # (..., 3) uint8: one per tile or LED
```

A palette is 2–8 colour stops spaced evenly round a loop: `at(0)` is the
first, and past the last it blends back to the first, so a value that keeps
rising cycles through the scheme without a jump. Blending is in linear light,
so the middle of red and green is a bright yellow. A palette change lands at
the next frame; [`chase.py`](../src/pi/animations/chase.py) and
[`waves.py`](../src/pi/animations/waves.py) follow it. Animations without a
`palette` param are unaffected.

**Triggers.** `ctx.triggers` is a tuple of `Trigger(slot, velocity, age_s)`
for the hits that arrived since the last frame: pads on the web UI,
notes, the API's `POST /api/transport/trigger`. Each is delivered to
exactly one frame - every animation rendered on it, so a crossfade's
incoming animation sees it too - and `age_s` says how long before that
frame is seen it arrived. `slot` is 0..15 and what it means is yours:
[`ripple.py`](../src/pi/animations/ripple.py) drops a ring at the slot's
place on a 4×4 grid, as bright as the velocity. Declare `triggers=True` and
the web UI shows trigger pads while your animation plays.

### Video clips

Real footage - surf, fire, drifting cloud - plays through the **Video**
animation. Import a clip on a laptop (it needs the `[preview]` extra, for
ffmpeg):

```bash
df2-pi video import surf.mp4 --name surf --start 4 --duration 30   # --crop 0.5,0.4,0.8 to frame it
df2-pi video list
```

The import crops the video square, averages each frame down to the floor's
136×136 cells in linear light and keeps the 3,840 that are LEDs, at 15 frames
a second (`--fps`). The top of the video is the top of the floor as
displayed. Clips land in `src/pi/media/` (or `$DF2_MEDIA`), which
`sync-to-pi.sh` copies to the Pi and git ignores - they are large, and
usually someone else's footage, so check its licence. Reload animations
for Video to list a new clip.

The picture is seen through the lattice of tile edges, so big, soft
movement reads best: surf shot from above, flames, cloud. Video blends
between stored frames in linear light, so slowed right down it still moves
smoothly, crossfades its end into its start (Loop fade) so the seam does not
show, and has Contrast, Black level - dark footage usually wants a little,
since the floor shows black poorly - Brightness and Rotation.

## 5. Structural access — the floor is 256 line segments

The floor is not an image with the middles missing. It is 64 tiles × 4 sides =
**256 runs of 15 LEDs** meeting at tile corners, and the interesting animations
are the ones that know it. `ctx.geometry` gives you that structure directly:

```python
geo = ctx.geometry
geo.edges                    # all 256 Edge objects: tile, side, 15 LEDs, start/end (x, y)
geo.edge(tile, Side.NORTH)   # one of them
geo.seams                    # 112 (Edge, Edge) pairs facing each other across a tile frame
geo.floor_ring               # the outer boundary: 480 LED indices, one ordered loop
geo.tile_ring(tile)          # one tile's 60 LEDs, clockwise from its top-left
geo.rails(Axis.X, i)         # a straight 120-LED line across the whole floor
geo.edge_graph()             # edges as a graph: walk(), shortest_path(), branch()
geo.led_positions            # (64, 60, 2) float (y, x) centre of every LED
```

Two things make this usable:

- **Edges are ordered in floor orientation, not chain orientation.** Every
  horizontal run goes west→east and every vertical run south→north, whatever
  way the WS2815 strip happens to wind on that side. So an effect travelling
  across a tile boundary keeps going instead of zigzagging.
- **Anything spanning tiles uses flat indices** — `tile * 60 + led` — which go
  straight into `frame.flat`, the `(3840, 3)` view of the frame:

```python
ring = geo.floor_ring
frame.flat[ring[(head - np.arange(40)) % 480]] = comet_colours   # chase.py
```

Seams matter because adjacent tiles do not share LEDs: they present two
parallel runs one cell apart, and `seams` pairs them so `a.leds[i]` is directly
opposite `b.leds[i]` (`seams.py`).

**The edge graph** is what a lightning bolt is made of. Edges are nodes,
adjacent where they meet at a tile corner:

```python
g = geo.edge_graph()
bolt = g.walk(start_edge, length=14, rng=ctx.np_rng, turn_bias=0.35)  # self-avoiding
forks = g.branch(bolt, ctx.np_rng, n=2)                                # lightning forks
frame.flat[bolt.leds()] = colour          # LEDs in travel order, no duplicates
bolt.positions()                          # (n, 2) float (x, y) of those LEDs
```

`turn_bias` 0 runs straight whenever it can; 1 turns at every corner.
`walk()` goes wherever the graph allows. For a path with a direction to it,
step corner by corner yourself: `g.edges_at(junction)` lists the edges
meeting at a tile corner and each edge's `junctions` say where it leads, so
you choose. [`lightning.py`](../src/pi/animations/lightning.py) does this so
its bolts only ever go down or sideways, and weights its forks toward the
ground.

**Continuous coordinates.** For effects that think geometrically, everything is
in cell units, `(x, y)`, 0–136 on both axes:

```python
frame.splat(x, y, colour, radius=4.0, falloff="gaussian", blend="add")
frame.line((x0, y0), (x1, y1), colour, width=1.5)
frame.circle(cx, cy, r, colour, width=1.0)
```

These rasterise onto **lit cells only** — a diagonal line lights whichever edge
LEDs it passes near and nothing in the dark interiors, which is what someone
standing on the floor sees. They mutate the frame, so draw onto your copy of
`previous`, never onto `previous`. For anything the primitives don't cover,
`led_positions` and a numpy expression give you a per-LED distance field that
indexes straight into `frame.data` ([`ripple.py`](../src/pi/animations/ripple.py)).

## 6. Parameters

```python
params={
    "speed": Param(float, default=1.0, min=0.1, max=5.0, label="Speed"),
    "mode":  Param(str, default="up", choices=["up", "down"]),
    "wobble": Param(bool, default=False),
}
```

`Param(type, default, min=, max=, choices=, label=, help=, role=, macro=, curve=)`. The specs are what
the web UI turns into controls automatically — a float with bounds becomes a
slider, `choices` a select, a bool a switch — and what `--param speed=2` and
playlist overrides are validated against. Read them as `ctx.params["speed"]`.
Playlists store only the values that differ from your defaults, so changing a
default in the file changes it everywhere it wasn't overridden.

### External control: roles, macros and curves

A MIDI knob, a DMX channel or an OSC fader sends a value with no idea what is
playing. Three optional `Param` fields let one control do something sensible
whatever animation is up:

```python
params={
    "speed": Param(float, default=120, min=10, max=480, curve="log"),   # role "speed", by its name
    "rate":  Param(float, default=1.5, min=0.1, max=10, role="density", curve="log"),
    "tail":  Param(int, default=40, min=2, max=200, macro=1),
}
```

- **`role`** is one of `speed`, `intensity`, `density`, `scale`, `hue`,
  `variation`. A control bound to a role reaches the playing animation's param
  with that role; an animation without one ignores it. A param *named* after a
  role has it automatically, so declare `role=` only to give a differently
  named param one — never rename a param to get a role, because playlists
  store overrides by name. Roles carry meaning, not units.
- **`macro=1..4`** puts a param under a general-purpose knob — the animation's
  most interesting control, role or not. Every starter animation fills macro 1;
  yours should too. The web UI badges it (M1).
- **`curve="log"`** maps a 0–1 control along a log scale — right for ranges
  that span decades, like a rate or a speed. The web UI's slider follows the
  same curve, so it agrees with the knob. Needs `min > 0`.

Each role and macro is claimed by at most one param per animation, and only a
param with something to map onto (numbers with `min` and `max`, `choices`, or a
bool) can claim one; both are checked when the file loads. A control value
lands where a web UI edit does — on top of the playlist's overrides.

Two more keywords describe how an animation responds to the music:
`sync="beat"` if it follows `ctx.beat`, and `triggers=True` if it reacts to
trigger hits. They are served to the UI for filtering and labelling.

Anything else you pass to `@animation(...)` — `preview_hint="loop"`,
`energy="low"` — is kept verbatim in `meta.extra` and served to the UI as-is.
`period=` is advisory: the natural loop length in seconds, so the playlist
editor can offer durations that don't cut mid-cycle.

## 7. The authoring loop

```bash
df2-pi play --no-hardware --terminal --animation mine              # full grid (needs ~136 columns)
df2-pi play --no-hardware --terminal=tiles --animation mine        # 8x8 view, fits anywhere
df2-pi play --no-hardware --window --animation mine                # a window with the LED bloom (pip install -e ".[preview]")
df2-pi play --no-hardware --animation mine --param speed=3 --record out.gif --frames 90
df2-pi animations -v                                               # what loaded, its params, what failed
```

Edit, save, run again. On the floor itself, drop `--no-hardware`. To put it in
a show: `df2-pi playlists add "Party" mine --duration 45 --param speed=2`.

## 8. Performance

The floor runs at **30 FPS: 33 ms per frame**, and that budget is shared:

| Phase | Typical | Notes |
| --- | --- | --- |
| `render` | 0.05–1 ms for the pack | **yours** |
| `encode` | < 1 ms | frame → bytes |
| `wire` | 0.5–19 ms | 32 B/row for uniform tiles up to 1,448 B/row fully addressed |
| tile-side | ~15 ms | after the last byte lands; not on the Pi's clock |

`df2-pi play` prints the frame count, drops, and latch jitter when it exits;
the admin page shows the same telemetry live (p50 / p95 / max per phase). A
render over a few milliseconds on the Pi is worth looking at; over ~10 ms it
will start dropping frames on a fully addressed floor.

What makes a render slow, in order of likelihood:

1. **Python loops over LEDs or cells.** 3,840 LEDs × 30 FPS is 115k iterations
   a second before you have done anything. Use numpy expressions over the whole
   frame (`plasma.py`, `ripple.py`), or over precomputed index arrays.
2. **Rebuilding the same lookup every frame.** Gather index arrays once on frame
   0 into `ctx.state` ([`seams.py`](../src/pi/animations/seams.py) does this
   for its 224 edges).
3. **Graph searches every frame.** `walk()` and `branch()` are cheap but not
   free; run them when a bolt strikes, not per frame.
4. **Full-frame allocation you didn't need.** `previous.copy()` is 11 KB and
   fine; building several 136×136 float images per frame adds up.

A frame that overruns is not an error — the clock skips the missed deadline and
counts it — but persistent overruns show up as a warning in the admin page and
as visible stutter on the floor.
