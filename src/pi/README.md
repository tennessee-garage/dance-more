# df2-pi

Raspberry Pi host controller for Dance Floor v2. This is Tier 1 of the
control hierarchy described in [docs/architecture.md](../../docs/architecture.md):
it generates animation frames and drives the Row Bus, the RS-485 link to the
8 row controllers, through the Pi hat. Wire format and command set are
specified in [docs/row-bus-protocol.md](../../docs/row-bus-protocol.md).

## Layout

```
src/df2_pi/
  protocol/     Row Bus wire format: constants, CRC-16, frame encode/decode
  transport/
    row_bus.py    One RS-485 chain: pyserial + XDIR direction control
    chain_map.py  Logical row (0-7) -> physical chain
    floor.py      The whole floor: routes rows to chains, fans out broadcasts
  geometry.py   FloorGeometry: what is an LED and where - the 136x136 cell
                grid, chain order, tile <-> cell <-> bus lookup tables
  pixels.py     TileFrame (one colour per tile) and PixelFrame (every LED,
                chain order); grid view, conversions, gain/blend, ownership
  edges.py      The floor as 256 runs of 15: Edge, seams, floor ring, rails,
                and EdgeGraph walks/paths for effects that travel the edges
  paint.py      splat / line / circle in continuous (x, y) cell coordinates,
                rasterised onto lit cells only
  gamma.py      The one gamma curve (linear <-> encoded bytes)
  encode.py     FrameEncoder: frame -> one SEND_DATA payload per row, with
                SET_COLOR for uniform tiles, effect entries, and the
                brightness / white-balance / gamma LUT
  effects.py    Effect: the tile effect-register value (id + 4 params)
  animation/    The authoring contract: @animation + Param (meta.py),
                FrameContext (context.py), one-file loading and
                AnimationRun (loader.py), AnimationRegistry with safe
                reload (registry.py)
  engine/
    clock.py    FrameClock: drift-free deadlines, hybrid sleep, overrun and
                stall policy, per-phase timing telemetry
    runner.py   Runner: walks a playlist, renders, crossfades, isolates a
                failing animation, and takes queued control commands at
                frame boundaries; RunnerState is the per-frame snapshot
  output/       Where frames go. sink.py: the Sink protocol, the latest-wins
                Mailbox and ThreadedSink observers (Null, Callback);
                hardware.py: HardwareSink, the one synchronous sink;
                preview.py: the binary preview wire format, PreviewSink with
                per-subscriber rate/format, RecorderSink; fanout.py: FanOut
  playlists/    PlaylistStore: playlists, entries, settings and the play
                log in SQLite (WAL, one connection per thread), resolved
                against the animation registry; schema.py migrations
  output/dev.py TerminalSink, WindowSink (pygame), FrameCollector with
                GIF / mp4 / .df2rec export; all draw origin bottom-left
  cli.py        Command-line entry point (`df2-pi`): play, animations,
                playlists, ledwalk, tilewalk, and the Row Bus admin commands
animations/     The animations themselves, one .py per animation; the
                filename stem is the id playlists reference
test/           Automated unit tests (pytest) - no hardware required
test/integration/
                Scripts that drive a real Pi + row controller + tile.
                See test/integration/README.md.
```

## Two-chain topology

The floor's 8 rows are split across **two RS-485 chains, driven concurrently**,
so the worst-case frame fits the 33 ms budget — see
[docs/row-bus-protocol.md](../../docs/row-bus-protocol.md) §1, which states the
budget and why the concurrency is a requirement rather than an optimization. A
single Cat5 run passes every row controller in physical order, each tapping the
opposite pair from its neighbour, so the split is **alternating**: rows 0,2,4,6
on chain 0, rows 1,3,5,7 on chain 1.

| | Chain 0 | Chain 1 |
| --- | --- | --- |
| UART | `uart0` → `/dev/ttyAMA0` | `uart2` → `/dev/ttyAMA2` |
| TX / RX | GPIO14 / GPIO15 | GPIO4 / GPIO5 |
| XDIR | GPIO17 (`RTS0`) | GPIO7 (`RTS2`) |

Needs both UARTs enabled in `/boot/firmware/config.txt`:

```
dtoverlay=uart0-pi5,ctsrts
dtoverlay=uart2-pi5,ctsrts
```

Row addresses stay global (`0x00`–`0x07`) — only the wiring is partitioned,
so row controller firmware is unaware there is more than one chain. `Floor`
routes unicast by row and fans `LATCH`/`BLACKOUT` out to every chain at once,
writing all chains before waiting on any so cross-chain skew stays in the
microseconds.

> **Not yet concurrent for `SEND_DATA`.** `Floor.broadcast()` overlaps the
> chains as described, but `Floor.send()` — and so `send_data()` — writes one
> chain and waits out its whole frame time before touching the other. At
> worst-case frame sizes that costs ~18.6 ms per floor update, doubling the
> Row Bus phase and putting 30 FPS out of reach. `RowBus.start_write()` /
> `finish_write()` already expose the split needed to fix it; the frame loop
> that will use it is part of the driver epic.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"            # + ".[preview]" for --window and --record to gif/mp4
```

## Usage

Playing:

```bash
df2-pi play                                  # the startup playlist, on the floor
df2-pi play --playlist "Party"
df2-pi play --animation rainbow_sweep --param speed=2   # one animation, looping

df2-pi play --no-hardware --terminal         # no floor: render in the terminal
df2-pi play --no-hardware --terminal=tiles   # ...as the 8x8 tile view
df2-pi play --animation ripple --window      # a real window (pip install -e ".[preview]")
df2-pi play --no-hardware --animation chase --record out.gif --frames 90

df2-pi animations -v                         # what was found, params, and load errors
df2-pi playlists                             # list; also show/create/add/move/remove/set-startup/delete
```

`--no-hardware` never imports the serial or GPIO stack, so all of that
runs on a laptop. Every dev renderer draws the floor with row 0 along the
bottom, as you see it standing at the rack.

To write an animation, copy [animations/solid.py](animations/solid.py) and
read [docs/animations.md](../../docs/animations.md); the rest of the
starter pack in [animations/](animations/) is the tutorial.

Verifying a build:

```bash
df2-pi ledwalk --row 0 --slot 0   # one LED at a time: LED 0 must light lower-left, then climb the left side
df2-pi tilewalk                   # one tile at a time: tile 0 nearest the Pi, 0-7 sweeping along row 0
```

Neither discovers anything - the LED winding and floor orientation are
fixed in `geometry.py` - so a tile or row that walks the wrong way is a
build fault to fix with a screwdriver, not a setting.

Row Bus admin:

```bash
df2-pi scan          # find responding row controllers on every chain
df2-pi status 0      # query one row
df2-pi version       # firmware versions across rows and tiles
df2-pi blackout      # black out the whole floor
```

`--chain PORT:XDIR_GPIO` (repeatable) overrides the two-chain defaults — use
it for a single-chain bench board:

```bash
df2-pi --chain /dev/ttyAMA0:23 scan
```

Or without installing the console script: `python3 -m df2_pi scan`.

## Testing

```bash
pytest
```

Hardware-in-the-loop scripts live in [test/integration/](test/integration/)
and are run manually on a Pi wired to a live Row Bus - see that directory's
README. The main one takes the same `--chain` flag:

```bash
python3 test/integration/test_row_bus_scan.py --chain /dev/ttyAMA0:23
```

## Syncing to the Pi

`tools/sync-to-pi.sh` pushes this directory's contents to a Pi over rsync
and, by default, watches for local changes and re-syncs automatically
(needs `fswatch`: `brew install fswatch`).

```bash
tools/sync-to-pi.sh              # sync once, then watch and auto-resync
tools/sync-to-pi.sh --once       # sync once and exit

# override the default target (garth@testing-pi:/home/garth/dance-floor)
PI_HOST=user@host PI_DEST=/path/on/pi tools/sync-to-pi.sh
```
