# Getting Started

How to get this repository running on your own machine: the host software
with no floor attached, the firmware test suites, and builds of both firmware
images. Nothing here needs hardware until [Running on real hardware](#running-on-real-hardware).

For what the pieces are and how they talk to each other, read
[architecture.md](architecture.md) first. Note it still describes a single
Row Bus; the floor runs two chains (see [row-bus-protocol.md](row-bus-protocol.md) §1).

## What's in the repo

| Path | What | Needs |
| --- | --- | --- |
| `src/pi/` | `df2-pi`, the Python host: animations, playlists, the web UI, the Row Bus driver | Python 3.10+ |
| `src/row/` | Row controller firmware (Xiao RP2350) | PlatformIO |
| `src/tile/` | Tile controller firmware (ATtiny3226) | PlatformIO |
| `src/common/tile_bus_protocol/` | Wire protocol shared by row and tile firmware | (built by both) |
| `src/simulation/` | Acrylic light-propagation simulator (Vite/TypeScript) | Node.js |
| `src/native-simulator/` | The same idea in Python/ModernGL, separate from everything else | Python |
| `pcb/` | KiCad projects for the boards | KiCad |

Most people only need `src/pi/`. That is where animations live, and with
`--no-hardware` the whole host runs on a laptop.

## Prerequisites

- **git**
- **Python 3.10 or newer**, for `src/pi/`
- **PlatformIO Core** (the `pio` command), only for firmware:
  `brew install platformio` or `pipx install platformio`. The VS Code
  PlatformIO extension works too. The first build of each environment
  downloads its toolchain, so it takes a few minutes.
- **A C++17 compiler and `make`**, for the firmware's native tests and
  integration mocks. On macOS that's the Xcode command-line tools
  (`xcode-select --install`).
- **Node.js**, only for `src/simulation/`.

Linux and macOS both work. Windows is untested.

```bash
git clone git@github.com:tennessee-garage/dance-more.git
cd dance-more
```

## The host (`src/pi`)

### Install

```bash
cd src/pi
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

`dev` pulls in pytest and everything `df2-pi serve` uses (the web UI, beat
sync, MIDI). Add `".[dev,preview]"` if you want `--window` or to record
GIF/mp4 files. The Raspberry Pi GPIO library is Linux-only and is skipped
on other platforms, which is fine with `--no-hardware`.

### Run the tests

```bash
pytest
```

No hardware is needed. A few slow timing tests only run with
`DF2_SLOW_TESTS=1`.

### Run it with no floor

`--no-hardware` never touches the serial port or GPIO, so these all work on
a laptop:

```bash
df2-pi play --no-hardware --terminal                 # the startup playlist, drawn in the terminal
df2-pi play --no-hardware --terminal=tiles           # ...as the 8x8 tile view
df2-pi play --no-hardware --animation chase --window # one animation in a window (needs [preview])
df2-pi animations -v                                 # what animations loaded, with their params
```

The admin UI, served locally:

```bash
df2-pi serve --no-hardware --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/>. You get the transport bar, a live WebGL
preview of the floor and the playlist editor, the same page the Pi serves.

Playlists and settings are stored in SQLite at
`~/.local/share/df2/df2.sqlite3`. Set `DF2_DB=/some/other.sqlite3` (or pass
`--db`) to keep a throwaway database while experimenting.

### Writing an animation

Copy [src/pi/animations/solid.py](../src/pi/animations/solid.py) to a new
file in the same folder. The filename is the animation's id. Then read
[animations.md](animations.md). The other files in `animations/` work as
worked examples. The loop is edit, save, run again:

```bash
df2-pi play --no-hardware --terminal=tiles --animation mine
```

§7 of animations.md, "The authoring loop", has the other ways to view it.

The full command reference (`play`, `serve`, `playlists`, the Row Bus admin
commands) is in [src/pi/README.md](../src/pi/README.md).

## The firmware (`src/row`, `src/tile`)

### Unit tests

These compile the firmware logic for your computer and run it there:

```bash
cd src/row  && pio test -e native
cd src/tile && pio test -e test_native
```

### Integration tests (no hardware)

These scripts build native mock row controllers and tiles, connect them
through a Unix-socket broker and check them against each other: tile
discovery, SENSE mapping and Row Bus broadcast. Run each one from its
project directory:

```bash
cd src/row  && python3 test/integration/test_native_discovery.py
cd src/row  && python3 test/integration/test_row_bus_broadcast.py
cd src/tile && python3 test/integration/test_sense_mapping.py
```

Each one rebuilds what it needs first (`make` for the broker, `pio run -e
native` for the mocks) and ends by printing `PASS`.

### Building the firmware images

```bash
cd src/tile && pio run -e ATtiny3226   # tile controller
cd src/row  && pio run -e row0         # row controller; the address is fixed at build time, row0-row7
```

Flashing needs the physical boards. Row controllers upload over USB with
`tools/flash_row.sh <row>`. Tiles use a serial UPDI programmer (see the
`upload_protocol` in [src/tile/platformio.ini](../src/tile/platformio.ini)).

If you change firmware, bump `ROW_FW_VERSION` and/or `TILE_FW_VERSION` before
you open a PR; CI checks this. The rules are in the repo's `CLAUDE.md`, under
Conventions.

## The simulators (optional)

Both simulators are for studying how light spreads through the acrylic. They
don't connect to the host or the firmware.

```bash
cd src/simulation && npm install && npm run dev        # browser, Vite dev server
```

```bash
cd src/native-simulator
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && python3 main.py    # see its README for controls
```

## Running on real hardware

The host runs on a Raspberry Pi 5 with the pi-hat, which drives the row
controllers over RS-485. Setting up a Pi involves UART overlays, disabling the
serial console and the systemd service. All of that is in
[src/pi/README.md](../src/pi/README.md), with the hardware-in-the-loop
scripts in [src/pi/test/integration/](../src/pi/test/integration/).
`src/pi/tools/sync-to-pi.sh` keeps a Pi's copy in step with your checkout
while you work.
