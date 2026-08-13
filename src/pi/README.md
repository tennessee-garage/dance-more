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
  cli.py        Command-line entry point (`df2-pi`)
test/           Automated unit tests (pytest) - no hardware required
test/integration/
                Scripts that drive a real Pi + row controller + tile.
                See test/integration/README.md.
```

## Two-chain topology

The floor's 8 rows are split across **two RS-485 chains** so the worst-case
frame fits the 33 ms budget (docs/row-bus-protocol.md §1). A single Cat5 run
passes every row controller in physical order, each tapping the opposite pair
from its neighbour, so the split is **alternating**: rows 0,2,4,6 on chain 0,
rows 1,3,5,7 on chain 1.

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

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
df2-pi scan          # find responding row controllers on every chain
df2-pi status 0      # query one row
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
