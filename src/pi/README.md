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
  transport/    Serial transport (pyserial) over the Pi hat's UART
  cli.py        Command-line entry point (`df2-pi`)
test/           Automated unit tests (pytest) - no hardware required
test/integration/
                Scripts that drive a real Pi + row controller + tile.
                See test/integration/README.md.
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
df2-pi --port /dev/serial0 status 0
```

Or without installing the console script:

```bash
python3 -m df2_pi status 0
```

## Testing

```bash
pytest
```

Hardware-in-the-loop scripts live in [test/integration/](test/integration/)
and are run manually on a Pi wired to a live Row Bus - see that directory's
README.

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
