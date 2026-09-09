# Dance Floor v2

An 8×8 grid of light-up acrylic tiles driven by a three-tier hierarchy:
**Raspberry Pi** (host) → **8 row controllers** (Xiao RP2350) → **64 tile
controllers** (ATtiny3224 + WS2815).

Design docs live in [docs/](docs/) — start with
[architecture.md](docs/architecture.md), then the two wire protocols
([row-bus-protocol.md](docs/row-bus-protocol.md),
[tile-bus-protocol.md](docs/tile-bus-protocol.md)). Don't restate protocol
detail here; those files are the source of truth.

## Repository map

| Path | What |
| --- | --- |
| `src/pi/` | Python host controller (`df2-pi`) — drives the Row Bus |
| `src/row/` | PlatformIO firmware, Xiao RP2350 row controller |
| `src/tile/` | PlatformIO firmware, ATtiny3224 tile controller |
| `src/common/tile_bus_protocol/` | Wire protocol shared by row + tile firmware |
| `src/simulation/` | Vite/TypeScript acrylic light-propagation simulator |
| `src/native-simulator/` | Standalone ModernGL simulator (unrelated to firmware) |
| `pcb/{pi-hat,row-controller,tile}/` | KiCad projects; versioned gerbers **are** committed |

## Build & test

```bash
# Pi host  (src/pi)
pip install -e ".[dev]"      # then: pytest        - no hardware needed
python3 test/integration/test_row_bus_scan.py --chain /dev/ttyAMA0:23   # on the Pi

# Row controller  (src/row)
pio test -e native                     # unit tests, no hardware
pio run  -e row3                       # production firmware, address baked in at build time (row0-row7)
pio run  -e seeed_xiao_rp2350_debug    # + counter dumps over the idle Tile Bus
pio run  -e seeed_xiao_rp2350_pin_probe # scans every GPIO for signal activity
tools/flash_row.sh 3                   # build + upload the row3 env

# Tile controller  (src/tile)
pio test -e test_native
pio run  -e ATtiny3224

# Simulator  (src/simulation)
npm run dev
```

`src/row` and `src/tile` also carry `test/integration/*.py` — host-side scripts
driving native mocks through a Unix-socket broker, no hardware required.

## Bench setup

A live Pi 5 is at **`garth@testing-pi`**. It is `src/pi` that syncs there, so
`~/dance-floor` is the *package* root — `test/` and `src/df2_pi/` sit directly
under it, not under a `src/pi/`. The venv is `venv`, not `.venv`.

```bash
src/pi/tools/sync-to-pi.sh          # watch + rsync on change
src/pi/tools/sync-to-pi.sh --once
ssh garth@testing-pi 'cd ~/dance-floor && ./venv/bin/pytest -q'
```

The venv is long-lived and does **not** track `pyproject.toml`. Adding a
dependency means `./venv/bin/pip install -e ".[dev]"` there before tests will
collect — otherwise the failure is a `ModuleNotFoundError` at import time that
looks like a broken test rather than a stale environment.

**The bench pi-hat is the old single-chain revision** — one transceiver, XDIR
on GPIO23. Host code defaults to the two-chain board (GPIO17/GPIO7), so bench
runs need `--chain /dev/ttyAMA0:23`. See the `bench-hardware-bodges` memory.

**Flashing a row controller is a physical round trip** and cannot be done
in-situ: power down the bench, pull the RP2350 from its socket, carry it to the
computer, connect USB, flash, carry back, reseat, re-power. Don't retry a
hardware test immediately after an upload — wait to be told it's back.

Pi one-time setup: serial console **disabled** (`raspi-config nonint
do_serial_cons 1`) or a getty holds the UART; chain 1 additionally needs
`dtoverlay=uart2-pi5,ctsrts` in `/boot/firmware/config.txt`.

## Things that will waste your time if you don't know them

Each is documented where it bites — this is just the index:

- **Row Bus is capped at 3.125 Mbps** by the Pi's 50 MHz UART clock. Asking for
  more fails *silently* (mute bus, looks like a broken wire).
  `docs/row-bus-protocol.md` §1.
- **The two Row Bus chains must be driven concurrently** — it is a design
  requirement, not an optimization anyone is free to skip. The pi-hat commits
  to it in copper (two transceivers into one 8P8C), and the 33 ms frame budget
  does not close without it: serializing the chains takes the worst-case floor
  update from ~33.6 ms to ~52 ms, i.e. 30 FPS to 19. `docs/row-bus-protocol.md`
  §1 and §8. `Floor.broadcast()` does this; `Floor.send()` does **not** yet.
- **LED geometry is 15/side, 60/tile, and everything derives from it.** Buffer
  and payload sizes are computed from `LEDS_PER_SIDE` in
  `src/common/tile_bus_protocol/protocol.h` (firmware) and `FloorGeometry`
  (host) — don't reintroduce literal 60/180/1448s. Tile Bus's 1-byte `LEN`
  caps a side at 20 LEDs at 3 bytes/LED; a `static_assert` enforces it.
- **Use `/dev/ttyAMA0`, never `/dev/serial0`** — on a Pi 5 that symlink is the
  SoC's internal console UART, not the header. `row_bus.py` docstring.
- **Never release RS-485 direction with `serial.flush()`** — `tcdrain()`
  overshoots ~6 ms. `row_bus.py` `start_write()`.
- **Driving GPIO14/15 as plain GPIOs permanently steals them from the UART**
  until re-muxed (`pinctrl set 14 a4`). Cost an hour of false readings once.
- **Verify board pin assignments against the KiCad netlist**, not the issue
  spec — `pins.h` had three rotated assignments that silently disabled the UART.
  `pio` can't catch this; the `static_assert`s in `pi_transport_rp2350.cpp` now can.
- Check part specs against the **real datasheet** before flagging anything.

## Conventions

- Branches: `feature/#<issue>-<slug>` where an issue exists.
- Firmware and host changes that share a protocol behaviour go in **separate
  commits**, one per side.
- **Bump the firmware version before opening a PR.** Any non-`.md` change under
  `src/row/` bumps `ROW_FW_VERSION`, under `src/tile/` bumps
  `TILE_FW_VERSION`, and anything under `src/common/tile_bus_protocol/` bumps
  **both** — it ships in both images. CI enforces this
  (`.github/workflows/fw-version-bump.yml`) but only on `pull_request`, so a
  push straight to `main` slips through and the constants silently stop
  meaning anything. Gaps are fine; going backwards is not.
- Row addresses are global `0x00`–`0x07` even though the floor runs two RS-485
  chains; only the wiring is partitioned, and row firmware is unaware of it.

## Known-stale docs

`communication.md`, `glossary.md`, `architecture.md` and
`hardware-row-controller.md` still describe a **single** Row Bus on a 4P4C
connector. The floor runs **two** chains over one RJ45 (rows 0,2,4,6 on
chain 0; 1,3,5,7 on chain 1) — see [docs/row-bus-protocol.md](docs/row-bus-protocol.md) §1
and [src/pi/README.md](src/pi/README.md), which are current. (§1 only became
current on this in Aug 2026; before that both this note and `src/pi/README.md`
pointed at a section that never mentioned chains.)
