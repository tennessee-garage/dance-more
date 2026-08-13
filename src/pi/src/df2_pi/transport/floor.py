"""Host-side view of the whole floor: N Row Bus chains behind one API.

Callers address logical rows 0-7 and never pick a chain; Floor routes each
unicast to the chain carrying that row, and fans broadcasts out to every
chain at once. See chain_map.py for the wiring split and
docs/row-bus-protocol.md for the command set.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from ..protocol.constants import ADDR_BROADCAST, Cmd, Resp
from ..protocol.frame import Frame
from .chain_map import RowChainMap
from .row_bus import DEFAULT_BAUDRATE, DEFAULT_XDIR_PIN, RowBus

# Chain 0 is the Pi's uart0 (GPIO14/15) with direction on RTS0 (GPIO17).
# Chain 1 is uart2 (GPIO4/5) with direction on RTS2 (GPIO7), enabled by
# `dtoverlay=uart2-pi5,ctsrts`. Both direction pins moved onto the UARTs'
# own RTS lines in the two-chain hat revision so the kernel's RS-485 mode
# can eventually drive them with hardware timing.
CHAIN0_PORT = "/dev/ttyAMA0"
CHAIN0_XDIR_PIN = 17
CHAIN1_PORT = "/dev/ttyAMA2"
CHAIN1_XDIR_PIN = 7

# Per-command response timeouts, docs/row-bus-protocol.md §7.
RESPONSE_TIMEOUT_S: dict[int, float] = {
    Cmd.TEST: 0.200,
    Cmd.STATUS: 0.020,
    Cmd.POWER: 0.020,
    Cmd.RE_DISCOVER: 0.020,
    Cmd.ERROR_LOG: 0.020,
}
DEFAULT_RESPONSE_TIMEOUT_S = 0.020
MAX_ATTEMPTS = 3  # 1 initial + 2 retries, per §7


@dataclass(frozen=True)
class ChainConfig:
    port: str
    xdir_pin: int | None = DEFAULT_XDIR_PIN
    baudrate: int = DEFAULT_BAUDRATE


def default_chain_configs(baudrate: int = DEFAULT_BAUDRATE) -> list[ChainConfig]:
    return [
        ChainConfig(CHAIN0_PORT, CHAIN0_XDIR_PIN, baudrate),
        ChainConfig(CHAIN1_PORT, CHAIN1_XDIR_PIN, baudrate),
    ]


class RowNotResponding(RuntimeError):
    """A unicast admin command got no valid reply within MAX_ATTEMPTS."""


class Floor:
    def __init__(
        self,
        chains: Sequence[ChainConfig] | None = None,
        chain_map: RowChainMap | None = None,
    ) -> None:
        configs = list(chains) if chains is not None else default_chain_configs()
        self._map = chain_map or RowChainMap.alternating(len(configs))

        if self._map.chain_count > len(configs):
            raise ValueError(
                f"chain map references {self._map.chain_count} chains "
                f"but only {len(configs)} were configured"
            )

        self._buses: list[RowBus] = []
        try:
            for cfg in configs:
                self._buses.append(
                    RowBus(port=cfg.port, baudrate=cfg.baudrate, xdir_pin=cfg.xdir_pin)
                )
        except Exception:
            # Don't leak already-opened ports/GPIOs if a later chain fails.
            self.close()
            raise

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        for bus in self._buses:
            try:
                bus.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        self._buses = []

    def __enter__(self) -> Floor:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- routing ---------------------------------------------------------

    @property
    def chain_map(self) -> RowChainMap:
        return self._map

    @property
    def chain_count(self) -> int:
        return len(self._buses)

    def bus_for(self, row: int) -> RowBus:
        return self._buses[self._map.chain_for(row)]

    def bus_for_chain(self, chain: int) -> RowBus:
        """Direct access to one chain's transport, for diagnostics that need
        to read a specific chain (e.g. after a broadcast, where replies can
        arrive on any chain and there is no single row to route by)."""
        return self._buses[chain]

    # ---- unicast ---------------------------------------------------------

    def send(self, row: int, cmd: int, payload: bytes = b"") -> None:
        """Fire-and-forget unicast to one row (SEND_DATA and friends)."""
        self.bus_for(row).send(row, cmd, payload)

    def request(self, row: int, cmd: int, payload: bytes = b"") -> Frame:
        """Unicast an admin command and return its reply, retrying per §7.

        Raises RowNotResponding after MAX_ATTEMPTS. Only the chain carrying
        `row` is touched, so a dead row can't stall the other chain.
        """
        bus = self.bus_for(row)
        timeout = RESPONSE_TIMEOUT_S.get(cmd, DEFAULT_RESPONSE_TIMEOUT_S)
        # Responses are 0x80 | cmd (§6). Commands with no defined response
        # code fall through with expected=None and accept any reply.
        try:
            expected: Resp | None = Resp(0x80 | cmd)
        except ValueError:
            expected = None

        for _ in range(MAX_ATTEMPTS):
            bus.send(row, cmd, payload)
            frame = bus.read_frame(timeout=timeout)
            if frame is None:
                continue
            if frame.addr != row:
                continue
            if expected is not None and frame.cmd != expected:
                continue
            return frame

        raise RowNotResponding(
            f"row 0x{row:02X} did not answer cmd 0x{cmd:02X} after {MAX_ATTEMPTS} attempts"
        )

    # ---- broadcast -------------------------------------------------------

    def broadcast(self, cmd: int, payload: bytes = b"") -> None:
        """Send one broadcast frame on every chain as near-simultaneously as
        the hardware allows.

        All chains are keyed up and written before waiting on any of them,
        so the skew between chains is the gap between write() calls (single
        -digit microseconds) rather than a full frame time per chain. That
        matters for LATCH, which is what makes every tile across the floor
        illuminate together.
        """
        data = Frame(ADDR_BROADCAST, cmd, payload).encode()

        deadlines = [bus.start_write(data) for bus in self._buses]
        latest = max(deadlines, default=0.0)
        while time.perf_counter() < latest:
            pass
        for bus in self._buses:
            bus.finish_write()

    def latch(self) -> None:
        self.broadcast(Cmd.LATCH)

    def blackout(self) -> None:
        self.broadcast(Cmd.BLACKOUT)

    def send_data(self, row: int, payload: bytes) -> None:
        self.send(row, Cmd.SEND_DATA, payload)

    # ---- discovery -------------------------------------------------------

    def scan(self, attempts: int = 2, timeout: float = 0.15) -> dict[int, int]:
        """Probe every row with STATUS; return {row: chain} for responders.

        Each row is probed only on the chain that should carry it, so a
        responder appearing here also confirms the wiring matches chain_map.
        """
        found: dict[int, int] = {}
        for row, chain in self._map.items():
            bus = self._buses[chain]
            for _ in range(attempts):
                bus.send(row, Cmd.STATUS)
                frame = bus.read_frame(timeout=timeout)
                if frame is not None and frame.cmd == Resp.STATUS_RESP and frame.addr == row:
                    found[row] = chain
                    break
        return found
