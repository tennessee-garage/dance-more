"""Floor.broadcast()'s admin-command guard - the only piece of Floor that's
pure logic. Everything else opens a real serial port and GPIO line, which
belongs in test/integration/, not here.

RowBus is mocked out entirely: this test is about whether broadcast() raises
before touching hardware, not about RowBus itself (see test_row_bus's
absence - there isn't one, for the same reason; it's exercised for real by
test/integration/test_row_bus_scan.py).
"""

from unittest.mock import MagicMock, patch

import pytest

from df2_pi.protocol.constants import Cmd
from df2_pi.transport.chain_map import RowChainMap
from df2_pi.transport.floor import BROADCAST_SAFE_CMDS, ChainConfig, Floor


def _mock_bus() -> MagicMock:
    bus = MagicMock()
    bus.start_write.return_value = 0.0  # already-elapsed deadline
    return bus


def make_floor(num_chains: int = 1) -> Floor:
    configs = [ChainConfig(f"/dev/fake{i}", xdir_pin=None) for i in range(num_chains)]
    with patch("df2_pi.transport.floor.RowBus", side_effect=lambda **_: _mock_bus()):
        return Floor(chains=configs, chain_map=RowChainMap.alternating(num_chains))


def test_broadcast_safe_set_is_exactly_latch_and_blackout():
    # This set is what actually decides collision safety - pin it down
    # explicitly rather than only testing through broadcast()'s exceptions.
    assert BROADCAST_SAFE_CMDS == {Cmd.LATCH, Cmd.BLACKOUT}


@pytest.mark.parametrize("cmd", [Cmd.TEST, Cmd.STATUS, Cmd.POWER, Cmd.RE_DISCOVER, Cmd.ERROR_LOG])
def test_broadcast_rejects_admin_commands(cmd):
    floor = make_floor()
    with pytest.raises(ValueError, match="not broadcast-safe"):
        floor.broadcast(cmd)
    floor.close()


def test_broadcast_rejects_before_touching_any_bus():
    # The guard must fire before start_write() - a caller shouldn't get a
    # half-sent broadcast out of a rejected call.
    floor = make_floor(num_chains=2)
    with pytest.raises(ValueError, match="not broadcast-safe"):
        floor.broadcast(Cmd.STATUS)
    for bus in floor._buses:  # noqa: SLF001 - asserting internal state is the point here
        bus.start_write.assert_not_called()
    floor.close()


def test_broadcast_allows_latch_and_blackout():
    floor = make_floor(num_chains=2)
    floor.broadcast(Cmd.LATCH)
    floor.broadcast(Cmd.BLACKOUT)
    for bus in floor._buses:  # noqa: SLF001
        assert bus.start_write.call_count == 2
        assert bus.finish_write.call_count == 2
    floor.close()


def test_latch_and_blackout_convenience_methods_go_through_the_guard():
    floor = make_floor()
    floor.latch()
    floor.blackout()
    bus = floor._buses[0]  # noqa: SLF001
    assert bus.start_write.call_count == 2
    floor.close()
