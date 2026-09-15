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


def _decode(data: bytes):
    from df2_pi.protocol.frame import FrameParser

    frames = list(FrameParser().feed_bytes(data))
    assert len(frames) == 1
    return frames[0]


def test_send_rows_starts_one_frame_per_chain_before_waiting_on_any():
    # Two chains, alternating: rows 0,2,4,6 on chain 0 and 1,3,5,7 on chain
    # 1, sent in 4 rounds of two. Within a round both start_write()s must
    # precede both finish_write()s - that is the concurrency the 33 ms
    # budget depends on.
    floor = make_floor(num_chains=2)
    order: list[tuple[str, int]] = []
    for chain, bus in enumerate(floor._buses):  # noqa: SLF001
        bus.start_write.side_effect = lambda data, c=chain: (order.append(("start", c)), 0.0)[1]
        bus.finish_write.side_effect = lambda c=chain: order.append(("finish", c))
    payloads = [bytes([row]) * 4 for row in range(8)]
    floor.send_rows(payloads)
    assert order == [("start", 0), ("start", 1), ("finish", 0), ("finish", 1)] * 4
    # each chain saw its rows in row order, as SEND_DATA frames to that row
    for chain, bus in enumerate(floor._buses):  # noqa: SLF001
        frames = [_decode(call.args[0]) for call in bus.start_write.call_args_list]
        assert [f.addr for f in frames] == [r for r in range(8) if r % 2 == chain]
        assert all(f.cmd == Cmd.SEND_DATA for f in frames)
        assert [f.payload for f in frames] == [payloads[r] for r in range(8) if r % 2 == chain]
    floor.close()


def test_send_rows_on_one_chain_is_sequential_in_row_order():
    floor = make_floor(num_chains=1)
    bus = floor._buses[0]  # noqa: SLF001
    floor.send_rows([bytes(4)] * 8)
    assert [_decode(c.args[0]).addr for c in bus.start_write.call_args_list] == list(range(8))
    assert bus.finish_write.call_count == 8
    floor.close()


def test_send_rows_releases_every_started_bus_if_the_wait_is_interrupted():
    floor = make_floor(num_chains=2)
    floor._buses[1].start_write.side_effect = RuntimeError("serial died")  # noqa: SLF001
    with pytest.raises(RuntimeError):
        floor.send_rows([bytes(4)] * 8)
    floor._buses[0].finish_write.assert_called_once()  # noqa: SLF001
    floor.close()
