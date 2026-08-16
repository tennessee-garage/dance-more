import pytest

from df2_pi.transport.chain_map import NUM_ROWS, RowChainMap


def test_alternating_splits_odd_even():
    m = RowChainMap.alternating(2)
    # The as-built wiring: one Cat5 run passes every row controller in
    # order, each tapping the opposite pair from its neighbour.
    assert m.rows_on(0) == [0, 2, 4, 6]
    assert m.rows_on(1) == [1, 3, 5, 7]


def test_alternating_balances_the_load():
    m = RowChainMap.alternating(2)
    # An unbalanced split would blow the frame budget on the busier chain.
    assert len(m.rows_on(0)) == len(m.rows_on(1)) == NUM_ROWS // 2


def test_chain_for_each_row():
    m = RowChainMap.alternating(2)
    for row in range(NUM_ROWS):
        assert m.chain_for(row) == row % 2


def test_single_chain_puts_everything_on_zero():
    m = RowChainMap.single_chain()
    assert m.rows_on(0) == list(range(NUM_ROWS))
    assert m.chain_count == 1


def test_chain_count():
    assert RowChainMap.alternating(2).chain_count == 2


def test_explicit_mapping_is_honoured():
    m = RowChainMap({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1})
    assert m.rows_on(0) == [0, 1, 2, 3]
    assert m.rows_on(1) == [4, 5, 6, 7]


def test_incomplete_mapping_rejected():
    with pytest.raises(ValueError, match="missing"):
        RowChainMap({0: 0, 1: 1})


def test_out_of_range_row_rejected():
    mapping = {row: 0 for row in range(NUM_ROWS)}
    mapping[99] = 1
    with pytest.raises(ValueError, match="outside"):
        RowChainMap(mapping)


def test_chain_for_rejects_unknown_row():
    m = RowChainMap.alternating(2)
    with pytest.raises(ValueError, match="out of range"):
        m.chain_for(NUM_ROWS)
