"""Which physical Row Bus chain carries each logical row.

The floor's 8 rows are split across 2 RS-485 chains so the worst-case frame
fits the 33 ms budget - see docs/row-bus-protocol.md §1. Row addresses stay
global (0x00-0x07); only the wiring is partitioned, so row controller
firmware is unaware there is more than one chain.

The split is **alternating**, not blocked: a single Cat5 run passes every row
controller in physical order, and each one taps the opposite pair from its
neighbour. So rows 0,2,4,6 sit on chain 0 and rows 1,3,5,7 on chain 1.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

NUM_ROWS = 8
NUM_CHAINS = 2


class RowChainMap:
    """Immutable logical-row -> chain-index lookup."""

    def __init__(self, mapping: Mapping[int, int]) -> None:
        missing = set(range(NUM_ROWS)) - set(mapping)
        if missing:
            raise ValueError(f"rows missing from chain map: {sorted(missing)}")
        extra = set(mapping) - set(range(NUM_ROWS))
        if extra:
            raise ValueError(f"chain map has rows outside 0-{NUM_ROWS - 1}: {sorted(extra)}")
        self._mapping = dict(mapping)

    @classmethod
    def alternating(cls, num_chains: int = NUM_CHAINS) -> RowChainMap:
        """The as-built wiring: consecutive rows land on alternating chains."""
        if num_chains < 1:
            raise ValueError("num_chains must be >= 1")
        return cls({row: row % num_chains for row in range(NUM_ROWS)})

    @classmethod
    def single_chain(cls) -> RowChainMap:
        """Everything on chain 0 - the pre-two-chain hat, still useful for
        bench work on the older board."""
        return cls({row: 0 for row in range(NUM_ROWS)})

    def chain_for(self, row: int) -> int:
        try:
            return self._mapping[row]
        except KeyError:
            raise ValueError(f"row {row} out of range 0-{NUM_ROWS - 1}") from None

    def rows_on(self, chain: int) -> list[int]:
        return sorted(row for row, c in self._mapping.items() if c == chain)

    @property
    def chain_count(self) -> int:
        return len(set(self._mapping.values()))

    def items(self) -> Iterator[tuple[int, int]]:
        return iter(sorted(self._mapping.items()))

    def __repr__(self) -> str:
        groups = ", ".join(
            f"chain {c}: {self.rows_on(c)}" for c in sorted(set(self._mapping.values()))
        )
        return f"RowChainMap({groups})"
