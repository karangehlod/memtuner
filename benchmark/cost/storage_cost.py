"""Storage cost calculator for database operations.

Computes costs based on read/write operations against memory stores.
"""

from __future__ import annotations

from benchmark.cost.models import CostEntry

# Fixed pricing for storage operations (per operation, in USD).
# These are approximations for benchmarking purposes.
STORAGE_PRICING: dict[str, float] = {
    "read": 0.000001,  # $1 per million reads
    "write": 0.000005,  # $5 per million writes
}


class StorageCostCalculator:
    """Calculates cost of storage operations during benchmarks.

    Uses fixed per-operation pricing for reads and writes.
    """

    def compute_read_cost(self, operation_count: int = 1) -> CostEntry:
        """Compute cost for read operations.

        Args:
            operation_count: Number of read operations.

        Returns:
            A CostEntry with the computed storage read cost.
        """
        total = operation_count * STORAGE_PRICING["read"]
        return CostEntry(
            source="storage_read",
            amount_usd=total,
            details={"operation_count": operation_count},
        )

    def compute_write_cost(self, operation_count: int = 1) -> CostEntry:
        """Compute cost for write operations.

        Args:
            operation_count: Number of write operations.

        Returns:
            A CostEntry with the computed storage write cost.
        """
        total = operation_count * STORAGE_PRICING["write"]
        return CostEntry(
            source="storage_write",
            amount_usd=total,
            details={"operation_count": operation_count},
        )
