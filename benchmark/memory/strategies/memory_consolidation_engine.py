"""Hierarchical memory consolidation engine for tier management."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np

MemoryTier = Literal["working", "episodic", "semantic"]
DecayStrategy = Literal["linear", "exponential", "power_law", "selective"]


@dataclass
class MemoryRecord:
    """Internal record of a memory with metadata."""
    memory_id: str
    content: dict[str, Any]
    tier: MemoryTier
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    value_score: float = 0.5  # Estimated value (0-1)
    archived: bool = False

    def age_days(self, reference_time: datetime | None = None) -> float:
        """Get age in days."""
        ref = reference_time or datetime.now()
        return (ref - self.created_at).total_seconds() / 86400

    def last_access_days(self, reference_time: datetime | None = None) -> float:
        """Get days since last access."""
        ref = reference_time or datetime.now()
        return (ref - self.last_accessed).total_seconds() / 86400


@dataclass
class ConsolidationMetrics:
    """Metrics about consolidation operations."""
    promoted: int = 0
    demoted: int = 0
    archived: int = 0
    reactivated: int = 0
    total_consolidations: int = 0
    elapsed_ms: float = 0.0


class MemoryConsolidationEngine:
    """Manages memory lifecycle across working/episodic/semantic tiers."""

    def __init__(
        self,
        working_capacity: int = 100,
        episodic_capacity: int = 1000,
        semantic_capacity: int = 10000,
    ):
        """Initialize consolidation engine with tier capacities.

        Args:
            working_capacity: Max items in working memory (hot)
            episodic_capacity: Max items in episodic memory
            semantic_capacity: Max items in semantic memory (cold)
        """
        self.capacities = {
            "working": working_capacity,
            "episodic": episodic_capacity,
            "semantic": semantic_capacity,
        }

        # Storage for each tier
        self._tiers: dict[MemoryTier, dict[str, MemoryRecord]] = {
            "working": {},
            "episodic": {},
            "semantic": {},
        }

        # Archive for removed items
        self._archive: dict[str, MemoryRecord] = {}

        # Memory ID to tier mapping for fast lookup
        self._id_to_tier: dict[str, MemoryTier] = {}

        # Consolidation metrics
        self.metrics = ConsolidationMetrics()

        # Decay configuration
        self._decay_strategy: DecayStrategy = "exponential"
        self._decay_lambda: float = 0.1

    def add_memory(
        self,
        memory_id: str,
        content: dict[str, Any],
        tier: MemoryTier = "working",
        value_score: float = 0.5,
    ) -> bool:
        """Add memory to specified tier.

        Args:
            memory_id: Unique memory identifier
            content: Memory content dict
            tier: Which tier to add to
            value_score: Initial value score (0-1)

        Returns:
            True if added, False if tier at capacity

        Raises:
            ValueError: If tier unknown or score invalid
        """
        if tier not in self.capacities:
            raise ValueError(f"Unknown tier: {tier}")

        if not (0.0 <= value_score <= 1.0):
            raise ValueError(f"Value score must be [0,1], got {value_score}")

        # Check capacity
        if len(self._tiers[tier]) >= self.capacities[tier]:
            return False

        # Create record
        now = datetime.now()
        record = MemoryRecord(
            memory_id=memory_id,
            content=content,
            tier=tier,
            created_at=now,
            last_accessed=now,
            value_score=value_score,
        )

        # Add to tier
        self._tiers[tier][memory_id] = record
        self._id_to_tier[memory_id] = tier

        return True

    def promote(
        self,
        memory_id: str,
        target_tier: MemoryTier | None = None,
    ) -> bool:
        """Promote memory to higher tier (working → episodic → semantic).

        Args:
            memory_id: Memory to promote
            target_tier: Specific tier (default: next tier up)

        Returns:
            True if promoted successfully

        Raises:
            ValueError: If memory not found
        """
        current_tier = self.get_memory_tier(memory_id)
        if current_tier is None:
            raise ValueError(f"Memory {memory_id} not found")

        # Determine target tier if not specified
        if target_tier is None:
            tier_order = ["working", "episodic", "semantic"]
            current_idx = tier_order.index(current_tier)
            if current_idx >= len(tier_order) - 1:
                return False  # Already at top
            target_tier = tier_order[current_idx + 1]

        # Move record
        record = self._tiers[current_tier].pop(memory_id)
        record.tier = target_tier
        record.last_accessed = datetime.now()

        self._tiers[target_tier][memory_id] = record
        self._id_to_tier[memory_id] = target_tier
        self.metrics.promoted += 1

        return True

    def demote(
        self,
        memory_id: str,
        target_tier: MemoryTier | None = None,
    ) -> bool:
        """Demote memory to lower tier (semantic → episodic → working).

        Args:
            memory_id: Memory to demote
            target_tier: Specific tier (default: next tier down)

        Returns:
            True if demoted successfully

        Raises:
            ValueError: If memory not found
        """
        current_tier = self.get_memory_tier(memory_id)
        if current_tier is None:
            raise ValueError(f"Memory {memory_id} not found")

        # Determine target tier if not specified
        if target_tier is None:
            tier_order = ["working", "episodic", "semantic"]
            current_idx = tier_order.index(current_tier)
            if current_idx <= 0:
                return False  # Already at bottom, move to archive
            target_tier = tier_order[current_idx - 1]

        # Move record
        record = self._tiers[current_tier].pop(memory_id)
        record.tier = target_tier
        record.last_accessed = datetime.now()

        self._tiers[target_tier][memory_id] = record
        self._id_to_tier[memory_id] = target_tier
        self.metrics.demoted += 1

        return True

    def archive(self, memory_id: str) -> bool:
        """Archive memory (move to long-term storage).

        Args:
            memory_id: Memory to archive

        Returns:
            True if archived

        Raises:
            ValueError: If memory not found
        """
        tier = self.get_memory_tier(memory_id)
        if tier is None:
            raise ValueError(f"Memory {memory_id} not found")

        record = self._tiers[tier].pop(memory_id)
        record.archived = True
        self._archive[memory_id] = record
        del self._id_to_tier[memory_id]
        self.metrics.archived += 1

        return True

    def reactivate(self, memory_id: str, target_tier: MemoryTier = "episodic") -> bool:
        """Reactivate archived memory.

        Args:
            memory_id: Archived memory to reactivate
            target_tier: Tier to restore to

        Returns:
            True if reactivated

        Raises:
            ValueError: If memory not archived
        """
        if memory_id not in self._archive:
            raise ValueError(f"Memory {memory_id} not archived")

        record = self._archive.pop(memory_id)
        record.archived = False
        record.tier = target_tier
        record.last_accessed = datetime.now()

        # Check target tier capacity
        if len(self._tiers[target_tier]) >= self.capacities[target_tier]:
            self._archive[memory_id] = record
            return False

        self._tiers[target_tier][memory_id] = record
        self._id_to_tier[memory_id] = target_tier
        self.metrics.reactivated += 1

        return True

    def get_memory_tier(self, memory_id: str) -> MemoryTier | None:
        """Get tier for a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Tier name or None if not found
        """
        return self._id_to_tier.get(memory_id)

    def consolidate(self, reference_time: datetime | None = None) -> ConsolidationMetrics:
        """Execute full consolidation pass.

        Applies decay, promotes high-value items, demotes low-value items.

        Args:
            reference_time: Time to use for age calculations

        Returns:
            Metrics about consolidation operations
        """
        import time
        start_time = time.time()

        ref_time = reference_time or datetime.now()

        # Apply decay to all tiers
        self._apply_decay(ref_time)

        # Promote high-value items
        self._promote_high_value_items(ref_time)

        # Demote low-value items
        self._demote_low_value_items(ref_time)

        # Enforce tier capacity limits
        self._enforce_capacity_limits(ref_time)

        self.metrics.total_consolidations += 1
        self.metrics.elapsed_ms = (time.time() - start_time) * 1000

        return self.metrics

    def set_decay_strategy(
        self,
        strategy: DecayStrategy,
        lambda_param: float = 0.1,
    ) -> None:
        """Set decay strategy for memory value.

        Args:
            strategy: Type of decay (linear, exponential, power_law, selective)
            lambda_param: Decay rate/lambda parameter
        """
        if strategy not in ["linear", "exponential", "power_law", "selective"]:
            raise ValueError(f"Unknown decay strategy: {strategy}")

        self._decay_strategy = strategy
        self._decay_lambda = lambda_param

    def get_tier_statistics(self, tier: MemoryTier) -> dict[str, Any]:
        """Get statistics for a tier.

        Args:
            tier: Tier to analyze

        Returns:
            Dict with size, avg_age, avg_value, etc.
        """
        if tier not in self.capacities:
            raise ValueError(f"Unknown tier: {tier}")

        memories = list(self._tiers[tier].values())

        if not memories:
            return {
                "size": 0,
                "capacity": self.capacities[tier],
                "utilization": 0.0,
                "avg_age_days": 0.0,
                "avg_value": 0.0,
                "avg_access_count": 0,
            }

        now = datetime.now()
        ages = [m.age_days(now) for m in memories]
        values = [m.value_score for m in memories]
        accesses = [m.access_count for m in memories]

        return {
            "size": len(memories),
            "capacity": self.capacities[tier],
            "utilization": len(memories) / self.capacities[tier],
            "avg_age_days": np.mean(ages),
            "avg_value": np.mean(values),
            "avg_access_count": np.mean(accesses),
            "min_value": np.min(values),
            "max_value": np.max(values),
        }

    def get_all_tier_statistics(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all tiers.

        Returns:
            Dict mapping tier → statistics
        """
        return {
            tier: self.get_tier_statistics(tier)
            for tier in ["working", "episodic", "semantic"]
        }

    def record_access(self, memory_id: str) -> bool:
        """Record access to a memory.

        Args:
            memory_id: Memory accessed

        Returns:
            True if recorded
        """
        tier = self.get_memory_tier(memory_id)
        if tier is None:
            return False

        record = self._tiers[tier][memory_id]
        record.access_count += 1
        record.last_accessed = datetime.now()

        return True

    def update_value_score(self, memory_id: str, score: float) -> bool:
        """Update value score for a memory.

        Args:
            memory_id: Memory to update
            score: New value score (0-1)

        Returns:
            True if updated

        Raises:
            ValueError: If score invalid
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Value score must be [0,1], got {score}")

        tier = self.get_memory_tier(memory_id)
        if tier is None:
            return False

        self._tiers[tier][memory_id].value_score = score
        return True

    # Private helper methods

    def _apply_decay(self, reference_time: datetime) -> None:
        """Apply decay function to all memory values."""
        for tier_memories in self._tiers.values():
            for record in tier_memories.values():
                age = record.age_days(reference_time)
                decay_factor = self._compute_decay(age)
                record.value_score *= decay_factor

    def _compute_decay(self, age_days: float) -> float:
        """Compute decay multiplier based on age."""
        if age_days < 0:
            age_days = 0

        if self._decay_strategy == "linear":
            # Linear: -lambda * age, clamped to 0
            return max(0.0, 1.0 - self._decay_lambda * age_days)

        elif self._decay_strategy == "exponential":
            # Exponential: e^(-lambda * age)
            return np.exp(-self._decay_lambda * age_days)

        elif self._decay_strategy == "power_law":
            # Power law: (1 + age)^(-lambda)
            return (1.0 + age_days) ** (-self._decay_lambda)

        elif self._decay_strategy == "selective":
            # Selective: only decay if > 1 day old
            if age_days > 1.0:
                return np.exp(-self._decay_lambda * (age_days - 1.0))
            return 1.0

        return 1.0

    def _promote_high_value_items(self, reference_time: datetime) -> None:
        """Promote high-value items from lower tiers."""
        # Promote working → episodic
        self._promote_tier_threshold("working", "episodic", value_threshold=0.7)
        # Promote episodic → semantic
        self._promote_tier_threshold("episodic", "semantic", value_threshold=0.8)

    def _promote_tier_threshold(
        self,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
        value_threshold: float,
    ) -> None:
        """Promote memories above value threshold."""
        candidates = [
            mem_id for mem_id, record in self._tiers[from_tier].items()
            if record.value_score >= value_threshold
        ]

        for mem_id in candidates:
            if len(self._tiers[to_tier]) < self.capacities[to_tier]:
                self.promote(mem_id, to_tier)

    def _demote_low_value_items(self, reference_time: datetime) -> None:
        """Demote low-value items to lower tiers."""
        # Demote semantic → episodic
        self._demote_tier_threshold("semantic", "episodic", value_threshold=0.2)
        # Demote episodic → working
        self._demote_tier_threshold("episodic", "working", value_threshold=0.3)

    def _demote_tier_threshold(
        self,
        from_tier: MemoryTier,
        to_tier: MemoryTier,
        value_threshold: float,
    ) -> None:
        """Demote memories below value threshold."""
        candidates = [
            mem_id for mem_id, record in self._tiers[from_tier].items()
            if record.value_score <= value_threshold
        ]

        for mem_id in candidates:
            if len(self._tiers[to_tier]) < self.capacities[to_tier]:
                self.demote(mem_id, to_tier)

    def _enforce_capacity_limits(self, reference_time: datetime) -> None:
        """Enforce capacity constraints, demoting overflow."""
        for tier in ["working", "episodic", "semantic"]:
            while len(self._tiers[tier]) > self.capacities[tier]:
                # Find lowest value memory
                lowest_id = min(
                    self._tiers[tier].keys(),
                    key=lambda mid: self._tiers[tier][mid].value_score,
                )

                # Try to demote
                if not self.demote(lowest_id):
                    # If can't demote (at bottom), archive
                    self.archive(lowest_id)
