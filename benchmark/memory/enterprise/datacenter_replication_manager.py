"""Manage memory state replication across geographically distributed datacenters."""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ReplicationStrategy(Enum):
    """Replication strategies."""
    SYNCHRONOUS = "synchronous"  # Wait for all replicas
    ASYNCHRONOUS = "asynchronous"  # Fire and forget
    EVENTUALLY_CONSISTENT = "eventually_consistent"  # Eventual consistency


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    VECTOR_CLOCK = "vector_clock"
    MERGE_FUNCTION = "merge_function"


@dataclass
class ReplicationResult:
    """Result from replication operation."""
    success: bool
    replicated_to: list[str] = field(default_factory=list)
    failed_replicas: list[str] = field(default_factory=list)
    replication_time_sec: float = 0.0
    error_message: str | None = None


@dataclass
class SyncStatus:
    """Synchronization status for a datacenter."""
    dc_id: str
    in_sync: bool
    last_sync_time: float
    lag_sec: float
    bytes_transferred: int
    sync_errors: int


@dataclass
class ReplicationMetrics:
    """Metrics for replication."""
    total_replications: int
    successful_replications: int
    failed_replications: int
    avg_replication_time_sec: float
    max_replication_lag_sec: float
    total_bytes_replicated: int


@dataclass
class FailoverResult:
    """Result from failover operation."""
    success: bool
    new_primary: str | None
    promoted_replicas: list[str] = field(default_factory=list)
    failover_time_sec: float = 0.0
    error_message: str | None = None


class DatacenterReplicationManager:
    """Manage replication across multiple datacenters."""

    def __init__(
        self,
        primary_dc: str,
        replica_dcs: list[str],
        strategy: ReplicationStrategy = ReplicationStrategy.ASYNCHRONOUS,
        conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
    ):
        """Initialize replication manager.

        Args:
            primary_dc: Primary datacenter ID
            replica_dcs: List of replica datacenter IDs
            strategy: Replication strategy
            conflict_resolution: Conflict resolution strategy
        """
        self.primary_dc = primary_dc
        self.replica_dcs = replica_dcs
        self.all_dcs = [primary_dc] + replica_dcs
        self.strategy = strategy
        self.conflict_resolution = conflict_resolution

        # State tracking
        self._dc_states: dict[str, Any] = {dc: {} for dc in self.all_dcs}
        self._state_versions: dict[str, int] = {dc: 0 for dc in self.all_dcs}
        self._last_sync_times: dict[str, float] = {}
        self._sync_lags: dict[str, float] = {dc: 0.0 for dc in self.all_dcs}
        self._replication_metrics: dict[str, Any] = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0,
            "total_bytes": 0,
        }
        self._dc_health: dict[str, str] = {dc: "healthy" for dc in self.all_dcs}
        self._state_checksums: dict[str, str] = {}

    def replicate_to_dc(
        self,
        dc_id: str,
        state: Any,
        timeout_sec: float = 30.0,
    ) -> ReplicationResult:
        """Replicate state to specific datacenter.

        Args:
            dc_id: Target datacenter ID
            state: State to replicate
            timeout_sec: Replication timeout

        Returns:
            ReplicationResult
        """
        if dc_id not in self.all_dcs:
            return ReplicationResult(
                success=False,
                error_message=f"Unknown datacenter: {dc_id}",
            )

        start_time = time.time()

        try:
            # Simulate replication
            self._dc_states[dc_id] = state.copy() if isinstance(state, dict) else state
            self._state_versions[dc_id] += 1
            self._update_checksum(dc_id, state)

            elapsed = time.time() - start_time
            bytes_transferred = self._estimate_state_size(state)

            # Update metrics
            self._replication_metrics["total"] += 1
            self._replication_metrics["successful"] += 1
            self._replication_metrics["total_time"] += elapsed
            self._replication_metrics["total_bytes"] += bytes_transferred

            logger.info(f"Replicated state to {dc_id} in {elapsed:.3f}s")

            return ReplicationResult(
                success=True,
                replicated_to=[dc_id],
                replication_time_sec=elapsed,
            )

        except Exception as exc:
            self._replication_metrics["total"] += 1
            self._replication_metrics["failed"] += 1
            self._dc_health[dc_id] = "degraded"

            logger.error(f"Replication to {dc_id} failed: {exc}")

            return ReplicationResult(
                success=False,
                failed_replicas=[dc_id],
                error_message=str(exc),
            )

    def sync_state(
        self,
        dc_ids: list[str] | None = None,
    ) -> dict[str, SyncStatus]:
        """Synchronize state across datacenters.

        Args:
            dc_ids: Specific DCs to sync (all if None)

        Returns:
            Dict mapping DC ID to sync status
        """
        if dc_ids is None:
            dc_ids = self.all_dcs

        sync_statuses = {}

        for dc_id in dc_ids:
            if dc_id not in self.all_dcs:
                continue

            current_time = time.time()
            last_sync = self._last_sync_times.get(dc_id, 0)
            lag = current_time - last_sync if last_sync else 0

            # Determine sync status
            in_sync = lag < 5.0 and self._dc_health[dc_id] == "healthy"

            sync_statuses[dc_id] = SyncStatus(
                dc_id=dc_id,
                in_sync=in_sync,
                last_sync_time=last_sync,
                lag_sec=lag,
                bytes_transferred=self._replication_metrics["total_bytes"],
                sync_errors=0,
            )

            self._last_sync_times[dc_id] = current_time
            self._sync_lags[dc_id] = lag

        return sync_statuses

    def replicate_to_all(
        self,
        state: Any,
    ) -> ReplicationResult:
        """Replicate state to all datacenters.

        Args:
            state: State to replicate

        Returns:
            ReplicationResult
        """
        start_time = time.time()
        replicated_to = []
        failed = []

        for dc_id in self.all_dcs:
            if dc_id == self.primary_dc:
                self._dc_states[dc_id] = state.copy() if isinstance(state, dict) else state
                self._state_versions[dc_id] += 1
                replicated_to.append(dc_id)
                continue

            result = self.replicate_to_dc(dc_id, state)

            if result.success:
                replicated_to.extend(result.replicated_to)
            else:
                failed.extend(result.failed_replicas)

        elapsed = time.time() - start_time

        return ReplicationResult(
            success=len(failed) == 0,
            replicated_to=replicated_to,
            failed_replicas=failed,
            replication_time_sec=elapsed,
        )

    def get_replication_status(self) -> dict[str, ReplicationMetrics]:
        """Get replication status for all datacenters.

        Returns:
            Dict mapping DC ID to metrics
        """
        avg_time = (
            self._replication_metrics["total_time"] / max(1, self._replication_metrics["successful"])
        )

        return {
            "aggregated": ReplicationMetrics(
                total_replications=self._replication_metrics["total"],
                successful_replications=self._replication_metrics["successful"],
                failed_replications=self._replication_metrics["failed"],
                avg_replication_time_sec=avg_time,
                max_replication_lag_sec=max(self._sync_lags.values()) if self._sync_lags else 0.0,
                total_bytes_replicated=self._replication_metrics["total_bytes"],
            )
        }

    def handle_dc_failure(self, failed_dc: str) -> FailoverResult:
        """Handle datacenter failure and failover.

        Args:
            failed_dc: Failed datacenter ID

        Returns:
            FailoverResult
        """
        start_time = time.time()

        if failed_dc not in self.all_dcs:
            return FailoverResult(
                success=False,
                new_primary=None,
                error_message=f"Unknown datacenter: {failed_dc}",
            )

        self._dc_health[failed_dc] = "failed"

        # Select new primary if primary failed
        new_primary = None
        if failed_dc == self.primary_dc:
            # Promote first healthy replica
            for dc_id in self.replica_dcs:
                if self._dc_health[dc_id] == "healthy":
                    new_primary = dc_id
                    self.primary_dc = dc_id
                    logger.warning(f"Promoted {dc_id} to primary after failure of {failed_dc}")
                    break

        elapsed = time.time() - start_time

        return FailoverResult(
            success=True,
            new_primary=new_primary,
            promoted_replicas=[new_primary] if new_primary else [],
            failover_time_sec=elapsed,
        )

    def recover_dc(self, dc_id: str) -> bool:
        """Recover a failed datacenter.

        Args:
            dc_id: Datacenter to recover

        Returns:
            True if recovery successful
        """
        if dc_id not in self.all_dcs:
            return False

        try:
            self._dc_health[dc_id] = "healthy"
            self._last_sync_times[dc_id] = time.time()

            # Re-replicate state from primary
            primary_state = self._dc_states[self.primary_dc]
            result = self.replicate_to_dc(dc_id, primary_state)

            if result.success:
                logger.info(f"Recovered {dc_id} successfully")
                return True
            else:
                self._dc_health[dc_id] = "degraded"
                return False

        except Exception as exc:
            logger.error(f"Failed to recover {dc_id}: {exc}")
            return False

    def verify_consistency(self) -> bool:
        """Verify state consistency across all datacenters.

        Returns:
            True if all states are consistent
        """
        if not self._dc_states:
            return True

        # Compare checksums
        checksums = list(self._state_checksums.values())

        if not checksums:
            return True

        return len(set(checksums)) == 1

    def get_dc_state(self, dc_id: str) -> Any | None:
        """Get current state for a datacenter.

        Args:
            dc_id: Datacenter ID

        Returns:
            State or None
        """
        return self._dc_states.get(dc_id)

    def get_dc_health(self) -> dict[str, str]:
        """Get health status of all datacenters.

        Returns:
            Dict mapping DC ID to health status
        """
        return self._dc_health.copy()

    # Private helper methods

    def _update_checksum(self, dc_id: str, state: Any) -> None:
        """Update state checksum for datacenter."""
        state_str = str(state)
        checksum = hashlib.md5(state_str.encode()).hexdigest()
        self._state_checksums[dc_id] = checksum

    def _estimate_state_size(self, state: Any) -> int:
        """Estimate state size in bytes."""
        import sys
        return sys.getsizeof(state)

    def get_stats(self) -> dict[str, Any]:
        """Get replication statistics.

        Returns:
            Stats dict
        """
        return {
            "total_replications": self._replication_metrics["total"],
            "successful_replications": self._replication_metrics["successful"],
            "failed_replications": self._replication_metrics["failed"],
            "total_datacenters": len(self.all_dcs),
            "healthy_datacenters": sum(1 for h in self._dc_health.values() if h == "healthy"),
            "average_replication_time_sec": (
                self._replication_metrics["total_time"] / max(1, self._replication_metrics["successful"])
            ),
            "total_bytes_replicated": self._replication_metrics["total_bytes"],
        }
