"""Dataset validator — structural integrity checks on gold datasets.

Validates that a GoldDataset is internally consistent before execution.
Catches corrupt or misconfigured datasets early with actionable diagnostics.
"""

from __future__ import annotations

from benchmark.application.errors import DatasetValidationError
from benchmark.gold.schema import GoldDataset


class DatasetValidator:
    """Validates gold dataset structural integrity before benchmark execution.

    Checks for:
    - Duplicate memory IDs
    - Orphaned expected IDs (referenced but never injected)
    - Negative event days
    - Invalid user references in queries
    - Empty expected sets
    """

    def validate(self, dataset: GoldDataset) -> None:
        """Run all validation checks on the dataset.

        Args:
            dataset: The gold dataset to validate.

        Raises:
            DatasetValidationError: If any integrity check fails,
                with all collected errors in the message.
        """
        errors: list[str] = []
        errors.extend(self._check_duplicate_memory_ids(dataset))
        errors.extend(self._check_orphaned_expected_ids(dataset))
        errors.extend(self._check_negative_days(dataset))
        errors.extend(self._check_invalid_user_references(dataset))
        errors.extend(self._check_empty_expected_sets(dataset))

        if errors:
            raise DatasetValidationError(
                dataset_path=dataset.scenario,
                errors=errors,
            )

    def _check_duplicate_memory_ids(self, dataset: GoldDataset) -> list[str]:
        """Check for duplicate memory IDs across all event days."""
        seen: dict[str, int] = {}
        for day_events in dataset.events:
            for event in day_events.memory_events:
                seen[event.id] = seen.get(event.id, 0) + 1
        return [
            f"Duplicate memory ID: '{mid}' (appears {count} times)"
            for mid, count in seen.items()
            if count > 1
        ]

    def _check_orphaned_expected_ids(self, dataset: GoldDataset) -> list[str]:
        """Check that every expected ID exists in the event pool."""
        all_memory_ids = {
            event.id for day_events in dataset.events for event in day_events.memory_events
        }
        errors: list[str] = []
        for query in dataset.queries:
            for expected_id in query.expected.memory_ids:
                if expected_id not in all_memory_ids:
                    errors.append(
                        f"Query '{query.query[:50]}' expects '{expected_id}' "
                        f"which does not exist in events"
                    )
        return errors

    def _check_negative_days(self, dataset: GoldDataset) -> list[str]:
        """Check that no event day is negative."""
        return [
            f"Event day {day_events.day} is negative"
            for day_events in dataset.events
            if day_events.day < 0
        ]

    def _check_invalid_user_references(self, dataset: GoldDataset) -> list[str]:
        """Check that every query user_id has at least one event."""
        event_users = {
            event.user_id for day_events in dataset.events for event in day_events.memory_events
        }
        errors: list[str] = []
        for query in dataset.queries:
            if query.user_id not in event_users:
                errors.append(f"Query user '{query.user_id}' has no events in dataset")
        return errors

    def _check_empty_expected_sets(self, dataset: GoldDataset) -> list[str]:
        """Check that every query has at least one expected memory ID."""
        return [
            f"Query '{query.query[:50]}' has empty expected set"
            for query in dataset.queries
            if not query.expected.memory_ids
        ]
