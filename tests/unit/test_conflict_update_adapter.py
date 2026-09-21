import pytest

from benchmark.gold.adapters.conflict_update_adapter import ConflictUpdateAdapter

pytestmark = pytest.mark.unit


def test_conflict_update_dataset_targets_current_evidence() -> None:
    dataset = ConflictUpdateAdapter().load()

    expected_by_task = {query.task_id: query.expected.memory_ids for query in dataset.queries}

    assert expected_by_task["address-current"] == ["address-v2"]
    assert expected_by_task["timezone-current"] == ["timezone-v2"]
    assert expected_by_task["address-deletion"] == ["address-deleted"]
    assert dataset.metadata["conflict_policy"] == "latest_authoritative_update_or_tombstone"


def test_conflict_update_dataset_is_reproducible() -> None:
    adapter = ConflictUpdateAdapter()
    first = adapter.load()
    second = adapter.load()

    assert adapter.fingerprint(first) == adapter.fingerprint(second)
