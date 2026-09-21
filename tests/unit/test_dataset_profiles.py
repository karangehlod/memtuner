import pytest

from benchmark.gold.dataset_profiles import get_profile

pytestmark = pytest.mark.unit


def test_conflict_update_profile_is_controlled_verification() -> None:
    profile = get_profile("conflict-update-controlled.json")

    assert profile is not None
    assert profile.evidence_tier == "controlled_verification"


def test_longmemeval_full_history_profile_is_agent_memory_core() -> None:
    profile = get_profile("longmemeval_small_gold.json")

    assert profile is not None
    assert profile.evidence_tier == "agent_memory_core"
