import json
from pathlib import Path

import pytest

from benchmark.packs.private.adapter import PrivateDataPack

pytestmark = pytest.mark.unit


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_production_trace_preserves_audit_metadata(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "events.jsonl",
        [{"memory_id": "m1", "user_id": "u1", "content": "new address", "day": 0, "source_record_id": "event-source"}],
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {
                "query_id": "q1",
                "user_id": "u1",
                "query_text": "What is the address?",
                "day": 1,
                "expected_memory_ids": ["m1"],
                "source_record_id": "query-source",
                "session_id": "validation-session",
                "relevance_judgment": "human-reviewed",
                "split": "validation",
            },
            {
                "query_id": "q2",
                "user_id": "u1",
                "query_text": "Confirm the address.",
                "day": 1,
                "expected_memory_ids": ["m1"],
                "source_record_id": "query-final-source",
                "session_id": "final-test-session",
                "relevance_judgment": "human-reviewed",
                "split": "final_test",
            },
        ],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "private-trace-v1",
                "dataset_tier": "production_trace",
                "provenance": "observed-production-traffic",
                "split_protocol": "user-session-final-test",
            }
        ),
        encoding="utf-8",
    )

    pack = PrivateDataPack()
    pack.load(tmp_path)
    dataset = pack.to_gold_dataset()

    assert dataset.metadata["dataset_tier"] == "production_trace"
    assert dataset.events[0].memory_events[0].provenance["source_record_id"] == "event-source"
    assert dataset.queries[1].provenance["split"] == "final_test"
    assert dataset.queries[1].provenance["session_id"] == "final-test-session"


def test_private_pack_rejects_cross_user_evidence(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "events.jsonl",
        [{"memory_id": "m1", "user_id": "owner", "content": "private", "day": 0}],
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [{"query_id": "q1", "user_id": "other", "query_text": "private?", "day": 1, "expected_memory_ids": ["m1"]}],
    )

    with pytest.raises(ValueError, match="another user's memory"):
        PrivateDataPack().load(tmp_path)


def test_production_trace_rejects_reused_validation_final_session(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "events.jsonl",
        [{"memory_id": "m1", "user_id": "u1", "content": "address", "day": 0, "source_record_id": "e1"}],
    )
    _write_jsonl(
        tmp_path / "queries.jsonl",
        [
            {"query_id": "q1", "user_id": "u1", "session_id": "shared", "query_text": "Address?", "day": 1, "expected_memory_ids": ["m1"], "source_record_id": "q1", "relevance_judgment": "human", "split": "validation"},
            {"query_id": "q2", "user_id": "u1", "session_id": "shared", "query_text": "Address again?", "day": 2, "expected_memory_ids": ["m1"], "source_record_id": "q2", "relevance_judgment": "human", "split": "final_test"},
        ],
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"schema_version": "private-trace-v1", "dataset_tier": "production_trace", "provenance": "observed", "split_protocol": "user-session-final-test"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disjoint user/session scopes"):
        PrivateDataPack().load(tmp_path)
