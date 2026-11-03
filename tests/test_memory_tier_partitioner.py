"""Locked Agile-V verification suite for Hot/Cold Memory Tier Partitioner (REQ-BM-TIER-01)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agy_brain_memory import (
    DEFAULT_HOT_MAX_AGE_DAYS as EXPORTED_HOT_MAX_AGE_DAYS,
    DEFAULT_PINNED_ARTIFACTS as EXPORTED_PINNED_ARTIFACTS,
    MemoryTierPartitioner as ExportedMemoryTierPartitioner,
    partition_brain_memory_tiers as exported_partition_tiers,
    search_tiered_brain as exported_search_tiered_brain,
)
from agy_brain_memory.indexer import BrainMemoryIndexer
from agy_brain_memory.memory_tier_partitioner import (
    DEFAULT_HOT_MAX_AGE_DAYS,
    DEFAULT_PINNED_ARTIFACTS,
    MemoryTierPartitioner,
    partition_brain_memory_tiers,
    search_tiered_brain,
)


def _create_conversation_fixture(
    brain_dir: Path,
    conv_id: str,
    mtime_sec: float,
    step_text: str,
    artifacts: dict[str, str] | None = None,
) -> Path:
    """Create a synthetic conversation directory with deterministic file mtimes."""
    conv_dir = brain_dir / conv_id
    log_dir = conv_dir / ".system_generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript = log_dir / "transcript.jsonl"
    step_payload = {
        "step_index": 1,
        "type": "PLANNER_RESPONSE",
        "status": "DONE",
        "created_at": "2026-10-01T00:00:00Z",
        "content": step_text,
    }
    transcript.write_text(json.dumps(step_payload) + "\n", encoding="utf-8")
    os.utime(transcript, (mtime_sec, mtime_sec))

    if artifacts:
        for fname, text in artifacts.items():
            art_path = conv_dir / fname
            art_path.write_text(text, encoding="utf-8")
            os.utime(art_path, (mtime_sec, mtime_sec))
    return conv_dir


def _build_three_tier_brain(tmp_path: Path) -> tuple[Path, BrainMemoryIndexer, float]:
    """Populate a 3-conversation SQLite brain database for REQ-BM-TIER-01 testing."""
    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "brain_index.sqlite"
    now_ts = 1_760_000_000.0
    day_sec = 86400.0

    _create_conversation_fixture(
        brain_dir,
        conv_id="conv_recent",
        mtime_sec=now_ts - (2.0 * day_sec),
        step_text="Implemented shared_keyword in recent hot session.",
    )
    _create_conversation_fixture(
        brain_dir,
        conv_id="conv_pinned_old",
        mtime_sec=now_ts - (60.0 * day_sec),
        step_text="Documented shared_keyword in pinned milestone session.",
        artifacts={"PROGRESS.md": "# Progress\nPinned milestone state with shared_keyword."},
    )
    _create_conversation_fixture(
        brain_dir,
        conv_id="conv_cold_archive",
        mtime_sec=now_ts - (90.0 * day_sec),
        step_text="Historical shared_keyword and ancient_archive_only_token in cold archive.",
    )

    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    sync_res = indexer.sync(force=True)
    assert sync_res["conversations_indexed"] == 3
    return db_path, indexer, now_ts


def test_req_bm_tier_01_partition_preserves_sync_and_classifies_hot_vs_cold(
    tmp_path: Path,
) -> None:
    """REQ-BM-TIER-01: Verify Hot/Cold classification rules and 7-column conversations sync safety."""
    assert DEFAULT_HOT_MAX_AGE_DAYS == EXPORTED_HOT_MAX_AGE_DAYS
    assert DEFAULT_PINNED_ARTIFACTS == EXPORTED_PINNED_ARTIFACTS
    assert "PROGRESS.md" in DEFAULT_PINNED_ARTIFACTS
    assert MemoryTierPartitioner is ExportedMemoryTierPartitioner
    assert partition_brain_memory_tiers is exported_partition_tiers
    assert search_tiered_brain is exported_search_tiered_brain

    db_path, indexer, now_ts = _build_three_tier_brain(tmp_path)

    report = partition_brain_memory_tiers(
        db_path,
        hot_max_age_days=14.0,
        reference_ts=now_ts,
    )
    assert report["total_conversations"] == 3
    assert report["hot_count"] == 2
    assert report["cold_count"] == 1
    assert "conv_recent" in report["hot_conv_ids"]
    assert "conv_pinned_old" in report["hot_conv_ids"]
    assert "conv_cold_archive" in report["cold_conv_ids"]

    serialized_report = json.dumps(report)
    assert "recent_activity" in serialized_report
    assert "pinned_artifact" in serialized_report
    assert "aged_out_archive" in serialized_report

    # Verify calling indexer.sync() afterwards does NOT fail or wipe out the 7-column schema
    resync = indexer.sync(force=True)
    assert resync["conversations_indexed"] == 3


def test_req_bm_tier_01_two_stage_search_short_circuits_hot_and_falls_back_to_cold(
    tmp_path: Path,
) -> None:
    """REQ-BM-TIER-01: Verify two-stage search short-circuits on Hot hits and falls back to Cold."""
    db_path, _, now_ts = _build_three_tier_brain(tmp_path)
    partitioner = MemoryTierPartitioner(db_path=db_path, hot_max_age_days=14.0)
    part_res = partitioner.partition(reference_ts=now_ts)
    assert part_res["hot_count"] == 2

    hot_res = search_tiered_brain(
        db_path,
        "shared_keyword",
        limit=2,
        short_circuit_on_hot=True,
        reference_ts=now_ts,
    )
    assert hot_res["hot_hits_count"] == 2
    assert hot_res["cold_tier_queried"] is False
    assert len(hot_res["results"]) == 2
    assert all(r["tier"] == "hot" for r in hot_res["results"])

    cold_res = search_tiered_brain(
        db_path,
        "ancient_archive_only_token",
        limit=5,
        include_cold_fallback=True,
        reference_ts=now_ts,
    )
    assert cold_res["hot_hits_count"] == 0
    assert cold_res["cold_tier_queried"] is True
    assert cold_res["cold_hits_count"] == 1
    assert len(cold_res["results"]) == 1
    assert cold_res["results"][0]["tier"] == "cold"
    assert cold_res["results"][0]["conv_id"] == "conv_cold_archive"
