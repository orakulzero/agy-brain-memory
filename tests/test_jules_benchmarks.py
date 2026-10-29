import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agy_brain_memory.benchmarks import (
    _write_transcript,
    _sandbox_1_incremental_wal_and_fts5,
    _sandbox_2_subagent_lineage,
    _sandbox_3_health_and_digest,
    _sandbox_4_plugin_and_mcp,
    run_benchmark_suite,
)


def validate_transcript_schema(transcript_path: Path) -> bool:
    """
    Validates the schema of a transcript JSON file.
    
    Args:
        transcript_path (Path): Path to the transcript JSON file.
    
    Returns:
        bool: True if the schema is valid, False otherwise.
    """
    if not transcript_path.exists():
        return False
    
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                if not isinstance(step, dict) or "step_index" not in step or "type" not in step:
                    return False
            except json.JSONDecodeError:
                return False
    
    return True


def test_corrupted_transcript_handled_gracefully(tmp_path: Path) -> None:
    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "test.sqlite"
    
    logs_dir = brain_dir / "conv_1" / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tpath = logs_dir / "transcript.jsonl"
    
    # Write corrupted JSON
    tpath.write_text("{corrupted: json}\n", encoding="utf-8")
    
    # Ensure indexer doesn't crash
    from agy_brain_memory.indexer import BrainMemoryIndexer
    try:
        indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
        indexer.sync()
    except Exception as e:
        pytest.fail(f"BrainMemoryIndexer crashed on corrupted transcript: {e}")
        
    # Ensure search doesn't crash
    from agy_brain_memory.search import search_brain
    try:
        search_brain(db_path=db_path, query="test")
    except Exception as e:
        pytest.fail(f"search_brain crashed on corrupted transcript: {e}")


def test_write_transcript(tmp_path: Path):
    steps = [{"step_index": 1, "type": "USER_INPUT"}]
    conv_dir = tmp_path / "conv_dir"
    tpath = _write_transcript(conv_dir, steps)
    
    assert tpath.exists()
    assert tpath.parent == conv_dir / ".system_generated" / "logs"
    assert tpath.name == "transcript.jsonl"
    assert json.loads(tpath.read_text()) == steps[0]


@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_success(mock_search, mock_indexer, tmp_path):
    mock_indexer_inst = MagicMock()
    mock_indexer.return_value = mock_indexer_inst
    
    # first sync, second sync
    mock_indexer_inst.sync.side_effect = [
        {"conversations_indexed": 5},
        {"conversations_indexed": 0, "conversations_skipped": 5}
    ]
    
    mock_search.return_value = {"results": [{"hit": 1}]}
    
    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    
    assert res["name"] == "SB1_IncrementalWAL_And_FTS5_Search"
    assert res["passed"] is True
    assert res["metrics"]["first_indexed"] == 5
    assert res["metrics"]["second_skipped"] == 5
    assert res["metrics"]["search_hits"] == 1


@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_failure(mock_search, mock_indexer, tmp_path):
    mock_indexer_inst = MagicMock()
    mock_indexer.return_value = mock_indexer_inst
    
    mock_indexer_inst.sync.side_effect = [
        {"conversations_indexed": 4},
        {"conversations_indexed": 0, "conversations_skipped": 5}
    ]
    
    mock_search.return_value = {"results": [{"hit": 1}]}
    
    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    
    assert res["passed"] is False


@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_success(mock_lineage, mock_indexer, tmp_path):
    mock_lineage.return_value = {
        "spawned_children": [{"child_conv_id": "conv_child_1", "child_outcome_summary": "22/22 tests passed"}]
    }
    
    res = _sandbox_2_subagent_lineage(tmp_path)
    
    assert res["name"] == "SB2_SubagentCallGraph_Lineage_And_Outcomes"
    assert res["passed"] is True
    assert res["metrics"]["edges_extracted"] == 1


@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_failure(mock_lineage, mock_indexer, tmp_path):
    mock_lineage.return_value = {
        "spawned_children": [{"child_conv_id": "wrong_child", "child_outcome_summary": "22/22 tests passed"}]
    }
    
    res = _sandbox_2_subagent_lineage(tmp_path)
    
    assert res["passed"] is False


@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_success(mock_digest, mock_health, tmp_path):
    mock_health.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": ["conv_bigint_err"]
    }
    mock_digest.return_value = {"char_count": 500}
    
    res = _sandbox_3_health_and_digest(tmp_path)
    
    assert res["name"] == "SB3_BrainHealth_And_CompactSessionDigest"
    assert res["passed"] is True
    assert res["metrics"]["digest_chars"] == 500
    assert res["metrics"]["bigint_detected"] == 1


@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_failure(mock_digest, mock_health, tmp_path):
    mock_health.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": []
    }
    mock_digest.return_value = {"char_count": 500}
    
    res = _sandbox_3_health_and_digest(tmp_path)
    
    assert res["passed"] is False


@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_success(mock_mcp, mock_val, mock_inst, tmp_path):
    mock_inst.return_value = {"plugin_dir": "test"}
    mock_val.return_value = {"valid": True, "oss_compliant": True}
    
    mock_mcp.side_effect = [
        {"result": {"isError": False}},  # sync
        {"result": {"isError": False}}   # search
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    
    assert res["name"] == "SB4_NativeL2Plugin_And_MCP_RoundTrip"
    assert res["passed"] is True
    assert res["metrics"]["plugin_valid"] is True
    assert res["metrics"]["oss_compliant"] is True


@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_failure(mock_mcp, mock_val, mock_inst, tmp_path):
    mock_inst.return_value = {"plugin_dir": "test"}
    mock_val.return_value = {"valid": False, "oss_compliant": True}
    
    mock_mcp.side_effect = [
        {"result": {"isError": False}},  # sync
        {"result": {"isError": False}}   # search
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    
    assert res["passed"] is False


@patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5")
@patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage")
@patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest")
@patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp")
def test_run_benchmark_suite_success(mock_sb4, mock_sb3, mock_sb2, mock_sb1):
    mock_sb1.return_value = {"passed": True}
    mock_sb2.return_value = {"passed": True}
    mock_sb3.return_value = {"passed": True}
    mock_sb4.return_value = {"passed": True}
    
    res = run_benchmark_suite()
    
    assert res["all_passed"] is True
    assert len(res["sandboxes"]) == 4


@patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5")
@patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage")
@patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest")
@patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp")
def test_run_benchmark_suite_failure(mock_sb4, mock_sb3, mock_sb2, mock_sb1):
    mock_sb1.return_value = {"passed": True}
    mock_sb2.return_value = {"passed": False}
    mock_sb3.return_value = {"passed": True}
    mock_sb4.return_value = {"passed": True}
    
    res = run_benchmark_suite()
    
    assert res["all_passed"] is False
    assert len(res["sandboxes"]) == 4
