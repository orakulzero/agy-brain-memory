"""Tests for agy_brain_memory.benchmarks."""
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

def test_write_transcript_empty(tmp_path: Path) -> None:
    tpath = _write_transcript(tmp_path, [])
    assert tpath.exists()
    assert tpath.read_text() == "\n"

def test_write_transcript_normal(tmp_path: Path) -> None:
    steps = [
        {"step_index": 1, "type": "USER_INPUT", "content": "Hello", "extra": None},
        {"step_index": 2, "type": "PLANNER_RESPONSE", "content": "World"},
    ]
    tpath = _write_transcript(tmp_path, steps)
    
    assert tpath.exists()
    assert tpath.parent.name == "logs"
    assert tpath.parent.parent.name == ".system_generated"
    
    lines = tpath.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed_1 = json.loads(lines[0])
    parsed_2 = json.loads(lines[1])
    assert parsed_1["content"] == "Hello"
    assert parsed_1["extra"] is None
    assert parsed_2["content"] == "World"

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_incremental_wal_and_fts5_pass(mock_search, mock_indexer, tmp_path):
    mock_indexer_inst = mock_indexer.return_value
    mock_indexer_inst.sync.side_effect = [
        {"conversations_indexed": 5},
        {"conversations_indexed": 0, "conversations_skipped": 5},
    ]
    mock_search.return_value = {"results": [{"id": 1}]}
    
    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    
    assert res["name"] == "SB1_IncrementalWAL_And_FTS5_Search"
    assert res["passed"] is True
    assert res["metrics"]["first_indexed"] == 5
    assert res["metrics"]["second_skipped"] == 5
    assert res["metrics"]["search_hits"] == 1
    assert "elapsed_ms" in res
    assert res["elapsed_ms"] >= 0

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_fail_sync1(mock_search, mock_indexer, tmp_path):
    mock_indexer_inst = mock_indexer.return_value
    mock_indexer_inst.sync.side_effect = [
        {"conversations_indexed": 4},
        {"conversations_indexed": 0, "conversations_skipped": 5},
    ]
    mock_search.return_value = {"results": [{"id": 1}]}
    
    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_fail_sync2(mock_search, mock_indexer, tmp_path):
    mock_indexer_inst = mock_indexer.return_value
    mock_indexer_inst.sync.side_effect = [
        {"conversations_indexed": 5},
        {"conversations_indexed": 1, "conversations_skipped": 4},
    ]
    mock_search.return_value = {"results": [{"id": 1}]}
    
    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_fail_search(mock_search, mock_indexer, tmp_path):
    mock_indexer_inst = mock_indexer.return_value
    mock_indexer_inst.sync.side_effect = [
        {"conversations_indexed": 5},
        {"conversations_indexed": 0, "conversations_skipped": 5},
    ]
    mock_search.return_value = {"results": []}
    
    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_subagent_lineage_pass(mock_lineage, mock_indexer, tmp_path):
    mock_lineage.return_value = {
        "spawned_children": [
            {
                "child_conv_id": "conv_child_1",
                "child_outcome_summary": "22/22 red baseline tests authored."
            }
        ]
    }
    
    res = _sandbox_2_subagent_lineage(tmp_path)
    
    assert res["name"] == "SB2_SubagentCallGraph_Lineage_And_Outcomes"
    assert res["passed"] is True
    assert res["metrics"]["edges_extracted"] == 1
    assert "elapsed_ms" in res

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_fail_wrong_child(mock_lineage, mock_indexer, tmp_path):
    mock_lineage.return_value = {
        "spawned_children": [
            {
                "child_conv_id": "wrong_child",
                "child_outcome_summary": "22/22 red baseline tests authored."
            }
        ]
    }
    
    res = _sandbox_2_subagent_lineage(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_fail_no_children(mock_lineage, mock_indexer, tmp_path):
    mock_lineage.return_value = {
        "spawned_children": []
    }
    
    res = _sandbox_2_subagent_lineage(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_fail_wrong_summary(mock_lineage, mock_indexer, tmp_path):
    mock_lineage.return_value = {
        "spawned_children": [
            {
                "child_conv_id": "conv_child_1",
                "child_outcome_summary": "0/22 tests authored."
            }
        ]
    }
    
    res = _sandbox_2_subagent_lineage(tmp_path)
    assert res["passed"] is False


@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_health_and_digest_pass(mock_digest, mock_health, tmp_path):
    mock_health.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage", "other"],
        "bigint_error_conversations": ["conv_bigint_err"],
    }
    mock_digest.return_value = {"char_count": 600}
    
    res = _sandbox_3_health_and_digest(tmp_path)
    
    assert res["name"] == "SB3_BrainHealth_And_CompactSessionDigest"
    assert res["passed"] is True
    assert res["metrics"]["digest_chars"] == 600
    assert res["metrics"]["bigint_detected"] == 1
    assert "elapsed_ms" in res

@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_fail_orphan(mock_digest, mock_health, tmp_path):
    mock_health.return_value = {
        "empty_or_orphan_dirs": ["other"],
        "bigint_error_conversations": ["conv_bigint_err"],
    }
    mock_digest.return_value = {"char_count": 500}
    
    res = _sandbox_3_health_and_digest(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_fail_bigint(mock_digest, mock_health, tmp_path):
    mock_health.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": ["other"],
    }
    mock_digest.return_value = {"char_count": 500}
    
    res = _sandbox_3_health_and_digest(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_fail_digest_size(mock_digest, mock_health, tmp_path):
    mock_health.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": ["conv_bigint_err"],
    }
    mock_digest.return_value = {"char_count": 601}
    
    res = _sandbox_3_health_and_digest(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_plugin_and_mcp_pass(mock_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "test"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    mock_mcp.side_effect = [
        {"result": {"isError": False, "data": "sync"}}, 
        {"result": {"isError": False, "data": "search"}}, 
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    
    assert res["name"] == "SB4_NativeL2Plugin_And_MCP_RoundTrip"
    assert res["passed"] is True
    assert res["metrics"]["plugin_valid"] is True
    assert res["metrics"]["oss_compliant"] is True
    assert "elapsed_ms" in res

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_fail_validation(mock_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "test"}
    mock_validate.return_value = {"valid": False, "oss_compliant": True}
    
    mock_mcp.side_effect = [
        {"result": {"isError": False}},
        {"result": {"isError": False}},
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_fail_oss(mock_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "test"}
    mock_validate.return_value = {"valid": True, "oss_compliant": False}
    
    mock_mcp.side_effect = [
        {"result": {"isError": False}},
        {"result": {"isError": False}},
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_fail_mcp_none(mock_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "test"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    mock_mcp.side_effect = [
        None,
        {"result": {"isError": False}},
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_fail_mcp_error(mock_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "test"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    mock_mcp.side_effect = [
        {"result": {"isError": False}},
        {"result": {"isError": True}},
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5")
@patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage")
@patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest")
@patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp")
def test_run_benchmark_suite_all_pass(mock_sb4, mock_sb3, mock_sb2, mock_sb1):
    mock_sb1.return_value = {"passed": True}
    mock_sb2.return_value = {"passed": True}
    mock_sb3.return_value = {"passed": True}
    mock_sb4.return_value = {"passed": True}
    
    res = run_benchmark_suite()
    assert res["all_passed"] is True
    assert len(res["sandboxes"]) == 4
    assert "total_elapsed_ms" in res
    assert mock_sb1.called
    assert mock_sb2.called
    assert mock_sb3.called
    assert mock_sb4.called

@patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5")
@patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage")
@patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest")
@patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp")
def test_run_benchmark_suite_some_fail(mock_sb4, mock_sb3, mock_sb2, mock_sb1):
    mock_sb1.return_value = {"passed": True}
    mock_sb2.return_value = {"passed": False}
    mock_sb3.return_value = {"passed": True}
    mock_sb4.return_value = {"passed": True}
    
    res = run_benchmark_suite()
    assert res["all_passed"] is False
    assert len(res["sandboxes"]) == 4
