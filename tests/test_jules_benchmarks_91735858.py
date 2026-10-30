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

def test_write_transcript_empty(tmp_path):
    conv_dir = tmp_path / "conv1"
    tpath = _write_transcript(conv_dir, [])
    assert tpath.exists()
    assert tpath.read_text(encoding="utf-8") == "\n"

def test_write_transcript_with_content(tmp_path):
    conv_dir = tmp_path / "conv2"
    steps = [{"hello": "world"}, {"foo": "bar"}]
    tpath = _write_transcript(conv_dir, steps)
    assert tpath.exists()
    lines = tpath.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"hello": "world"}
    assert json.loads(lines[1]) == {"foo": "bar"}

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_pass(mock_search, mock_indexer_cls, tmp_path):
    mock_indexer = MagicMock()
    # first sync, second sync
    mock_indexer.sync.side_effect = [
        {"conversations_indexed": 5},
        {"conversations_indexed": 0, "conversations_skipped": 5}
    ]
    mock_indexer_cls.return_value = mock_indexer
    mock_search.return_value = {"results": [{"id": 1}]}

    result = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    assert result["name"] == "SB1_IncrementalWAL_And_FTS5_Search"
    assert result["passed"] is True
    assert result["metrics"]["first_indexed"] == 5
    assert result["metrics"]["second_skipped"] == 5
    assert result["metrics"]["search_hits"] == 1

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_fail_conditions(mock_search, mock_indexer_cls, tmp_path):
    mock_indexer = MagicMock()
    mock_indexer.sync.side_effect = [
        {"conversations_indexed": 4}, # wrong
        {"conversations_indexed": 0, "conversations_skipped": 5}
    ]
    mock_indexer_cls.return_value = mock_indexer
    mock_search.return_value = {"results": []} # wrong

    result = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    assert result["passed"] is False

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_pass(mock_get_subagent_lineage, mock_indexer_cls, tmp_path):
    mock_get_subagent_lineage.return_value = {
        "spawned_children": [
            {
                "child_conv_id": "conv_child_1",
                "child_outcome_summary": "22/22 tests passed"
            }
        ]
    }
    result = _sandbox_2_subagent_lineage(tmp_path)
    assert result["passed"] is True
    assert result["metrics"]["edges_extracted"] == 1

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_fail_no_children(mock_get_subagent_lineage, mock_indexer_cls, tmp_path):
    mock_get_subagent_lineage.return_value = {
        "spawned_children": []
    }
    result = _sandbox_2_subagent_lineage(tmp_path)
    assert result["passed"] is False

@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_pass(mock_export, mock_analyze, tmp_path):
    mock_analyze.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": ["conv_bigint_err"]
    }
    mock_export.return_value = {"char_count": 500}

    result = _sandbox_3_health_and_digest(tmp_path)
    assert result["passed"] is True

@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_fail(mock_export, mock_analyze, tmp_path):
    mock_analyze.return_value = {
        "empty_or_orphan_dirs": [],
        "bigint_error_conversations": []
    }
    mock_export.return_value = {"char_count": 601}

    result = _sandbox_3_health_and_digest(tmp_path)
    assert result["passed"] is False

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_pass(mock_handle_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "/tmp/plugin"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    # Needs two calls to handle_mcp_request (sync, search)
    mock_handle_mcp.side_effect = [
        {"result": {"success": True}},
        {"result": {"success": True}}
    ]

    result = _sandbox_4_plugin_and_mcp(tmp_path)
    assert result["passed"] is True

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_fail_isError(mock_handle_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "/tmp/plugin"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    # sync error
    mock_handle_mcp.side_effect = [
        {"result": {"isError": True}},
        {"result": {"success": True}}
    ]

    result = _sandbox_4_plugin_and_mcp(tmp_path)
    assert result["passed"] is False

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

@patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5")
@patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage")
@patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest")
@patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp")
def test_run_benchmark_suite_one_fails(mock_sb4, mock_sb3, mock_sb2, mock_sb1):
    mock_sb1.return_value = {"passed": True}
    mock_sb2.return_value = {"passed": False}
    mock_sb3.return_value = {"passed": True}
    mock_sb4.return_value = {"passed": True}

    res = run_benchmark_suite()
    assert res["all_passed"] is False
