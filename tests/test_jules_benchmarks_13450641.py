"""Tests for agy_brain_memory/benchmarks.py."""

from __future__ import annotations

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
    conv_dir = tmp_path / "conv_empty"
    tpath = _write_transcript(conv_dir, [])
    assert tpath.exists()
    assert tpath.read_text(encoding="utf-8") == "\n"

def test_write_transcript_boundary_chars(tmp_path: Path) -> None:
    conv_dir = tmp_path / "conv_chars"
    steps = [
        {"content": "Hello \u4f60\u597d"},
        {"content": "Emoji \U0001f600"}
    ]
    tpath = _write_transcript(conv_dir, steps)
    assert tpath.exists()
    text = tpath.read_text(encoding="utf-8")
    assert "你好" in text
    assert "😀" in text
    assert len(text.splitlines()) == 2

@patch("agy_brain_memory.benchmarks.search_brain")
@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
def test_sandbox_1_pass(mock_indexer_cls, mock_search, tmp_path: Path) -> None:
    mock_indexer = MagicMock()
    mock_indexer_cls.return_value = mock_indexer
    mock_indexer.sync.side_effect = [
        {"conversations_indexed": 5, "conversations_skipped": 0},
        {"conversations_indexed": 0, "conversations_skipped": 5},
    ]
    mock_search.return_value = {"results": ["hit1"]}

    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    assert res["passed"] is True
    assert res["name"] == "SB1_IncrementalWAL_And_FTS5_Search"
    assert res["metrics"]["first_indexed"] == 5
    assert res["metrics"]["second_skipped"] == 5
    assert res["metrics"]["search_hits"] == 1
    
    # Invariant: transcript files are created
    for i in range(5):
        assert (tmp_path / f"sb1_brain/conv_{i}/.system_generated/logs/transcript.jsonl").exists()

@patch("agy_brain_memory.benchmarks.search_brain")
@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
def test_sandbox_1_fail_boundary(mock_indexer_cls, mock_search, tmp_path: Path) -> None:
    mock_indexer = MagicMock()
    mock_indexer_cls.return_value = mock_indexer
    # Second sync indexed > 0
    mock_indexer.sync.side_effect = [
        {"conversations_indexed": 5, "conversations_skipped": 0},
        {"conversations_indexed": 1, "conversations_skipped": 4},
    ]
    mock_search.return_value = {"results": ["hit1"]}

    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
def test_sandbox_2_pass(mock_indexer_cls, mock_get_lineage, tmp_path: Path) -> None:
    mock_indexer_cls.return_value = MagicMock()
    mock_get_lineage.return_value = {
        "spawned_children": [
            {
                "child_conv_id": "conv_child_1",
                "child_outcome_summary": "22/22 tests passed."
            }
        ]
    }

    res = _sandbox_2_subagent_lineage(tmp_path)
    assert res["passed"] is True
    assert res["name"] == "SB2_SubagentCallGraph_Lineage_And_Outcomes"
    assert res["metrics"]["edges_extracted"] == 1

@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
def test_sandbox_2_fail_edge(mock_indexer_cls, mock_get_lineage, tmp_path: Path) -> None:
    mock_indexer_cls.return_value = MagicMock()
    # Missing summary match "22/22"
    mock_get_lineage.return_value = {
        "spawned_children": [
            {
                "child_conv_id": "conv_child_1",
                "child_outcome_summary": "21/22 tests passed."
            }
        ]
    }

    res = _sandbox_2_subagent_lineage(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.export_session_digest")
@patch("agy_brain_memory.benchmarks.analyze_brain_health")
def test_sandbox_3_pass(mock_analyze, mock_export, tmp_path: Path) -> None:
    mock_analyze.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": ["conv_bigint_err"]
    }
    mock_export.return_value = {"char_count": 600}

    res = _sandbox_3_health_and_digest(tmp_path)
    assert res["passed"] is True
    assert res["name"] == "SB3_BrainHealth_And_CompactSessionDigest"
    assert res["metrics"]["digest_chars"] == 600
    assert res["metrics"]["bigint_detected"] == 1

@patch("agy_brain_memory.benchmarks.export_session_digest")
@patch("agy_brain_memory.benchmarks.analyze_brain_health")
def test_sandbox_3_fail_boundary(mock_analyze, mock_export, tmp_path: Path) -> None:
    mock_analyze.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": ["conv_bigint_err"]
    }
    mock_export.return_value = {"char_count": 601} # > 600

    res = _sandbox_3_health_and_digest(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks.handle_mcp_request")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
def test_sandbox_4_pass(mock_install, mock_validate, mock_mcp, tmp_path: Path) -> None:
    mock_install.return_value = {"plugin_dir": "fake_dir"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    mock_mcp.side_effect = [
        {"result": {"isError": False}}, # sync_resp
        {"result": {"isError": False}}, # search_resp
    ]

    res = _sandbox_4_plugin_and_mcp(tmp_path)
    assert res["passed"] is True
    assert res["name"] == "SB4_NativeL2Plugin_And_MCP_RoundTrip"
    assert res["metrics"]["plugin_valid"] is True

@patch("agy_brain_memory.benchmarks.handle_mcp_request")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
def test_sandbox_4_fail_none_resp(mock_install, mock_validate, mock_mcp, tmp_path: Path) -> None:
    mock_install.return_value = {"plugin_dir": "fake_dir"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    # Test boundary of None response
    mock_mcp.side_effect = [
        None, 
        {"result": {"isError": False}},
    ]

    res = _sandbox_4_plugin_and_mcp(tmp_path)
    assert res["passed"] is False

@patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp")
@patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest")
@patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage")
@patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5")
def test_run_benchmark_suite_all_pass(mock_sb1, mock_sb2, mock_sb3, mock_sb4) -> None:
    mock_sb1.return_value = {"passed": True}
    mock_sb2.return_value = {"passed": True}
    mock_sb3.return_value = {"passed": True}
    mock_sb4.return_value = {"passed": True}

    res = run_benchmark_suite()
    assert res["all_passed"] is True
    assert len(res["sandboxes"]) == 4

@patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp")
@patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest")
@patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage")
@patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5")
def test_run_benchmark_suite_one_fail(mock_sb1, mock_sb2, mock_sb3, mock_sb4) -> None:
    mock_sb1.return_value = {"passed": True}
    mock_sb2.return_value = {"passed": False}
    mock_sb3.return_value = {"passed": True}
    mock_sb4.return_value = {"passed": True}

    res = run_benchmark_suite()
    assert res["all_passed"] is False
    assert len(res["sandboxes"]) == 4
