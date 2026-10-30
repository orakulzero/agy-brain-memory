import pytest
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from agy_brain_memory.benchmarks import (
    _write_transcript,
    _sandbox_1_incremental_wal_and_fts5,
    _sandbox_2_subagent_lineage,
    _sandbox_3_health_and_digest,
    _sandbox_4_plugin_and_mcp,
    run_benchmark_suite
)

def test_write_transcript_basic(tmp_path: Path):
    conv_dir = tmp_path / "conv"
    steps = [
        {"step": 1, "content": "hello", "type": "USER_INPUT"},
        {"step": 2, "content": "world", "status": "DONE"}
    ]
    tpath = _write_transcript(conv_dir, steps)
    assert tpath.exists()
    assert tpath.is_file()
    assert tpath.parent.name == "logs"
    assert tpath.parent.parent.name == ".system_generated"
    
    # Read back to verify JSONL
    content = tpath.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == steps[0]
    assert json.loads(lines[1]) == steps[1]

def test_write_transcript_empty(tmp_path: Path):
    conv_dir = tmp_path / "conv_empty"
    tpath = _write_transcript(conv_dir, [])
    assert tpath.exists()
    content = tpath.read_text(encoding="utf-8")
    assert content == "\n" # The implementation does `"\n".join(lines) + "\n"` -> "" + "\n" -> "\n"

def test_write_transcript_unicode(tmp_path: Path):
    conv_dir = tmp_path / "conv_uni"
    steps = [{"content": "こんにちは世界"}]
    tpath = _write_transcript(conv_dir, steps)
    content = tpath.read_text(encoding="utf-8")
    assert "こんにちは世界" in content

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_pass(mock_search_brain, mock_indexer, tmp_path: Path):
    mock_indexer_instance = mock_indexer.return_value
    mock_indexer_instance.sync.side_effect = [
        {"conversations_indexed": 5},
        {"conversations_indexed": 0, "conversations_skipped": 5}
    ]
    mock_search_brain.return_value = {"results": [{"id": 1}]}

    result = _sandbox_1_incremental_wal_and_fts5(tmp_path)

    assert result["name"] == "SB1_IncrementalWAL_And_FTS5_Search"
    assert result["passed"] is True
    assert result["metrics"]["first_indexed"] == 5
    assert result["metrics"]["second_skipped"] == 5
    assert result["metrics"]["search_hits"] == 1
    assert "elapsed_ms" in result
    assert isinstance(result["elapsed_ms"], float)

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.search_brain")
def test_sandbox_1_fail_sync(mock_search_brain, mock_indexer, tmp_path: Path):
    mock_indexer_instance = mock_indexer.return_value
    mock_indexer_instance.sync.side_effect = [
        {"conversations_indexed": 4}, # Should be 5
        {"conversations_indexed": 0, "conversations_skipped": 5}
    ]
    mock_search_brain.return_value = {"results": [{"id": 1}]}

    result = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    assert result["passed"] is False

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_pass(mock_get_lineage, mock_indexer, tmp_path: Path):
    mock_get_lineage.return_value = {
        "spawned_children": [
            {"child_conv_id": "conv_child_1", "child_outcome_summary": "22/22 red baseline tests authored."}
        ]
    }
    result = _sandbox_2_subagent_lineage(tmp_path)

    assert result["name"] == "SB2_SubagentCallGraph_Lineage_And_Outcomes"
    assert result["passed"] is True
    assert result["metrics"]["edges_extracted"] == 1

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_fail_missing_child(mock_get_lineage, mock_indexer, tmp_path: Path):
    mock_get_lineage.return_value = {
        "spawned_children": [] # Should have 1 child
    }
    result = _sandbox_2_subagent_lineage(tmp_path)
    assert result["passed"] is False

@patch("agy_brain_memory.benchmarks.BrainMemoryIndexer")
@patch("agy_brain_memory.benchmarks.get_subagent_lineage")
def test_sandbox_2_fail_wrong_text(mock_get_lineage, mock_indexer, tmp_path: Path):
    mock_get_lineage.return_value = {
        "spawned_children": [
            {"child_conv_id": "conv_child_1", "child_outcome_summary": "21/22 red baseline tests authored."} # Wrong summary
        ]
    }
    result = _sandbox_2_subagent_lineage(tmp_path)
    assert result["passed"] is False

@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_pass(mock_export, mock_analyze, tmp_path: Path):
    mock_analyze.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage", "other"],
        "bigint_error_conversations": ["conv_bigint_err"]
    }
    mock_export.return_value = {"char_count": 600}

    result = _sandbox_3_health_and_digest(tmp_path)

    assert result["name"] == "SB3_BrainHealth_And_CompactSessionDigest"
    assert result["passed"] is True
    assert result["metrics"]["digest_chars"] == 600
    assert result["metrics"]["bigint_detected"] == 1

@patch("agy_brain_memory.benchmarks.analyze_brain_health")
@patch("agy_brain_memory.benchmarks.export_session_digest")
def test_sandbox_3_fail_size(mock_export, mock_analyze, tmp_path: Path):
    mock_analyze.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": ["conv_bigint_err"]
    }
    mock_export.return_value = {"char_count": 601} # Over limit

    result = _sandbox_3_health_and_digest(tmp_path)
    assert result["passed"] is False

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_pass(mock_handle_mcp, mock_validate, mock_install, tmp_path: Path):
    mock_install.return_value = {"plugin_dir": "some/dir"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    mock_handle_mcp.side_effect = [
        {"result": {"success": True}}, # sync_resp
        {"result": {"success": True}}  # search_resp
    ]

    result = _sandbox_4_plugin_and_mcp(tmp_path)

    assert result["name"] == "SB4_NativeL2Plugin_And_MCP_RoundTrip"
    assert result["passed"] is True
    assert result["metrics"]["plugin_valid"] is True
    assert result["metrics"]["oss_compliant"] is True

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_fail_mcp_error(mock_handle_mcp, mock_validate, mock_install, tmp_path: Path):
    mock_install.return_value = {"plugin_dir": "some/dir"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    mock_handle_mcp.side_effect = [
        {"result": {"isError": True}}, # Error here
        {"result": {"isError": False}}
    ]

    result = _sandbox_4_plugin_and_mcp(tmp_path)
    assert result["passed"] is False

@patch("agy_brain_memory.benchmarks.install_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.validate_brain_memory_plugin")
@patch("agy_brain_memory.benchmarks.handle_mcp_request")
def test_sandbox_4_fail_mcp_none(mock_handle_mcp, mock_validate, mock_install, tmp_path: Path):
    mock_install.return_value = {"plugin_dir": "some/dir"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    # Sync fails by returning None
    mock_handle_mcp.side_effect = [
        None,
        {"result": {"isError": False}}
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

    result = run_benchmark_suite()

    assert result["all_passed"] is True
    assert len(result["sandboxes"]) == 4

@patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5")
@patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage")
@patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest")
@patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp")
def test_run_benchmark_suite_one_fails(mock_sb4, mock_sb3, mock_sb2, mock_sb1):
    mock_sb1.return_value = {"passed": True}
    mock_sb2.return_value = {"passed": False} # Fails
    mock_sb3.return_value = {"passed": True}
    mock_sb4.return_value = {"passed": True}

    result = run_benchmark_suite()

    assert result["all_passed"] is False
    assert len(result["sandboxes"]) == 4

def test_run_benchmark_suite_tmpdir_cleanup():
    # Make sure run_benchmark_suite cleans up the temp dir
    with patch("agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5") as mock_sb1, \
         patch("agy_brain_memory.benchmarks._sandbox_2_subagent_lineage") as mock_sb2, \
         patch("agy_brain_memory.benchmarks._sandbox_3_health_and_digest") as mock_sb3, \
         patch("agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp") as mock_sb4:
        
        tmp_dirs = []
        
        def mock_sb_call(root):
            tmp_dirs.append(root)
            return {"passed": True}
            
        mock_sb1.side_effect = mock_sb_call
        mock_sb2.return_value = {"passed": True}
        mock_sb3.return_value = {"passed": True}
        mock_sb4.return_value = {"passed": True}
        
        run_benchmark_suite()
        
        assert len(tmp_dirs) == 1
        assert not tmp_dirs[0].exists()
