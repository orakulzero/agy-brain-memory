import pytest
from unittest.mock import MagicMock, patch

import agy_brain_memory.benchmarks
from agy_brain_memory.benchmarks import (
    _write_transcript,
    _sandbox_1_incremental_wal_and_fts5,
    _sandbox_2_subagent_lineage,
    _sandbox_3_health_and_digest,
    _sandbox_4_plugin_and_mcp,
    run_benchmark_suite,
)

def test_write_transcript(tmp_path):
    steps = [{"foo": "bar"}, {"baz": "qux"}]
    tpath = _write_transcript(tmp_path, steps)
    
    assert tpath.exists()
    assert tpath.parent.name == "logs"
    assert tpath.parent.parent.name == ".system_generated"
    
    content = tpath.read_text(encoding="utf-8")
    assert '{"foo": "bar"}' in content
    assert '{"baz": "qux"}' in content

@patch('agy_brain_memory.benchmarks.BrainMemoryIndexer')
@patch('agy_brain_memory.benchmarks.search_brain')
def test_sandbox_1_incremental_wal_and_fts5_success(mock_search_brain, MockIndexer, tmp_path):
    mock_indexer = MockIndexer.return_value
    mock_indexer.sync.side_effect = [
        {"conversations_indexed": 5},
        {"conversations_indexed": 0, "conversations_skipped": 5}
    ]
    
    mock_search_brain.return_value = {"results": [{"id": 1}]}
    
    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    
    assert res["name"] == "SB1_IncrementalWAL_And_FTS5_Search"
    assert res["passed"] is True
    assert res["metrics"]["first_indexed"] == 5
    assert res["metrics"]["second_skipped"] == 5
    assert res["metrics"]["search_hits"] == 1

@patch('agy_brain_memory.benchmarks.BrainMemoryIndexer')
@patch('agy_brain_memory.benchmarks.search_brain')
def test_sandbox_1_incremental_wal_and_fts5_failure(mock_search_brain, MockIndexer, tmp_path):
    mock_indexer = MockIndexer.return_value
    mock_indexer.sync.side_effect = [
        {"conversations_indexed": 5},
        {"conversations_indexed": 1, "conversations_skipped": 4} # Error, should be 0 indexed
    ]
    
    mock_search_brain.return_value = {"results": []} # Error, should be >= 1
    
    res = _sandbox_1_incremental_wal_and_fts5(tmp_path)
    
    assert res["name"] == "SB1_IncrementalWAL_And_FTS5_Search"
    assert res["passed"] is False

@patch('agy_brain_memory.benchmarks.BrainMemoryIndexer')
@patch('agy_brain_memory.benchmarks.get_subagent_lineage')
def test_sandbox_2_subagent_lineage_success(mock_get_lineage, MockIndexer, tmp_path):
    mock_get_lineage.return_value = {
        "spawned_children": [
            {
                "child_conv_id": "conv_child_1",
                "child_outcome_summary": "Successfully ran 22/22 red baseline tests."
            }
        ]
    }
    
    res = _sandbox_2_subagent_lineage(tmp_path)
    
    assert res["name"] == "SB2_SubagentCallGraph_Lineage_And_Outcomes"
    assert res["passed"] is True
    assert res["metrics"]["edges_extracted"] == 1

@patch('agy_brain_memory.benchmarks.BrainMemoryIndexer')
@patch('agy_brain_memory.benchmarks.get_subagent_lineage')
def test_sandbox_2_subagent_lineage_failure(mock_get_lineage, MockIndexer, tmp_path):
    mock_get_lineage.return_value = {
        "spawned_children": [
            {
                "child_conv_id": "conv_child_2", # Wrong ID
                "child_outcome_summary": "Ran 22/22 tests."
            }
        ]
    }
    
    res = _sandbox_2_subagent_lineage(tmp_path)
    
    assert res["name"] == "SB2_SubagentCallGraph_Lineage_And_Outcomes"
    assert res["passed"] is False

@patch('agy_brain_memory.benchmarks.analyze_brain_health')
@patch('agy_brain_memory.benchmarks.export_session_digest')
def test_sandbox_3_health_and_digest_success(mock_export, mock_analyze, tmp_path):
    mock_analyze.return_value = {
        "empty_or_orphan_dirs": ["tempmediaStorage"],
        "bigint_error_conversations": ["conv_bigint_err"]
    }
    
    mock_export.return_value = {
        "char_count": 500 # <= 600
    }
    
    res = _sandbox_3_health_and_digest(tmp_path)
    
    assert res["name"] == "SB3_BrainHealth_And_CompactSessionDigest"
    assert res["passed"] is True
    assert res["metrics"]["digest_chars"] == 500
    assert res["metrics"]["bigint_detected"] == 1

@patch('agy_brain_memory.benchmarks.analyze_brain_health')
@patch('agy_brain_memory.benchmarks.export_session_digest')
def test_sandbox_3_health_and_digest_failure(mock_export, mock_analyze, tmp_path):
    mock_analyze.return_value = {
        "empty_or_orphan_dirs": [],
        "bigint_error_conversations": []
    }
    
    mock_export.return_value = {
        "char_count": 601 # > 600
    }
    
    res = _sandbox_3_health_and_digest(tmp_path)
    
    assert res["name"] == "SB3_BrainHealth_And_CompactSessionDigest"
    assert res["passed"] is False

@patch('agy_brain_memory.benchmarks.install_brain_memory_plugin')
@patch('agy_brain_memory.benchmarks.validate_brain_memory_plugin')
@patch('agy_brain_memory.benchmarks.handle_mcp_request')
def test_sandbox_4_plugin_and_mcp_success(mock_handle_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "test_dir"}
    mock_validate.return_value = {"valid": True, "oss_compliant": True}
    
    mock_handle_mcp.side_effect = [
        {"result": {"isError": False}}, # sync_resp
        {"result": {"isError": False}}  # search_resp
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    
    assert res["name"] == "SB4_NativeL2Plugin_And_MCP_RoundTrip"
    assert res["passed"] is True
    assert res["metrics"]["plugin_valid"] is True
    assert res["metrics"]["oss_compliant"] is True

@patch('agy_brain_memory.benchmarks.install_brain_memory_plugin')
@patch('agy_brain_memory.benchmarks.validate_brain_memory_plugin')
@patch('agy_brain_memory.benchmarks.handle_mcp_request')
def test_sandbox_4_plugin_and_mcp_failure(mock_handle_mcp, mock_validate, mock_install, tmp_path):
    mock_install.return_value = {"plugin_dir": "test_dir"}
    mock_validate.return_value = {"valid": True, "oss_compliant": False}
    
    mock_handle_mcp.side_effect = [
        {"result": {"isError": True}}, # sync_resp
        {"result": {"isError": False}}  # search_resp
    ]
    
    res = _sandbox_4_plugin_and_mcp(tmp_path)
    
    assert res["name"] == "SB4_NativeL2Plugin_And_MCP_RoundTrip"
    assert res["passed"] is False

@patch('agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5')
@patch('agy_brain_memory.benchmarks._sandbox_2_subagent_lineage')
@patch('agy_brain_memory.benchmarks._sandbox_3_health_and_digest')
@patch('agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp')
def test_run_benchmark_suite_all_passed(sb4, sb3, sb2, sb1):
    sb1.return_value = {"passed": True}
    sb2.return_value = {"passed": True}
    sb3.return_value = {"passed": True}
    sb4.return_value = {"passed": True}
    
    res = run_benchmark_suite()
    
    assert res["all_passed"] is True
    assert len(res["sandboxes"]) == 4

@patch('agy_brain_memory.benchmarks._sandbox_1_incremental_wal_and_fts5')
@patch('agy_brain_memory.benchmarks._sandbox_2_subagent_lineage')
@patch('agy_brain_memory.benchmarks._sandbox_3_health_and_digest')
@patch('agy_brain_memory.benchmarks._sandbox_4_plugin_and_mcp')
def test_run_benchmark_suite_some_failed(sb4, sb3, sb2, sb1):
    sb1.return_value = {"passed": True}
    sb2.return_value = {"passed": False}
    sb3.return_value = {"passed": True}
    sb4.return_value = {"passed": True}
    
    res = run_benchmark_suite()
    
    assert res["all_passed"] is False
    assert len(res["sandboxes"]) == 4
