"""Tests for the Stdio JSON-RPC 2.0 MCP Server (agy_brain_memory)."""

import json
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from agy_brain_memory.mcp_server import handle_mcp_request, run_stdio_server


def test_mcp_server_initialize():
    req = {"method": "initialize", "id": 1}
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "agy-brain-memory-mcp"
    assert resp["result"]["serverInfo"]["version"] == "1.0.0"


def test_mcp_server_initialized_notification():
    req = {"method": "notifications/initialized"}
    resp = handle_mcp_request(req)
    assert resp is None


def test_mcp_server_tools_list():
    req = {"method": "tools/list", "id": 2}
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["id"] == 2
    assert "tools" in resp["result"]
    tools = resp["result"]["tools"]
    assert len(tools) == 5
    tool_names = [t["name"] for t in tools]
    assert "agy_brain_sync" in tool_names
    assert "agy_brain_subagent_lineage" in tool_names


def test_mcp_server_unknown_method():
    req = {"method": "unknown_method", "id": 3}
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["error"]["code"] == -32601
    assert "Method not found" in resp["error"]["message"]


@patch("agy_brain_memory.mcp_server.BrainMemoryIndexer")
def test_mcp_server_unknown_tool(mock_indexer):
    req = {
        "method": "tools/call",
        "id": 4,
        "params": {
            "name": "unknown_tool",
            "arguments": {}
        }
    }
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["result"]["isError"] is True
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content["error"] == "Unknown tool: unknown_tool"


@patch("agy_brain_memory.mcp_server.BrainMemoryIndexer")
@patch("agy_brain_memory.mcp_server.get_subagent_lineage")
def test_subagent_lineage_invalid_conv_id_error(mock_get_lineage, mock_indexer):
    """Test edge case where agy_brain_subagent_lineage is called with an invalid conv_id 
    to ensure proper error handling and response format.
    """
    mock_get_lineage.side_effect = ValueError("Invalid conv_id format")

    req = {
        "method": "tools/call",
        "id": 5,
        "params": {
            "name": "agy_brain_subagent_lineage",
            "arguments": {
                "conv_id": "invalid-uuid"
            }
        }
    }
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 5
    assert resp["result"]["isError"] is True
    
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "Invalid conv_id format" in content["error"]


@patch("agy_brain_memory.mcp_server.BrainMemoryIndexer")
@patch("agy_brain_memory.mcp_server.get_subagent_lineage")
def test_subagent_lineage_missing_conv_id_error(mock_get_lineage, mock_indexer):
    """Test boundary case where conv_id is entirely omitted."""
    mock_get_lineage.side_effect = ValueError("conv_id is required")

    req = {
        "method": "tools/call",
        "id": 6,
        "params": {
            "name": "agy_brain_subagent_lineage",
            "arguments": {}
        }
    }
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["result"]["isError"] is True
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "conv_id is required" in content["error"]


@patch("agy_brain_memory.mcp_server.BrainMemoryIndexer")
@patch("agy_brain_memory.mcp_server.get_subagent_lineage")
def test_subagent_lineage_success(mock_get_lineage, mock_indexer):
    mock_get_lineage.return_value = {"lineage": ["node1", "node2"]}
    req = {
        "method": "tools/call",
        "id": 7,
        "params": {
            "name": "agy_brain_subagent_lineage",
            "arguments": {
                "conv_id": "valid-uuid"
            }
        }
    }
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["result"]["isError"] is False
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content == {"lineage": ["node1", "node2"]}
    mock_get_lineage.assert_called_once()
    assert mock_get_lineage.call_args.kwargs["conv_id"] == "valid-uuid"


@patch("agy_brain_memory.mcp_server.BrainMemoryIndexer")
def test_brain_sync_success(mock_indexer):
    instance = mock_indexer.return_value
    instance.sync.return_value = {"status": "synced"}
    req = {
        "method": "tools/call",
        "id": 8,
        "params": {
            "name": "agy_brain_sync",
            "arguments": {
                "force": True
            }
        }
    }
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["result"]["isError"] is False
    content = json.loads(resp["result"]["content"][0]["text"])
    assert content == {"status": "synced"}
    instance.sync.assert_called_once_with(force=True)


@patch("agy_brain_memory.mcp_server.BrainMemoryIndexer")
@patch("agy_brain_memory.mcp_server.search_brain")
def test_brain_search_success(mock_search_brain, mock_indexer):
    mock_search_brain.return_value = {"results": []}
    req = {
        "method": "tools/call",
        "id": 9,
        "params": {
            "name": "agy_brain_search",
            "arguments": {
                "query": "test query",
                "limit": 5,
                "include_artifacts": False
            }
        }
    }
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["result"]["isError"] is False
    mock_search_brain.assert_called_once()
    kwargs = mock_search_brain.call_args.kwargs
    assert kwargs["query"] == "test query"
    assert kwargs["limit"] == 5
    assert kwargs["include_artifacts"] is False
    assert kwargs["step_type"] is None


@patch("agy_brain_memory.mcp_server.BrainMemoryIndexer")
@patch("agy_brain_memory.mcp_server.export_session_digest")
def test_session_digest_success(mock_export, mock_indexer):
    mock_export.return_value = {"digest": "content"}
    req = {
        "method": "tools/call",
        "id": 10,
        "params": {
            "name": "agy_brain_session_digest",
            "arguments": {
                "conv_id": "test-id",
                "max_chars": 1000
            }
        }
    }
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["result"]["isError"] is False
    mock_export.assert_called_once()
    assert mock_export.call_args.kwargs["conv_id"] == "test-id"
    assert mock_export.call_args.kwargs["max_chars"] == 1000


@patch("agy_brain_memory.mcp_server.BrainMemoryIndexer")
@patch("agy_brain_memory.mcp_server.analyze_brain_health")
def test_health_report_success(mock_analyze, mock_indexer):
    mock_analyze.return_value = {"health": "good"}
    req = {
        "method": "tools/call",
        "id": 11,
        "params": {
            "name": "agy_brain_health_report",
            "arguments": {}
        }
    }
    resp = handle_mcp_request(req)
    assert resp is not None
    assert resp["result"]["isError"] is False
    mock_analyze.assert_called_once()


def test_run_stdio_server():
    # Provide multiple valid and invalid requests through stdin
    reqs = [
        json.dumps({"method": "notifications/initialized"}),
        "invalid_json",
        json.dumps({"method": "initialize", "id": 99}),
        ""
    ]
    input_str = "\n".join(reqs) + "\n"
    
    with patch("sys.stdin", StringIO(input_str)), patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        run_stdio_server()
        
    output = mock_stdout.getvalue().strip()
    assert output
    resp = json.loads(output.splitlines()[-1])
    assert resp["id"] == 99
    assert resp["result"]["serverInfo"]["name"] == "agy-brain-memory-mcp"
