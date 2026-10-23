"""Stdio JSON-RPC 2.0 MCP Server for Antigravity Brain Memory & Subagent Lineage."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from agy_brain_memory.compactor import analyze_brain_health, export_session_digest
from agy_brain_memory.indexer import BrainMemoryIndexer
from agy_brain_memory.search import get_subagent_lineage, search_brain

SERVER_NAME = "agy-brain-memory-mcp"
SERVER_VERSION = "1.0.0"

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "agy_brain_sync",
        "description": "Incrementally index Antigravity brain transcripts and Markdown artifacts into SQLite WAL + FTS5.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force full re-index of all transcripts even if unchanged.",
                }
            },
        },
    },
    {
        "name": "agy_brain_search",
        "description": "Search indexed Antigravity brain conversations and artifacts using sanitized FTS5 BM25 ranking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query across conversation steps, thinking blocks, tool calls, and artifacts.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10).",
                },
                "step_type": {
                    "type": "string",
                    "description": "Optional step type filter (e.g. USER_INPUT, PLANNER_RESPONSE).",
                },
                "include_artifacts": {
                    "type": "boolean",
                    "description": "Also search Markdown artifacts (default true).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "agy_brain_subagent_lineage",
        "description": "Reconstruct parent-to-child subagent call-graph trees and child outcomes for a conversation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conv_id": {
                    "type": "string",
                    "description": "Conversation ID to inspect parent, spawned children, and descendants.",
                },
                "role_query": {
                    "type": "string",
                    "description": "Optional substring filter on subagent Role or TypeName.",
                },
            },
        },
    },
    {
        "name": "agy_brain_session_digest",
        "description": "Export a compact Markdown handoff digest of a conversation within a strict character budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conv_id": {
                    "type": "string",
                    "description": "Conversation ID to summarize.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum character budget for the Markdown digest (default 2000).",
                },
            },
            "required": ["conv_id"],
        },
    },
    {
        "name": "agy_brain_health_report",
        "description": "Analyze Antigravity brain storage health, detecting orphaned folders, BigInt crashes, and stalled sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def default_brain_dir() -> Path:
    return Path.home() / ".gemini" / "antigravity" / "brain"


def default_db_path() -> Path:
    return Path.home() / ".gemini" / "antigravity" / "brain_index.sqlite"


def handle_mcp_request(
    req: dict[str, Any],
    brain_dir: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Process a single JSON-RPC 2.0 MCP request and return the response dictionary."""
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params") or {}

    bdir = Path(brain_dir) if brain_dir is not None else default_brain_dir()
    dpath = Path(db_path) if db_path is not None else default_db_path()

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                },
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments") or {}
        try:
            indexer = BrainMemoryIndexer(db_path=dpath, brain_dir=bdir)
            if tool_name == "agy_brain_sync":
                payload = indexer.sync(force=bool(args.get("force", False)))
            elif tool_name == "agy_brain_search":
                indexer.sync(force=False)
                payload = search_brain(
                    db_path=dpath,
                    query=str(args.get("query", "")),
                    limit=int(args.get("limit", 10)),
                    step_type=args.get("step_type"),
                    include_artifacts=bool(args.get("include_artifacts", True)),
                )
            elif tool_name == "agy_brain_subagent_lineage":
                indexer.sync(force=False)
                payload = get_subagent_lineage(
                    db_path=dpath,
                    conv_id=args.get("conv_id"),
                    role_query=args.get("role_query"),
                )
            elif tool_name == "agy_brain_session_digest":
                indexer.sync(force=False)
                payload = export_session_digest(
                    db_path=dpath,
                    conv_id=str(args.get("conv_id", "")),
                    max_chars=int(args.get("max_chars", 2000)),
                )
            elif tool_name == "agy_brain_health_report":
                payload = analyze_brain_health(brain_dir=bdir, db_path=dpath)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                            }
                        ],
                        "isError": True,
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(payload, indent=2),
                        }
                    ],
                    "isError": False,
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error": str(exc)}),
                        }
                    ],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def run_stdio_server(
    brain_dir: Path | str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Run the JSON-RPC 2.0 MCP server over standard input/output."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_mcp_request(req, brain_dir=brain_dir, db_path=db_path)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
