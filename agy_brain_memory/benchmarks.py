"""4-Sandbox End-to-End Benchmark & Verification Suite for agy-brain-memory."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from agy_brain_memory.compactor import analyze_brain_health, export_session_digest
from agy_brain_memory.indexer import BrainMemoryIndexer
from agy_brain_memory.mcp_server import handle_mcp_request
from agy_brain_memory.plugin_installer import (
    install_brain_memory_plugin,
    validate_brain_memory_plugin,
)
from agy_brain_memory.search import get_subagent_lineage, search_brain


def _write_transcript(conv_dir: Path, steps: list[dict[str, Any]]) -> Path:
    logs_dir = conv_dir / ".system_generated" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tpath = logs_dir / "transcript.jsonl"
    lines = [json.dumps(s, ensure_ascii=False) for s in steps]
    tpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tpath


def _sandbox_1_incremental_wal_and_fts5(root: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    brain_dir = root / "sb1_brain"
    db_path = root / "sb1.sqlite"

    for idx in range(5):
        _write_transcript(
            brain_dir / f"conv_{idx}",
            [
                {
                    "step_index": 1,
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "content": f"ArchitectingHookMultiplexer_{idx} with SQLite WAL",
                },
                {
                    "step_index": 2,
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "content": f"Completed HookMultiplexer_{idx} (CRIT-01) SQLite WAL verification.",
                },
            ],
        )

    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    first_sync = indexer.sync()
    second_sync = indexer.sync()
    search_res = search_brain(
        db_path=db_path, query="HookMultiplexer_3: (CRIT-01) [SQLite]"
    )
    passed = (
        first_sync["conversations_indexed"] == 5
        and second_sync["conversations_indexed"] == 0
        and second_sync["conversations_skipped"] == 5
        and len(search_res["results"]) >= 1
    )
    return {
        "name": "SB1_IncrementalWAL_And_FTS5_Search",
        "passed": passed,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "metrics": {
            "first_indexed": first_sync["conversations_indexed"],
            "second_skipped": second_sync["conversations_skipped"],
            "search_hits": len(search_res["results"]),
        },
    }


def _sandbox_2_subagent_lineage(root: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    brain_dir = root / "sb2_brain"
    db_path = root / "sb2.sqlite"

    _write_transcript(
        brain_dir / "conv_parent",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Launching subagent",
                "tool_calls": [
                    {
                        "name": "invoke_subagent",
                        "args": {
                            "Role": "Test-Designer",
                            "TypeName": "self",
                            "Prompt": "Write red tests",
                        },
                    }
                ],
            },
            {
                "step_index": 2,
                "type": "GENERIC",
                "status": "DONE",
                "content": 'Created the following subagent:\n{"conversationId": "conv_child_1"}',
            },
        ],
    )
    _write_transcript(
        brain_dir / "conv_child_1",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "22/22 red baseline tests authored.",
            }
        ],
    )

    BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir).sync()
    lineage = get_subagent_lineage(db_path=db_path, conv_id="conv_parent")
    children = lineage["spawned_children"]
    passed = (
        len(children) == 1
        and children[0]["child_conv_id"] == "conv_child_1"
        and "22/22" in children[0]["child_outcome_summary"]
    )
    return {
        "name": "SB2_SubagentCallGraph_Lineage_And_Outcomes",
        "passed": passed,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "metrics": {"edges_extracted": len(children)},
    }


def _sandbox_3_health_and_digest(root: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    brain_dir = root / "sb3_brain"
    db_path = root / "sb3.sqlite"

    (brain_dir / "tempmediaStorage").mkdir(parents=True)
    _write_transcript(
        brain_dir / "conv_healthy",
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Generate compact digest " + ("token " * 200),
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Digest ready.",
            },
        ],
    )
    _write_transcript(
        brain_dir / "conv_bigint_err",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "ERROR",
                "content": "TypeError: Do not know how to serialize a BigInt",
            }
        ],
    )

    health = analyze_brain_health(brain_dir=brain_dir, db_path=db_path)
    digest = export_session_digest(
        db_path=db_path, conv_id="conv_healthy", max_chars=600
    )
    passed = (
        "tempmediaStorage" in health["empty_or_orphan_dirs"]
        and "conv_bigint_err" in health["bigint_error_conversations"]
        and digest["char_count"] <= 600
    )
    return {
        "name": "SB3_BrainHealth_And_CompactSessionDigest",
        "passed": passed,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "metrics": {
            "digest_chars": digest["char_count"],
            "bigint_detected": len(health["bigint_error_conversations"]),
        },
    }


def _sandbox_4_plugin_and_mcp(root: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    plugins_root = root / "sb4_plugins"
    inst = install_brain_memory_plugin(plugins_root)
    val = validate_brain_memory_plugin(inst["plugin_dir"])

    brain_dir = root / "sb4_brain"
    db_path = root / "sb4.sqlite"
    _write_transcript(
        brain_dir / "conv_mcp_sb4",
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Testing MCP round trip in SB4",
            }
        ],
    )

    sync_resp = handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "agy_brain_sync", "arguments": {"force": True}},
        },
        brain_dir=brain_dir,
        db_path=db_path,
    )
    search_resp = handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "agy_brain_search",
                "arguments": {"query": "SB4", "limit": 5},
            },
        },
        brain_dir=brain_dir,
        db_path=db_path,
    )
    passed = (
        val["valid"] is True
        and val["oss_compliant"] is True
        and sync_resp is not None
        and not sync_resp["result"].get("isError", False)
        and search_resp is not None
        and not search_resp["result"].get("isError", False)
    )
    return {
        "name": "SB4_NativeL2Plugin_And_MCP_RoundTrip",
        "passed": passed,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "metrics": {"plugin_valid": val["valid"], "oss_compliant": val["oss_compliant"]},
    }


def run_benchmark_suite() -> dict[str, Any]:
    """Run all 4 isolated sandboxes and return aggregate benchmark results."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="agy_brain_bench_", ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp)
        sandboxes = [
            _sandbox_1_incremental_wal_and_fts5(root),
            _sandbox_2_subagent_lineage(root),
            _sandbox_3_health_and_digest(root),
            _sandbox_4_plugin_and_mcp(root),
        ]
    all_passed = all(sb["passed"] for sb in sandboxes)
    return {
        "all_passed": all_passed,
        "total_elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "sandboxes": sandboxes,
    }
