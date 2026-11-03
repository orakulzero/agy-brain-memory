"""ISO 9001:2015 Agile V Verification Suite for Context Window Token Profiler (REQ-BM-PROFILER-01)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_req_bm_profiler_01_categorizes_tokens_and_identifies_top_hogs(
    tmp_path: Path,
) -> None:
    import agy_brain_memory
    from agy_brain_memory.context_token_profiler import (
        ContextTokenProfiler,
        profile_session_context,
    )

    assert getattr(agy_brain_memory, "__v3_generation__", None) == "3.0.0"
    assert hasattr(agy_brain_memory, "ContextTokenProfiler")
    assert hasattr(agy_brain_memory, "profile_session_context")

    steps = [
        {
            "step_index": 1,
            "type": "USER_INPUT",
            "category": "user_input",
            "content": "Inspect the project structure and database tables.",
        },
        {
            "step_index": 2,
            "type": "PLANNER_RESPONSE",
            "category": "thinking",
            "thinking": "Analyzing step sequence " * 25,
            "content": "Proceeding with inspection.",
        },
        {
            "step_index": 3,
            "type": "TOOL_RESPONSE",
            "tool_name": "read_file",
            "category": "file_reads",
            "content": "A" * 40000,
        },
        {
            "step_index": 4,
            "type": "TOOL_RESPONSE",
            "tool_name": "mcp_supabase_execute_sql",
            "category": "mcp_responses",
            "content": "B" * 20000,
        },
        {
            "step_index": 5,
            "type": "TOOL_RESPONSE",
            "tool_name": "run_command",
            "category": "tool_output",
            "content": "pytest passed\n" * 40,
        },
    ]

    report = profile_session_context(steps, top_n=3, token_budget=12000)
    assert report["requirement_id"] == "REQ-BM-PROFILER-01"
    assert report["total_tokens"] >= 15000
    assert report["over_budget"] is True
    for expected_cat in (
        "tool_output",
        "file_reads",
        "mcp_responses",
        "thinking",
        "user_input",
    ):
        assert expected_cat in report["category_tokens"]
        assert report["category_tokens"][expected_cat] > 0

    assert len(report["top_hogs"]) == 3
    assert report["top_hogs"][0]["step_index"] == 3
    assert report["top_hogs"][0]["category"] == "file_reads"
    assert len(report["compaction_plan"]) >= 1
    assert report["compaction_plan"][0]["estimated_savings_tokens"] > 0


def test_req_bm_profiler_01_streaming_jsonl_tolerates_malformed_lines(
    tmp_path: Path,
) -> None:
    from agy_brain_memory.context_token_profiler import ContextTokenProfiler

    jsonl_file = tmp_path / "session_transcript.jsonl"
    raw_lines = [
        json.dumps({"step_index": 1, "type": "USER_INPUT", "content": "Hello world"}),
        '{"step_index": 99, broken',
        json.dumps(
            {
                "step_index": 2,
                "type": "TOOL_RESPONSE",
                "tool_name": "read_file",
                "content": "X" * 8000,
            }
        ),
        json.dumps(
            {
                "step_index": 3,
                "type": "PLANNER_RESPONSE",
                "thinking": "Reasoning trace " * 20,
                "content": "Done.",
            }
        ),
    ]
    jsonl_file.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    profiler = ContextTokenProfiler(top_n=2, token_budget=50000)
    res = profiler.profile_jsonl(str(jsonl_file))
    assert res["requirement_id"] == "REQ-BM-PROFILER-01"
    assert res.get("malformed_lines_skipped", 0) == 1
    valid_count = res.get("steps_profiled", res.get("step_count", res.get("valid_steps", 0)))
    assert valid_count == 3
    assert res["total_tokens"] >= 2000
