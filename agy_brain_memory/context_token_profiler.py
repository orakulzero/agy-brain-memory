"""Context Window Token & Cost Profiler across Session Steps (REQ-BM-PROFILER-01)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

CATEGORIES: tuple[str, ...] = (
    "tool_output",
    "file_reads",
    "mcp_responses",
    "thinking",
    "user_input",
)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _classify_step_category(step: Mapping[str, Any]) -> str:
    explicit = str(step.get("category") or "").strip()
    if explicit in CATEGORIES:
        return explicit

    tool_name = str(step.get("tool_name") or "").strip().lower()
    step_type = str(step.get("type") or "").strip().upper()

    if tool_name.startswith("mcp_") or tool_name == "call_mcp_tool":
        return "mcp_responses"
    if tool_name in ("read_file", "view_file", "read_resource"):
        return "file_reads"
    if step_type == "USER_INPUT":
        return "user_input"
    if step_type == "PLANNER_RESPONSE" or step.get("thinking"):
        return "thinking"
    return "tool_output"


def _extract_step_text(step: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("content", "thinking", "tool_calls", "output"):
        val = step.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
        elif val is not None and not isinstance(val, str):
            parts.append(json.dumps(val, ensure_ascii=False))
    return "\n".join(parts)


def profile_session_context(
    steps: Iterable[Mapping[str, Any]],
    top_n: int = 5,
    token_budget: int = 120000,
) -> dict[str, Any]:
    """REQ-BM-PROFILER-01: Profile token distribution across categories and rank top context hogs."""
    category_tokens: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    step_records: list[dict[str, Any]] = []
    total_tokens = 0
    valid_steps = 0

    for idx, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        valid_steps += 1
        step_index = int(step.get("step_index", idx + 1))
        category = _classify_step_category(step)
        raw_text = _extract_step_text(step)
        tokens = _estimate_tokens(raw_text)

        category_tokens[category] = category_tokens.get(category, 0) + tokens
        total_tokens += tokens

        step_records.append(
            {
                "step_index": step_index,
                "type": str(step.get("type") or "UNKNOWN"),
                "tool_name": str(step.get("tool_name") or ""),
                "category": category,
                "tokens": tokens,
            }
        )

    ranked_hogs = sorted(
        step_records,
        key=lambda item: (item["tokens"], -item["step_index"]),
        reverse=True,
    )[: max(0, int(top_n))]

    compaction_plan: list[dict[str, Any]] = []
    for hog in ranked_hogs:
        if hog["tokens"] <= 0:
            continue
        savings = max(1, int(hog["tokens"] * 0.8))
        compaction_plan.append(
            {
                "step_index": hog["step_index"],
                "category": hog["category"],
                "action": f"compact_{hog['category']}",
                "current_tokens": hog["tokens"],
                "estimated_savings_tokens": savings,
            }
        )

    budget_val = max(1, int(token_budget))
    return {
        "requirement_id": "REQ-BM-PROFILER-01",
        "total_tokens": total_tokens,
        "token_budget": budget_val,
        "over_budget": total_tokens > budget_val,
        "steps_profiled": valid_steps,
        "step_count": valid_steps,
        "valid_steps": valid_steps,
        "category_tokens": category_tokens,
        "top_hogs": ranked_hogs,
        "compaction_plan": compaction_plan,
    }


class ContextTokenProfiler:
    """REQ-BM-PROFILER-01: Streaming context window token profiler for JSONL transcripts and step lists."""

    def __init__(self, top_n: int = 5, token_budget: int = 120000) -> None:
        self.top_n = int(top_n)
        self.token_budget = int(token_budget)

    def profile_steps(self, steps: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        return profile_session_context(
            steps=steps,
            top_n=self.top_n,
            token_budget=self.token_budget,
        )

    def profile_jsonl(self, jsonl_path: Path | str) -> dict[str, Any]:
        path = Path(jsonl_path)
        malformed_skipped = 0
        parsed_steps: list[dict[str, Any]] = []

        if path.is_file():
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                        if isinstance(obj, dict):
                            parsed_steps.append(obj)
                        else:
                            malformed_skipped += 1
                    except ValueError:
                        malformed_skipped += 1

        report = self.profile_steps(parsed_steps)
        report["malformed_lines_skipped"] = malformed_skipped
        return report
