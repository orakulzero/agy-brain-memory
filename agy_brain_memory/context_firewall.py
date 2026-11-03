"""Subagent Context Firewall & Knapsack Budget Allocation.

Traceability:
- REQ-BM-FIREWALL-01: Subagent context firewall isolating external references >500 lines and log dumps.

Quality Framework: ISO 9001:2015, ISO/IEC 42001:2023, Agile V.
Hygiene: 100% English-only (0 Cyrillic characters).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from agy_brain_memory.compactor import allocate_context_budget


def evaluate_context_firewall(
    content: str = "",
    content_type: str = "text",
    max_lines: int = 500,
) -> Dict[str, Any]:
    """Evaluate context content for subagent isolation or linear folding (REQ-BM-FIREWALL-01)."""
    lines = len(content.splitlines()) if isinstance(content, str) else 0
    is_log = (
        content_type in ("log_dump", "log", "traceback")
        or ("traceback (most recent call last)" in content.lower())
    )
    criteria: List[str] = []
    requires_isolation = False

    if lines > max_lines:
        requires_isolation = True
        criteria.append("context_firewall_gt_500_lines")
    if is_log:
        requires_isolation = True
        criteria.append("raw_log_dump_quarantine")

    if requires_isolation:
        action = "spawn_subagent"
    else:
        action = "fold_in_process"
        criteria.append("fold_linear_in_process")

    return {
        "action": action,
        "requires_isolation": requires_isolation,
        "is_log_dump": is_log,
        "estimated_lines": lines,
        "max_lines": max_lines,
        "criteria_triggered": criteria,
        "content_type": content_type,
    }


def should_isolate_to_subagent(
    content_or_lines: str | int,
    content_type: str = "text",
    max_lines: int = 500,
) -> bool:
    """Predicate indicating whether context payload requires subagent isolation (REQ-BM-FIREWALL-01)."""
    if isinstance(content_or_lines, int):
        return content_or_lines > max_lines or content_type == "log_dump"
    return evaluate_context_firewall(
        content=str(content_or_lines),
        content_type=content_type,
        max_lines=max_lines,
    )["requires_isolation"]


class ContextFirewallRouter:
    """Decision router for orchestrator context protection (REQ-BM-FIREWALL-01)."""

    def __init__(self, max_in_process_lines: int = 500) -> None:
        self.max_in_process_lines = max_in_process_lines

    def evaluate(
        self,
        estimated_lines: int | None = None,
        source_type: str = "general",
        content: str | None = None,
    ) -> Dict[str, Any]:
        cnt = content or ""
        lines = estimated_lines if estimated_lines is not None else len(cnt.splitlines())
        is_log = (
            source_type in ("log_dump", "log", "traceback")
            or ("traceback (most recent call last)" in cnt.lower())
        )
        criteria: List[str] = []
        requires_isolation = False

        if lines > self.max_in_process_lines:
            requires_isolation = True
            criteria.append("context_firewall_gt_500_lines")
        if is_log:
            requires_isolation = True
            criteria.append("raw_log_dump_quarantine")

        if requires_isolation:
            action = "spawn_subagent"
        else:
            action = "fold_in_process"
            criteria.append("fold_linear_in_process")

        return {
            "action": action,
            "requires_isolation": requires_isolation,
            "is_log_dump": is_log,
            "estimated_lines": lines,
            "max_in_process_lines": self.max_in_process_lines,
            "criteria_triggered": criteria,
            "source_type": source_type,
        }


allocate_context_knapsack = allocate_context_budget


class SubagentContextFirewall:
    """Context firewall controller managing subagent dispatch boundaries (REQ-BM-FIREWALL-01)."""

    def __init__(self, max_lines: int = 500) -> None:
        self.router = ContextFirewallRouter(max_in_process_lines=max_lines)

    def evaluate(self, content: str, content_type: str = "text") -> Dict[str, Any]:
        return self.router.evaluate(content=content, source_type=content_type)

    def should_isolate(self, content_or_lines: str | int, content_type: str = "text") -> bool:
        return should_isolate_to_subagent(content_or_lines, content_type=content_type, max_lines=self.router.max_in_process_lines)

    @staticmethod
    def allocate_knapsack(candidates: List[Dict[str, Any]], max_token_budget: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        return allocate_context_budget(candidates, max_token_budget=max_token_budget)
