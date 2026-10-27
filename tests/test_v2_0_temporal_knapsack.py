"""Tests for Antigravity v2.0.0 Temporal Knowledge Graph (REQ-BM-201)

and Utility-Density Knapsack Context Budget Allocator (REQ-BM-202).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

import agy_brain_memory

try:
    from agy_brain_memory import (
        TemporalEntityGraph,
        allocate_context_budget,
        extract_temporal_knowledge_graph,
    )
except ImportError:
    try:
        from agy_brain_memory.knowledge_graph import (
            TemporalEntityGraph,
            extract_temporal_knowledge_graph,
        )
        from agy_brain_memory.compactor import allocate_context_budget
    except ImportError:
        def allocate_context_budget(*args: Any, **kwargs: Any) -> Any:
            from agy_brain_memory.compactor import allocate_context_budget as fn
            return fn(*args, **kwargs)

        class TemporalEntityGraph:  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                from agy_brain_memory.knowledge_graph import TemporalEntityGraph as cls
                self._impl = cls(*args, **kwargs)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._impl, name)

        def extract_temporal_knowledge_graph(*args: Any, **kwargs: Any) -> Any:
            from agy_brain_memory.knowledge_graph import extract_temporal_knowledge_graph as fn
            return fn(*args, **kwargs)


# ============================================================================
# 1. Version and Re-export Tests (REQ-BM-201, REQ-BM-202)
# ============================================================================


def test_v2_0_brain_memory_version_and_exports() -> None:
    """Verify package version is 2.0.0 and v2.0.0 functions are exported in agy_brain_memory."""
    assert agy_brain_memory.__version__ == "2.0.0", "Package version must be bumped to 2.0.0"
    assert hasattr(agy_brain_memory, "TemporalEntityGraph")
    assert hasattr(agy_brain_memory, "extract_temporal_knowledge_graph")
    assert hasattr(agy_brain_memory, "allocate_context_budget")
    all_symbols = getattr(agy_brain_memory, "__all__", [])
    assert "TemporalEntityGraph" in all_symbols
    assert "extract_temporal_knowledge_graph" in all_symbols
    assert "allocate_context_budget" in all_symbols


# ============================================================================
# 2. Temporal Knowledge Graph Tests (REQ-BM-201)
# ============================================================================


def test_req201_temporal_graph_entity_addition_and_lookup() -> None:
    """REQ-BM-201: Adding entity nodes records metadata, properties, and timestamps."""
    graph = TemporalEntityGraph()
    res = graph.add_entity(
        entity_id="agent:architect",
        entity_type="Agent",
        properties={"role": "Chief Architect", "status": "active"},
        timestamp=1000.0,
    )
    assert isinstance(res, dict)
    assert res.get("entity_id") == "agent:architect"
    assert res.get("entity_type") == "Agent"

    # Export and verify entity is stored
    data = graph.export_graph_json()
    assert "agent:architect" in data["entities"]
    stored = data["entities"]["agent:architect"]
    assert stored["properties"]["role"] == "Chief Architect"
    assert stored["timestamp"] == 1000.0


def test_req201_temporal_graph_relation_valid_interval() -> None:
    """REQ-BM-201: Adding temporal relation stores [valid_from, valid_to] and confidence."""
    graph = TemporalEntityGraph()
    graph.add_entity("agent:writer", "Agent")
    graph.add_entity("file:tests/test_x.py", "File")

    rel = graph.add_relation(
        subject="agent:writer",
        predicate="edits",
        object_="file:tests/test_x.py",
        valid_from=100.0,
        valid_to=200.0,
        confidence=0.95,
        provenance={"step_id": 42},
    )
    assert isinstance(rel, dict)
    assert rel.get("subject") == "agent:writer"
    assert rel.get("predicate") == "edits"
    assert rel.get("object") == "file:tests/test_x.py"
    assert rel.get("valid_from") == 100.0
    assert rel.get("valid_to") == 200.0
    assert rel.get("confidence") == 0.95


def test_req201_temporal_graph_point_in_time_query() -> None:
    """REQ-BM-201: Point-in-time query retrieves facts where valid_from <= as_of < valid_to."""
    graph = TemporalEntityGraph()
    graph.add_entity("agent:verifier", "Agent")
    graph.add_entity("cert:v1", "Certificate")
    graph.add_entity("cert:v2", "Certificate")

    # cert:v1 valid in [100.0, 300.0)
    graph.add_relation("agent:verifier", "holds", "cert:v1", valid_from=100.0, valid_to=300.0)
    # cert:v2 valid in [300.0, None) (open interval)
    graph.add_relation("agent:verifier", "holds", "cert:v2", valid_from=300.0, valid_to=None)

    # At t=50: nothing active
    facts_50 = graph.query_point_in_time(as_of=50.0)
    assert len(facts_50) == 0

    # At t=200: cert:v1 is active
    facts_200 = graph.query_point_in_time(as_of=200.0)
    assert len(facts_200) == 1
    assert facts_200[0]["object"] == "cert:v1"

    # At t=350: cert:v2 is active
    facts_350 = graph.query_point_in_time(as_of=350.0)
    assert len(facts_350) == 1
    assert facts_350[0]["object"] == "cert:v2"


def test_req201_temporal_graph_point_in_time_entity_and_predicate_filters() -> None:
    """REQ-BM-201: Point-in-time query filters by entity_id and predicate."""
    graph = TemporalEntityGraph()
    graph.add_relation("agent:a", "spawns", "agent:b", valid_from=10.0, valid_to=50.0)
    graph.add_relation("agent:a", "writes", "file:doc.md", valid_from=10.0, valid_to=50.0)
    graph.add_relation("agent:c", "spawns", "agent:d", valid_from=10.0, valid_to=50.0)

    # Filter by entity_id
    res_a = graph.query_point_in_time(as_of=25.0, entity_id="agent:a")
    assert len(res_a) == 2
    assert all(r["subject"] == "agent:a" for r in res_a)

    # Filter by predicate
    res_spawns = graph.query_point_in_time(as_of=25.0, predicate="spawns")
    assert len(res_spawns) == 2
    assert all(r["predicate"] == "spawns" for r in res_spawns)


def test_req201_temporal_graph_entity_chronological_history() -> None:
    """REQ-BM-201: Querying entity history returns chronologically ordered timeline of relations."""
    graph = TemporalEntityGraph()
    graph.add_entity("agent:worker", "Agent")
    graph.add_relation("agent:worker", "step", "init", valid_from=10.0, valid_to=20.0)
    graph.add_relation("agent:worker", "step", "build", valid_from=30.0, valid_to=40.0)
    graph.add_relation("agent:worker", "step", "verify", valid_from=20.0, valid_to=30.0)

    history = graph.query_entity_history(entity_id="agent:worker")
    assert len(history) == 3
    # Check chronological ordering by valid_from
    assert history[0]["object"] == "init"
    assert history[1]["object"] == "verify"
    assert history[2]["object"] == "build"


def test_req201_temporal_graph_mermaid_export() -> None:
    """REQ-BM-201: export_graph_mermaid produces valid flowchart syntax with edges."""
    graph = TemporalEntityGraph()
    graph.add_entity("agent:orchestrator", "Agent")
    graph.add_entity("agent:worker", "Agent")
    graph.add_relation("agent:orchestrator", "dispatches", "agent:worker", valid_from=1.0)

    mermaid_code = graph.export_graph_mermaid(direction="LR")
    assert isinstance(mermaid_code, str)
    assert mermaid_code.startswith("flowchart") or mermaid_code.startswith("graph")
    assert "agent:orchestrator" in mermaid_code or "orchestrator" in mermaid_code
    assert "agent:worker" in mermaid_code or "worker" in mermaid_code
    assert "dispatches" in mermaid_code


def test_req201_temporal_graph_json_export_and_roundtrip() -> None:
    """REQ-BM-201: export_graph_json yields complete serialization with entities and relations."""
    graph = TemporalEntityGraph()
    graph.add_entity("node:1", "TypeA", {"weight": 10})
    graph.add_entity("node:2", "TypeB", {"weight": 20})
    graph.add_relation("node:1", "links", "node:2", valid_from=5.0, valid_to=15.0)

    exported = graph.export_graph_json()
    assert isinstance(exported, dict)
    assert "entities" in exported
    assert "relations" in exported
    assert len(exported["entities"]) == 2
    assert len(exported["relations"]) == 1


def test_req201_extract_temporal_graph_from_session_steps() -> None:
    """REQ-BM-201: extract_temporal_knowledge_graph parses structured step dicts into graph."""
    steps = [
        {
            "step_id": 1,
            "sender": "agent:orchestrator",
            "recipient": "agent:worker_1",
            "type": "dispatch",
            "timestamp": 100.0,
        },
        {
            "step_id": 2,
            "sender": "agent:worker_1",
            "target_file": "src/module.py",
            "type": "edit",
            "timestamp": 120.0,
        },
        {
            "step_id": 3,
            "sender": "agent:worker_1",
            "test_target": "tests/test_module.py",
            "type": "verify",
            "outcome": "PASS",
            "timestamp": 140.0,
        },
    ]
    graph = extract_temporal_knowledge_graph(steps)
    assert isinstance(graph, TemporalEntityGraph)
    data = graph.export_graph_json()
    assert len(data["entities"]) >= 3
    assert len(data["relations"]) >= 2


def test_req201_extract_temporal_graph_with_time_window_filtering() -> None:
    """REQ-BM-201: extract_temporal_knowledge_graph filters steps within (start, end) window."""
    steps = [
        {"step_id": 1, "sender": "agent:a", "type": "step", "timestamp": 10.0},
        {"step_id": 2, "sender": "agent:b", "type": "step", "timestamp": 50.0},
        {"step_id": 3, "sender": "agent:c", "type": "step", "timestamp": 100.0},
    ]
    # Filter window: [40.0, 60.0]
    graph = extract_temporal_knowledge_graph(steps, time_window=(40.0, 60.0))
    data = graph.export_graph_json()
    # Only agent:b was within window
    assert "agent:b" in data["entities"]
    assert "agent:a" not in data["entities"]
    assert "agent:c" not in data["entities"]


# ============================================================================
# 3. Knapsack Context Budget Allocator Tests (REQ-BM-202)
# ============================================================================


def test_req202_knapsack_optimal_dp_selection() -> None:
    """REQ-BM-202: 0/1 knapsack selects subset maximizing utility under budget limit."""
    items = [
        {"id": "doc1", "token_cost": 100, "utility": 60},
        {"id": "doc2", "token_cost": 200, "utility": 100},
        {"id": "doc3", "token_cost": 300, "utility": 120},
    ]
    # Budget = 500: optimal is doc2 (200, 100) + doc3 (300, 120) = cost 500, utility 220
    selected, telemetry = allocate_context_budget(items, max_token_budget=500)
    assert isinstance(selected, list)
    assert isinstance(telemetry, dict)
    sel_ids = {x["id"] for x in selected}
    assert sel_ids == {"doc2", "doc3"}
    assert telemetry["total_utility"] == 220
    assert telemetry["allocated_tokens"] == 500
    assert telemetry["budget_limit"] == 500
    assert telemetry["selected_count"] == 2
    assert telemetry["rejected_count"] == 1


def test_req202_knapsack_mandatory_items_priority_override() -> None:
    """REQ-BM-202: Mandatory items are preserved unconditionally before budget allocation."""
    items = [
        {"id": "mand1", "token_cost": 200, "utility": 10, "mandatory": True},
        {"id": "opt1", "token_cost": 250, "utility": 100, "mandatory": False},
        {"id": "opt2", "token_cost": 250, "utility": 90, "mandatory": False},
    ]
    # Budget = 500. mand1 costs 200, leaving 300 remaining budget.
    # Between opt1 (250) and opt2 (250), opt1 has higher utility (100 vs 90).
    selected, telemetry = allocate_context_budget(items, max_token_budget=500)
    sel_ids = {x["id"] for x in selected}
    assert "mand1" in sel_ids, "Mandatory item must be included"
    assert "opt1" in sel_ids, "Higher utility item must be selected for remaining budget"
    assert "opt2" not in sel_ids, "Lower utility item must be excluded when budget exhausted"
    assert telemetry["allocated_tokens"] == 450
    assert telemetry["total_utility"] == 110


def test_req202_knapsack_greedy_and_edge_cases() -> None:
    """REQ-BM-202: Greedy approximation, empty items, and budget overflow edge cases."""
    # 1. Empty items
    empty_sel, empty_tel = allocate_context_budget([], max_token_budget=1000)
    assert empty_sel == []
    assert empty_tel["allocated_tokens"] == 0
    assert empty_tel["total_utility"] == 0

    # 2. Zero budget with no mandatory items
    items = [{"id": "a", "token_cost": 10, "utility": 50}]
    zero_sel, zero_tel = allocate_context_budget(items, max_token_budget=0)
    assert zero_sel == []
    assert zero_tel["rejected_count"] == 1

    # 3. Greedy approximation by utility density
    dense_items = [
        {"id": "d1", "token_cost": 100, "utility": 300},  # density 3.0
        {"id": "d2", "token_cost": 100, "utility": 200},  # density 2.0
        {"id": "d3", "token_cost": 100, "utility": 100},  # density 1.0
    ]
    g_sel, g_tel = allocate_context_budget(dense_items, max_token_budget=200, greedy_approximation=True)
    g_ids = {x["id"] for x in g_sel}
    assert g_ids == {"d1", "d2"}
    assert g_tel["allocated_tokens"] == 200
    assert g_tel["algorithm"] in ("greedy_density", "greedy")
