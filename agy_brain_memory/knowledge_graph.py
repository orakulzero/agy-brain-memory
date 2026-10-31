"""Temporal Knowledge Graph & Point-in-Time Fact Query Engine (REQ-BM-201).

Provides directed temporal multigraph tracking entity states, valid-time intervals,
point-in-time state queries, chronological histories, and Mermaid export.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from typing import Any


class TemporalEntityGraph:
    """Directed temporal multigraph storing entities and time-bounded assertions."""

    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.relations: list[dict[str, Any]] = []

    def add_entity(
        self,
        entity_id: str,
        entity_type: str = "Entity",
        properties: dict[str, Any] | None = None,
        timestamp: float = 0.0,
    ) -> dict[str, Any]:
        """Record or update an entity node in the knowledge graph."""
        node = {
            "entity_id": str(entity_id),
            "entity_type": str(entity_type),
            "properties": dict(properties or {}),
            "timestamp": float(timestamp),
        }
        self.entities[str(entity_id)] = node
        return node

    def add_relation(
        self,
        subject: str,
        predicate: str,
        object_: str,
        valid_from: float = 0.0,
        valid_to: float | None = None,
        confidence: float = 1.0,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a time-scoped relationship assertion between two entities."""
        sub_str = str(subject)
        obj_str = str(object_)
        if sub_str not in self.entities:
            self.add_entity(sub_str, entity_type="Entity", timestamp=valid_from)
        if obj_str not in self.entities:
            self.add_entity(obj_str, entity_type="Entity", timestamp=valid_from)

        v_to = float(valid_to) if valid_to is not None else None
        rel = {
            "subject": sub_str,
            "predicate": str(predicate),
            "object": obj_str,
            "valid_from": float(valid_from),
            "valid_to": v_to,
            "confidence": float(confidence),
            "provenance": dict(provenance or {}),
        }
        self.relations.append(rel)
        return rel

    def query_point_in_time(
        self,
        as_of: float,
        entity_id: str | None = None,
        predicate: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve all facts whose valid interval covers `as_of` (valid_from <= as_of < valid_to)."""
        t = float(as_of)
        results: list[dict[str, Any]] = []
        for r in self.relations:
            if r["valid_from"] <= t and (r["valid_to"] is None or t < r["valid_to"]):
                if entity_id is not None and r["subject"] != entity_id and r["object"] != entity_id:
                    continue
                if predicate is not None and r["predicate"] != predicate:
                    continue
                results.append(copy.deepcopy(r))
        return results

    def query_entity_history(self, entity_id: str) -> list[dict[str, Any]]:
        """Retrieve chronological relationship history for an entity sorted by valid_from."""
        eid = str(entity_id)
        matches = [
            copy.deepcopy(r)
            for r in self.relations
            if r["subject"] == eid or r["object"] == eid
        ]
        matches.sort(key=lambda r: (r["valid_from"], str(r.get("valid_to") or "")))
        return matches

    def export_graph_json(self) -> dict[str, Any]:
        """Export serialized representation of entities and relations."""
        return {
            "entities": copy.deepcopy(self.entities),
            "relations": copy.deepcopy(self.relations),
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
        }

    def export_graph_mermaid(self, direction: str = "LR") -> str:
        """Export graph as Mermaid flowchart syntax."""
        lines = [f"flowchart {direction}"]

        def _clean_id(raw: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", raw)

        for eid in sorted(self.entities):
            cid = _clean_id(eid)
            lines.append(f'    {cid}["{eid}"]')

        for r in self.relations:
            s_id = _clean_id(r["subject"])
            o_id = _clean_id(r["object"])
            pred = r["predicate"]
            lines.append(f'    {s_id} -->|"{pred}"| {o_id}')

        return "\n".join(lines)


def extract_temporal_knowledge_graph(
    steps_or_db: Any,
    time_window: tuple[float, float] | None = None,
) -> TemporalEntityGraph:
    """Extract a TemporalEntityGraph from session step records or database entries."""
    graph = TemporalEntityGraph()
    if not steps_or_db:
        return graph

    steps: Sequence[dict[str, Any]] = steps_or_db if isinstance(steps_or_db, (list, tuple)) else []
    for step in steps:
        if not isinstance(step, dict):
            continue
        ts = float(step.get("timestamp", 0.0))
        if time_window is not None:
            w_start, w_end = time_window
            if ts < w_start or ts > w_end:
                continue

        sender = step.get("sender")
        recipient = step.get("recipient")
        target_file = step.get("target_file")
        test_target = step.get("test_target")
        step_type = str(step.get("type", "step"))
        step_id = step.get("step_id")
        prov = {"step_id": step_id} if step_id is not None else {}

        if sender:
            graph.add_entity(str(sender), entity_type="Agent", timestamp=ts)

        if recipient:
            graph.add_entity(str(recipient), entity_type="Agent", timestamp=ts)
            if sender:
                graph.add_relation(str(sender), step_type, str(recipient), valid_from=ts, provenance=prov)

        if target_file:
            graph.add_entity(str(target_file), entity_type="File", timestamp=ts)
            if sender:
                graph.add_relation(str(sender), step_type, str(target_file), valid_from=ts, provenance=prov)

        if test_target:
            graph.add_entity(str(test_target), entity_type="Test", timestamp=ts)
            prov_t = dict(prov)
            if "outcome" in step:
                prov_t["outcome"] = step["outcome"]
            if sender:
                graph.add_relation(str(sender), step_type, str(test_target), valid_from=ts, provenance=prov_t)

        if sender and not (recipient or target_file or test_target):
            graph.add_relation(str(sender), step_type, f"step_{step_id or ts}", valid_from=ts, provenance=prov)

    return graph


__all__ = ["TemporalEntityGraph", "extract_temporal_knowledge_graph"]
