"""BM25 Full-Text Search, FTS5 Syntax Sanitizer, and Subagent Call-Graph Lineage."""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def sanitize_fts5_query(raw_query: str) -> str:
    """Sanitize arbitrary user query strings into safe FTS5 quoted token expressions."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", raw_query or "")
    if not tokens:
        return '""'
    return " ".join(f'"{tok}"' for tok in tokens)


def _open_db(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(db_path).resolve()))
    conn.row_factory = sqlite3.Row
    return conn


def search_diff_tours(
    db_path: Path | str,
    query: str,
    limit: int = 10,
    dimension: str | None = None,
) -> dict[str, Any]:
    """Search indexed diff-tour v1 records using SQLite FTS5 BM25 ranking."""
    sanitized = sanitize_fts5_query(query)
    tour_hits: list[dict[str, Any]] = []

    db_file = Path(db_path).resolve()
    if not db_file.is_file() or sanitized == '""':
        return {
            "query": query,
            "sanitized_query": sanitized,
            "results": [],
        }

    with contextlib.closing(_open_db(db_file)) as conn:
        rows = conn.execute(
            """
            SELECT dt.conv_id, dt.tour_id, dt.filename, dt.title, dt.description,
                   dt.step_count, dt.dimensions_json, dt.content_json,
                   snippet(diff_tours_fts, 5, '[', ']', '...', 24) AS snip,
                   bm25(diff_tours_fts) AS rank_score
            FROM diff_tours_fts
            JOIN diff_tours dt ON dt.conv_id = diff_tours_fts.conv_id AND dt.tour_id = diff_tours_fts.tour_id
            WHERE diff_tours_fts MATCH ?
            ORDER BY rank_score ASC
            LIMIT ?
            """,
            (sanitized, int(limit) * 3 if dimension else int(limit)),
        ).fetchall()

        for r in rows:
            dims = json.loads(str(r["dimensions_json"] or "[]"))
            if dimension and dimension not in dims:
                continue
            cid = str(r["conv_id"])
            tour_hits.append(
                {
                    "conv_id": cid,
                    "conversation_link": f"conversation://{cid}",
                    "tour_id": str(r["tour_id"]),
                    "filename": str(r["filename"]),
                    "title": str(r["title"]),
                    "description": str(r["description"]),
                    "step_count": int(r["step_count"]),
                    "dimensions": dims,
                    "snippet": str(r["snip"]),
                    "score": float(r["rank_score"]),
                }
            )
            if len(tour_hits) >= int(limit):
                break

    return {
        "query": query,
        "sanitized_query": sanitized,
        "results": tour_hits,
    }


def search_capa_records(
    db_path: Path | str,
    query: str,
    limit: int = 10,
    record_type: str | None = None,
) -> dict[str, Any]:
    """Search indexed ISO 19011 / CAPA records using SQLite FTS5 BM25 ranking."""
    sanitized = sanitize_fts5_query(query)
    capa_hits: list[dict[str, Any]] = []

    db_file = Path(db_path).resolve()
    if not db_file.is_file() or sanitized == '""':
        return {
            "query": query,
            "sanitized_query": sanitized,
            "results": [],
        }

    with contextlib.closing(_open_db(db_file)) as conn:
        if record_type:
            rows = conn.execute(
                """
                SELECT cr.conv_id, cr.record_type, cr.record_id, cr.title, cr.details_json, cr.created_at,
                       snippet(capa_records_fts, 4, '[', ']', '...', 24) AS snip,
                       bm25(capa_records_fts) AS rank_score
                FROM capa_records_fts
                JOIN capa_records cr ON cr.conv_id = capa_records_fts.conv_id AND cr.record_id = capa_records_fts.record_id
                WHERE capa_records_fts MATCH ? AND cr.record_type = ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (sanitized, record_type, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT cr.conv_id, cr.record_type, cr.record_id, cr.title, cr.details_json, cr.created_at,
                       snippet(capa_records_fts, 4, '[', ']', '...', 24) AS snip,
                       bm25(capa_records_fts) AS rank_score
                FROM capa_records_fts
                JOIN capa_records cr ON cr.conv_id = capa_records_fts.conv_id AND cr.record_id = capa_records_fts.record_id
                WHERE capa_records_fts MATCH ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (sanitized, int(limit)),
            ).fetchall()

        for r in rows:
            cid = str(r["conv_id"])
            capa_hits.append(
                {
                    "conv_id": cid,
                    "conversation_link": f"conversation://{cid}",
                    "record_type": str(r["record_type"]),
                    "record_id": str(r["record_id"]),
                    "title": str(r["title"]),
                    "snippet": str(r["snip"]),
                    "created_at": str(r["created_at"]),
                    "score": float(r["rank_score"]),
                }
            )

    return {
        "query": query,
        "sanitized_query": sanitized,
        "results": capa_hits,
    }


def search_brain(
    db_path: Path | str,
    query: str,
    limit: int = 10,
    step_type: str | None = None,
    include_artifacts: bool = True,
    include_tours: bool = False,
    include_capa: bool = False,
) -> dict[str, Any]:
    """Search indexed brain steps, Markdown artifacts, diff tours, and CAPA records."""
    sanitized = sanitize_fts5_query(query)
    step_hits: list[dict[str, Any]] = []
    artifact_hits: list[dict[str, Any]] = []

    db_file = Path(db_path).resolve()
    if not db_file.is_file() or sanitized == '""':
        out: dict[str, Any] = {
            "query": query,
            "sanitized_query": sanitized,
            "results": [],
            "artifact_results": [],
        }
        if include_tours:
            out["diff_tour_results"] = []
        if include_capa:
            out["capa_results"] = []
        return out

    with contextlib.closing(_open_db(db_file)) as conn:
        if step_type:
            rows = conn.execute(
                """
                SELECT conv_id, step_index, type, status,
                       snippet(steps_fts, 4, '[', ']', '...', 24) AS snip,
                       bm25(steps_fts) AS rank_score
                FROM steps_fts
                WHERE steps_fts MATCH ? AND type = ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (sanitized, step_type, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT conv_id, step_index, type, status,
                       snippet(steps_fts, 4, '[', ']', '...', 24) AS snip,
                       bm25(steps_fts) AS rank_score
                FROM steps_fts
                WHERE steps_fts MATCH ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (sanitized, int(limit)),
            ).fetchall()

        for r in rows:
            cid = str(r["conv_id"])
            step_hits.append(
                {
                    "conv_id": cid,
                    "conversation_link": f"conversation://{cid}",
                    "step_index": int(r["step_index"]),
                    "type": str(r["type"]),
                    "status": str(r["status"]),
                    "snippet": str(r["snip"]),
                    "score": float(r["rank_score"]),
                }
            )

        if include_artifacts:
            art_rows = conn.execute(
                """
                SELECT conv_id, filename,
                       snippet(artifacts_fts, 2, '[', ']', '...', 24) AS snip,
                       bm25(artifacts_fts) AS rank_score
                FROM artifacts_fts
                WHERE artifacts_fts MATCH ?
                ORDER BY rank_score ASC
                LIMIT ?
                """,
                (sanitized, int(limit)),
            ).fetchall()
            for ar in art_rows:
                cid = str(ar["conv_id"])
                artifact_hits.append(
                    {
                        "conv_id": cid,
                        "conversation_link": f"conversation://{cid}",
                        "filename": str(ar["filename"]),
                        "snippet": str(ar["snip"]),
                        "score": float(ar["rank_score"]),
                    }
                )

    out = {
        "query": query,
        "sanitized_query": sanitized,
        "results": step_hits,
        "artifact_results": artifact_hits,
    }

    if include_tours:
        t_res = search_diff_tours(db_path, query, limit=limit)
        out["diff_tour_results"] = t_res.get("results", [])

    if include_capa:
        c_res = search_capa_records(db_path, query, limit=limit)
        out["capa_results"] = c_res.get("results", [])

    return out


def _row_to_edge(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "parent_conv_id": str(row["parent_conv_id"]),
        "child_conv_id": str(row["child_conv_id"]),
        "step_index": int(row["step_index"]),
        "role": str(row["role"]),
        "type_name": str(row["type_name"]),
        "prompt": str(row["prompt"]),
        "child_status": str(row["child_status"]),
        "child_outcome_summary": str(row["child_outcome_summary"]),
    }


def get_subagent_lineage(
    db_path: Path | str,
    conv_id: str | None = None,
    role_query: str | None = None,
) -> dict[str, Any]:
    """Reconstruct parent-to-child subagent lineage, transitive descendants, and role-filtered edges."""
    db_file = Path(db_path).resolve()
    if not db_file.is_file():
        return {
            "conv_id": conv_id,
            "role_query": role_query,
            "spawned_children": [],
            "spawned_by_parent": None,
            "descendants": [],
            "edges": [],
        }

    with contextlib.closing(_open_db(db_file)) as conn:
        all_rows = conn.execute(
            "SELECT * FROM subagent_edges ORDER BY id ASC"
        ).fetchall()
        all_edges = [_row_to_edge(r) for r in all_rows]

    spawned_children: list[dict[str, Any]] = []
    spawned_by_parent: dict[str, Any] | None = None
    descendants: list[dict[str, Any]] = []

    if conv_id:
        spawned_children = [
            e for e in all_edges if e["parent_conv_id"] == conv_id
        ]
        parent_matches = [
            e for e in all_edges if e["child_conv_id"] == conv_id
        ]
        spawned_by_parent = parent_matches[0] if parent_matches else None

        # BFS traversal for multi-hop descendants
        visited: set[str] = {conv_id}
        queue: list[str] = [conv_id]
        while queue:
            curr = queue.pop(0)
            for edge in all_edges:
                if edge["parent_conv_id"] == curr:
                    descendants.append(edge)
                    child = edge["child_conv_id"]
                    if child not in visited:
                        visited.add(child)
                        queue.append(child)

    filtered_edges = all_edges
    if conv_id:
        filtered_edges = [
            e
            for e in filtered_edges
            if e["parent_conv_id"] == conv_id or e["child_conv_id"] == conv_id
        ]
    if role_query:
        rq_low = role_query.lower().strip()
        filtered_edges = [
            e
            for e in filtered_edges
            if rq_low in e["role"].lower() or rq_low in e["prompt"].lower()
        ]

    return {
        "conv_id": conv_id,
        "role_query": role_query,
        "spawned_children": spawned_children,
        "spawned_by_parent": spawned_by_parent,
        "descendants": descendants,
        "edges": filtered_edges,
    }


def get_conversation_summary(db_path: Path | str, conv_id: str) -> dict[str, Any]:
    """Reconstruct a structured timeline summary for a conversation ID."""
    db_file = Path(db_path).resolve()
    if not db_file.is_file():
        return {
            "conv_id": conv_id,
            "conversation_link": f"conversation://{conv_id}",
            "step_count": 0,
            "user_requests": [],
            "artifacts": [],
            "spawned_subagents": [],
            "last_status": "UNKNOWN",
            "has_bigint_error": False,
        }

    with contextlib.closing(_open_db(db_file)) as conn:
        conv_row = conn.execute(
            "SELECT * FROM conversations WHERE conv_id = ?", (conv_id,)
        ).fetchone()
        user_rows = conn.execute(
            """
            SELECT content FROM steps
            WHERE conv_id = ? AND type = 'USER_INPUT'
            ORDER BY step_index ASC
            """,
            (conv_id,),
        ).fetchall()
        art_rows = conn.execute(
            "SELECT filename FROM artifacts WHERE conv_id = ? ORDER BY filename ASC",
            (conv_id,),
        ).fetchall()
        edge_rows = conn.execute(
            "SELECT * FROM subagent_edges WHERE parent_conv_id = ? ORDER BY step_index ASC",
            (conv_id,),
        ).fetchall()

    return {
        "conv_id": conv_id,
        "conversation_link": f"conversation://{conv_id}",
        "step_count": int(conv_row["step_count"]) if conv_row else 0,
        "user_requests": [str(r["content"]) for r in user_rows],
        "artifacts": [str(a["filename"]) for a in art_rows],
        "spawned_subagents": [_row_to_edge(e) for e in edge_rows],
        "last_status": str(conv_row["last_status"]) if conv_row else "UNKNOWN",
        "has_bigint_error": bool(conv_row["has_bigint_error"]) if conv_row else False,
    }


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _safe_mermaid_id(conv_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", conv_id)
    return f"n_{sanitized}"


def analyze_subagent_graph_topology(
    db_path: Path | str,
    root_conv_id: str | None = None,
) -> dict[str, Any]:
    """Analyze directed subagent graph topology, depth, fanout, orphan nodes, cycles, and roles."""
    db_file = Path(db_path).resolve()
    if not db_file.is_file():
        return {
            "total_nodes": 0,
            "max_depth": 0,
            "max_fanout": 0,
            "orphan_nodes": [],
            "has_cycles": False,
            "roles_distribution": {},
            "root_conv_id": root_conv_id,
        }

    with contextlib.closing(_open_db(db_file)) as conn:
        all_conv_ids: set[str] = set()
        if _table_exists(conn, "conversations"):
            c_rows = conn.execute("SELECT conv_id FROM conversations").fetchall()
            all_conv_ids = {str(r["conv_id"]) for r in c_rows}

        raw_edges: list[tuple[str, str, str, int]] = []
        if _table_exists(conn, "subagent_edges"):
            e_rows = conn.execute(
                "SELECT parent_conv_id, child_conv_id, role, step_index FROM subagent_edges ORDER BY id ASC"
            ).fetchall()
            raw_edges = [
                (
                    str(r["parent_conv_id"]),
                    str(r["child_conv_id"]),
                    str(r["role"]),
                    int(r["step_index"]),
                )
                for r in e_rows
            ]

    # Full graph mappings
    full_outgoing: dict[str, list[tuple[str, str, int]]] = {}
    for parent, child, role, s_idx in raw_edges:
        full_outgoing.setdefault(parent, []).append((child, role, s_idx))

    if root_conv_id is not None:
        reachable: set[str] = set()
        if root_conv_id in all_conv_ids or root_conv_id in full_outgoing or any(e[1] == root_conv_id for e in raw_edges):
            reachable.add(root_conv_id)
            queue = [root_conv_id]
            while queue:
                curr = queue.pop(0)
                for ch, _, _ in full_outgoing.get(curr, []):
                    if ch not in reachable:
                        reachable.add(ch)
                        queue.append(ch)
        nodes = reachable
        active_edges = [e for e in raw_edges if e[0] in reachable and e[1] in reachable]
    else:
        nodes = all_conv_ids | {e[0] for e in raw_edges} | {e[1] for e in raw_edges}
        active_edges = raw_edges

    adj: dict[str, list[str]] = {n: [] for n in nodes}
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    roles_distribution: dict[str, int] = {}

    for parent, child, role, _ in active_edges:
        if parent in adj and child in adj:
            adj[parent].append(child)
            in_degree[child] = in_degree.get(child, 0) + 1
            r_name = role.strip() or "unknown"
            roles_distribution[r_name] = roles_distribution.get(r_name, 0) + 1

    total_nodes = len(nodes)
    max_fanout = max((len(adj[n]) for n in nodes), default=0)
    orphan_nodes = sorted([n for n in nodes if in_degree.get(n, 0) == 0])

    # Cycle detection via 3-color DFS (0=unvisited, 1=visiting, 2=visited)
    color: dict[str, int] = {n: 0 for n in nodes}
    has_cycles = False

    def dfs_cycle(u: str) -> None:
        nonlocal has_cycles
        color[u] = 1
        for v in adj.get(u, []):
            if color.get(v, 0) == 1:
                has_cycles = True
            elif color.get(v, 0) == 0:
                dfs_cycle(v)
        color[u] = 2

    for n in sorted(nodes):
        if color.get(n, 0) == 0:
            dfs_cycle(n)

    # Max depth computation (longest path in DAG or longest simple path if cyclic)
    if not has_cycles:
        memo_depth: dict[str, int] = {}

        def get_dag_depth(u: str) -> int:
            if u in memo_depth:
                return memo_depth[u]
            children = adj.get(u, [])
            if not children:
                memo_depth[u] = 0
                return 0
            d = 1 + max(get_dag_depth(v) for v in children)
            memo_depth[u] = d
            return d

        max_depth = max((get_dag_depth(n) for n in nodes), default=0) if active_edges else 0
    else:
        def longest_simple_path(u: str, path: set[str]) -> int:
            best = 0
            for v in adj.get(u, []):
                if v not in path:
                    best = max(best, 1 + longest_simple_path(v, path | {v}))
            return best

        max_depth = max((longest_simple_path(n, {n}) for n in nodes), default=0)

    return {
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "max_fanout": max_fanout,
        "orphan_nodes": orphan_nodes,
        "has_cycles": has_cycles,
        "roles_distribution": roles_distribution,
        "root_conv_id": root_conv_id,
    }


def export_lineage_mermaid(
    db_path: Path | str,
    root_conv_id: str | None = None,
    direction: str = "TD",
) -> str:
    """Export directed subagent call-graph lineage to valid Mermaid flowchart diagram."""
    db_file = Path(db_path).resolve()
    dir_clean = direction.strip().upper() if direction else "TD"
    if not db_file.is_file():
        return f"flowchart {dir_clean}\n"

    with contextlib.closing(_open_db(db_file)) as conn:
        all_conv_ids: set[str] = set()
        if _table_exists(conn, "conversations"):
            c_rows = conn.execute("SELECT conv_id FROM conversations").fetchall()
            all_conv_ids = {str(r["conv_id"]) for r in c_rows}

        raw_edges: list[tuple[str, str, str, int]] = []
        if _table_exists(conn, "subagent_edges"):
            e_rows = conn.execute(
                "SELECT parent_conv_id, child_conv_id, role, step_index FROM subagent_edges ORDER BY id ASC"
            ).fetchall()
            raw_edges = [
                (
                    str(r["parent_conv_id"]),
                    str(r["child_conv_id"]),
                    str(r["role"]),
                    int(r["step_index"]),
                )
                for r in e_rows
            ]

    # Map each child to its invoked role
    node_roles: dict[str, str] = {}
    for _, child, role, _ in raw_edges:
        if child not in node_roles and role.strip():
            node_roles[child] = role.strip()

    full_outgoing: dict[str, list[tuple[str, str, int]]] = {}
    for parent, child, role, s_idx in raw_edges:
        full_outgoing.setdefault(parent, []).append((child, role, s_idx))

    if root_conv_id is not None:
        reachable: set[str] = set()
        if root_conv_id in all_conv_ids or root_conv_id in full_outgoing or any(e[1] == root_conv_id for e in raw_edges):
            reachable.add(root_conv_id)
            queue = [root_conv_id]
            while queue:
                curr = queue.pop(0)
                for ch, _, _ in full_outgoing.get(curr, []):
                    if ch not in reachable:
                        reachable.add(ch)
                        queue.append(ch)
        nodes = reachable
        active_edges = [e for e in raw_edges if e[0] in reachable and e[1] in reachable]
    else:
        nodes = all_conv_ids | {e[0] for e in raw_edges} | {e[1] for e in raw_edges}
        active_edges = raw_edges

    lines: list[str] = [f"flowchart {dir_clean}"]
    if not nodes and not active_edges:
        return "\n".join(lines) + "\n"

    # Identify any isolated nodes with no incoming and no outgoing edges
    connected_nodes = {e[0] for e in active_edges} | {e[1] for e in active_edges}
    isolated_nodes = sorted(nodes - connected_nodes)
    for iso in isolated_nodes:
        safe_nid = _safe_mermaid_id(iso)
        iso_role = node_roles.get(iso, "root").replace('"', "'")
        lines.append(f'    {safe_nid}["[{iso_role}](conversation://{iso})"]')

    for parent, child, role, step_idx in active_edges:
        p_nid = _safe_mermaid_id(parent)
        c_nid = _safe_mermaid_id(child)
        p_role = node_roles.get(parent, "root").replace('"', "'")
        c_role = node_roles.get(child, role).replace('"', "'")
        clean_role = role.replace('"', "'").strip()
        edge_label = f"step {step_idx}: {clean_role}" if step_idx is not None else clean_role
        lines.append(
            f'    {p_nid}["[{p_role}](conversation://{parent})"] -->|{edge_label}| {c_nid}["[{c_role}](conversation://{child})"]'
        )

    return "\n".join(lines) + "\n"

