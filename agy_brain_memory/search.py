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
