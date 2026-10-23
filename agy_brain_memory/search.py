"""BM25 Full-Text Search, FTS5 Syntax Sanitizer, and Subagent Call-Graph Lineage."""

from __future__ import annotations

import contextlib
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


def search_brain(
    db_path: Path | str,
    query: str,
    limit: int = 10,
    step_type: str | None = None,
    include_artifacts: bool = True,
) -> dict[str, Any]:
    """Search indexed brain steps and Markdown artifacts using SQLite FTS5 BM25 ranking."""
    sanitized = sanitize_fts5_query(query)
    step_hits: list[dict[str, Any]] = []
    artifact_hits: list[dict[str, Any]] = []

    db_file = Path(db_path).resolve()
    if not db_file.is_file() or sanitized == '""':
        return {
            "query": query,
            "sanitized_query": sanitized,
            "results": [],
            "artifact_results": [],
        }

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

    return {
        "query": query,
        "sanitized_query": sanitized,
        "results": step_hits,
        "artifact_results": artifact_hits,
    }


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
