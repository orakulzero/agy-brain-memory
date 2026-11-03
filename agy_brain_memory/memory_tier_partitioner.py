"""Hot/Cold Memory Tier Partitioner and Two-Stage Short-Circuit Retriever (REQ-BM-TIER-01)."""

from __future__ import annotations

import contextlib
from pathlib import Path
import sqlite3
import time
from typing import Any, Sequence

from agy_brain_memory.search import sanitize_fts5_query

DEFAULT_HOT_MAX_AGE_DAYS: float = 14.0
DEFAULT_PINNED_ARTIFACTS: tuple[str, ...] = (
    "PROGRESS.md",
    "TASK.md",
    "DECISIONS.md",
    "walkthrough.md",
    "implementation_plan.md",
)
HOT_SCORE_BOOST: float = 1.25
COLD_SCORE_DECAY: float = 0.75


def _open_db(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(db_path).resolve()))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tier_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_tiers (
            conv_id TEXT PRIMARY KEY,
            tier TEXT NOT NULL,
            reason TEXT NOT NULL,
            age_days REAL NOT NULL,
            has_pinned_artifact INTEGER NOT NULL,
            updated_at REAL NOT NULL
        );
        """
    )


def partition_brain_memory_tiers(
    db_path: Path | str,
    hot_max_age_days: float = DEFAULT_HOT_MAX_AGE_DAYS,
    pinned_artifacts: Sequence[str] = DEFAULT_PINNED_ARTIFACTS,
    reference_ts: float | None = None,
) -> dict[str, Any]:
    """REQ-BM-TIER-01: Classify indexed conversations into Hot and Cold tiers without altering the 7-column conversations schema."""
    db_file = Path(db_path).resolve()
    ref_ts = float(reference_ts) if reference_ts is not None else time.time()
    pinned_lower = {name.lower() for name in pinned_artifacts}

    if not db_file.is_file():
        return {
            "req_id": "REQ-BM-TIER-01",
            "total_conversations": 0,
            "hot_count": 0,
            "cold_count": 0,
            "hot_conv_ids": [],
            "cold_conv_ids": [],
            "classifications": {},
        }

    hot_conv_ids: list[str] = []
    cold_conv_ids: list[str] = []
    classifications: dict[str, dict[str, Any]] = {}
    tier_rows: list[tuple[str, str, str, float, int, float]] = []

    with contextlib.closing(_open_db(db_file)) as conn, conn:
        _ensure_tier_table(conn)
        conv_rows = conn.execute(
            "SELECT conv_id, mtime_ns FROM conversations ORDER BY conv_id ASC"
        ).fetchall()
        art_rows = conn.execute(
            "SELECT conv_id, filename FROM artifacts"
        ).fetchall()

        artifacts_by_conv: dict[str, set[str]] = {}
        for arow in art_rows:
            cid = str(arow["conv_id"])
            artifacts_by_conv.setdefault(cid, set()).add(str(arow["filename"]).lower())

        for crow in conv_rows:
            cid = str(crow["conv_id"])
            raw_mtime = int(crow["mtime_ns"])
            mtime_sec = raw_mtime / 1_000_000_000.0 if raw_mtime > 10_000_000_000 else float(raw_mtime)
            age_days = max(0.0, (ref_ts - mtime_sec) / 86400.0)
            has_pinned = bool(artifacts_by_conv.get(cid, set()) & pinned_lower)

            if age_days <= float(hot_max_age_days):
                tier = "hot"
                reason = "recent_activity"
                hot_conv_ids.append(cid)
            elif has_pinned:
                tier = "hot"
                reason = "pinned_artifact"
                hot_conv_ids.append(cid)
            else:
                tier = "cold"
                reason = "aged_out_archive"
                cold_conv_ids.append(cid)

            classifications[cid] = {
                "conv_id": cid,
                "tier": tier,
                "reason": reason,
                "age_days": round(age_days, 4),
                "has_pinned_artifact": has_pinned,
            }
            tier_rows.append((cid, tier, reason, age_days, int(has_pinned), ref_ts))

        conn.execute("DELETE FROM conversation_tiers")
        if tier_rows:
            conn.executemany(
                """
                INSERT INTO conversation_tiers (
                    conv_id, tier, reason, age_days, has_pinned_artifact, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                tier_rows,
            )

    return {
        "req_id": "REQ-BM-TIER-01",
        "total_conversations": len(conv_rows),
        "hot_count": len(hot_conv_ids),
        "cold_count": len(cold_conv_ids),
        "hot_conv_ids": hot_conv_ids,
        "cold_conv_ids": cold_conv_ids,
        "classifications": classifications,
    }


def _query_tier_fts(
    conn: sqlite3.Connection,
    sanitized_query: str,
    tier_conv_ids: Sequence[str],
    tier_label: str,
    score_multiplier: float,
    limit: int,
) -> list[dict[str, Any]]:
    if not tier_conv_ids or limit <= 0 or sanitized_query == '""':
        return []
    placeholders = ",".join("?" for _ in tier_conv_ids)
    params: list[Any] = [sanitized_query, *tier_conv_ids, int(limit)]
    rows = conn.execute(
        f"""
        SELECT conv_id, step_index, type, status,
               snippet(steps_fts, 4, '[', ']', '...', 24) AS snip,
               bm25(steps_fts) AS rank_score
        FROM steps_fts
        WHERE steps_fts MATCH ? AND conv_id IN ({placeholders})
        ORDER BY rank_score ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    hits: list[dict[str, Any]] = []
    for r in rows:
        cid = str(r["conv_id"])
        raw_score = abs(float(r["rank_score"]))
        hits.append(
            {
                "conv_id": cid,
                "conversation_link": f"conversation://{cid}",
                "step_index": int(r["step_index"]),
                "type": str(r["type"]),
                "status": str(r["status"]),
                "snippet": str(r["snip"]),
                "tier": tier_label,
                "score": round(raw_score * score_multiplier, 6),
            }
        )
    return hits


def search_tiered_brain(
    db_path: Path | str,
    query: str,
    limit: int = 10,
    hot_max_age_days: float = DEFAULT_HOT_MAX_AGE_DAYS,
    pinned_artifacts: Sequence[str] = DEFAULT_PINNED_ARTIFACTS,
    short_circuit_on_hot: bool = True,
    include_cold_fallback: bool = True,
    reference_ts: float | None = None,
) -> dict[str, Any]:
    """REQ-BM-TIER-01: Execute two-stage Hot-first FTS5 search with optional Cold archive fallback."""
    part = partition_brain_memory_tiers(
        db_path=db_path,
        hot_max_age_days=hot_max_age_days,
        pinned_artifacts=pinned_artifacts,
        reference_ts=reference_ts,
    )
    sanitized = sanitize_fts5_query(query)
    db_file = Path(db_path).resolve()
    if not db_file.is_file() or sanitized == '""':
        return {
            "query": query,
            "sanitized_query": sanitized,
            "hot_hits_count": 0,
            "cold_hits_count": 0,
            "cold_tier_queried": False,
            "results": [],
        }

    with contextlib.closing(_open_db(db_file)) as conn:
        hot_hits = _query_tier_fts(
            conn, sanitized, part["hot_conv_ids"], "hot", HOT_SCORE_BOOST, int(limit)
        )
        cold_hits: list[dict[str, Any]] = []
        cold_tier_queried = False
        if include_cold_fallback and not (short_circuit_on_hot and len(hot_hits) > 0):
            cold_tier_queried = True
            remaining = max(0, int(limit) - len(hot_hits))
            cold_hits = _query_tier_fts(
                conn, sanitized, part["cold_conv_ids"], "cold", COLD_SCORE_DECAY, remaining
            )

    results = (hot_hits + cold_hits)[: int(limit)]
    return {
        "query": query,
        "sanitized_query": sanitized,
        "hot_hits_count": len(hot_hits),
        "cold_hits_count": len(cold_hits),
        "cold_tier_queried": cold_tier_queried,
        "results": results,
    }


class MemoryTierPartitioner:
    """REQ-BM-TIER-01: Stateful manager for Hot/Cold brain memory tier partitioning and retrieval."""

    def __init__(
        self,
        db_path: Path | str,
        hot_max_age_days: float = DEFAULT_HOT_MAX_AGE_DAYS,
        pinned_artifacts: Sequence[str] = DEFAULT_PINNED_ARTIFACTS,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.hot_max_age_days = float(hot_max_age_days)
        self.pinned_artifacts = tuple(pinned_artifacts)

    def partition(self, reference_ts: float | None = None) -> dict[str, Any]:
        return partition_brain_memory_tiers(
            db_path=self.db_path,
            hot_max_age_days=self.hot_max_age_days,
            pinned_artifacts=self.pinned_artifacts,
            reference_ts=reference_ts,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        short_circuit_on_hot: bool = True,
        include_cold_fallback: bool = True,
        reference_ts: float | None = None,
    ) -> dict[str, Any]:
        return search_tiered_brain(
            db_path=self.db_path,
            query=query,
            limit=limit,
            hot_max_age_days=self.hot_max_age_days,
            pinned_artifacts=self.pinned_artifacts,
            short_circuit_on_hot=short_circuit_on_hot,
            include_cold_fallback=include_cold_fallback,
            reference_ts=reference_ts,
        )
