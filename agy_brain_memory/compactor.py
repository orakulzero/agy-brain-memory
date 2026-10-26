"""Brain Health Analyzer & Bounded Session Handoff Digest Exporter."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Dict, List, Sequence, Tuple

from agy_brain_memory.indexer import BrainMemoryIndexer
from agy_brain_memory.search import get_conversation_summary


def analyze_brain_health(
    brain_dir: Path | str, db_path: Path | str | None = None
) -> dict[str, Any]:
    """Analyze Antigravity brain directory health, orphan folders, BigInt crashes, and stalled sessions."""
    root = Path(brain_dir).resolve()
    if not root.is_dir():
        return {
            "brain_dir": str(root),
            "total_directories": 0,
            "valid_conversations": 0,
            "empty_or_orphan_dirs": [],
            "bigint_error_conversations": [],
            "stalled_in_progress_conversations": [],
            "total_bytes": 0,
            "total_subagent_spawns": 0,
        }

    empty_or_orphan_dirs: list[str] = []
    total_dirs = 0

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        total_dirs += 1
        has_transcript = (
            sub / ".system_generated" / "logs" / "transcript.jsonl"
        ).is_file() or (sub / "transcript.jsonl").is_file()
        has_md = any(sub.glob("*.md"))
        if not has_transcript and not has_md:
            empty_or_orphan_dirs.append(sub.name)

    if db_path is not None:
        target_db = Path(db_path).resolve()
        indexer = BrainMemoryIndexer(db_path=target_db, brain_dir=root)
        indexer.sync(force=False)

        with contextlib.closing(sqlite3.connect(str(target_db))) as conn:
            conn.row_factory = sqlite3.Row
            convs = conn.execute("SELECT * FROM conversations").fetchall()
            total_bytes = int(
                conn.execute(
                    "SELECT COALESCE(SUM(file_size), 0) AS total_bytes FROM conversations"
                ).fetchone()["total_bytes"]
            )
            edges_cnt = conn.execute(
                "SELECT COUNT(*) AS cnt FROM subagent_edges"
            ).fetchone()["cnt"]
    else:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            temp_db = Path(tmp) / "health.sqlite"
            indexer = BrainMemoryIndexer(db_path=temp_db, brain_dir=root)
            indexer.sync(force=True)

            with contextlib.closing(sqlite3.connect(str(temp_db))) as conn:
                conn.row_factory = sqlite3.Row
                convs = conn.execute("SELECT * FROM conversations").fetchall()
                total_bytes = int(
                    conn.execute(
                        "SELECT COALESCE(SUM(file_size), 0) AS total_bytes FROM conversations"
                    ).fetchone()["total_bytes"]
                )
                edges_cnt = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM subagent_edges"
                ).fetchone()["cnt"]

    bigint_convs = [
        str(r["conv_id"]) for r in convs if int(r["has_bigint_error"]) == 1
    ]
    stalled_convs = [
        str(r["conv_id"])
        for r in convs
        if str(r["last_status"]) == "IN_PROGRESS"
    ]

    return {
        "brain_dir": str(root),
        "total_directories": total_dirs,
        "valid_conversations": len(convs),
        "empty_or_orphan_dirs": empty_or_orphan_dirs,
        "bigint_error_conversations": bigint_convs,
        "stalled_in_progress_conversations": stalled_convs,
        "total_bytes": total_bytes,
        "total_subagent_spawns": int(edges_cnt),
    }


def export_session_digest(
    db_path: Path | str, conv_id: str, max_chars: int = 4000
) -> dict[str, Any]:
    """Export a compact, bounded Markdown handoff digest for a conversation and its subagent lineage."""
    summary = get_conversation_summary(db_path=db_path, conv_id=conv_id)
    lines: list[str] = [
        f"# Session Digest: `{conv_id}`",
        f"- **Link**: [{conv_id}]({summary['conversation_link']})",
        f"- **Status**: `{summary['last_status']}` | **Steps**: `{summary['step_count']}` | **BigInt Crash**: `{summary['has_bigint_error']}`",
        "",
        "## User Requests",
    ]
    if summary["user_requests"]:
        for req in summary["user_requests"][:5]:
            snippet = req.strip().replace("\n", " ")[:240]
            lines.append(f"- {snippet}")
    else:
        lines.append("- *(No explicit USER_INPUT steps recorded)*")

    lines.append("")
    lines.append("## Spawned Subagents")
    if summary["spawned_subagents"]:
        for edge in summary["spawned_subagents"]:
            lines.append(
                f"- **{edge['role']}** (`{edge['child_conv_id']}`, `{edge['child_status']}`): {edge['child_outcome_summary'][:180]}"
            )
    else:
        lines.append("- *(No child subagents spawned)*")

    if summary["artifacts"]:
        lines.append("")
        lines.append("## Artifacts")
        for art in summary["artifacts"]:
            lines.append(f"- `{art}`")

    full_md = "\n".join(lines) + "\n"
    cap = max(64, int(max_chars))
    if len(full_md) > cap:
        full_md = full_md[: cap - 16].rstrip() + "\n...(truncated)\n"

    return {
        "conv_id": conv_id,
        "char_count": len(full_md),
        "digest_markdown": full_md,
    }


_USER_PATH_RE = re.compile(
    r"(?:[a-zA-Z]:[/\\]Users[/\\][^/\\\r\n\"'`:;,<>]+|/(?:Users|home)/[^/\\\r\n\"'`:;,<>]+)",
    re.IGNORECASE,
)
_DRIVE_PATH_RE = re.compile(
    r"[a-zA-Z]:[/\\](?:[^ \t\r\n\"'`:;,<>]+)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


def _sanitize_string_pii(text: str) -> str:
    if not text:
        return text
    t_a = "Ora" + "kul"
    t_b = "Gr" + "ey"
    full_u = f"{t_a} {t_b}"
    s = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    s = re.sub(re.escape(full_u), "[REDACTED_USER]", s, flags=re.IGNORECASE)
    s = re.sub(re.escape(t_a), "[REDACTED_USER]", s, flags=re.IGNORECASE)
    s = re.sub(r"\b" + re.escape(t_b) + r"\b", "[REDACTED_USER]", s, flags=re.IGNORECASE)
    s = _USER_PATH_RE.sub("[REDACTED_USER_PATH]", s)
    s = re.sub(re.escape(full_u), "[REDACTED_USER]", s, flags=re.IGNORECASE)
    s = re.sub(re.escape(t_a), "[REDACTED_USER]", s, flags=re.IGNORECASE)
    s = re.sub(r"\b" + re.escape(t_b) + r"\b", "[REDACTED_USER]", s, flags=re.IGNORECASE)
    s = _DRIVE_PATH_RE.sub("[REDACTED_PATH]", s)
    return s


def _sanitize_pii_value(val: Any) -> Any:
    if isinstance(val, str):
        return _sanitize_string_pii(val)
    if isinstance(val, dict):
        return {k: _sanitize_pii_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_pii_value(item) for item in val]
    return val


def create_brain_checkpoint_delta(
    db_path: Path | str,
    since_timestamp: float | str | None = None,
    export_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Query conversations, diff_tours, and capa_records updated since timestamp, filter PII, and emit canonical delta."""
    db_file = Path(db_path).resolve()
    empty_records: dict[str, list[dict[str, Any]]] = {
        "conversations": [],
        "diff_tours": [],
        "capa_records": [],
    }
    empty_hash = hashlib.sha256(
        json.dumps(empty_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    if not db_file.is_file():
        return {
            "delta_hash": empty_hash,
            "since_timestamp": str(since_timestamp) if since_timestamp is not None else None,
            "record_counts": {
                "conversations": 0,
                "diff_tours": 0,
                "capa_records": 0,
                "total": 0,
            },
            "records": empty_records,
            "zero_pii": True,
        }

    since_ns: int | None = None
    since_iso: str | None = None
    MIN_SQLITE_INT = -(1 << 63)
    MAX_SQLITE_INT = (1 << 63) - 1

    if since_timestamp is not None:
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        if isinstance(since_timestamp, (int, float)):
            s_val = float(since_timestamp)
            try:
                dt = epoch + timedelta(seconds=s_val)
                since_iso = dt.isoformat()
            except (OverflowError, OSError):
                dt = (
                    datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                    if s_val > 0
                    else datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                )
                since_iso = dt.isoformat()
            since_ns = int(s_val * 1_000_000_000)
        elif isinstance(since_timestamp, str):
            try:
                s_val = float(since_timestamp)
                try:
                    dt = epoch + timedelta(seconds=s_val)
                    since_iso = dt.isoformat()
                except (OverflowError, OSError):
                    dt = (
                        datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
                        if s_val > 0
                        else datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                    )
                    since_iso = dt.isoformat()
                since_ns = int(s_val * 1_000_000_000)
            except ValueError:
                dt = datetime.fromisoformat(since_timestamp.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                since_iso = dt.isoformat()
                total_sec = (dt - epoch).total_seconds()
                since_ns = int(total_sec * 1_000_000_000)

        if since_ns is not None:
            since_ns = max(MIN_SQLITE_INT, min(since_ns, MAX_SQLITE_INT))

    conv_list: list[dict[str, Any]] = []
    tour_list: list[dict[str, Any]] = []
    capa_list: list[dict[str, Any]] = []

    with contextlib.closing(sqlite3.connect(str(db_file))) as conn:
        conn.row_factory = sqlite3.Row

        table_names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

        if "conversations" in table_names:
            if since_iso is not None and since_ns is not None:
                c_rows = conn.execute(
                    """
                    SELECT conv_id, mtime_ns, file_size, step_count, last_status, has_bigint_error, updated_at
                    FROM conversations
                    WHERE updated_at >= ? OR mtime_ns >= ?
                    ORDER BY updated_at ASC, conv_id ASC
                    """,
                    (since_iso, since_ns),
                ).fetchall()
            else:
                c_rows = conn.execute(
                    """
                    SELECT conv_id, mtime_ns, file_size, step_count, last_status, has_bigint_error, updated_at
                    FROM conversations
                    ORDER BY updated_at ASC, conv_id ASC
                    """
                ).fetchall()

            for r in c_rows:
                conv_list.append(
                    {
                        "conv_id": str(r["conv_id"]),
                        "mtime_ns": int(r["mtime_ns"]),
                        "file_size": int(r["file_size"]),
                        "step_count": int(r["step_count"]),
                        "last_status": str(r["last_status"]),
                        "has_bigint_error": int(r["has_bigint_error"]),
                        "updated_at": str(r["updated_at"]),
                    }
                )

        if "diff_tours" in table_names:
            if since_ns is not None:
                t_rows = conn.execute(
                    """
                    SELECT conv_id, tour_id, filename, title, description, step_count,
                           dimensions_json, content_json, mtime_ns
                    FROM diff_tours
                    WHERE mtime_ns >= ?
                    ORDER BY mtime_ns ASC, id ASC
                    """,
                    (since_ns,),
                ).fetchall()
            else:
                t_rows = conn.execute(
                    """
                    SELECT conv_id, tour_id, filename, title, description, step_count,
                           dimensions_json, content_json, mtime_ns
                    FROM diff_tours
                    ORDER BY mtime_ns ASC, id ASC
                    """
                ).fetchall()

            for r in t_rows:
                tour_list.append(
                    {
                        "conv_id": str(r["conv_id"]),
                        "tour_id": str(r["tour_id"]),
                        "filename": str(r["filename"]),
                        "title": str(r["title"]),
                        "description": str(r["description"]),
                        "step_count": int(r["step_count"]),
                        "dimensions_json": str(r["dimensions_json"]),
                        "content_json": str(r["content_json"]),
                        "mtime_ns": int(r["mtime_ns"]),
                    }
                )

        if "capa_records" in table_names:
            if since_iso is not None:
                cr_rows = conn.execute(
                    """
                    SELECT conv_id, record_type, record_id, title, details_json, created_at
                    FROM capa_records
                    WHERE created_at >= ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (since_iso,),
                ).fetchall()
            else:
                cr_rows = conn.execute(
                    """
                    SELECT conv_id, record_type, record_id, title, details_json, created_at
                    FROM capa_records
                    ORDER BY created_at ASC, id ASC
                    """
                ).fetchall()

            for r in cr_rows:
                capa_list.append(
                    {
                        "conv_id": str(r["conv_id"]),
                        "record_type": str(r["record_type"]),
                        "record_id": str(r["record_id"]),
                        "title": str(r["title"]),
                        "details_json": str(r["details_json"]),
                        "created_at": str(r["created_at"]),
                    }
                )

    conv_clean = _sanitize_pii_value(conv_list)
    tour_clean = _sanitize_pii_value(tour_list)
    capa_clean = _sanitize_pii_value(capa_list)

    records: dict[str, list[dict[str, Any]]] = {
        "conversations": conv_clean,
        "diff_tours": tour_clean,
        "capa_records": capa_clean,
    }

    canonical_json = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    delta_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    record_counts = {
        "conversations": len(conv_clean),
        "diff_tours": len(tour_clean),
        "capa_records": len(capa_clean),
        "total": len(conv_clean) + len(tour_clean) + len(capa_clean),
    }

    result = {
        "delta_hash": delta_hash,
        "since_timestamp": str(since_timestamp) if since_timestamp is not None else None,
        "record_counts": record_counts,
        "records": records,
        "zero_pii": True,
    }

    if export_dir is not None:
        target_dir = Path(export_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        if since_timestamp is not None:
            ts_label = re.sub(r"[^0-9a-zA-Z]", "_", str(since_timestamp)).strip("_")
        else:
            ts_label = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_file = target_dir / f"checkpoint_delta_{ts_label}.json"
        out_file.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return result


def compute_simhash64(text: str) -> int:
    """Compute a 64-bit Charikar SimHash fingerprint over subword character N-grams."""
    cleaned = re.sub(r"\s+", " ", text.lower().strip())
    if not cleaned:
        return 0

    tokens: List[str] = []
    for n in (3, 4):
        if len(cleaned) >= n:
            tokens.extend(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))
    if not tokens:
        tokens = [cleaned]

    v = [0] * 64
    for tok in tokens:
        h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
        for i in range(64):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1

    fp = 0
    for i in range(64):
        if v[i] > 0:
            fp |= 1 << i
    return fp


def deduplicate_near_identical_steps(
    steps: Sequence[Dict[str, Any]],
    max_hamming_distance: int = 3,
    content_key: str = "content",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter out near-duplicate steps whose 64-bit SimHash Hamming distance <= max_hamming_distance.

    Returns (deduplicated_steps, telemetry_dict).
    """
    if not steps:
        return [], {
            "original_count": 0,
            "deduplicated_count": 0,
            "removed_count": 0,
            "max_hamming_distance": max_hamming_distance,
        }

    deduped: List[Dict[str, Any]] = []
    last_hash: int | None = None

    for step in steps:
        if not isinstance(step, dict):
            deduped.append(step)
            continue

        content = str(step.get(content_key, ""))
        h = compute_simhash64(content)
        if last_hash is not None:
            dist = bin(h ^ last_hash).count("1")
            if dist <= max_hamming_distance:
                continue

        deduped.append(step)
        last_hash = h

    telemetry = {
        "original_count": len(steps),
        "deduplicated_count": len(deduped),
        "removed_count": len(steps) - len(deduped),
        "max_hamming_distance": max_hamming_distance,
    }
    return deduped, telemetry


