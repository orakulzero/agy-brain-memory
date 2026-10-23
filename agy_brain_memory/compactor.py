"""Brain Health Analyzer & Bounded Session Handoff Digest Exporter."""

from __future__ import annotations

import contextlib
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

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
