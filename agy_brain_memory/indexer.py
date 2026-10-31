"""Incremental SQLite WAL + FTS5 Brain Transcript & Artifact Indexer."""

from __future__ import annotations

import contextlib
import datetime
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

_CONV_ID_RE = re.compile(r'"conversationId"\s*:\s*"([^"]+)"')

MAX_STEP_CONTENT_CHARS = 8192
MAX_ARTIFACT_CONTENT_CHARS = 32768


def _unquote_arg(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if len(s) >= 2 and (
        (s.startswith('"') and s.endswith('"'))
        or (s.startswith("'") and s.endswith("'"))
    ):
        return s[1:-1]
    return s


def _connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


class BrainMemoryIndexer:
    """Incremental SQLite WAL + FTS5 indexer for Antigravity brain transcripts and artifacts."""

    def __init__(self, db_path: Path | str, brain_dir: Path | str) -> None:
        self.db_path = Path(db_path).resolve()
        self.brain_dir = Path(brain_dir).resolve()
        self._init_schema()

    def _init_schema(self) -> None:
        with contextlib.closing(_connect_db(self.db_path)) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conv_id TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    step_count INTEGER NOT NULL,
                    last_status TEXT NOT NULL,
                    has_bigint_error INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    content TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_steps_conv_id
                    ON steps(conv_id, step_index DESC);

                CREATE VIRTUAL TABLE IF NOT EXISTS steps_fts USING fts5(
                    conv_id UNINDEXED,
                    step_index UNINDEXED,
                    type UNINDEXED,
                    status UNINDEXED,
                    content
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    content TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_conv_id
                    ON artifacts(conv_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
                    conv_id UNINDEXED,
                    filename,
                    content
                );

                CREATE TABLE IF NOT EXISTS subagent_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_conv_id TEXT NOT NULL,
                    child_conv_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    type_name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    child_status TEXT NOT NULL,
                    child_outcome_summary TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_edges_parent_conv_id
                    ON subagent_edges(parent_conv_id);

                CREATE INDEX IF NOT EXISTS idx_edges_child_conv_id
                    ON subagent_edges(child_conv_id);

                CREATE TABLE IF NOT EXISTS diff_tours (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL,
                    tour_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    step_count INTEGER NOT NULL,
                    dimensions_json TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_diff_tours_conv_id
                    ON diff_tours(conv_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS diff_tours_fts USING fts5(
                    conv_id UNINDEXED,
                    tour_id UNINDEXED,
                    filename,
                    title,
                    description,
                    content
                );

                CREATE TABLE IF NOT EXISTS capa_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL,
                    record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_capa_records_conv_id
                    ON capa_records(conv_id);

                CREATE INDEX IF NOT EXISTS idx_capa_records_type
                    ON capa_records(record_type);

                CREATE VIRTUAL TABLE IF NOT EXISTS capa_records_fts USING fts5(
                    conv_id UNINDEXED,
                    record_type UNINDEXED,
                    record_id UNINDEXED,
                    title,
                    content
                );
                """
            )

    def sync(self, force: bool = False) -> dict[str, Any]:
        """Incrementally index modified or new conversations and resolve subagent outcomes."""
        t0 = time.perf_counter()
        total_seen = 0
        conversations_indexed = 0
        conversations_skipped = 0
        steps_indexed = 0
        artifacts_indexed = 0
        edges_indexed = 0
        diff_tours_indexed = 0
        capa_records_indexed = 0

        if not self.brain_dir.is_dir():
            return {
                "total_conversations_seen": 0,
                "conversations_indexed": 0,
                "conversations_skipped": 0,
                "steps_indexed": 0,
                "artifacts_indexed": 0,
                "subagent_edges_indexed": 0,
                "diff_tours_indexed": 0,
                "capa_records_indexed": 0,
                "elapsed_ms": 0.0,
            }

        with contextlib.closing(_connect_db(self.db_path)) as conn, conn:
            existing_rows: dict[str, tuple[int, int]] = {}
            last_status_by_conv: dict[str, str] = {}
            for row in conn.execute(
                "SELECT conv_id, mtime_ns, file_size, last_status FROM conversations"
            ).fetchall():
                cid = str(row["conv_id"])
                existing_rows[cid] = (int(row["mtime_ns"]), int(row["file_size"]))
                last_status_by_conv[cid] = str(row["last_status"])

            step_rows: list[tuple[Any, ...]] = []
            step_fts_rows: list[tuple[Any, ...]] = []
            artifact_rows: list[tuple[Any, ...]] = []
            artifact_fts_rows: list[tuple[Any, ...]] = []
            diff_tour_rows: list[tuple[Any, ...]] = []
            diff_tour_fts_rows: list[tuple[Any, ...]] = []
            capa_record_rows: list[tuple[Any, ...]] = []
            capa_record_fts_rows: list[tuple[Any, ...]] = []
            edge_rows_to_insert: list[tuple[Any, ...]] = []
            conv_rows: list[tuple[Any, ...]] = []
            convs_to_delete: list[tuple[str]] = []

            last_non_empty_content_by_conv: dict[str, str] = {}
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            for sub in sorted(self.brain_dir.iterdir()):
                if not sub.is_dir():
                    continue
                conv_id = sub.name
                candidates = [
                    sub / ".system_generated" / "logs" / "transcript.jsonl",
                    sub / "transcript.jsonl",
                ]
                transcript_file = next((c for c in candidates if c.is_file()), None)
                md_files = sorted(p for p in sub.glob("*.md") if p.is_file())
                tour_files = sorted(
                    p for p in sub.rglob("*")
                    if p.is_file() and (p.name.endswith(".tour.json") or p.name.startswith("diff_tour") or ".diff-tour" in p.parts)
                )
                capa_files = sorted(
                    p for p in sub.rglob("*")
                    if p.is_file() and p.name in (
                        "evidence_units.jsonl",
                        "failure_ledger.jsonl",
                        "audit_chain.json",
                        "expel_rules.json",
                        "capa_optoprime_report.json",
                    )
                )

                if transcript_file is None and not md_files and not tour_files and not capa_files:
                    continue

                total_seen += 1
                tracked_files = ([transcript_file] if transcript_file else []) + md_files + tour_files + capa_files
                max_mtime_ns = max(p.stat().st_mtime_ns for p in tracked_files)
                total_size = sum(p.stat().st_size for p in tracked_files)

                if not force and existing_rows.get(conv_id) == (max_mtime_ns, total_size):
                    conversations_skipped += 1
                    continue

                if conv_id in existing_rows:
                    convs_to_delete.append((conv_id,))

                step_count = 0
                last_status = "DONE"
                has_bigint = 0
                last_non_empty = ""
                pending_subagents: list[dict[str, Any]] = []
                conv_edges_by_child: dict[str, tuple[Any, ...]] = {}

                if transcript_file is not None:
                    raw_lines = transcript_file.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    for idx, raw_line in enumerate(raw_lines, start=1):
                        s = raw_line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue

                        step_idx = int(obj.get("step_index", idx))
                        step_type = str(obj.get("type", "GENERIC"))
                        status = str(obj.get("status", "DONE"))
                        created_at = str(obj.get("created_at", ""))
                        content = str(obj.get("content", "") or "")

                        last_status = status
                        if content:
                            last_non_empty = content[:500]
                        if (
                            "Do not know how to serialize a BigInt" in content
                            or "BigInt" in content
                        ):
                            has_bigint = 1

                        stored_content = content[:MAX_STEP_CONTENT_CHARS]
                        step_rows.append(
                            (conv_id, step_idx, step_type, status, created_at, stored_content)
                        )
                        step_fts_rows.append(
                            (conv_id, str(step_idx), step_type, status, stored_content)
                        )
                        step_count += 1
                        steps_indexed += 1

                        tool_calls = obj.get("tool_calls")
                        if isinstance(tool_calls, list):
                            for tc in tool_calls:
                                if (
                                    isinstance(tc, dict)
                                    and tc.get("name") == "invoke_subagent"
                                ):
                                    tc_args = tc.get("args") or {}
                                    if isinstance(tc_args, dict):
                                        pending_subagents.append(
                                            {
                                                "step_index": step_idx,
                                                "role": _unquote_arg(
                                                    tc_args.get("Role", "subagent")
                                                ),
                                                "type_name": _unquote_arg(
                                                    tc_args.get("TypeName", "self")
                                                ),
                                                "prompt": _unquote_arg(
                                                    tc_args.get("Prompt", "")
                                                )[:2000],
                                            }
                                        )

                        if step_type != "USER_INPUT" and (
                            "Created the following subagent" in content
                            or (pending_subagents and "conversationId" in content)
                        ):
                            m = _CONV_ID_RE.search(content)
                            if m:
                                child_id = m.group(1).strip()
                                call_meta = (
                                    pending_subagents.pop(0)
                                    if pending_subagents
                                    else {
                                        "step_index": step_idx,
                                        "role": "subagent",
                                        "type_name": "self",
                                        "prompt": "",
                                    }
                                )
                                candidate_edge = (
                                    conv_id,
                                    child_id,
                                    int(call_meta["step_index"]),
                                    str(call_meta["role"]),
                                    str(call_meta["type_name"]),
                                    str(call_meta["prompt"]),
                                    "UNKNOWN",
                                    "",
                                )
                                existing_edge = conv_edges_by_child.get(child_id)
                                if existing_edge is None or (
                                    existing_edge[3] == "subagent"
                                    and not existing_edge[5]
                                    and (candidate_edge[3] != "subagent" or candidate_edge[5])
                                ):
                                    conv_edges_by_child[child_id] = candidate_edge

                if conv_edges_by_child:
                    edge_rows_to_insert.extend(conv_edges_by_child.values())
                    edges_indexed += len(conv_edges_by_child)

                for md_path in md_files:
                    md_content = md_path.read_text(encoding="utf-8", errors="replace")[
                        :MAX_ARTIFACT_CONTENT_CHARS
                    ]
                    md_mtime = md_path.stat().st_mtime_ns
                    artifact_rows.append((conv_id, md_path.name, md_mtime, md_content))
                    artifact_fts_rows.append((conv_id, md_path.name, md_content))
                    artifacts_indexed += 1

                for tour_path in tour_files:
                    try:
                        tour_data = json.loads(tour_path.read_text(encoding="utf-8", errors="replace"))
                        if isinstance(tour_data, dict):
                            tour_title = str(tour_data.get("title") or tour_path.stem)
                            tour_desc = str(tour_data.get("description") or "")
                            tour_steps = tour_data.get("steps") or []
                            step_cnt = len(tour_steps) if isinstance(tour_steps, list) else 0
                            dims: set[str] = set()
                            text_parts = [tour_title, tour_desc]
                            if isinstance(tour_steps, list):
                                for st in tour_steps:
                                    if isinstance(st, dict):
                                        d = st.get("dimension") or st.get("review_dimension")
                                        if d:
                                            dims.add(str(d))
                                        f = st.get("file") or st.get("file_path")
                                        if f:
                                            text_parts.append(str(f))
                                        c = st.get("description") or st.get("comment") or st.get("body")
                                        if c:
                                            text_parts.append(str(c))
                            dims_json = json.dumps(sorted(dims))
                            content_text = " ".join(text_parts)[:MAX_ARTIFACT_CONTENT_CHARS]
                            t_mtime = tour_path.stat().st_mtime_ns
                            tour_id = str(tour_data.get("id") or tour_path.stem)
                            diff_tour_rows.append((conv_id, tour_id, tour_path.name, tour_title, tour_desc, step_cnt, dims_json, json.dumps(tour_data)[:MAX_ARTIFACT_CONTENT_CHARS], t_mtime))
                            diff_tour_fts_rows.append((conv_id, tour_id, tour_path.name, tour_title, tour_desc, content_text))
                            diff_tours_indexed += 1
                    except Exception:
                        pass

                for c_path in capa_files:
                    try:
                        raw_content = c_path.read_text(encoding="utf-8", errors="replace")
                        if c_path.name == "evidence_units.jsonl":
                            for line in raw_content.splitlines():
                                if not line.strip():
                                    continue
                                try:
                                    u = json.loads(line)
                                    if isinstance(u, dict):
                                        u_id = str(u.get("unit_id") or u.get("id") or "EV-UNKNOWN")
                                        u_cmd = str(u.get("command") or "")
                                        u_title = f"Evidence Unit {u_id}: {u_cmd}"[:200]
                                        u_dt = str(u.get("created_at") or now_iso)
                                        capa_record_rows.append((conv_id, "evidence_unit", u_id, u_title, line[:MAX_ARTIFACT_CONTENT_CHARS], u_dt))
                                        capa_record_fts_rows.append((conv_id, "evidence_unit", u_id, u_title, line[:MAX_ARTIFACT_CONTENT_CHARS]))
                                        capa_records_indexed += 1
                                except Exception:
                                    pass
                        elif c_path.name == "failure_ledger.jsonl":
                            for line in raw_content.splitlines():
                                if not line.strip():
                                    continue
                                try:
                                    f = json.loads(line)
                                    if isinstance(f, dict):
                                        f_id = str(f.get("node_id") or f.get("id") or "FAIL-UNKNOWN")
                                        f_reason = str(f.get("reason") or "")
                                        f_title = f"Failure: {f_id} - {f_reason}"[:200]
                                        f_dt = str(f.get("created_at") or now_iso)
                                        capa_record_rows.append((conv_id, "failure_ledger", f_id, f_title, line[:MAX_ARTIFACT_CONTENT_CHARS], f_dt))
                                        capa_record_fts_rows.append((conv_id, "failure_ledger", f_id, f_title, line[:MAX_ARTIFACT_CONTENT_CHARS]))
                                        capa_records_indexed += 1
                                except Exception:
                                    pass
                        elif c_path.name == "audit_chain.json":
                            try:
                                ac = json.loads(raw_content)
                                ac_id = str(ac.get("root_hash") or "audit_chain")
                                ac_title = "ISO 19011 W3C PROV-DM Audit Chain"
                                capa_record_rows.append((conv_id, "audit_chain", ac_id, ac_title, raw_content[:MAX_ARTIFACT_CONTENT_CHARS], now_iso))
                                capa_record_fts_rows.append((conv_id, "audit_chain", ac_id, ac_title, raw_content[:MAX_ARTIFACT_CONTENT_CHARS]))
                                capa_records_indexed += 1
                            except Exception:
                                pass
                        elif c_path.name in ("expel_rules.json", "capa_optoprime_report.json"):
                            try:
                                cr_id = c_path.stem
                                cr_title = f"CAPA Rules {cr_id}"
                                capa_record_rows.append((conv_id, "capa_rule", cr_id, cr_title, raw_content[:MAX_ARTIFACT_CONTENT_CHARS], now_iso))
                                capa_record_fts_rows.append((conv_id, "capa_rule", cr_id, cr_title, raw_content[:MAX_ARTIFACT_CONTENT_CHARS]))
                                capa_records_indexed += 1
                            except Exception:
                                pass
                    except Exception:
                        pass

                conv_rows.append(
                    (
                        conv_id,
                        max_mtime_ns,
                        total_size,
                        step_count,
                        last_status,
                        has_bigint,
                        now_iso,
                    )
                )
                last_status_by_conv[conv_id] = last_status
                last_non_empty_content_by_conv[conv_id] = last_non_empty
                conversations_indexed += 1

            if convs_to_delete:
                conn.executemany(
                    "DELETE FROM conversations WHERE conv_id = ?", convs_to_delete
                )
                conn.executemany("DELETE FROM steps WHERE conv_id = ?", convs_to_delete)
                conn.executemany(
                    "DELETE FROM steps_fts WHERE conv_id = ?", convs_to_delete
                )
                conn.executemany(
                    "DELETE FROM artifacts WHERE conv_id = ?", convs_to_delete
                )
                conn.executemany(
                    "DELETE FROM artifacts_fts WHERE conv_id = ?", convs_to_delete
                )
                conn.executemany(
                    "DELETE FROM diff_tours WHERE conv_id = ?", convs_to_delete
                )
                conn.executemany(
                    "DELETE FROM diff_tours_fts WHERE conv_id = ?", convs_to_delete
                )
                conn.executemany(
                    "DELETE FROM capa_records WHERE conv_id = ?", convs_to_delete
                )
                conn.executemany(
                    "DELETE FROM capa_records_fts WHERE conv_id = ?", convs_to_delete
                )
                conn.executemany(
                    "DELETE FROM subagent_edges WHERE parent_conv_id = ?",
                    convs_to_delete,
                )

            if conv_rows:
                conn.executemany(
                    """
                    INSERT INTO conversations (
                        conv_id, mtime_ns, file_size, step_count,
                        last_status, has_bigint_error, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    conv_rows,
                )
            if step_rows:
                conn.executemany(
                    """
                    INSERT INTO steps (conv_id, step_index, type, status, created_at, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    step_rows,
                )
                conn.executemany(
                    """
                    INSERT INTO steps_fts (conv_id, step_index, type, status, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    step_fts_rows,
                )
            if artifact_rows:
                conn.executemany(
                    """
                    INSERT INTO artifacts (conv_id, filename, mtime_ns, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    artifact_rows,
                )
                conn.executemany(
                    """
                    INSERT INTO artifacts_fts (conv_id, filename, content)
                    VALUES (?, ?, ?)
                    """,
                    artifact_fts_rows,
                )
            if diff_tour_rows:
                conn.executemany(
                    """
                    INSERT INTO diff_tours (
                        conv_id, tour_id, filename, title, description,
                        step_count, dimensions_json, content_json, mtime_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    diff_tour_rows,
                )
            if diff_tour_fts_rows:
                conn.executemany(
                    """
                    INSERT INTO diff_tours_fts (conv_id, tour_id, filename, title, description, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    diff_tour_fts_rows,
                )
            if capa_record_rows:
                conn.executemany(
                    """
                    INSERT INTO capa_records (
                        conv_id, record_type, record_id, title, details_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    capa_record_rows,
                )
            if capa_record_fts_rows:
                conn.executemany(
                    """
                    INSERT INTO capa_records_fts (conv_id, record_type, record_id, title, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    capa_record_fts_rows,
                )
            if edge_rows_to_insert:
                conn.executemany(
                    """
                    INSERT INTO subagent_edges (
                        parent_conv_id, child_conv_id, step_index,
                        role, type_name, prompt, child_status, child_outcome_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    edge_rows_to_insert,
                )

            # Second pass: resolve child_status and child_outcome_summary when conversations changed
            if conversations_indexed > 0:
                edge_rows = conn.execute(
                    "SELECT id, child_conv_id FROM subagent_edges"
                ).fetchall()
                edge_updates: list[tuple[str, str, int]] = []
                for erow in edge_rows:
                    edge_id = int(erow["id"])
                    cid = str(erow["child_conv_id"])
                    c_status = last_status_by_conv.get(cid, "UNKNOWN")
                    if cid in last_non_empty_content_by_conv:
                        outcome = last_non_empty_content_by_conv[cid]
                    else:
                        last_step_row = conn.execute(
                            """
                            SELECT content FROM steps
                            WHERE conv_id = ? AND content != ''
                            ORDER BY step_index DESC, id DESC LIMIT 1
                            """,
                            (cid,),
                        ).fetchone()
                        outcome = (
                            str(last_step_row["content"])[:500] if last_step_row else ""
                        )
                        last_non_empty_content_by_conv[cid] = outcome
                    edge_updates.append((c_status, outcome, edge_id))

                if edge_updates:
                    conn.executemany(
                        """
                        UPDATE subagent_edges
                        SET child_status = ?, child_outcome_summary = ?
                        WHERE id = ?
                        """,
                        edge_updates,
                    )

            conn.commit()

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "total_conversations_seen": total_seen,
            "conversations_indexed": conversations_indexed,
            "conversations_skipped": conversations_skipped,
            "steps_indexed": steps_indexed,
            "artifacts_indexed": artifacts_indexed,
            "subagent_edges_indexed": edges_indexed,
            "diff_tours_indexed": diff_tours_indexed,
            "capa_records_indexed": capa_records_indexed,
            "elapsed_ms": elapsed_ms,
        }

    def index_diff_tour(
        self,
        conv_id: str,
        tour_data: dict[str, Any],
        filename: str = "tour.json",
    ) -> int:
        """Directly insert a structured diff-tour v1 record into the index."""
        tour_title = str(tour_data.get("title") or Path(filename).stem)
        tour_desc = str(tour_data.get("description") or "")
        tour_steps = tour_data.get("steps") or []
        step_cnt = len(tour_steps) if isinstance(tour_steps, list) else 0
        dims: set[str] = set()
        text_parts = [tour_title, tour_desc]
        if isinstance(tour_steps, list):
            for st in tour_steps:
                if isinstance(st, dict):
                    d = st.get("dimension") or st.get("review_dimension")
                    if d:
                        dims.add(str(d))
                    f = st.get("file") or st.get("file_path")
                    if f:
                        text_parts.append(str(f))
                    c = st.get("description") or st.get("comment") or st.get("body")
                    if c:
                        text_parts.append(str(c))
        dims_json = json.dumps(sorted(dims))
        content_text = " ".join(text_parts)[:MAX_ARTIFACT_CONTENT_CHARS]
        now_ns = time.time_ns()
        tour_id = str(tour_data.get("id") or Path(filename).stem)

        with contextlib.closing(_connect_db(self.db_path)) as conn, conn:
            cur = conn.execute(
                """
                INSERT INTO diff_tours (
                    conv_id, tour_id, filename, title, description,
                    step_count, dimensions_json, content_json, mtime_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (conv_id, tour_id, filename, tour_title, tour_desc, step_cnt, dims_json, json.dumps(tour_data)[:MAX_ARTIFACT_CONTENT_CHARS], now_ns),
            )
            inserted_id = int(cur.lastrowid or 0)
            conn.execute(
                """
                INSERT INTO diff_tours_fts (conv_id, tour_id, filename, title, description, content)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conv_id, tour_id, filename, tour_title, tour_desc, content_text),
            )
            conn.commit()
            return inserted_id

    def index_capa_record(
        self,
        conv_id: str,
        record_type: str,
        record_id: str,
        title: str,
        details: dict[str, Any] | str,
        created_at: str = "",
    ) -> int:
        """Directly insert an ISO 19011 / CAPA record into the index."""
        details_str = json.dumps(details) if isinstance(details, dict) else str(details)
        dt = created_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
        with contextlib.closing(_connect_db(self.db_path)) as conn, conn:
            cur = conn.execute(
                """
                INSERT INTO capa_records (
                    conv_id, record_type, record_id, title, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conv_id, record_type, record_id, title, details_str[:MAX_ARTIFACT_CONTENT_CHARS], dt),
            )
            inserted_id = int(cur.lastrowid or 0)
            conn.execute(
                """
                INSERT INTO capa_records_fts (conv_id, record_type, record_id, title, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conv_id, record_type, record_id, title, details_str[:MAX_ARTIFACT_CONTENT_CHARS]),
            )
            conn.commit()
            return inserted_id

def validate_transcript_schema(transcript_path: Path) -> bool:
    """Validate JSONL transcript schema (requires dict rows with step_index and type)."""
    if not transcript_path.exists():
        return False
    try:
        with transcript_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                if not raw_line.strip():
                    continue
                step = json.loads(raw_line)
                if not isinstance(step, dict) or "step_index" not in step or "type" not in step:
                    return False
        return True
    except (OSError, json.JSONDecodeError):
        return False
