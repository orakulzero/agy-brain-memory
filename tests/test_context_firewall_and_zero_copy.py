"""Agile V Verification Tests for Context Firewall & Zero-Copy SQLite WAL Indexer.

Traceability:
- REQ-BM-ZEROCOPY-01: Zero-copy SQLite WAL FTS5 cross-session indexer PRAGMAs.
- REQ-BM-FIREWALL-01: Subagent context firewall isolating reads >500 lines and log dumps.

Quality Framework: ISO 9001:2015, ISO/IEC 42001:2023, Agile V Role Separation.
Hygiene: 100% English-only (0 Cyrillic characters).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import sqlite3
from typing import Any, Dict, List
import pytest

from agy_brain_memory.compactor import allocate_context_budget
from agy_brain_memory.indexer import _connect_db, BrainMemoryIndexer

try:
    from agy_brain_memory.context_firewall import (
        ContextFirewallRouter,
        evaluate_context_firewall,
    )
except ImportError:
    ContextFirewallRouter = None
    evaluate_context_firewall = None


# ============================================================================
# REQ-BM-ZEROCOPY-01: Zero-Copy SQLite WAL FTS5 Indexer PRAGMAs
# ============================================================================


def test_req_bm_zerocopy_01_database_pragmas_configured(tmp_path: Path):
    """Verify _connect_db configures WAL, mmap_size, temp_store, and synchronous (REQ-BM-ZEROCOPY-01)."""
    db_file = tmp_path / "test_brain.db"
    with contextlib.closing(_connect_db(db_file)) as conn:
        j_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0].upper()
        sync_mode = conn.execute("PRAGMA synchronous;").fetchone()[0]
        mmap_sz = conn.execute("PRAGMA mmap_size;").fetchone()[0]
        temp_st = conn.execute("PRAGMA temp_store;").fetchone()[0]

        assert j_mode == "WAL", f"Expected WAL journal mode, got {j_mode}"
        assert sync_mode in (1, "1", "NORMAL"), f"Expected NORMAL synchronous mode, got {sync_mode}"
        assert mmap_sz >= 268435456, f"Expected mmap_size >= 268435456 (256MB), got {mmap_sz}"
        assert temp_st in (2, "2", "MEMORY"), f"Expected temp_store = MEMORY (2), got {temp_st}"


def test_req_bm_zerocopy_01_fts5_indexing_and_fast_search(tmp_path: Path):
    """Verify BrainMemoryIndexer creates FTS5 tables and executes zero-copy queries (REQ-BM-ZEROCOPY-01)."""
    db_file = tmp_path / "brain_fts.db"
    brain_dir = tmp_path / "brain_dir"
    brain_dir.mkdir(parents=True, exist_ok=True)

    indexer = BrainMemoryIndexer(db_path=db_file, brain_dir=brain_dir)
    assert db_file.exists()

    with contextlib.closing(_connect_db(db_file)) as conn, conn:
        conn.execute(
            "INSERT INTO steps (conv_id, step_index, type, status, created_at, content) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("conv_001", 1, "USER_INPUT", "DONE", "2026-11-01T00:00:00Z", "Implement zero copy WAL FTS5 indexer"),
        )
        conn.execute(
            "INSERT INTO steps_fts (conv_id, step_index, type, status, content) "
            "VALUES (?, ?, ?, ?, ?)",
            ("conv_001", 1, "USER_INPUT", "DONE", "Implement zero copy WAL FTS5 indexer"),
        )

        rows = conn.execute(
            "SELECT conv_id, content FROM steps_fts WHERE steps_fts MATCH 'zero copy' ORDER BY rank"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "conv_001"
        assert "zero copy WAL" in rows[0][1]


def test_req_bm_zerocopy_01_concurrent_readers_in_wal_mode(tmp_path: Path):
    """Verify concurrent zero-copy reads do not block on concurrent connections (REQ-BM-ZEROCOPY-01)."""
    db_file = tmp_path / "wal_concurrency.db"
    with contextlib.closing(_connect_db(db_file)) as conn1, contextlib.closing(_connect_db(db_file)) as conn2:
        conn1.execute("CREATE TABLE IF NOT EXISTS test_data (id INT, val TEXT);")
        conn1.execute("INSERT INTO test_data VALUES (1, 'alpha');")
        conn1.commit()

        row = conn2.execute("SELECT val FROM test_data WHERE id = 1;").fetchone()
        assert row is not None
        assert row[0] == "alpha"


# ============================================================================
# REQ-BM-FIREWALL-01: Subagent Context Firewall & 0/1 Knapsack Budgeting
# ============================================================================


def test_req_bm_firewall_01_module_contract_and_presence():
    """Verify agy_brain_memory.context_firewall contract symbols are implemented (REQ-BM-FIREWALL-01)."""
    assert ContextFirewallRouter is not None, (
        "Contract violation: agy_brain_memory.context_firewall.ContextFirewallRouter must be implemented"
    )
    assert evaluate_context_firewall is not None, (
        "Contract violation: agy_brain_memory.context_firewall.evaluate_context_firewall must be implemented"
    )


def test_req_bm_firewall_01_isolates_reads_greater_than_500_lines():
    """Verify reading context >500 lines routes to subagent to protect orchestrator (REQ-BM-FIREWALL-01)."""
    assert ContextFirewallRouter is not None
    router = ContextFirewallRouter(max_in_process_lines=500)
    decision = router.evaluate(estimated_lines=501, source_type="reference_file")
    assert decision["action"] == "spawn_subagent"
    assert "context_firewall_gt_500_lines" in decision["criteria_triggered"]
    assert decision["requires_isolation"] is True


def test_req_bm_firewall_01_allows_in_process_reads_within_500_lines():
    """Verify reading context <=500 lines is folded in-process with zero overhead (REQ-BM-FIREWALL-01)."""
    assert ContextFirewallRouter is not None
    router = ContextFirewallRouter(max_in_process_lines=500)
    decision = router.evaluate(estimated_lines=350, source_type="code_snippet")
    assert decision["action"] == "fold_in_process"
    assert decision["requires_isolation"] is False
    assert "fold_linear_in_process" in decision["criteria_triggered"]


def test_req_bm_firewall_01_isolates_raw_log_dumps():
    """Verify raw multi-thousand-line tracebacks and log dumps are quarantined (REQ-BM-FIREWALL-01)."""
    assert evaluate_context_firewall is not None
    dump_text = "\n".join(f"Traceback (most recent call last): frame {i}" for i in range(700))
    res = evaluate_context_firewall(content=dump_text, content_type="log_dump")
    assert res["action"] == "spawn_subagent"
    assert res["is_log_dump"] is True
    assert res["requires_isolation"] is True


def test_req_bm_firewall_01_knapsack_budget_protects_mandatory_instructions():
    """Verify 0/1 knapsack context budgeting preserves mandatory items under tight token budget (REQ-BM-FIREWALL-01)."""
    candidates = [
        {"id": "system_prompt", "utility": 100, "token_cost": 400, "mandatory": True},
        {"id": "task_spec", "utility": 90, "token_cost": 300, "mandatory": True},
        {"id": "small_context", "utility": 40, "token_cost": 150, "mandatory": False},
        {"id": "large_dump", "utility": 30, "token_cost": 800, "mandatory": False},
    ]
    # Budget 900 tokens. Mandatory consumes 700. Remaining 200 fits small_context (150).
    selected, telemetry = allocate_context_budget(candidates, max_token_budget=900)
    sel_ids = {item["id"] for item in selected}
    assert "system_prompt" in sel_ids
    assert "task_spec" in sel_ids
    assert "small_context" in sel_ids
    assert "large_dump" not in sel_ids
    assert telemetry["allocated_tokens"] <= 900
