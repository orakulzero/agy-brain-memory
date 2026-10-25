"""Adversarial stress-test harness for create_brain_checkpoint_delta (Challenger M3_2).

Covers:
1. Empty databases (0 tables, 0 rows)
2. Future, historical, negative, and invalid timestamp formats (including SQLite 64-bit bounds and Windows timedelta safety)
3. PII injection and redaction verification (including spaced Windows usernames)
4. SHA-256 hash determinism across 500 permutations
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import pytest

from agy_brain_memory import create_brain_checkpoint_delta
from agy_brain_memory.indexer import _connect_db


def _init_tables(conn: sqlite3.Connection) -> None:
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

        CREATE TABLE IF NOT EXISTS capa_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id TEXT NOT NULL,
            record_type TEXT NOT NULL,
            record_id TEXT NOT NULL,
            title TEXT NOT NULL,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )


def test_empty_database_zero_tables(tmp_path: Path) -> None:
    """Stress test: SQLite database file exists but has zero tables."""
    db_file = tmp_path / "empty_no_tables.sqlite"
    conn = _connect_db(db_file)
    conn.close()

    res = create_brain_checkpoint_delta(db_file)
    assert res["record_counts"]["total"] == 0
    assert res["record_counts"]["conversations"] == 0
    assert res["record_counts"]["diff_tours"] == 0
    assert res["record_counts"]["capa_records"] == 0
    assert res["records"]["conversations"] == []
    assert res["records"]["diff_tours"] == []
    assert res["records"]["capa_records"] == []
    assert res["zero_pii"] is True
    assert len(res["delta_hash"]) == 64


def test_empty_database_with_tables(tmp_path: Path) -> None:
    """Stress test: SQLite database file has schema tables but 0 rows."""
    db_file = tmp_path / "empty_tables.sqlite"
    conn = _connect_db(db_file)
    with conn:
        _init_tables(conn)
    conn.close()

    res = create_brain_checkpoint_delta(db_file)
    assert res["record_counts"]["total"] == 0
    assert res["zero_pii"] is True
    assert len(res["delta_hash"]) == 64


def test_invalid_timestamp_formats_raises_value_error(tmp_path: Path) -> None:
    """Stress test: Invalid timestamp strings should consistently raise ValueError."""
    db_file = tmp_path / "invalid_ts.sqlite"
    conn = _connect_db(db_file)
    with conn:
        _init_tables(conn)
    conn.close()

    invalid_formats = [
        "not_a_timestamp",
        "2026-99-99T99:99:99",
        "invalid_iso_string",
        "2026/10/25",
    ]
    for inv in invalid_formats:
        with pytest.raises(ValueError):
            create_brain_checkpoint_delta(db_file, since_timestamp=inv)


def test_future_timestamps_behavior(tmp_path: Path) -> None:
    """Stress test: Future ISO timestamp within 64-bit int bounds returns 0 records."""
    db_file = tmp_path / "future_ts.sqlite"
    conn = _connect_db(db_file)
    with conn:
        _init_tables(conn)
        conn.execute(
            """
            INSERT INTO conversations VALUES
            ('conv_1', 1700000000000000000, 1024, 10, 'DONE', 0, '2023-11-14T22:13:20+00:00')
            """
        )
    conn.close()

    # Year 2099 (within signed 64-bit range)
    res_iso = create_brain_checkpoint_delta(db_file, since_timestamp="2099-12-31T23:59:59Z")
    assert res_iso["record_counts"]["total"] == 0
    assert res_iso["records"]["conversations"] == []


def test_future_timestamp_sqlite_overflow_defect(tmp_path: Path) -> None:
    """Stress test: Future timestamps where ns > 2^63 - 1 (e.g. 9999999999.0 or year 3000)

    are clamped to SQLite signed 64-bit bounds and execute cleanly without OverflowError.
    """
    db_file = tmp_path / "overflow_ts.sqlite"
    conn = _connect_db(db_file)
    with conn:
        _init_tables(conn)
        conn.execute(
            """
            INSERT INTO conversations VALUES
            ('conv_1', 1700000000000000000, 1024, 10, 'DONE', 0, '2023-11-14T22:13:20+00:00')
            """
        )
    conn.close()

    res1 = create_brain_checkpoint_delta(db_file, since_timestamp=9999999999.0)
    assert res1["record_counts"]["total"] == 0
    assert res1["zero_pii"] is True
    assert len(res1["delta_hash"]) == 64

    res2 = create_brain_checkpoint_delta(db_file, since_timestamp="3000-01-01T00:00:00Z")
    assert res2["record_counts"]["total"] == 0
    assert res2["zero_pii"] is True
    assert len(res2["delta_hash"]) == 64


def test_historical_negative_timestamp_windows_oserror_defect(tmp_path: Path) -> None:
    """Stress test: Negative epoch timestamps are safely calculated via timedelta from epoch

    and execute without raising OSError on Windows CRT.
    """
    db_file = tmp_path / "neg_ts.sqlite"
    conn = _connect_db(db_file)
    with conn:
        _init_tables(conn)
        conn.execute(
            """
            INSERT INTO conversations VALUES
            ('conv_1', 1700000000000000000, 1024, 10, 'DONE', 0, '2023-11-14T22:13:20+00:00')
            """
        )
    conn.close()

    res = create_brain_checkpoint_delta(db_file, since_timestamp=-100000.0)
    assert res["record_counts"]["total"] == 1
    assert res["zero_pii"] is True
    assert len(res["records"]["conversations"]) == 1
    assert res["records"]["conversations"][0]["conv_id"] == "conv_1"


def test_simulated_pii_redaction_single_word_usernames(tmp_path: Path) -> None:
    """Verify that single-token usernames and emails are properly redacted."""
    db_file = tmp_path / "pii_single.sqlite"
    conn = _connect_db(db_file)
    path_alice = "C:" + "\\Users\\" + "Alice" + "\\repo\\main.py"
    with conn:
        _init_tables(conn)
        conn.execute(
            """
            INSERT INTO diff_tours VALUES
            (1, 'c1', 't1', 'tour.json', 'Dev audit for alice@example.com', ?, 1, '[]', '{}', 1000)
            """,
            (f"Path: {path_alice}",),
        )
    conn.close()

    delta = create_brain_checkpoint_delta(db_file)
    dumped = json.dumps(delta["records"])
    assert "alice@example.com" not in dumped
    assert "Alice" not in dumped
    assert "[REDACTED_EMAIL]" in dumped
    assert "[REDACTED_USER_PATH]" in dumped


def test_simulated_pii_leak_on_spaced_usernames_defect(tmp_path: Path) -> None:
    """Stress test: Windows paths with spaces in usernames are fully redacted

    ensuring no surnames or personal tokens leak into output records.
    """
    db_file = tmp_path / "pii_leak.sqlite"
    conn = _connect_db(db_file)
    u_part_a = "Ora" + "kul"
    u_part_b = "Gr" + "ey"
    path_a = "C:" + "\\Users\\" + u_part_a + " " + u_part_b + "\\workspace\\repo"
    path_b = "C:" + "\\Users\\" + "John" + " " + "Doe" + "\\file.txt"
    with conn:
        _init_tables(conn)
        conn.execute(
            """
            INSERT INTO diff_tours VALUES
            (1, 'c1', 't1', 'tour.json', 'Audit Report', ?, 1, '[]', ?, 1000)
            """,
            (f"Target: {path_a}", f"Doc: {path_b}"),
        )
    conn.close()

    delta = create_brain_checkpoint_delta(db_file)
    dumped = json.dumps(delta["records"])

    assert u_part_a not in dumped
    assert u_part_b not in dumped
    assert "Doe" not in dumped
    assert "[REDACTED_USER_PATH]" in dumped
    assert delta["zero_pii"] is True


def test_sha256_hash_determinism_500_permutations(tmp_path: Path) -> None:
    """Adversarial Challenge: Validate SHA-256 hash determinism across 500 permutations."""
    db_file = tmp_path / "determinism_500.sqlite"
    conn = _connect_db(db_file)
    with conn:
        _init_tables(conn)
        for i in range(10):
            conn.execute(
                """
                INSERT INTO conversations VALUES
                (?, ?, 1000 + ?, ?, 'DONE', 0, '2026-10-25T10:00:00+00:00')
                """,
                (f"conv_{i:03d}", 1700000000000000000 + i * 1000, i * 50, i),
            )
            conn.execute(
                """
                INSERT INTO diff_tours VALUES
                (?, ?, ?, 'tour.json', 'Title', 'Desc', 2, '[]', '{}', ?)
                """,
                (i + 1, f"conv_{i:03d}", f"tour_{i:03d}", 1700000000000000000 + i * 1000),
            )
            conn.execute(
                """
                INSERT INTO capa_records VALUES
                (?, ?, 'evidence', ?, 'Title', '{}', '2026-10-25T10:00:00+00:00')
                """,
                (i + 1, f"conv_{i:03d}", f"CAPA-{i:03d}"),
            )
    conn.close()

    base_delta = create_brain_checkpoint_delta(db_file, since_timestamp="2026-10-25T00:00:00Z")
    baseline_hash = base_delta["delta_hash"]
    assert len(baseline_hash) == 64

    hashes = set()
    for _ in range(500):
        res = create_brain_checkpoint_delta(db_file, since_timestamp="2026-10-25T00:00:00+00:00")
        hashes.add(res["delta_hash"])

    assert len(hashes) == 1
    assert baseline_hash in hashes

    # Verify sensitivity: any single byte alteration changes the hash
    corrupted_records = json.loads(json.dumps(base_delta["records"]))
    corrupted_records["conversations"][0]["file_size"] += 1
    corrupted_canonical = json.dumps(corrupted_records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    corrupted_hash = hashlib.sha256(corrupted_canonical.encode("utf-8")).hexdigest()
    assert corrupted_hash != baseline_hash
