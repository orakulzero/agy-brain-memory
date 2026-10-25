"""v1.2.0 Subagent Lineage Topology, Mermaid Exporter, and Checkpoint Delta Tests."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3

from agy_brain_memory import (
    analyze_subagent_graph_topology,
    create_brain_checkpoint_delta,
    export_lineage_mermaid,
)
from agy_brain_memory.indexer import _connect_db


def _init_test_db(db_path: Path) -> sqlite3.Connection:
    conn = _connect_db(db_path)
    with conn:
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
    return conn


def test_subagent_graph_topology_tree_metrics(tmp_path: Path) -> None:
    db_file = tmp_path / "topology_tree.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        conn.executemany(
            """
            INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
            VALUES (?, ?, ?, ?, 'self', 'test prompt', 'DONE', 'summary')
            """,
            [
                ("conv_root", "conv_child_1", 1, "Planner"),
                ("conv_child_1", "conv_child_2", 2, "Explorer"),
                ("conv_child_2", "conv_child_3", 3, "Implementer"),
            ],
        )
    conn.close()

    res = analyze_subagent_graph_topology(db_file)
    assert res["total_nodes"] == 4
    assert res["max_depth"] == 3
    assert res["max_fanout"] == 1
    assert res["orphan_nodes"] == ["conv_root"]
    assert res["has_cycles"] is False
    assert res["roles_distribution"] == {
        "Planner": 1,
        "Explorer": 1,
        "Implementer": 1,
    }
    assert res["root_conv_id"] is None


def test_subagent_graph_topology_wide_fanout(tmp_path: Path) -> None:
    db_file = tmp_path / "topology_fanout.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        edges = [
            ("conv_parent", f"conv_child_{i}", i, "Worker", "self", "prompt", "DONE", "ok")
            for i in range(5)
        ]
        conn.executemany(
            """
            INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            edges,
        )
    conn.close()

    res = analyze_subagent_graph_topology(db_file)
    assert res["total_nodes"] == 6
    assert res["max_depth"] == 1
    assert res["max_fanout"] == 5
    assert res["orphan_nodes"] == ["conv_parent"]
    assert res["has_cycles"] is False
    assert res["roles_distribution"] == {"Worker": 5}


def test_subagent_graph_topology_detects_cycles(tmp_path: Path) -> None:
    db_file = tmp_path / "topology_cycle.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        conn.executemany(
            """
            INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
            VALUES (?, ?, ?, ?, 'self', 'cycle test', 'DONE', 'ok')
            """,
            [
                ("node_alpha", "node_beta", 1, "RoleA"),
                ("node_beta", "node_gamma", 2, "RoleB"),
                ("node_gamma", "node_alpha", 3, "RoleC"),
            ],
        )
    conn.close()

    res = analyze_subagent_graph_topology(db_file)
    assert res["has_cycles"] is True
    assert res["total_nodes"] == 3
    assert res["max_depth"] >= 2
    assert res["orphan_nodes"] == []


def test_subagent_graph_topology_orphan_detection(tmp_path: Path) -> None:
    db_file = tmp_path / "topology_orphans.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        conn.executemany(
            """
            INSERT INTO conversations (conv_id, mtime_ns, file_size, step_count, last_status, has_bigint_error, updated_at)
            VALUES (?, 1000, 100, 5, 'DONE', 0, '2026-10-25T10:00:00Z')
            """,
            [("conv_isolated_1",), ("conv_isolated_2",)],
        )
        conn.executemany(
            """
            INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
            VALUES (?, ?, 1, 'Worker', 'self', 'p', 'DONE', 'ok')
            """,
            [
                ("tree1_root", "tree1_child"),
                ("tree2_root", "tree2_child"),
            ],
        )
    conn.close()

    res = analyze_subagent_graph_topology(db_file)
    assert res["total_nodes"] == 6
    assert res["orphan_nodes"] == [
        "conv_isolated_1",
        "conv_isolated_2",
        "tree1_root",
        "tree2_root",
    ]
    assert res["has_cycles"] is False


def test_subagent_graph_topology_root_filtering(tmp_path: Path) -> None:
    db_file = tmp_path / "topology_filter.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        conn.executemany(
            """
            INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
            VALUES (?, ?, ?, 'Role', 'self', 'p', 'DONE', 'ok')
            """,
            [
                ("root_A", "child_A1", 1),
                ("child_A1", "child_A2", 2),
                ("root_B", "child_B1", 1),
            ],
        )
    conn.close()

    res_a = analyze_subagent_graph_topology(db_file, root_conv_id="root_A")
    assert res_a["total_nodes"] == 3
    assert res_a["max_depth"] == 2
    assert res_a["root_conv_id"] == "root_A"
    assert res_a["orphan_nodes"] == ["root_A"]

    res_b = analyze_subagent_graph_topology(db_file, root_conv_id="root_B")
    assert res_b["total_nodes"] == 2
    assert res_b["max_depth"] == 1
    assert res_b["root_conv_id"] == "root_B"


def test_export_lineage_mermaid_structure_and_links(tmp_path: Path) -> None:
    db_file = tmp_path / "mermaid.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        conn.execute(
            """
            INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
            VALUES ('parent_99', 'child_88', 4, 'Code-Auditor', 'self', 'audit', 'DONE', 'ok')
            """
        )
    conn.close()

    mermaid_td = export_lineage_mermaid(db_file, direction="TD")
    assert mermaid_td.startswith("flowchart TD")
    assert "conversation://parent_99" in mermaid_td
    assert "conversation://child_88" in mermaid_td
    assert "[root](conversation://parent_99)" in mermaid_td
    assert "[Code-Auditor](conversation://child_88)" in mermaid_td
    assert "step 4: Code-Auditor" in mermaid_td

    mermaid_lr = export_lineage_mermaid(db_file, direction="LR")
    assert mermaid_lr.startswith("flowchart LR")


def test_export_lineage_mermaid_with_root_filtering(tmp_path: Path) -> None:
    db_file = tmp_path / "mermaid_filter.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        conn.executemany(
            """
            INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
            VALUES (?, ?, 1, 'Worker', 'self', 'p', 'DONE', 'ok')
            """,
            [
                ("tree_alpha", "child_alpha"),
                ("tree_beta", "child_beta"),
            ],
        )
    conn.close()

    mermaid = export_lineage_mermaid(db_file, root_conv_id="tree_alpha")
    assert "conversation://tree_alpha" in mermaid
    assert "conversation://child_alpha" in mermaid
    assert "tree_beta" not in mermaid
    assert "child_beta" not in mermaid


def test_create_brain_checkpoint_delta_incremental_window(tmp_path: Path) -> None:
    db_file = tmp_path / "delta_window.sqlite"
    conn = _init_test_db(db_file)
    t_old_ns = int(datetime.fromisoformat("2026-10-25T08:00:00+00:00").timestamp() * 1_000_000_000)
    t_new_ns = int(datetime.fromisoformat("2026-10-25T14:00:00+00:00").timestamp() * 1_000_000_000)
    with conn:
        conn.executemany(
            """
            INSERT INTO conversations (conv_id, mtime_ns, file_size, step_count, last_status, has_bigint_error, updated_at)
            VALUES (?, ?, 200, 10, 'DONE', 0, ?)
            """,
            [
                ("conv_old", t_old_ns, "2026-10-25T08:00:00+00:00"),
                ("conv_new", t_new_ns, "2026-10-25T14:00:00+00:00"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO diff_tours (conv_id, tour_id, filename, title, description, step_count, dimensions_json, content_json, mtime_ns)
            VALUES (?, ?, 'tour.json', 'Tour Title', 'Tour Desc', 2, '["L1"]', '{}', ?)
            """,
            [
                ("conv_old", "tour_old", t_old_ns),
                ("conv_new", "tour_new", t_new_ns),
            ],
        )
        conn.executemany(
            """
            INSERT INTO capa_records (conv_id, record_type, record_id, title, details_json, created_at)
            VALUES (?, 'evidence_unit', ?, 'Title', '{}', ?)
            """,
            [
                ("conv_old", "capa_old", "2026-10-25T08:00:00+00:00"),
                ("conv_new", "capa_new", "2026-10-25T14:00:00+00:00"),
            ],
        )
    conn.close()

    res = create_brain_checkpoint_delta(db_file, since_timestamp="2026-10-25T10:00:00+00:00")
    assert res["record_counts"]["conversations"] == 1
    assert res["record_counts"]["diff_tours"] == 1
    assert res["record_counts"]["capa_records"] == 1
    assert res["record_counts"]["total"] == 3

    assert res["records"]["conversations"][0]["conv_id"] == "conv_new"
    assert res["records"]["diff_tours"][0]["tour_id"] == "tour_new"
    assert res["records"]["capa_records"][0]["record_id"] == "capa_new"
    assert res["zero_pii"] is True
    assert len(res["delta_hash"]) == 64


def test_create_brain_checkpoint_delta_deterministic_sha256_and_zero_pii(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "delta_pii.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        conn.execute(
            """
            INSERT INTO conversations (conv_id, mtime_ns, file_size, step_count, last_status, has_bigint_error, updated_at)
            VALUES ('conv_pii', 3000, 100, 2, 'DONE', 0, '2026-10-25T12:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO diff_tours (conv_id, tour_id, filename, title, description, step_count, dimensions_json, content_json, mtime_ns)
            VALUES ('conv_pii', 'tour_pii', 'tour.json', 'Dev audit for user@secretcorp.com', 'File located at D:\\\\Private\\\\sample.py', 1, '[]', '{}', 3000)
            """
        )
        conn.execute(
            """
            INSERT INTO capa_records (conv_id, record_type, record_id, title, details_json, created_at)
            VALUES ('conv_pii', 'evidence_unit', 'EV-01', 'Report for engineer@corp.org', '{"user_path": "C:\\\\Users\\\\TestUser\\\\repo"}', '2026-10-25T12:00:00+00:00')
            """
        )
    conn.close()

    export_dir = tmp_path / "exports"
    res1 = create_brain_checkpoint_delta(db_file, export_dir=export_dir)
    res2 = create_brain_checkpoint_delta(db_file)

    assert res1["delta_hash"] == res2["delta_hash"]
    assert res1["zero_pii"] is True

    exported_files = list(export_dir.glob("checkpoint_delta_*.json"))
    assert len(exported_files) == 1
    content = exported_files[0].read_text(encoding="utf-8")
    assert "user@secretcorp.com" not in content
    assert "engineer@corp.org" not in content
    assert "[REDACTED_EMAIL]" in content


def test_v1_2_empty_and_missing_db_edge_cases(tmp_path: Path) -> None:
    missing_db = tmp_path / "does_not_exist.sqlite"

    topo = analyze_subagent_graph_topology(missing_db)
    assert topo["total_nodes"] == 0
    assert topo["max_depth"] == 0
    assert topo["orphan_nodes"] == []
    assert topo["has_cycles"] is False

    mermaid = export_lineage_mermaid(missing_db)
    assert mermaid.startswith("flowchart TD")

    delta = create_brain_checkpoint_delta(missing_db)
    assert delta["record_counts"]["total"] == 0
    assert delta["records"]["conversations"] == []
    assert delta["zero_pii"] is True
    assert len(delta["delta_hash"]) == 64
