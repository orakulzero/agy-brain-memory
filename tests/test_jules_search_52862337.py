import json
import sqlite3
from pathlib import Path

import pytest

from agy_brain_memory.search import (
    sanitize_fts5_query,
    search_diff_tours,
    search_capa_records,
    search_brain,
    get_subagent_lineage,
    get_conversation_summary,
    analyze_subagent_graph_topology,
)

# --- sanitize_fts5_query tests ---

def test_sanitize_fts5_query_empty():
    assert sanitize_fts5_query("") == '""'

def test_sanitize_fts5_query_only_special_chars():
    assert sanitize_fts5_query("!@#$%^&*()") == '""'

def test_sanitize_fts5_query_none():
    assert sanitize_fts5_query(None) == '""'

def test_sanitize_fts5_query_normal():
    assert sanitize_fts5_query("hello world") == '"hello" "world"'

def test_sanitize_fts5_query_mixed():
    assert sanitize_fts5_query("hello! @world#") == '"hello" "world"'

def test_sanitize_fts5_query_numbers():
    assert sanitize_fts5_query("123 456") == '"123" "456"'


# --- database fixtures ---

@pytest.fixture
def mock_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE conversations (
            conv_id TEXT PRIMARY KEY,
            step_count INTEGER,
            last_status TEXT,
            has_bigint_error INTEGER
        );
        INSERT INTO conversations (conv_id, step_count, last_status, has_bigint_error)
        VALUES ('conv1', 5, 'COMPLETED', 0);
        INSERT INTO conversations (conv_id, step_count, last_status, has_bigint_error)
        VALUES ('conv2', 2, 'IN_PROGRESS', 1);

        CREATE VIRTUAL TABLE diff_tours_fts USING fts5(conv_id, tour_id, col2, col3, col4, content);
        CREATE TABLE diff_tours (
            conv_id TEXT, tour_id TEXT, filename TEXT, title TEXT, description TEXT,
            step_count INTEGER, dimensions_json TEXT, content_json TEXT
        );
        INSERT INTO diff_tours_fts (conv_id, tour_id, content) VALUES ('conv1', 't1', 'hello world');
        INSERT INTO diff_tours (conv_id, tour_id, filename, title, description, step_count, dimensions_json, content_json)
        VALUES ('conv1', 't1', 'f1.py', 'title1', 'desc1', 1, '["dim1", "dim2"]', '{}');

        CREATE VIRTUAL TABLE capa_records_fts USING fts5(conv_id, record_id, col2, col3, content);
        CREATE TABLE capa_records (
            conv_id TEXT, record_type TEXT, record_id TEXT, title TEXT, details_json TEXT, created_at TEXT
        );
        INSERT INTO capa_records_fts (conv_id, record_id, content) VALUES ('conv1', 'r1', 'capa content hello');
        INSERT INTO capa_records (conv_id, record_type, record_id, title, details_json, created_at)
        VALUES ('conv1', 'type1', 'r1', 'title1', '{}', '2023-01-01');

        CREATE VIRTUAL TABLE steps_fts USING fts5(conv_id, step_index, type, status, content);
        CREATE TABLE steps (
            conv_id TEXT, step_index INTEGER, type TEXT, status TEXT, content TEXT
        );
        INSERT INTO steps_fts (conv_id, step_index, type, status, content) VALUES ('conv1', 1, 'USER_INPUT', 'OK', 'step content hello');
        INSERT INTO steps (conv_id, step_index, type, status, content)
        VALUES ('conv1', 1, 'USER_INPUT', 'OK', 'user request 1');

        CREATE VIRTUAL TABLE artifacts_fts USING fts5(conv_id, filename, content);
        CREATE TABLE artifacts (
            conv_id TEXT, filename TEXT, content TEXT
        );
        INSERT INTO artifacts_fts (conv_id, filename, content) VALUES ('conv1', 'art1.md', 'artifact hello');
        INSERT INTO artifacts (conv_id, filename, content)
        VALUES ('conv1', 'art1.md', 'artifact hello');

        CREATE TABLE subagent_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_conv_id TEXT,
            child_conv_id TEXT,
            step_index INTEGER,
            role TEXT,
            type_name TEXT,
            prompt TEXT,
            child_status TEXT,
            child_outcome_summary TEXT
        );
        INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
        VALUES ('conv1', 'conv2', 1, 'researcher', 'typeA', 'do research', 'DONE', 'summary');
        INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
        VALUES ('conv2', 'conv3', 1, 'coder', 'typeB', 'write code', 'DONE', 'summary');
        INSERT INTO subagent_edges (parent_conv_id, child_conv_id, step_index, role, type_name, prompt, child_status, child_outcome_summary)
        VALUES ('conv3', 'conv1', 1, 'tester', 'typeC', 'test code', 'DONE', 'summary'); -- creates a cycle
        """
    )
    conn.commit()
    conn.close()
    return db_path

# --- search tests ---

def test_search_diff_tours(mock_db):
    res = search_diff_tours(mock_db, "hello")
    assert len(res["results"]) == 1
    assert res["results"][0]["conv_id"] == "conv1"
    assert res["results"][0]["tour_id"] == "t1"

def test_search_diff_tours_empty_query(mock_db):
    res = search_diff_tours(mock_db, "!@#")
    assert len(res["results"]) == 0
    assert res["sanitized_query"] == '""'

def test_search_diff_tours_invalid_db():
    res = search_diff_tours("invalid.db", "hello")
    assert len(res["results"]) == 0

def test_search_diff_tours_dimension_filter(mock_db):
    res = search_diff_tours(mock_db, "hello", dimension="dim1")
    assert len(res["results"]) == 1

    res = search_diff_tours(mock_db, "hello", dimension="dim3")
    assert len(res["results"]) == 0

def test_search_capa_records(mock_db):
    res = search_capa_records(mock_db, "hello")
    assert len(res["results"]) == 1
    assert res["results"][0]["record_id"] == "r1"

def test_search_capa_records_empty_query(mock_db):
    res = search_capa_records(mock_db, "!@#")
    assert len(res["results"]) == 0

def test_search_capa_records_invalid_db():
    res = search_capa_records("invalid.db", "hello")
    assert len(res["results"]) == 0

def test_search_capa_records_type_filter(mock_db):
    res = search_capa_records(mock_db, "hello", record_type="type1")
    assert len(res["results"]) == 1

    res = search_capa_records(mock_db, "hello", record_type="type2")
    assert len(res["results"]) == 0

def test_search_brain(mock_db):
    res = search_brain(mock_db, "hello")
    assert len(res["results"]) == 1
    assert len(res["artifact_results"]) == 1
    assert "diff_tour_results" not in res

    res = search_brain(mock_db, "hello", include_tours=True, include_capa=True)
    assert len(res["diff_tour_results"]) == 1
    assert len(res["capa_results"]) == 1

def test_search_brain_empty_query(mock_db):
    res = search_brain(mock_db, "!@#", include_tours=True, include_capa=True)
    assert len(res["results"]) == 0
    assert len(res["artifact_results"]) == 0
    assert len(res["diff_tour_results"]) == 0
    assert len(res["capa_results"]) == 0

def test_search_brain_invalid_db():
    res = search_brain("invalid.db", "hello", include_tours=True, include_capa=True)
    assert len(res["results"]) == 0
    assert len(res["artifact_results"]) == 0
    assert len(res["diff_tour_results"]) == 0
    assert len(res["capa_results"]) == 0

def test_search_brain_step_type_filter(mock_db):
    res = search_brain(mock_db, "hello", step_type="USER_INPUT")
    assert len(res["results"]) == 1

    res = search_brain(mock_db, "hello", step_type="AI_OUTPUT")
    assert len(res["results"]) == 0

# --- graph tests ---

def test_get_subagent_lineage(mock_db):
    res = get_subagent_lineage(mock_db, conv_id="conv1")
    assert res["conv_id"] == "conv1"
    assert len(res["spawned_children"]) == 1
    assert res["spawned_children"][0]["child_conv_id"] == "conv2"
    assert len(res["descendants"]) >= 2 # conv2, conv3, conv1
    assert len(res["edges"]) == 2 # 1->2, 3->1

def test_get_subagent_lineage_invalid_db():
    res = get_subagent_lineage("invalid.db")
    assert len(res["spawned_children"]) == 0

def test_get_subagent_lineage_role_query(mock_db):
    res = get_subagent_lineage(mock_db, role_query="researcher")
    assert len(res["edges"]) == 1
    assert res["edges"][0]["role"] == "researcher"

def test_get_conversation_summary(mock_db):
    res = get_conversation_summary(mock_db, "conv1")
    assert res["conv_id"] == "conv1"
    assert res["step_count"] == 5
    assert res["last_status"] == "COMPLETED"
    assert res["has_bigint_error"] is False
    assert len(res["user_requests"]) == 1
    assert len(res["artifacts"]) == 1
    assert len(res["spawned_subagents"]) == 1

    res2 = get_conversation_summary(mock_db, "conv2")
    assert res2["step_count"] == 2
    assert res2["last_status"] == "IN_PROGRESS"
    assert res2["has_bigint_error"] is True

def test_get_conversation_summary_invalid_db():
    res = get_conversation_summary("invalid.db", "conv1")
    assert res["step_count"] == 0
    assert res["last_status"] == "UNKNOWN"

def test_analyze_subagent_graph_topology(mock_db):
    res = analyze_subagent_graph_topology(mock_db)
    assert res["total_nodes"] == 3 # conv1, conv2, conv3
    assert res["max_fanout"] == 1
    assert res["has_cycles"] is True
    assert "researcher" in res["roles_distribution"]
    assert "coder" in res["roles_distribution"]

def test_analyze_subagent_graph_topology_root(mock_db):
    res = analyze_subagent_graph_topology(mock_db, root_conv_id="conv1")
    assert res["total_nodes"] == 3
    assert res["has_cycles"] is True
    
def test_analyze_subagent_graph_topology_invalid_db():
    res = analyze_subagent_graph_topology("invalid.db")
    assert res["total_nodes"] == 0
