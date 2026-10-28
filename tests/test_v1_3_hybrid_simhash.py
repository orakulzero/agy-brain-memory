"""Unit and integration test suite for agy-brain-memory v1.3.0:
HybridBrainRetriever (BM25 + 3/4-gram TF-IDF cosine with RRF k=60),
hybrid_search_memory, 64-bit Charikar SimHash, and near-duplicate step deduplication.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Sequence, Tuple, Union

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe dynamic import resolution for Agile V TDD test collection
try:
    from agy_brain_memory import (
        HybridBrainRetriever,
        __version__,
        compute_simhash64,
        deduplicate_near_identical_steps,
        hybrid_search_memory,
    )
except ImportError:
    try:
        from agy_brain_memory.search import HybridBrainRetriever, hybrid_search_memory
        from agy_brain_memory.compactor import (
            compute_simhash64,
            deduplicate_near_identical_steps,
        )
        from agy_brain_memory import __version__
    except ImportError:
        HybridBrainRetriever = None
        hybrid_search_memory = None
        compute_simhash64 = None
        deduplicate_near_identical_steps = None
        __version__ = "1.2.0"


@pytest.fixture
def sample_corpus() -> List[Dict[str, str]]:
    """Standard multi-domain corpus for memory retrieval tests."""
    return [
        {
            "id": "doc_topology",
            "content": "Subagent call graph topology analyzer with cycle detection in SQLite WAL",
        },
        {
            "id": "doc_merkle",
            "content": "Mandatory RFC 6962 Merkle tree audit trail verification and domain separation",
        },
        {
            "id": "doc_deadlock",
            "content": "Single process hook multiplexer and Windows worktree deadlock healer",
        },
        {
            "id": "doc_cad",
            "content": "CAD blueprint normocontrol and 300 DPI rasterization pipeline",
        },
        {
            "id": "doc_secrets",
            "content": "Autonomous secret shield with Windows DPAPI quarantine and entropy scanner",
        },
    ]


def _init_test_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
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


# ============================================================================
# 1. HybridBrainRetriever Tests
# ============================================================================


def test_v1_3_retriever_exact_keyword_retrieval(sample_corpus: List[Dict[str, str]]) -> None:
    """Verify exact keyword query retrieves the exact document at rank 1 with positive scores."""
    assert HybridBrainRetriever is not None, "HybridBrainRetriever must be implemented and exported"
    retriever = HybridBrainRetriever()
    retriever.index_documents(sample_corpus)

    results = retriever.search("RFC 6962 Merkle tree audit", top_k=3)
    assert len(results) > 0
    top_hit = results[0]

    assert top_hit["id"] == "doc_merkle"
    assert top_hit["score"] > 0.0
    assert top_hit["bm25_score"] > 0.0
    assert top_hit["semantic_score"] > 0.0
    assert "RFC 6962 Merkle tree" in top_hit["content"]


def test_v1_3_retriever_typo_recovery_deadlock(sample_corpus: List[Dict[str, str]]) -> None:
    """Verify typographical errors (e.g. 'deadlok') recover the target document via subword N-grams."""
    assert HybridBrainRetriever is not None, "HybridBrainRetriever must be implemented and exported"
    retriever = HybridBrainRetriever()
    retriever.index_documents(sample_corpus)

    results = retriever.search("deadlok healer", top_k=3)
    assert len(results) > 0
    top_hit = results[0]

    assert top_hit["id"] == "doc_deadlock"
    assert top_hit["semantic_score"] > 0.05
    assert "deadlock healer" in top_hit["content"].lower()


def test_v1_3_retriever_morphological_variants(sample_corpus: List[Dict[str, str]]) -> None:
    """Verify inflected and morphological variants successfully retrieve target documents."""
    assert HybridBrainRetriever is not None, "HybridBrainRetriever must be implemented and exported"
    retriever = HybridBrainRetriever()
    retriever.index_documents(sample_corpus)

    results = retriever.search("normocontrols rasterizing", top_k=2)
    assert len(results) > 0
    assert results[0]["id"] == "doc_cad"
    assert results[0]["semantic_score"] > 0.0


def test_v1_3_retriever_empty_corpus_and_queries(sample_corpus: List[Dict[str, str]]) -> None:
    """Verify empty corpus or empty/whitespace queries return empty lists without raising exceptions."""
    assert HybridBrainRetriever is not None, "HybridBrainRetriever must be implemented and exported"
    retriever = HybridBrainRetriever()
    assert retriever.search("any query") == []

    retriever.index_documents([])
    assert retriever.search("any query") == []

    retriever.index_documents(sample_corpus)
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("\r\n\t") == []


def test_v1_3_retriever_top_k_bounds_and_ranking_order(sample_corpus: List[Dict[str, str]]) -> None:
    """Verify top_k constraints, negative values, and strict descending rank order."""
    assert HybridBrainRetriever is not None, "HybridBrainRetriever must be implemented and exported"
    retriever = HybridBrainRetriever()
    retriever.index_documents(sample_corpus)

    assert retriever.search("pipeline", top_k=0) == []
    assert retriever.search("pipeline", top_k=-3) == []

    hits_2 = retriever.search("pipeline", top_k=2)
    assert len(hits_2) <= 2
    if len(hits_2) > 1:
        assert hits_2[0]["score"] >= hits_2[1]["score"]

    hits_oversized = retriever.search("pipeline", top_k=100)
    assert len(hits_oversized) <= len(sample_corpus)


def test_v1_3_retriever_string_and_dict_documents() -> None:
    """Verify retriever accepts both raw string sequences and structured dictionaries."""
    assert HybridBrainRetriever is not None, "HybridBrainRetriever must be implemented and exported"
    raw_texts = [
        "First log of autonomous agent session",
        "Second trace of compiler diagnostics",
        "Third record of network latency metrics",
    ]
    retriever = HybridBrainRetriever()
    retriever.index_documents(raw_texts)

    results = retriever.search("compiler diagnostics", top_k=1)
    assert len(results) == 1
    assert "compiler diagnostics" in results[0]["content"]


# ============================================================================
# 2. hybrid_search_memory Helper Tests
# ============================================================================


def test_v1_3_hybrid_search_memory_in_memory_sequence(sample_corpus: List[Dict[str, str]]) -> None:
    """Verify hybrid_search_memory helper function works directly on document sequences."""
    assert hybrid_search_memory is not None, "hybrid_search_memory must be implemented and exported"
    hits = hybrid_search_memory(sample_corpus, query="secret shield DPAPI", top_k=2)
    assert len(hits) > 0
    assert hits[0]["id"] == "doc_secrets"
    assert "DPAPI" in hits[0]["content"]


def test_v1_3_hybrid_search_memory_sqlite_db_integration(tmp_path: Path) -> None:
    """Verify hybrid_search_memory extracts records from a SQLite database and ranks them."""
    assert hybrid_search_memory is not None, "hybrid_search_memory must be implemented and exported"
    db_file = tmp_path / "test_brain_search.sqlite"
    conn = _init_test_db(db_file)
    with conn:
        conn.execute(
            """
            INSERT INTO diff_tours (conv_id, tour_id, filename, title, description, step_count, dimensions_json, content_json, mtime_ns)
            VALUES ('conv_01', 'tour_deadlock', 'tour.json', 'Deadlock Healer Tour', 'Automatic unfreezing of git worktree locks', 3, '["L2"]', '{}', 1000)
            """
        )
        conn.execute(
            """
            INSERT INTO capa_records (conv_id, record_type, record_id, title, details_json, created_at)
            VALUES ('conv_01', 'capa_merkle', 'CAPA-001', 'Merkle Audit Log', '{"proof": "valid"}', '2026-10-26T12:00:00Z')
            """
        )
    conn.close()

    hits = hybrid_search_memory(db_file, query="deadlock unfreezing", top_k=2)
    assert len(hits) > 0
    top_content = str(hits[0].get("content") or hits[0].get("title", ""))
    assert "deadlock" in top_content.lower()


# ============================================================================
# 3. 64-bit Charikar SimHash Tests
# ============================================================================


def test_v1_3_compute_simhash64_deterministic_and_bounds() -> None:
    """Verify compute_simhash64 returns a 64-bit unsigned integer with 100% determinism."""
    assert compute_simhash64 is not None, "compute_simhash64 must be implemented and exported"
    sample_text = "Autonomous subagent graph execution with SQLite WAL storage."
    h1 = compute_simhash64(sample_text)
    h2 = compute_simhash64(sample_text)

    assert isinstance(h1, int)
    assert 0 <= h1 < (1 << 64)
    assert h1 == h2

    # Empty string handling
    h_empty = compute_simhash64("")
    assert isinstance(h_empty, int)
    assert 0 <= h_empty < (1 << 64)


def test_v1_3_compute_simhash64_hamming_distance_sensitivity() -> None:
    """Verify small textual deltas produce low Hamming distance while different texts produce high distance."""
    assert compute_simhash64 is not None, "compute_simhash64 must be implemented and exported"
    text_base = "Execute task step one with compiler diagnostic validation"
    text_near = "Execute task step one with compiler diagnostic verification"
    text_far = "Prepare gourmet espresso with steamed almond milk and vanilla beans"

    h_base = compute_simhash64(text_base)
    h_near = compute_simhash64(text_near)
    h_far = compute_simhash64(text_far)

    dist_near = bin(h_base ^ h_near).count("1")
    dist_far = bin(h_base ^ h_far).count("1")

    # Near duplicates must have significantly lower Hamming distance than unrelated texts
    assert dist_near <= 10
    assert dist_far >= 15
    assert dist_near < dist_far


# ============================================================================
# 4. Near-Duplicate Step Deduplication Tests
# ============================================================================


def test_v1_3_deduplicate_near_identical_steps_collapses_duplicates() -> None:
    """Verify deduplicate_near_identical_steps eliminates rapid redundant agent status loops."""
    assert (
        deduplicate_near_identical_steps is not None
    ), "deduplicate_near_identical_steps must be implemented and exported"

    steps = [
        {"id": 1, "content": "Waiting for background task compiler build to complete iteration 1"},
        {"id": 2, "content": "Waiting for background task compiler build to complete iteration 2"},
        {"id": 3, "content": "Waiting for background task compiler build to complete iteration 3"},
        {"id": 4, "content": "Compiler build finished successfully with 0 warnings"},
        {"id": 5, "content": "Executing test suite across all verified package targets"},
    ]

    res = deduplicate_near_identical_steps(steps, max_hamming_distance=6)
    if isinstance(res, tuple):
        deduped_steps, telemetry = res
        assert isinstance(telemetry, dict)
    else:
        deduped_steps = res

    # The 3 waiting steps should be collapsed into the first one
    assert len(deduped_steps) < len(steps)
    assert deduped_steps[0]["id"] == 1
    contents = [s["content"] for s in deduped_steps]
    assert any("finished successfully" in c for c in contents)
    assert any("Executing test suite" in c for c in contents)


def test_v1_3_deduplicate_near_identical_steps_custom_key_and_idempotency() -> None:
    """Verify custom content_key handling, empty lists, and idempotency."""
    assert (
        deduplicate_near_identical_steps is not None
    ), "deduplicate_near_identical_steps must be implemented and exported"

    # Empty list
    res_empty = deduplicate_near_identical_steps([])
    deduped_empty = res_empty[0] if isinstance(res_empty, tuple) else res_empty
    assert deduped_empty == []

    # Custom key
    steps_custom = [
        {"msg": "Syncing plugin repository manifests"},
        {"msg": "Syncing plugin repository manifests now"},
        {"msg": "Unrelated diagnostic report generated"},
    ]
    res_custom = deduplicate_near_identical_steps(
        steps_custom, max_hamming_distance=4, content_key="msg"
    )
    deduped_custom = res_custom[0] if isinstance(res_custom, tuple) else res_custom
    assert len(deduped_custom) == 2

    # Idempotency
    res_again = deduplicate_near_identical_steps(
        deduped_custom, max_hamming_distance=4, content_key="msg"
    )
    deduped_again = res_again[0] if isinstance(res_again, tuple) else res_again
    assert deduped_again == deduped_custom


def test_v1_3_package_exports_and_version() -> None:
    """Verify v1.3.0 version bump and public API surface exports."""
    assert __version__ in ("1.3.0", "2.0.0"), "Package __version__ must be bumped to 2.0.0"
    import agy_brain_memory

    expected_exports = [
        "HybridBrainRetriever",
        "hybrid_search_memory",
        "compute_simhash64",
        "deduplicate_near_identical_steps",
    ]
    for export in expected_exports:
        assert hasattr(agy_brain_memory, export), f"{export} must be exported in agy_brain_memory"
        assert export in agy_brain_memory.__all__, f"{export} must be included in __all__"
