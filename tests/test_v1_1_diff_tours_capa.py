"""Tests for diff-tour v1 and ISO 19011 / CAPA indexing and full-text search."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from agy_brain_memory.indexer import BrainMemoryIndexer
from agy_brain_memory.search import (
    search_brain,
    search_capa_records,
    search_diff_tours,
)


def test_index_and_search_diff_tours(tmp_path: Path) -> None:
    brain_dir = tmp_path / "brain"
    brain_dir.mkdir(parents=True, exist_ok=True)
    conv_dir = brain_dir / "conv_tour_test"
    conv_dir.mkdir()

    # Create a tour JSON file inside conversation directory
    tour_data = {
        "id": "tour_senior_audit",
        "title": "Senior Architecture Review",
        "description": "Guided walkthrough of security boundaries and failure modes",
        "steps": [
            {
                "file": "src/security.py",
                "line_start": 10,
                "line_end": 25,
                "dimension": "L4_TRUST",
                "epistemic_basis": "fact",
                "description": "HMAC Gate 1 cryptographic validation.",
            },
            {
                "file": "src/resilience.py",
                "line_start": 30,
                "line_end": 50,
                "dimension": "L2_FAILURE",
                "epistemic_basis": "inferred",
                "description": "Circuit breaker timeout handling.",
            },
        ],
    }
    tour_file = conv_dir / "review.tour.json"
    tour_file.write_text(json.dumps(tour_data), encoding="utf-8")

    db_path = tmp_path / "brain_memory.db"
    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    sync_res = indexer.sync()

    assert sync_res["diff_tours_indexed"] == 1

    # Search for cryptographic
    res_trust = search_diff_tours(db_path, "cryptographic")
    assert len(res_trust["results"]) == 1
    hit = res_trust["results"][0]
    assert hit["tour_id"] == "tour_senior_audit"
    assert hit["title"] == "Senior Architecture Review"
    assert "L4_TRUST" in hit["dimensions"]
    assert "L2_FAILURE" in hit["dimensions"]

    # Filter by dimension L4_TRUST
    res_dim = search_diff_tours(db_path, "Review", dimension="L4_TRUST")
    assert len(res_dim["results"]) == 1

    # Filter by non-existent dimension L5_SCALE
    res_scale = search_diff_tours(db_path, "Review", dimension="L5_SCALE")
    assert len(res_scale["results"]) == 0

    # Search via search_brain with include_tours=True
    brain_res = search_brain(db_path, "cryptographic", include_tours=True)
    assert len(brain_res["diff_tour_results"]) == 1


def test_index_and_search_capa_records(tmp_path: Path) -> None:
    brain_dir = tmp_path / "brain"
    brain_dir.mkdir(parents=True, exist_ok=True)
    conv_dir = brain_dir / "conv_capa_test"
    conv_dir.mkdir()

    # Create evidence units and failure ledger
    sg_dir = conv_dir / ".supergraph"
    sg_dir.mkdir()

    ev_file = sg_dir / "evidence_units.jsonl"
    ev_file.write_text(
        json.dumps({
            "unit_id": "EV-001",
            "command": "pytest tests/test_crypto.py",
            "exit_code": 0,
            "stdout_sha256": "abcdef1234567890",
        }) + "\n",
        encoding="utf-8",
    )

    fail_file = sg_dir / "failure_ledger.jsonl"
    fail_file.write_text(
        json.dumps({
            "node_id": "TEST_SYNTHESIS",
            "reason": "SubprocessTimeoutExpired after 120 seconds",
            "exit_code": 124,
        }) + "\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "brain_memory.db"
    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    sync_res = indexer.sync()

    assert sync_res["capa_records_indexed"] == 2

    # Search evidence units
    res_ev = search_capa_records(db_path, "pytest", record_type="evidence_unit")
    assert len(res_ev["results"]) == 1
    assert res_ev["results"][0]["record_id"] == "EV-001"

    # Search failure ledger
    res_fail = search_capa_records(db_path, "SubprocessTimeoutExpired")
    assert len(res_fail["results"]) == 1
    assert res_fail["results"][0]["record_id"] == "TEST_SYNTHESIS"

    # Search via search_brain with include_capa=True
    brain_res = search_brain(db_path, "SubprocessTimeoutExpired", include_capa=True)
    assert len(brain_res["capa_results"]) == 1


def test_programmatic_indexing_api(tmp_path: Path) -> None:
    db_path = tmp_path / "brain_direct.db"
    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=tmp_path / "empty")

    tour_id = indexer.index_diff_tour(
        conv_id="conv_prog",
        tour_data={
            "id": "tour_direct_1",
            "title": "Direct Index Tour",
            "description": "Programmatic tour insertion test",
            "steps": [
                {
                    "file": "direct.py",
                    "line_start": 1,
                    "line_end": 5,
                    "dimension": "L7_TESTS",
                    "epistemic_basis": "fact",
                    "description": "Direct testing assertion.",
                }
            ],
        },
    )
    assert tour_id > 0

    capa_id = indexer.index_capa_record(
        conv_id="conv_prog",
        record_type="capa_rule",
        record_id="RULE-99",
        title="Always enforce non-zero timeouts",
        details={"weight": 1.5, "target": "network_sockets"},
    )
    assert capa_id > 0

    t_res = search_diff_tours(db_path, "Programmatic")
    assert len(t_res["results"]) == 1

    c_res = search_capa_records(db_path, "timeouts", record_type="capa_rule")
    assert len(c_res["results"]) == 1
