"""Locked Agile-V verification tests for SMT State Invariant Guard (REQ-BM-SMT-01)."""

from __future__ import annotations

import copy
from agy_brain_memory.smt_state_invariant_guard import (
    SMTStateInvariantGuard,
    StateInvariantRule,
)


def test_compile_to_smtlib2_generates_valid_assertions() -> None:
    """REQ-BM-SMT-01: Verify compile_to_smtlib2 generates valid SMT-LIB2 QF_LIA assertions."""
    guard = SMTStateInvariantGuard()
    pre_state = {"version": 1, "node_count": 10, "orphan_edges": 0, "tokens": 500, "max_tokens": 2000}
    post_state = {"version": 2, "node_count": 12, "orphan_edges": 0, "tokens": 650, "max_tokens": 2000}
    smt_str = guard.compile_to_smtlib2(pre_state=pre_state, post_state=post_state)
    assert isinstance(smt_str, str)
    assert "(set-logic QF_LIA)" in smt_str
    assert "(declare-const" in smt_str
    assert "(assert" in smt_str
    assert "(check-sat)" in smt_str


def test_verify_state_transition_sat_valid() -> None:
    """REQ-BM-SMT-01: Verify monotonic valid state transition returns SAT_VALID and SHA-256 proof digest."""
    guard = SMTStateInvariantGuard()
    pre_state = {"version": 1, "node_count": 5, "orphan_edges": 0, "tokens": 400, "max_tokens": 2000}
    post_state = {"version": 2, "node_count": 8, "orphan_edges": 0, "tokens": 900, "max_tokens": 2000}
    res = guard.verify_state_transition(pre_state=pre_state, post_state=post_state)
    assert res["sat"] is True
    assert res["status"] == "SAT_VALID"
    assert res["violated_invariants"] == []
    assert res["counterexample"] == {}
    assert isinstance(res["proof_digest_sha256"], str)
    assert len(res["proof_digest_sha256"]) == 64


def test_verify_state_transition_detects_version_regression_and_orphans() -> None:
    """REQ-BM-SMT-01: Verify invalid transition with version regression and orphan edges returns UNSAT_VIOLATION."""
    guard = SMTStateInvariantGuard()
    pre_state = {"version": 5, "node_count": 10, "orphan_edges": 0, "tokens": 400, "max_tokens": 2000}
    post_state = {"version": 3, "node_count": 10, "orphan_edges": 2, "tokens": 450, "max_tokens": 2000}
    res = guard.verify_state_transition(pre_state=pre_state, post_state=post_state)
    assert res["sat"] is False
    assert res["status"] == "UNSAT_VIOLATION"
    assert "version_monotonicity" in res["violated_invariants"]
    assert "referential_integrity" in res["violated_invariants"]
    assert bool(res["counterexample"])


def test_verify_state_transition_custom_invariant_rules() -> None:
    """REQ-BM-SMT-01: Verify custom StateInvariantRule enforcement on state transitions."""
    custom_rule = StateInvariantRule(
        rule_id="INV-CUSTOM-01",
        description="Checkpoint count must not drop",
        precondition_expr="pre_checkpoints >= 0",
        postcondition_expr="post_checkpoints >= pre_checkpoints",
    )
    guard = SMTStateInvariantGuard(custom_rules=[custom_rule])
    base_pre = {"version": 1, "node_count": 5, "orphan_edges": 0, "tokens": 200, "max_tokens": 1000, "checkpoints": 4}
    bad_post = {"version": 2, "node_count": 6, "orphan_edges": 0, "tokens": 250, "max_tokens": 1000, "checkpoints": 2}
    res_bad = guard.verify_state_transition(pre_state=base_pre, post_state=bad_post)
    assert res_bad["sat"] is False
    assert res_bad["status"] == "UNSAT_VIOLATION"
    assert "INV-CUSTOM-01" in res_bad["violated_invariants"]

    good_post = {"version": 2, "node_count": 6, "orphan_edges": 0, "tokens": 250, "max_tokens": 1000, "checkpoints": 5}
    res_good = guard.verify_state_transition(pre_state=base_pre, post_state=good_post)
    assert res_good["sat"] is True
    assert res_good["status"] == "SAT_VALID"
    assert "INV-CUSTOM-01" not in res_good["violated_invariants"]


def test_commit_guarded_transition_commits_valid_mutation() -> None:
    """REQ-BM-SMT-01: Verify commit_guarded_transition applies valid mutation in-place."""
    guard = SMTStateInvariantGuard()
    store = {"version": 1, "node_count": 10, "orphan_edges": 0, "tokens": 300, "max_tokens": 2000}
    result = guard.commit_guarded_transition(
        store,
        lambda s: {**s, "version": s["version"] + 1, "node_count": s["node_count"] + 1},
    )
    assert result["committed"] is True
    assert result["status"] == "SAT_VALID"
    assert store["version"] == 2
    assert store["node_count"] == 11


def test_commit_guarded_transition_aborts_and_rolls_back_invalid_mutation() -> None:
    """REQ-BM-SMT-01: Verify commit_guarded_transition aborts and preserves store on UNSAT_VIOLATION."""
    guard = SMTStateInvariantGuard()
    store = {"version": 4, "node_count": 10, "orphan_edges": 0, "tokens": 300, "max_tokens": 2000}
    snapshot = copy.deepcopy(store)
    result = guard.commit_guarded_transition(
        store,
        lambda s: {**s, "version": s["version"] - 1, "orphan_edges": 3},
    )
    assert result["committed"] is False
    assert result["status"] == "UNSAT_VIOLATION"
    assert store == snapshot
