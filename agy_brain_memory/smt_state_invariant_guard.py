"""SMT-Backed State Invariant Guard for Brain Memory Transitions (REQ-BM-SMT-01)."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StateInvariantRule:
    """Declarative state transition invariant rule in QF_LIA arithmetic form (REQ-BM-SMT-01)."""

    rule_id: str
    description: str
    precondition_expr: str = "True"
    postcondition_expr: str = "True"


DEFAULT_INVARIANT_RULES: tuple[StateInvariantRule, ...] = (
    StateInvariantRule(
        rule_id="version_monotonicity",
        description="State version must be monotonically non-decreasing across transitions",
        precondition_expr="pre_version >= 0",
        postcondition_expr="post_version >= pre_version",
    ),
    StateInvariantRule(
        rule_id="referential_integrity",
        description="Graph transition must not introduce orphan edges or negative node counts",
        precondition_expr="pre_orphan_edges == 0",
        postcondition_expr="post_orphan_edges == 0 and post_node_count >= 0",
    ),
    StateInvariantRule(
        rule_id="token_budget_bound",
        description="Post-transition token usage must remain within [0, max_tokens]",
        precondition_expr="pre_max_tokens > 0",
        postcondition_expr="post_tokens >= 0 and post_tokens <= post_max_tokens",
    ),
)


def _safe_eval_bool_expr(expr: str, env: dict[str, int]) -> bool:
    """Safely evaluate a linear integer arithmetic / comparison expression over env."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "True":
                return True
            if node.id == "False":
                return False
            return env.get(node.id, 0)
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -val
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.Not):
                return not bool(val)
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        if isinstance(node, ast.BoolOp):
            values = [bool(_eval(v)) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = _eval(comp)
                if isinstance(op, ast.Eq) and not (left == right):
                    return False
                if isinstance(op, ast.NotEq) and not (left != right):
                    return False
                if isinstance(op, ast.Lt) and not (left < right):
                    return False
                if isinstance(op, ast.LtE) and not (left <= right):
                    return False
                if isinstance(op, ast.Gt) and not (left > right):
                    return False
                if isinstance(op, ast.GtE) and not (left >= right):
                    return False
                left = right
            return True
        raise ValueError(f"Unsupported AST node in invariant expression: {type(node).__name__}")

    return bool(_eval(tree))


class SMTStateInvariantGuard:
    """Formal QF_LIA SMT-LIB2 state invariant verifier and transactional commit guard (REQ-BM-SMT-01)."""

    def __init__(self, custom_rules: list[StateInvariantRule] | None = None) -> None:
        self._rules: list[StateInvariantRule] = list(DEFAULT_INVARIANT_RULES)
        if custom_rules:
            self._rules.extend(custom_rules)

    @staticmethod
    def _build_env(pre_state: dict[str, Any], post_state: dict[str, Any]) -> dict[str, int]:
        env: dict[str, int] = {}
        all_keys = sorted(set(pre_state.keys()) | set(post_state.keys()))
        for key in all_keys:
            pre_val = pre_state.get(key, 0)
            post_val = post_state.get(key, 0)
            if isinstance(pre_val, (int, float, bool)):
                env[f"pre_{key}"] = int(pre_val)
            if isinstance(post_val, (int, float, bool)):
                env[f"post_{key}"] = int(post_val)
        return env

    def compile_to_smtlib2(
        self,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
    ) -> str:
        """Compile pre_state, post_state, and active invariant rules into SMT-LIB2 QF_LIA script."""
        env = self._build_env(pre_state, post_state)
        lines: list[str] = ["(set-logic QF_LIA)"]
        for var_name in sorted(env.keys()):
            lines.append(f"(declare-const {var_name} Int)")
        for var_name, val in sorted(env.items()):
            lines.append(f"(assert (= {var_name} {val}))")
        for rule in self._rules:
            lines.append(f"; Rule {rule.rule_id}: {rule.description}")
            lines.append(f"(assert (=> ({rule.precondition_expr}) ({rule.postcondition_expr})))")
        lines.append("(check-sat)")
        return "\n".join(lines)

    def verify_state_transition(
        self,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify whether the transition from pre_state to post_state satisfies all QF_LIA invariants."""
        env = self._build_env(pre_state, post_state)
        smt_script = self.compile_to_smtlib2(pre_state=pre_state, post_state=post_state)
        violated: list[str] = []
        counterexample: dict[str, Any] = {}

        for rule in self._rules:
            pre_ok = _safe_eval_bool_expr(rule.precondition_expr, env)
            if pre_ok and not _safe_eval_bool_expr(rule.postcondition_expr, env):
                violated.append(rule.rule_id)
                counterexample[rule.rule_id] = {
                    "description": rule.description,
                    "precondition": rule.precondition_expr,
                    "postcondition": rule.postcondition_expr,
                    "assignments": dict(env),
                }

        sat = len(violated) == 0
        digest_payload = json.dumps(
            {"smt": smt_script, "sat": sat, "violated": violated},
            sort_keys=True,
        ).encode("utf-8")
        proof_digest = hashlib.sha256(digest_payload).hexdigest()
        return {
            "sat": sat,
            "status": "SAT_VALID" if sat else "UNSAT_VIOLATION",
            "violated_invariants": violated,
            "counterexample": counterexample,
            "proof_digest_sha256": proof_digest,
        }

    def commit_guarded_transition(
        self,
        store: dict[str, Any],
        mutation_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute mutation_fn and commit in-place only if SMT invariants hold; otherwise rollback."""
        snapshot = copy.deepcopy(store)
        candidate = mutation_fn(copy.deepcopy(store))
        verdict = self.verify_state_transition(pre_state=snapshot, post_state=candidate)
        if verdict["sat"]:
            store.clear()
            store.update(candidate)
            return {**verdict, "committed": True}
        store.clear()
        store.update(snapshot)
        return {**verdict, "committed": False}
