from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
VALID_TOOL_RE = re.compile(r"^[a-zA-Z0-9_.\-:]+$")

# Dynamically construct forbidden strings at runtime so this test file itself
# never contains any raw personal path, drive, or email substrings.
TOKEN_USER_NAME = "Ora" + "kul"
TOKEN_FULL_USER = TOKEN_USER_NAME + " " + "Grey"
TOKEN_USERS_WIN = "Users\\" + TOKEN_USER_NAME
TOKEN_USERS_POSIX = "Users/" + TOKEN_USER_NAME
TOKEN_EXT_FOLDER = "ANTI" + "GRA" + "VITY"
TOKEN_EXT_WIN = "e:\\" + TOKEN_EXT_FOLDER
TOKEN_EXT_POSIX = "e:/" + TOKEN_EXT_FOLDER
TOKEN_PERSONAL_EMAIL = "kuro" + "ora" + "kul" + "grey" + "@gmail.com"
EXPECTED_OSS_EMAIL = "129319184+orakulzero@users.noreply.github.com"


def _git_output(*args: str, cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return proc.stdout


def _git_ls_files() -> list[str]:
    out = _git_output("ls-files")
    return [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]


def _iter_tracked_text_files() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for rel_path in _git_ls_files():
        full_path = REPO_ROOT / rel_path
        if not full_path.is_file():
            continue
        try:
            raw = full_path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        items.append((rel_path, raw.decode("utf-8", errors="replace")))
    return items


def _write_transcript(conv_dir: Path, steps: list[dict]) -> Path:
    log_dir = conv_dir / ".system_generated" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = log_dir / "transcript.jsonl"
    lines = [json.dumps(step, ensure_ascii=False) for step in steps]
    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return transcript_path


# ============================================================================
# REQ-001: Incremental SQLite WAL + FTS5 Brain Indexer (agy_brain_memory.indexer)
# ============================================================================


def test_req001_incremental_wal_fts5_sync_and_mtime_skip(tmp_path: Path) -> None:
    from agy_brain_memory.indexer import BrainMemoryIndexer

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "brain_index.sqlite"

    conv1 = brain_dir / "conv_alpha"
    _write_transcript(
        conv1,
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Design the HookMultiplexer architecture",
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Implemented single-process HookMultiplexer with P10..P50 priority.",
            },
        ],
    )
    (conv1 / "architecture_plan.md").write_text(
        "# Architecture Plan\nSingle-process dispatcher with SQLite FTS5.",
        encoding="utf-8",
    )

    conv2 = brain_dir / "conv_beta"
    _write_transcript(
        conv2,
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Fix git worktree deadlock",
            }
        ],
    )

    # Empty non-conversation folder that must be skipped cleanly
    (brain_dir / "tempmediaStorage").mkdir(parents=True)

    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    res1 = indexer.sync(force=False)
    assert res1["conversations_indexed"] == 2
    assert res1["steps_indexed"] == 3
    assert res1["artifacts_indexed"] >= 1

    # Second sync without file changes must skip unchanged conversations via mtime_ns + file_size
    res2 = indexer.sync(force=False)
    assert res2["conversations_indexed"] == 0
    assert res2["conversations_skipped"] == 2


def test_req001_indexer_extracts_subagent_edges_and_outcomes(tmp_path: Path) -> None:
    from agy_brain_memory.indexer import BrainMemoryIndexer
    from agy_brain_memory.search import get_subagent_lineage

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "brain_index.sqlite"

    parent_dir = brain_dir / "conv_parent"
    _write_transcript(
        parent_dir,
        [
            {
                "step_index": 10,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Launching isolated test designer subagent.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "invoke_subagent",
                        "args": {
                            "Role": '"Test-Designer"',
                            "TypeName": '"self"',
                            "Prompt": '"Write 20 pytest functions for BL-20261023"',
                        },
                    }
                ],
            },
            {
                "step_index": 11,
                "type": "GENERIC",
                "status": "DONE",
                "content": 'Created the following subagent:\n{\n "conversationId": "conv_child_1"\n}',
            },
        ],
    )

    child_dir = brain_dir / "conv_child_1"
    _write_transcript(
        child_dir,
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Write 20 pytest functions for BL-20261023",
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "20 tests written and verified with SHA-256 hash.",
            },
        ],
    )

    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    sync_res = indexer.sync()
    assert sync_res["subagent_edges_indexed"] == 1

    lineage = get_subagent_lineage(db_path=db_path, conv_id="conv_parent")
    assert len(lineage["spawned_children"]) == 1
    edge = lineage["spawned_children"][0]
    assert edge["parent_conv_id"] == "conv_parent"
    assert edge["child_conv_id"] == "conv_child_1"
    assert edge["role"] == "Test-Designer"
    assert edge["type_name"] == "self"
    assert "Write 20 pytest functions" in edge["prompt"]
    assert edge["child_status"] == "DONE"
    assert "20 tests written and verified" in edge["child_outcome_summary"]


def test_req001_indexer_tracks_error_and_bigint_steps(tmp_path: Path) -> None:
    from agy_brain_memory.indexer import BrainMemoryIndexer
    from agy_brain_memory.search import get_conversation_summary

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "brain_index.sqlite"

    conv_err = brain_dir / "conv_bigint_crash"
    _write_transcript(
        conv_err,
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Run heavy subagent",
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "status": "ERROR",
                "content": "TypeError: Do not know how to serialize a BigInt",
            },
        ],
    )

    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    indexer.sync()

    summary = get_conversation_summary(db_path=db_path, conv_id="conv_bigint_crash")
    assert summary["has_bigint_error"] is True
    assert summary["last_status"] == "ERROR"


# ============================================================================
# REQ-002: BM25 Full-Text Search & Subagent Call-Graph Lineage (agy_brain_memory.search)
# ============================================================================


def test_req002_fts5_search_handles_hyphens_and_special_syntax(tmp_path: Path) -> None:
    from agy_brain_memory.indexer import BrainMemoryIndexer
    from agy_brain_memory.search import sanitize_fts5_query, search_brain

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "brain_index.sqlite"

    conv = brain_dir / "conv_search_1"
    _write_transcript(
        conv,
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Fixed agy-doctor-mux (CRIT-01) fail-fast short-circuit in HookMultiplexer.",
            }
        ],
    )

    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    indexer.sync()

    sanitized = sanitize_fts5_query("agy-doctor-mux: (CRIT-01) [HookMultiplexer] *")
    assert sanitized

    res = search_brain(
        db_path=db_path,
        query="agy-doctor-mux: (CRIT-01) [HookMultiplexer]",
        limit=5,
    )
    assert len(res["results"]) >= 1
    top = res["results"][0]
    assert top["conv_id"] == "conv_search_1"
    assert top["conversation_link"] == "conversation://conv_search_1"


def test_req002_fts5_search_filters_by_step_type_and_artifacts(tmp_path: Path) -> None:
    from agy_brain_memory.indexer import BrainMemoryIndexer
    from agy_brain_memory.search import search_brain

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "brain_index.sqlite"

    conv = brain_dir / "conv_filter"
    _write_transcript(
        conv,
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Build PlanarianRollbackEngine for statepoints",
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "PlanarianRollbackEngine restores clean snapshots.",
            },
        ],
    )
    (conv / "walkthrough.md").write_text(
        "# Walkthrough\nVerified QuantumAuditChainHash in artifact.",
        encoding="utf-8",
    )

    BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir).sync()

    only_user = search_brain(
        db_path=db_path,
        query="PlanarianRollbackEngine",
        step_type="USER_INPUT",
    )
    assert len(only_user["results"]) == 1
    assert only_user["results"][0]["type"] == "USER_INPUT"

    art_res = search_brain(
        db_path=db_path,
        query="QuantumAuditChainHash",
        include_artifacts=True,
    )
    assert len(art_res["artifact_results"]) == 1
    assert art_res["artifact_results"][0]["conv_id"] == "conv_filter"


def test_req002_subagent_lineage_graph_parent_to_child_and_outcomes(
    tmp_path: Path,
) -> None:
    from agy_brain_memory.indexer import BrainMemoryIndexer
    from agy_brain_memory.search import get_subagent_lineage

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "brain_index.sqlite"

    # Root spawns conv_td and conv_ba; conv_ba spawns conv_verifier
    _write_transcript(
        brain_dir / "conv_root",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Spawning Test-Designer",
                "tool_calls": [
                    {
                        "name": "invoke_subagent",
                        "args": {
                            "Role": "Agile-V Test-Designer",
                            "TypeName": "self",
                            "Prompt": "Design red baseline tests",
                        },
                    }
                ],
            },
            {
                "step_index": 2,
                "type": "GENERIC",
                "status": "DONE",
                "content": 'Created the following subagent:\n{"conversationId": "conv_td"}',
            },
            {
                "step_index": 3,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Spawning Build-Agent",
                "tool_calls": [
                    {
                        "name": "invoke_subagent",
                        "args": {
                            "Role": "Agile-V Build-Agent",
                            "TypeName": "self",
                            "Prompt": "Implement green synthesis",
                        },
                    }
                ],
            },
            {
                "step_index": 4,
                "type": "GENERIC",
                "status": "DONE",
                "content": 'Created the following subagent:\n{"conversationId": "conv_ba"}',
            },
        ],
    )
    _write_transcript(
        brain_dir / "conv_td",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Red baseline 22 tests ready.",
            }
        ],
    )
    _write_transcript(
        brain_dir / "conv_ba",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Delegating verification",
                "tool_calls": [
                    {
                        "name": "invoke_subagent",
                        "args": {
                            "Role": "Independent Verifier",
                            "TypeName": "terse-reviewer",
                            "Prompt": "Verify all 22 tests pass",
                        },
                    }
                ],
            },
            {
                "step_index": 2,
                "type": "GENERIC",
                "status": "DONE",
                "content": 'Created the following subagent:\n{"conversationId": "conv_verifier"}',
            },
        ],
    )
    _write_transcript(
        brain_dir / "conv_verifier",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "22/22 passed with zero findings.",
            }
        ],
    )

    BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir).sync()

    root_lineage = get_subagent_lineage(db_path=db_path, conv_id="conv_root")
    child_ids = {c["child_conv_id"] for c in root_lineage["spawned_children"]}
    assert child_ids == {"conv_td", "conv_ba"}
    descendant_ids = {d["child_conv_id"] for d in root_lineage["descendants"]}
    assert "conv_verifier" in descendant_ids

    # Query from child perspective
    ba_lineage = get_subagent_lineage(db_path=db_path, conv_id="conv_ba")
    assert ba_lineage["spawned_by_parent"]["parent_conv_id"] == "conv_root"
    assert ba_lineage["spawned_children"][0]["child_conv_id"] == "conv_verifier"

    # Filter by role_query
    role_filtered = get_subagent_lineage(db_path=db_path, role_query="Verifier")
    assert len(role_filtered["edges"]) == 1
    assert role_filtered["edges"][0]["child_conv_id"] == "conv_verifier"


def test_req002_get_conversation_summary_reconstructs_timeline(tmp_path: Path) -> None:
    from agy_brain_memory.indexer import BrainMemoryIndexer
    from agy_brain_memory.search import get_conversation_summary

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "brain_index.sqlite"

    conv = brain_dir / "conv_timeline"
    _write_transcript(
        conv,
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "<USER_REQUEST>Build agy-brain-memory indexer</USER_REQUEST>",
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Indexer completed.",
            },
        ],
    )
    (conv / "task.md").write_text("# Task Checklist\n- [x] Done", encoding="utf-8")

    BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir).sync()
    summary = get_conversation_summary(db_path=db_path, conv_id="conv_timeline")
    assert summary["conv_id"] == "conv_timeline"
    assert summary["conversation_link"] == "conversation://conv_timeline"
    assert summary["step_count"] == 2
    assert any("agy-brain-memory" in r for r in summary["user_requests"])
    assert "task.md" in summary["artifacts"]
    assert summary["last_status"] == "DONE"


# ============================================================================
# REQ-003: Brain Health Analyzer & Session Handoff Digest (agy_brain_memory.compactor)
# ============================================================================


def test_req003_analyze_brain_health_identifies_orphans_bigint_and_stalled(
    tmp_path: Path,
) -> None:
    from agy_brain_memory.compactor import analyze_brain_health

    brain_dir = tmp_path / "brain"
    (brain_dir / "tempmediaStorage").mkdir(parents=True)
    _write_transcript(
        brain_dir / "conv_ok",
        [{"step_index": 1, "type": "PLANNER_RESPONSE", "status": "DONE", "content": "ok"}],
    )
    _write_transcript(
        brain_dir / "conv_bigint",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "ERROR",
                "content": "TypeError: Do not know how to serialize a BigInt",
            }
        ],
    )
    _write_transcript(
        brain_dir / "conv_stalled",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "IN_PROGRESS",
                "content": "Running command...",
            }
        ],
    )

    health = analyze_brain_health(brain_dir=brain_dir, db_path=tmp_path / "idx.sqlite")
    assert health["valid_conversations"] == 3
    assert "tempmediaStorage" in health["empty_or_orphan_dirs"]
    assert "conv_bigint" in health["bigint_error_conversations"]
