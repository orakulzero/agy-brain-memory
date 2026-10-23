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
    assert "conv_stalled" in health["stalled_in_progress_conversations"]


def test_req003_export_session_digest_respects_max_chars_and_includes_lineage(
    tmp_path: Path,
) -> None:
    from agy_brain_memory.compactor import export_session_digest
    from agy_brain_memory.indexer import BrainMemoryIndexer

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "idx.sqlite"

    _write_transcript(
        brain_dir / "conv_digest",
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Create full architecture digest " + ("detail " * 300),
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Spawning subagent",
                "tool_calls": [
                    {
                        "name": "invoke_subagent",
                        "args": {
                            "Role": "Explorer",
                            "TypeName": "research",
                            "Prompt": "Survey brain logs",
                        },
                    }
                ],
            },
            {
                "step_index": 3,
                "type": "GENERIC",
                "status": "DONE",
                "content": 'Created the following subagent:\n{"conversationId": "conv_sub_exp"}',
            },
        ],
    )
    _write_transcript(
        brain_dir / "conv_sub_exp",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Found 66 conversations.",
            }
        ],
    )

    BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir).sync()
    digest = export_session_digest(
        db_path=db_path, conv_id="conv_digest", max_chars=800
    )
    assert digest["conv_id"] == "conv_digest"
    assert digest["char_count"] <= 800
    assert len(digest["digest_markdown"]) <= 800
    assert "conv_digest" in digest["digest_markdown"]


# ============================================================================
# REQ-004: Turnkey Level-2 (L2) Native Antigravity Plugin Installer
# ============================================================================


def test_req004_install_and_validate_native_l2_brain_memory_plugin(
    tmp_path: Path,
) -> None:
    from agy_brain_memory.plugin_installer import (
        install_brain_memory_plugin,
        validate_brain_memory_plugin,
    )

    plugins_root = tmp_path / "plugins"
    res = install_brain_memory_plugin(plugins_root)
    plugin_dir = plugins_root / "agy-brain-memory"
    assert res["plugin_dir"] == str(plugin_dir)

    manifest = json.loads((plugin_dir / "plugin.json").read_text(encoding="utf-8"))
    allowed_keys = {
        "name",
        "displayName",
        "description",
        "version",
        "logo",
        "suggestedPrompts",
    }
    assert set(manifest.keys()).issubset(allowed_keys)
    assert manifest["name"] == "agy-brain-memory"

    hooks_data = json.loads((plugin_dir / "hooks.json").read_text(encoding="utf-8"))
    assert "hooks" in hooks_data

    assert (plugin_dir / "assets" / "logo.svg").is_file()
    assert (plugin_dir / "mcp_config.json").is_file()
    assert (plugin_dir / "skills" / "agy-brain-memory" / "SKILL.md").is_file()
    assert (plugin_dir / "agents" / "agy-brain-memory-agent.md").is_file()
    assert (plugin_dir / "rules" / "AGENTS.md").is_file()
    assert not (
        (plugin_dir / "rules" / "AGENTS.md")
        .read_text(encoding="utf-8")
        .lstrip()
        .startswith("---")
    )

    val = validate_brain_memory_plugin(plugin_dir)
    assert val["valid"] is True, f"Plugin validation failed: {val}"
    assert val["oss_compliant"] is True
    assert val["errors"] == []


def test_req004_validate_plugin_detects_malformed_hooks_or_manifest(
    tmp_path: Path,
) -> None:
    from agy_brain_memory.plugin_installer import (
        install_brain_memory_plugin,
        validate_brain_memory_plugin,
    )

    plugins_root = tmp_path / "plugins"
    install_brain_memory_plugin(plugins_root)
    plugin_dir = plugins_root / "agy-brain-memory"

    # Corrupt hooks.json by removing top-level "hooks" key and add discarded key to plugin.json
    (plugin_dir / "hooks.json").write_text(
        json.dumps({"custom-wrapper": {"Stop": []}}), encoding="utf-8"
    )
    manifest = json.loads((plugin_dir / "plugin.json").read_text(encoding="utf-8"))
    manifest["author"] = "discarded"
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")

    val = validate_brain_memory_plugin(plugin_dir)
    assert val["valid"] is False
    assert len(val["errors"]) >= 2


# ============================================================================
# REQ-005: Unified CLI, Stdio MCP Server & 4-Sandbox Benchmark Suite
# ============================================================================


def test_req005_cli_sync_search_lineage_digest_and_health(tmp_path: Path) -> None:
    from agy_brain_memory.cli import run_cli

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "cli_idx.sqlite"
    _write_transcript(
        brain_dir / "conv_cli_1",
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Searchable token BravoDeltaOmega",
            },
            {
                "step_index": 2,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Spawning helper",
                "tool_calls": [
                    {
                        "name": "invoke_subagent",
                        "args": {
                            "Role": "CLI-Subagent",
                            "TypeName": "self",
                            "Prompt": "Check BravoDeltaOmega",
                        },
                    }
                ],
            },
            {
                "step_index": 3,
                "type": "GENERIC",
                "status": "DONE",
                "content": 'Created the following subagent:\n{"conversationId": "conv_cli_child"}',
            },
        ],
    )
    _write_transcript(
        brain_dir / "conv_cli_child",
        [
            {
                "step_index": 1,
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Child finished BravoDeltaOmega.",
            }
        ],
    )

    base_args = ["--brain-dir", str(brain_dir), "--db-path", str(db_path)]
    sync_res = run_cli([*base_args, "sync"])
    assert sync_res["conversations_indexed"] == 2

    search_res = run_cli([*base_args, "search", "--query", "BravoDeltaOmega"])
    assert len(search_res["results"]) >= 1

    lin_res = run_cli([*base_args, "lineage", "--conv-id", "conv_cli_1"])
    assert len(lin_res["spawned_children"]) == 1

    dig_res = run_cli([*base_args, "digest", "--conv-id", "conv_cli_1", "--max-chars", "1200"])
    assert "conv_cli_1" in dig_res["digest_markdown"]

    health_res = run_cli([*base_args, "health"])
    assert health_res["valid_conversations"] == 2


def test_req005_cli_install_plugin_and_benchmark(tmp_path: Path) -> None:
    from agy_brain_memory.cli import run_cli

    inst_res = run_cli(
        ["install-plugin", "--plugins-root", str(tmp_path / "plugins")]
    )
    assert inst_res["valid"] is True

    bench_res = run_cli(["benchmark"])
    assert bench_res["all_passed"] is True


def test_req005_mcp_server_initialize_and_tools_list(tmp_path: Path) -> None:
    from agy_brain_memory.mcp_server import handle_mcp_request

    init_resp = handle_mcp_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        brain_dir=tmp_path / "brain",
        db_path=tmp_path / "mcp.sqlite",
    )
    assert init_resp["result"]["serverInfo"]["name"] == "agy-brain-memory-mcp"

    list_resp = handle_mcp_request(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        brain_dir=tmp_path / "brain",
        db_path=tmp_path / "mcp.sqlite",
    )
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    expected = {
        "agy_brain_sync",
        "agy_brain_search",
        "agy_brain_subagent_lineage",
        "agy_brain_session_digest",
        "agy_brain_health_report",
    }
    assert expected == tool_names


def test_req005_mcp_server_all_five_tools_round_trip(tmp_path: Path) -> None:
    from agy_brain_memory.mcp_server import handle_mcp_request

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "mcp.sqlite"
    _write_transcript(
        brain_dir / "conv_mcp",
        [
            {
                "step_index": 1,
                "type": "USER_INPUT",
                "status": "DONE",
                "content": "Verify MCP brain search tool",
            }
        ],
    )

    calls = [
        ("agy_brain_sync", {"force": True}),
        ("agy_brain_search", {"query": "MCP brain search", "limit": 5}),
        ("agy_brain_subagent_lineage", {"conv_id": "conv_mcp"}),
        ("agy_brain_session_digest", {"conv_id": "conv_mcp", "max_chars": 1000}),
        ("agy_brain_health_report", {}),
    ]
    for idx, (tool_name, args) in enumerate(calls, start=10):
        resp = handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": idx,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            },
            brain_dir=brain_dir,
            db_path=db_path,
        )
        assert "result" in resp, f"Missing result for {tool_name}: {resp}"
        assert resp["result"].get("isError", False) is False


def test_req005_benchmark_suite_four_sandboxes_pass() -> None:
    from agy_brain_memory.benchmarks import run_benchmark_suite

    res = run_benchmark_suite()
    assert res["all_passed"] is True
    assert len(res["sandboxes"]) == 4
    expected = {
        "SB1_IncrementalWAL_And_FTS5_Search",
        "SB2_SubagentCallGraph_Lineage_And_Outcomes",
        "SB3_BrainHealth_And_CompactSessionDigest",
        "SB4_NativeL2Plugin_And_MCP_RoundTrip",
    }
    actual = {sb["name"] for sb in res["sandboxes"]}
    assert actual == expected
    for sb in res["sandboxes"]:
        assert sb["passed"] is True


# ============================================================================
# REQ-006: 5-Gate Git OSS Publication Readiness (agy_brain_memory.oss_audit)
# ============================================================================


def test_req006_oss_audit_git_repository_passes() -> None:
    from agy_brain_memory.oss_audit import audit_git_repository

    report = audit_git_repository(REPO_ROOT)
    assert report["passed"] is True, f"OSS repository audit failed: {report}"


def test_req006_required_oss_files_and_mit_license_tracked() -> None:
    required_files = {
        "LICENSE",
        "README.md",
        "pyproject.toml",
        ".gitignore",
        ".github/workflows/ci.yml",
    }
    tracked = set(_git_ls_files())
    missing = required_files - tracked
    assert not missing, f"Required OSS files missing from git index: {sorted(missing)}"

    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8", errors="replace")
    assert "MIT License" in license_text
    assert "orakulzero" in license_text
    assert "Permission is hereby granted, free of charge" in license_text


def test_req006_zero_cyrillic_in_tracked_files_and_git_history() -> None:
    offenders: list[str] = []
    for rel_path, content in _iter_tracked_text_files():
        matches = CYRILLIC_RE.findall(content)
        if matches:
            offenders.append(f"{rel_path} ({len(matches)} Cyrillic chars)")
    assert not offenders, f"Tracked files contain Cyrillic characters: {offenders}"

    history_diffs = _git_output("log", "-p", "--all")
    assert not bool(CYRILLIC_RE.search(history_diffs)), (
        "Git commit history contains Cyrillic characters"
    )


def test_req006_zero_personal_paths_or_external_drives_in_repo_and_history() -> None:
    forbidden = (
        TOKEN_FULL_USER.lower(),
        TOKEN_USERS_WIN.lower(),
        TOKEN_USERS_POSIX.lower(),
        TOKEN_EXT_WIN.lower(),
        TOKEN_EXT_POSIX.lower(),
        TOKEN_PERSONAL_EMAIL.lower(),
    )
    offenders: list[str] = []
    for rel_path, content in _iter_tracked_text_files():
        lowered = content.lower()
        for pat in forbidden:
            if pat in lowered:
                offenders.append(f"{rel_path} (matched: {pat!r})")
                break
    assert not offenders, f"Tracked files contain forbidden personal/drive paths: {offenders}"

    history_diffs = _git_output("log", "-p", "--all").lower()
    for pat in forbidden:
        assert pat not in history_diffs, (
            f"Git commit history diffs contain forbidden token: {pat!r}"
        )


def test_req006_git_commit_authors_locked_to_orakulzero() -> None:
    emails_out = _git_output("log", "--all", "--format=%ae")
    emails = [line.strip() for line in emails_out.splitlines() if line.strip()]
    assert emails, "Repository must have at least one git commit"
    invalid = [e for e in emails if e != EXPECTED_OSS_EMAIL]
    assert not invalid, (
        f"Git commit authors do not match {EXPECTED_OSS_EMAIL}: {set(invalid)}"
    )


def test_req006_all_mcp_composite_tool_names_under_64_chars() -> None:
    from agy_brain_memory.mcp_server import handle_mcp_request

    list_resp = handle_mcp_request(
        {"jsonrpc": "2.0", "id": 99, "method": "tools/list", "params": {}}
    )
    server_name = "agy-brain-memory-mcp"
    for tool in list_resp["result"]["tools"]:
        composite = f"mcp_{server_name}_{tool['name']}"
        assert len(composite) <= 64, f"Composite MCP tool name exceeds 64 chars: {composite}"
        assert VALID_TOOL_RE.match(composite), f"Invalid MCP tool chars: {composite}"


# ============================================================================
# REQ-007: Live Stress-Test Performance Regression Suite (BUG-BRAIN-STRESS-01..03)
# ============================================================================


def test_req007_stress_01_sqlite_indexes_exist_on_steps_and_edges(
    tmp_path: Path,
) -> None:
    import sqlite3
    from agy_brain_memory.indexer import BrainMemoryIndexer

    brain_dir = tmp_path / "brain"
    brain_dir.mkdir(parents=True)
    db_path = tmp_path / "idx.sqlite"

    BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    indexed_tables = {r[1] for r in rows}
    assert "steps" in indexed_tables, (
        f"Missing secondary index on steps(conv_id); found: {rows}"
    )
    assert "subagent_edges" in indexed_tables, (
        f"Missing secondary index on subagent_edges; found: {rows}"
    )


def test_req007_stress_02_analyze_brain_health_skips_deep_rglob_media_files(
    tmp_path: Path,
) -> None:
    from agy_brain_memory.compactor import analyze_brain_health

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "idx.sqlite"
    conv_dir = brain_dir / "conv_1"
    _write_transcript(
        conv_dir,
        [{"step_index": 1, "type": "USER_INPUT", "status": "DONE", "content": "hi"}],
    )
    deep_file = conv_dir / ".system_generated" / "steps" / "1" / "output.bin"
    deep_file.parent.mkdir(parents=True, exist_ok=True)
    deep_file.write_bytes(b"X" * 50000)

    health = analyze_brain_health(brain_dir=brain_dir, db_path=db_path)
    assert health["valid_conversations"] == 1
    assert 0 < health["total_bytes"] < 10000, (
        f"analyze_brain_health must sum indexed file_size without deep rglob('*'), got {health['total_bytes']}"
    )


def test_req007_stress_03_large_batch_indexing_throughput(tmp_path: Path) -> None:
    from agy_brain_memory.indexer import BrainMemoryIndexer

    brain_dir = tmp_path / "brain"
    db_path = tmp_path / "idx.sqlite"

    large_payload = "word " * 8000  # 40KB payload
    for c_idx in range(40):
        steps = [
            {
                "step_index": s_idx,
                "type": "PLANNER_RESPONSE" if s_idx % 2 else "USER_INPUT",
                "status": "DONE",
                "content": large_payload if s_idx == 1 else f"Step {s_idx} content in conv {c_idx}",
            }
            for s_idx in range(1, 41)
        ]
        _write_transcript(brain_dir / f"conv_{c_idx:03d}", steps)

    indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
    res1 = indexer.sync(force=False)
    assert res1["conversations_indexed"] == 40
    assert res1["steps_indexed"] == 1600
    assert res1["elapsed_ms"] < 2500.0, (
        f"Initial batch sync too slow: {res1['elapsed_ms']} ms"
    )

    res2 = indexer.sync(force=False)
    assert res2["conversations_indexed"] == 0
    assert res2["conversations_skipped"] == 40
    assert res2["elapsed_ms"] < 150.0, (
        f"Incremental mtime skip sync too slow: {res2['elapsed_ms']} ms"
    )

