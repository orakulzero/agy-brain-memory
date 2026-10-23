"""Unified Command-Line Interface for agy-brain-memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agy_brain_memory.benchmarks import run_benchmark_suite
from agy_brain_memory.compactor import analyze_brain_health, export_session_digest
from agy_brain_memory.indexer import BrainMemoryIndexer
from agy_brain_memory.mcp_server import default_brain_dir, default_db_path
from agy_brain_memory.oss_audit import audit_git_repository
from agy_brain_memory.plugin_installer import install_brain_memory_plugin
from agy_brain_memory.search import get_subagent_lineage, search_brain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agy-brain-memory",
        description="Antigravity Brain Cross-Conversation FTS5 Memory & Subagent Lineage Engine",
    )
    parser.add_argument(
        "--brain-dir",
        type=str,
        default=str(default_brain_dir()),
        help="Path to Antigravity brain directory",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(default_db_path()),
        help="Path to SQLite WAL + FTS5 index file",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Incrementally sync brain transcripts and artifacts")
    p_sync.add_argument("--force", action="store_true", help="Force full re-index")

    p_search = sub.add_parser("search", help="Search conversations and artifacts via FTS5")
    p_search.add_argument("--query", required=True, help="FTS5 search query")
    p_search.add_argument("--limit", type=int, default=10, help="Maximum results")
    p_search.add_argument("--step-type", default=None, help="Filter by step type")

    p_lineage = sub.add_parser("lineage", help="Inspect subagent parent-child call graph")
    p_lineage.add_argument("--conv-id", default=None, help="Conversation ID")
    p_lineage.add_argument("--role", default=None, help="Filter by subagent Role or TypeName")

    p_digest = sub.add_parser("digest", help="Export compact session handoff digest")
    p_digest.add_argument("--conv-id", required=True, help="Conversation ID")
    p_digest.add_argument("--max-chars", type=int, default=2000, help="Max character budget")

    sub.add_parser("health", help="Analyze brain storage health and detect anomalies")

    p_inst = sub.add_parser(
        "install-plugin", help="Install turnkey Level-2 Antigravity plugin"
    )
    p_inst.add_argument(
        "--plugins-root",
        default=str(Path.home() / ".gemini" / "config" / "plugins"),
        help="Target Antigravity plugins root directory",
    )

    sub.add_parser("benchmark", help="Run 4-sandbox verification benchmark suite")

    p_audit = sub.add_parser("oss-audit", help="Run 5-Gate Git OSS release audit")
    p_audit.add_argument(
        "--repo-path",
        default=str(Path(__file__).resolve().parent.parent),
        help="Path to git repository root",
    )

    return parser


def run_cli(argv: list[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments, execute the requested command, and return the JSON result."""
    parser = build_parser()
    args = parser.parse_args(argv)

    brain_dir = Path(args.brain_dir)
    db_path = Path(args.db_path)

    if args.command == "sync":
        indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
        return indexer.sync(force=bool(args.force))

    if args.command == "search":
        indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
        indexer.sync(force=False)
        return search_brain(
            db_path=db_path,
            query=args.query,
            limit=args.limit,
            step_type=args.step_type,
        )

    if args.command == "lineage":
        indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
        indexer.sync(force=False)
        return get_subagent_lineage(
            db_path=db_path,
            conv_id=args.conv_id,
            role_query=args.role,
        )

    if args.command == "digest":
        indexer = BrainMemoryIndexer(db_path=db_path, brain_dir=brain_dir)
        indexer.sync(force=False)
        return export_session_digest(
            db_path=db_path,
            conv_id=args.conv_id,
            max_chars=args.max_chars,
        )

    if args.command == "health":
        return analyze_brain_health(brain_dir=brain_dir, db_path=db_path)

    if args.command == "install-plugin":
        return install_brain_memory_plugin(Path(args.plugins_root))

    if args.command == "benchmark":
        return run_benchmark_suite()

    if args.command == "oss-audit":
        return audit_git_repository(Path(args.repo_path))

    raise ValueError(f"Unsupported command: {args.command}")


def main() -> int:
    res = run_cli(sys.argv[1:])
    sys.stdout.write(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
