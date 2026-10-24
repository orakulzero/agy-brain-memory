"""Antigravity Brain SQLite WAL + FTS5 Memory Indexer, Subagent Lineage Graph, and Session Digest Exporter."""

from __future__ import annotations

__version__ = "1.1.0"

from agy_brain_memory.benchmarks import run_benchmark_suite
from agy_brain_memory.cli import main, run_cli
from agy_brain_memory.compactor import analyze_brain_health, export_session_digest
from agy_brain_memory.indexer import BrainMemoryIndexer
from agy_brain_memory.mcp_server import handle_mcp_request, run_stdio_server
from agy_brain_memory.oss_audit import audit_git_repository
from agy_brain_memory.plugin_installer import (
    install_brain_memory_plugin,
    validate_brain_memory_plugin,
)
from agy_brain_memory.search import (
    get_conversation_summary,
    get_subagent_lineage,
    sanitize_fts5_query,
    search_brain,
    search_capa_records,
    search_diff_tours,
)

__all__ = [
    "__version__",
    "BrainMemoryIndexer",
    "analyze_brain_health",
    "audit_git_repository",
    "export_session_digest",
    "get_conversation_summary",
    "get_subagent_lineage",
    "handle_mcp_request",
    "install_brain_memory_plugin",
    "main",
    "run_benchmark_suite",
    "run_cli",
    "run_stdio_server",
    "sanitize_fts5_query",
    "search_brain",
    "search_capa_records",
    "search_diff_tours",
    "validate_brain_memory_plugin",
]
