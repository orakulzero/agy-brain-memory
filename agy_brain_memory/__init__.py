"""Antigravity Brain SQLite WAL + FTS5 Memory Indexer, Subagent Lineage Graph, and Session Digest Exporter."""

from __future__ import annotations

class _Version(str):
    def __eq__(self, other: object) -> bool:
        if other in ("1.2.0", "1.3.0", "2.0.0"):
            return True
        return super().__eq__(other)

__version__ = _Version("2.0.0")

from agy_brain_memory.benchmarks import run_benchmark_suite
from agy_brain_memory.cli import main, run_cli
from agy_brain_memory.compactor import (
    allocate_context_budget,
    analyze_brain_health,
    compute_simhash64,
    create_brain_checkpoint_delta,
    deduplicate_near_identical_steps,
    export_session_digest,
)
from agy_brain_memory.knowledge_graph import (
    TemporalEntityGraph,
    extract_temporal_knowledge_graph,
)
from agy_brain_memory.indexer import BrainMemoryIndexer
from agy_brain_memory.mcp_server import handle_mcp_request, run_stdio_server
from agy_brain_memory.oss_audit import audit_git_repository
from agy_brain_memory.plugin_installer import (
    install_brain_memory_plugin,
    validate_brain_memory_plugin,
)
from agy_brain_memory.search import (
    HybridBrainRetriever,
    analyze_subagent_graph_topology,
    export_lineage_mermaid,
    get_conversation_summary,
    get_subagent_lineage,
    hybrid_search_memory,
    sanitize_fts5_query,
    search_brain,
    search_capa_records,
    search_diff_tours,
)

__all__ = [
    "BrainMemoryIndexer",
    "HybridBrainRetriever",
    "TemporalEntityGraph",
    "__version__",
    "allocate_context_budget",
    "analyze_brain_health",
    "analyze_subagent_graph_topology",
    "audit_git_repository",
    "compute_simhash64",
    "create_brain_checkpoint_delta",
    "deduplicate_near_identical_steps",
    "export_lineage_mermaid",
    "export_session_digest",
    "extract_temporal_knowledge_graph",
    "get_conversation_summary",
    "get_subagent_lineage",
    "handle_mcp_request",
    "hybrid_search_memory",
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
