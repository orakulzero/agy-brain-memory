"""AGY Brain SQLite WAL + FTS5 Memory Indexer, Subagent Lineage Graph, and Session Digest Exporter."""

from __future__ import annotations

__version__ = "2.0.0"
__v3_generation__ = "3.0.0"

from agy_brain_memory.compactor import (
    allocate_context_budget,
    analyze_brain_health,
    compute_simhash64,
    create_brain_checkpoint_delta,
    deduplicate_near_identical_steps,
    export_session_digest,
)
from agy_brain_memory.context_token_profiler import (
    ContextTokenProfiler,
    profile_session_context,
)
from agy_brain_memory.knowledge_graph import (
    TemporalEntityGraph,
    extract_temporal_knowledge_graph,
)
from agy_brain_memory.indexer import BrainMemoryIndexer
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
from agy_brain_memory.smt_state_invariant_guard import (
    SMTStateInvariantGuard,
    StateInvariantRule,
)
from agy_brain_memory.memory_tier_partitioner import (
    COLD_SCORE_DECAY,
    DEFAULT_HOT_MAX_AGE_DAYS,
    DEFAULT_PINNED_ARTIFACTS,
    HOT_SCORE_BOOST,
    MemoryTierPartitioner,
    partition_brain_memory_tiers,
    search_tiered_brain,
)

__all__ = [
    "BrainMemoryIndexer",
    "COLD_SCORE_DECAY",
    "ContextTokenProfiler",
    "DEFAULT_HOT_MAX_AGE_DAYS",
    "DEFAULT_PINNED_ARTIFACTS",
    "HOT_SCORE_BOOST",
    "HybridBrainRetriever",
    "MemoryTierPartitioner",
    "SMTStateInvariantGuard",
    "StateInvariantRule",
    "TemporalEntityGraph",
    "__v3_generation__",
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
    "partition_brain_memory_tiers",
    "profile_session_context",
    "run_benchmark_suite",
    "run_cli",
    "run_stdio_server",
    "sanitize_fts5_query",
    "search_brain",
    "search_capa_records",
    "search_diff_tours",
    "search_tiered_brain",
    "validate_brain_memory_plugin",
]


def __getattr__(name: str):
    if name == "run_benchmark_suite":
        return run_benchmark_suite
    if name == "main":
        from agy_brain_memory.cli import main
        return main
    if name == "run_cli":
        from agy_brain_memory.cli import run_cli
        return run_cli
    if name == "handle_mcp_request":
        from agy_brain_memory.mcp_server import handle_mcp_request
        return handle_mcp_request
    if name == "run_stdio_server":
        from agy_brain_memory.mcp_server import run_stdio_server
        return run_stdio_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
