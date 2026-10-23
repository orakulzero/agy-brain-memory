# agy-brain-memory

**Antigravity Brain Cross-Conversation FTS5 Memory, Subagent Lineage & Context Compaction Engine** (`v1.0.0`).

[![CI](https://github.com/orakulzero/agy-brain-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/orakulzero/agy-brain-memory/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Google Antigravity stores session transcripts (`transcript.jsonl`) and Markdown artifacts across UUID directories in `~/.gemini/antigravity/brain/<conv-id>`. Without an index, searching past sessions or reconstructing parent-to-child `invoke_subagent` trees requires brute-force linear scans across hundreds of megabytes of JSONL files.

`agy-brain-memory` solves this with zero third-party runtime dependencies (100% Python 3.10+ standard library `sqlite3` WAL + FTS5):

1. **Incremental SQLite WAL + FTS5 Brain Indexer (`agy_brain_memory.indexer`)**:
   - Tracks `mtime_ns` and `file_size` per conversation transcript so unchanged sessions are skipped in sub-millisecond time.
   - Tolerates malformed or mid-write truncated JSONL lines without crashing.
   - Indexes conversation steps, reasoning summaries, tool calls, and top-level Markdown artifacts (`task.md`, `implementation_plan.md`, `walkthrough.md`).
2. **FTS5 BM25 Cross-Conversation Search & Subagent Lineage (`agy_brain_memory.search`)**:
   - Sanitizes arbitrary punctuation/operator queries (`:`, `-`, `()`, `[]`, `*`) into safe FTS5 token queries.
   - Reconstructs parent-to-child `invoke_subagent` trees (`parent_conv_id -> child_conv_id`), subagent roles, prompts, and child final outcomes.
3. **Brain Health Analyzer & Compact Session Digest (`agy_brain_memory.compactor`)**:
   - Identifies empty/orphaned directories, `BigInt` serialization failures, and stalled `IN_PROGRESS` sessions.
   - Exports token-budgeted Markdown handoff digests (`export_session_digest`) with `conversation://<conv-id>` links.
4. **Turnkey Level-2 (L2) Antigravity Plugin & Stdio MCP Server (`agy_brain_memory.plugin_installer`, `agy_brain_memory.mcp_server`)**:
   - Exposes 5 MCP tools (`agy_brain_sync`, `agy_brain_search`, `agy_brain_subagent_lineage`, `agy_brain_session_digest`, `agy_brain_health_report`) with composite tool names guaranteed `<= 64` characters.

## Quickstart

```bash
# 1. Sync ~/.gemini/antigravity/brain into SQLite WAL + FTS5
python -m agy_brain_memory.cli sync

# 2. Search across all conversations and artifacts
python -m agy_brain_memory.cli search --query "HookMultiplexer CRIT-01"

# 3. Inspect subagent parent-child call graph
python -m agy_brain_memory.cli lineage --conv-id <conversation-uuid>

# 4. Export a compact session handoff digest (<= 2000 chars)
python -m agy_brain_memory.cli digest --conv-id <conversation-uuid> --max-chars 2000

# 5. Run brain storage health diagnostics
python -m agy_brain_memory.cli health

# 6. Install turnkey Level-2 native Antigravity plugin
python -m agy_brain_memory.cli install-plugin

# 7. Run the 4-sandbox benchmark suite
python -m agy_brain_memory.cli benchmark
```

## License

MIT License — Copyright (c) 2026 `orakulzero`.
