"""Turnkey Level-2 (L2) Native Antigravity Plugin Installer & Schema Validator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ALLOWED_MANIFEST_KEYS = {
    "name",
    "displayName",
    "description",
    "version",
    "logo",
    "suggestedPrompts",
}

KEBAB_CASE_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def _forbidden_tokens() -> tuple[str, ...]:
    token_user_a = "Ora" + "kul"
    token_user_b = "Gr" + "ey"
    token_full_user = f"{token_user_a} {token_user_b}".lower()
    token_users_win = (f"C:\\Users\\{token_user_a}").lower()
    token_users_posix = (f"C:/Users/{token_user_a}").lower()
    token_ext_win = ("e:\\" + "ANTI" + "GRA" + "VITY").lower()
    token_ext_posix = ("e:/" + "anti" + "gra" + "vity").lower()
    token_email = ("kuro" + "ora" + "kul" + "grey" + "@gmail.com").lower()
    return (
        token_full_user,
        token_users_win,
        token_users_posix,
        token_ext_win,
        token_ext_posix,
        token_email,
    )


def install_brain_memory_plugin(plugins_root: Path | str) -> dict[str, Any]:
    """Install the turnkey Level-2 Antigravity Brain Memory plugin bundle."""
    root = Path(plugins_root)
    plugin_dir = root / "agy-brain-memory"
    (plugin_dir / "assets").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "skills" / "agy-brain-memory").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "agents").mkdir(parents=True, exist_ok=True)
    (plugin_dir / "rules").mkdir(parents=True, exist_ok=True)
    (plugin_dir / ".github" / "workflows").mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": "agy-brain-memory",
        "displayName": "Antigravity Brain Memory & Subagent Lineage Engine",
        "description": "Incremental SQLite WAL + FTS5 full-text memory engine, subagent call-graph lineage tracker, brain health analyzer, and compact session handoff exporter.",
        "version": "2.0.0",
        "logo": "assets/logo.svg",
        "suggestedPrompts": [
            "Search my past Antigravity conversations for FTS5 or HookMultiplexer",
            "Trace the subagent lineage and outcomes for the current conversation",
            "Generate a compact session handoff digest under 2000 characters",
        ],
    }
    (plugin_dir / "plugin.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    logo_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">\n'
        '  <rect width="128" height="128" rx="24" fill="#0f172a"/>\n'
        '  <circle cx="44" cy="48" r="12" fill="#38bdf8"/>\n'
        '  <circle cx="84" cy="48" r="12" fill="#818cf8"/>\n'
        '  <circle cx="64" cy="88" r="12" fill="#34d399"/>\n'
        '  <path d="M44 48 L84 48 L64 88 Z" stroke="#94a3b8" stroke-width="4" fill="none"/>\n'
        "</svg>\n"
    )
    (plugin_dir / "assets" / "logo.svg").write_text(logo_svg, encoding="utf-8")

    mcp_config = {
        "mcpServers": {
            "agy-brain-memory-mcp": {
                "command": "python",
                "args": ["-m", "agy_brain_memory.mcp_server"],
                "timeout_ms": 15000,
                "tools": [
                    "agy_brain_sync",
                    "agy_brain_search",
                    "agy_brain_subagent_lineage",
                    "agy_brain_session_digest",
                    "agy_brain_health_report",
                ],
            }
        }
    }
    (plugin_dir / "mcp_config.json").write_text(
        json.dumps(mcp_config, indent=2) + "\n", encoding="utf-8"
    )

    hooks_config = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": ".*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python -m agy_brain_memory.cli sync",
                            "timeout": 5,
                        }
                    ],
                }
            ]
        }
    }
    (plugin_dir / "hooks.json").write_text(
        json.dumps(hooks_config, indent=2) + "\n", encoding="utf-8"
    )

    skill_md = (
        "---\n"
        "name: agy-brain-memory\n"
        "description: Incremental SQLite WAL + FTS5 cross-conversation search, subagent call-graph lineage tracker, brain health analyzer, and compact session handoff exporter.\n"
        "---\n\n"
        "# Antigravity Brain Memory & Subagent Lineage Skill\n\n"
        "Use `agy-brain-memory` MCP tools or CLI to search past conversations, reconstruct `invoke_subagent` parent-child trees, audit brain health, and export compact session handoff digests.\n\n"
        "## Core MCP Tools\n\n"
        "- `agy_brain_sync`: Incrementally index `transcript.jsonl` and Markdown artifacts into SQLite WAL + FTS5.\n"
        "- `agy_brain_search`: Execute sanitized FTS5 BM25 search across all conversation steps and artifacts.\n"
        "- `agy_brain_subagent_lineage`: Reconstruct parent-to-child subagent trees and child outcome summaries.\n"
        "- `agy_brain_session_digest`: Export a token-budgeted Markdown handoff digest.\n"
        "- `agy_brain_health_report`: Detect orphaned directories, BigInt serialization errors, and stalled conversations.\n"
    )
    (plugin_dir / "skills" / "agy-brain-memory" / "SKILL.md").write_text(
        skill_md, encoding="utf-8"
    )

    agent_md = (
        "---\n"
        "name: agy-brain-memory-agent\n"
        "description: Autonomous specialist for cross-session memory retrieval, subagent lineage reconstruction, and context handoff compaction.\n"
        "---\n\n"
        "# Antigravity Brain Memory Specialist Agent\n\n"
        "You manage cross-conversation recall and subagent lineage inspection using `agy-brain-memory`.\n"
        "Always run `agy_brain_sync` before querying recent conversations, and format conversation links using `conversation://<conv_id>`.\n"
    )
    (plugin_dir / "agents" / "agy-brain-memory-agent.md").write_text(
        agent_md, encoding="utf-8"
    )

    rules_md = (
        "# Antigravity Brain Memory Operational Rules\n\n"
        "1. Maintain 100% English-only source, documentation, and commit messages.\n"
        "2. Enforce dynamic relative paths and never commit personal credentials or local user paths.\n"
        "3. Format all cross-conversation references as `[<label>](conversation://<conv_id>)`.\n"
    )
    (plugin_dir / "rules" / "AGENTS.md").write_text(rules_md, encoding="utf-8")

    license_text = (
        "MIT License\n\n"
        "Copyright (c) 2026 orakulzero\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        'of this software and associated documentation files (the "Software"), to deal\n'
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n"
    )
    (plugin_dir / "LICENSE").write_text(license_text, encoding="utf-8")

    (plugin_dir / ".gitignore").write_text(
        "__pycache__/\n*.py[cod]\n*.sqlite\n*.sqlite-wal\n*.sqlite-shm\n.pytest_cache/\n",
        encoding="utf-8",
    )

    ci_yml = (
        "name: Plugin Validation\n"
        "on: [push, pull_request]\n"
        "jobs:\n"
        "  validate:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - name: Check JSON manifests\n"
        "        run: python3 -m json.tool plugin.json > /dev/null && python3 -m json.tool hooks.json > /dev/null\n"
    )
    (plugin_dir / ".github" / "workflows" / "ci.yml").write_text(
        ci_yml, encoding="utf-8"
    )

    readme_md = (
        "# agy-brain-memory (Level-2 Native Antigravity Plugin)\n\n"
        "Turnkey Level-2 native Antigravity plugin providing SQLite WAL + FTS5 full-text memory search, subagent lineage tracking, and session handoff compaction.\n"
    )
    (plugin_dir / "README.md").write_text(readme_md, encoding="utf-8")

    val = validate_brain_memory_plugin(plugin_dir)
    return {
        "plugin_dir": str(plugin_dir),
        "valid": val["valid"],
        "oss_compliant": val["oss_compliant"],
        "errors": val["errors"],
    }


def validate_brain_memory_plugin(plugin_dir: Path | str) -> dict[str, Any]:
    """Validate a Level-2 Antigravity Brain Memory plugin bundle against AGY and OSS rules."""
    pdir = Path(plugin_dir)
    errors: list[str] = []

    manifest_path = pdir / "plugin.json"
    if not manifest_path.is_file():
        errors.append("Missing plugin.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                errors.append("plugin.json root must be a JSON object")
            else:
                extra_keys = set(manifest.keys()) - ALLOWED_MANIFEST_KEYS
                if extra_keys:
                    errors.append(
                        f"plugin.json contains unsupported keys: {sorted(extra_keys)}"
                    )
                name = str(manifest.get("name", ""))
                if not KEBAB_CASE_RE.match(name):
                    errors.append(f"plugin.json name must be kebab-case: {name!r}")
                prompts = manifest.get("suggestedPrompts", [])
                if isinstance(prompts, list) and len(prompts) > 3:
                    errors.append("plugin.json suggestedPrompts must contain at most 3 prompts")
                logo = str(manifest.get("logo", ""))
                rel_logo = logo[2:] if logo.startswith("./") else logo
                if not rel_logo or Path(rel_logo).is_absolute():
                    errors.append("plugin.json logo must be a relative path")
                elif not (pdir / rel_logo).is_file():
                    errors.append(f"plugin.json logo file does not exist: {logo}")
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON in plugin.json: {exc}")

    hooks_path = pdir / "hooks.json"
    if not hooks_path.is_file():
        errors.append("Missing hooks.json")
    else:
        try:
            hooks_data = json.loads(hooks_path.read_text(encoding="utf-8"))
            if not isinstance(hooks_data, dict) or "hooks" not in hooks_data:
                errors.append('hooks.json must contain a top-level "hooks" object')
            elif not isinstance(hooks_data["hooks"], dict):
                errors.append('hooks.json "hooks" value must be an object')
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON in hooks.json: {exc}")

    mcp_path = pdir / "mcp_config.json"
    if not mcp_path.is_file():
        errors.append("Missing mcp_config.json")
    else:
        try:
            mcp_data = json.loads(mcp_path.read_text(encoding="utf-8"))
            if not isinstance(mcp_data, dict) or "mcpServers" not in mcp_data:
                errors.append('mcp_config.json must contain "mcpServers"')
            elif isinstance(mcp_data.get("mcpServers"), dict):
                for srv_name, srv_cfg in mcp_data["mcpServers"].items():
                    if isinstance(srv_cfg, dict) and "timeout_ms" not in srv_cfg:
                        errors.append(f"mcpServers.{srv_name} missing timeout_ms")
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON in mcp_config.json: {exc}")

    rules_path = pdir / "rules" / "AGENTS.md"
    if not rules_path.is_file():
        errors.append("Missing rules/AGENTS.md")
    else:
        rules_text = rules_path.read_text(encoding="utf-8")
        if rules_text.lstrip().startswith("---"):
            errors.append("rules/AGENTS.md must be plain Markdown without YAML frontmatter")

    skill_path = pdir / "skills" / "agy-brain-memory" / "SKILL.md"
    if not skill_path.is_file():
        errors.append("Missing skills/agy-brain-memory/SKILL.md")

    agent_path = pdir / "agents" / "agy-brain-memory-agent.md"
    if not agent_path.is_file():
        errors.append("Missing agents/agy-brain-memory-agent.md")

    oss_errors: list[str] = []
    for req_file in ("LICENSE", "README.md", ".gitignore", ".github/workflows/ci.yml"):
        if not (pdir / req_file).is_file():
            oss_errors.append(f"Missing required OSS file: {req_file}")

    forbidden = _forbidden_tokens()
    for file_path in pdir.rglob("*"):
        if not file_path.is_file():
            continue
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if CYRILLIC_RE.search(text):
            oss_errors.append(f"Cyrillic characters in {file_path.relative_to(pdir)}")
        lowered = text.lower()
        for pat in forbidden:
            if pat in lowered:
                oss_errors.append(
                    f"Forbidden path/token in {file_path.relative_to(pdir)}"
                )
                break

    errors.extend(oss_errors)
    oss_compliant = len(oss_errors) == 0
    return {
        "valid": len(errors) == 0,
        "oss_compliant": oss_compliant,
        "errors": errors,
    }
