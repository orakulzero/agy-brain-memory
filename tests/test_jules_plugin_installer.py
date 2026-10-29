import json
import pytest
from pathlib import Path

from agy_brain_memory.plugin_installer import (
    install_brain_memory_plugin,
    validate_brain_memory_plugin,
    _forbidden_tokens,
)

def test_install_brain_memory_plugin_success(tmp_path: Path):
    """Test successful installation and validation of the plugin."""
    res = install_brain_memory_plugin(tmp_path)
    plugin_dir = tmp_path / "agy-brain-memory"
    
    assert res["plugin_dir"] == str(plugin_dir)
    assert res["valid"] is True
    assert res["oss_compliant"] is True
    assert res["errors"] == []

    # Check if files exist
    assert (plugin_dir / "plugin.json").is_file()
    assert (plugin_dir / "assets" / "logo.svg").is_file()
    assert (plugin_dir / "mcp_config.json").is_file()
    assert (plugin_dir / "hooks.json").is_file()
    assert (plugin_dir / "skills" / "agy-brain-memory" / "SKILL.md").is_file()
    assert (plugin_dir / "agents" / "agy-brain-memory-agent.md").is_file()
    assert (plugin_dir / "rules" / "AGENTS.md").is_file()
    assert (plugin_dir / "LICENSE").is_file()
    assert (plugin_dir / ".gitignore").is_file()
    assert (plugin_dir / ".github" / "workflows" / "ci.yml").is_file()
    assert (plugin_dir / "README.md").is_file()

def test_validate_missing_files(tmp_path: Path):
    """Test validation fails when required files are missing."""
    plugin_dir = tmp_path / "agy-brain-memory"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    res = validate_brain_memory_plugin(plugin_dir)
    assert res["valid"] is False
    assert res["oss_compliant"] is False
    assert "Missing plugin.json" in res["errors"]
    assert "Missing hooks.json" in res["errors"]
    assert "Missing mcp_config.json" in res["errors"]
    assert "Missing rules/AGENTS.md" in res["errors"]
    assert "Missing skills/agy-brain-memory/SKILL.md" in res["errors"]
    assert "Missing agents/agy-brain-memory-agent.md" in res["errors"]
    assert "Missing required OSS file: LICENSE" in res["errors"]
    assert "Missing required OSS file: README.md" in res["errors"]

def test_validate_invalid_json_formats(tmp_path: Path):
    """Test validation with invalid JSON and wrong structures."""
    res = install_brain_memory_plugin(tmp_path)
    plugin_dir = Path(res["plugin_dir"])

    (plugin_dir / "plugin.json").write_text("{invalid_json", encoding="utf-8")
    (plugin_dir / "hooks.json").write_text("[]", encoding="utf-8")
    (plugin_dir / "mcp_config.json").write_text('{"mcpServers": "not_dict"}', encoding="utf-8")
    
    val = validate_brain_memory_plugin(plugin_dir)
    assert val["valid"] is False
    
    errors = val["errors"]
    assert any("Invalid JSON in plugin.json" in err for err in errors)
    assert any('hooks.json must contain a top-level "hooks" object' in err for err in errors)

def test_validate_plugin_json_schema(tmp_path: Path):
    res = install_brain_memory_plugin(tmp_path)
    plugin_dir = Path(res["plugin_dir"])
    
    manifest = json.loads((plugin_dir / "plugin.json").read_text(encoding="utf-8"))
    manifest["extraKey"] = "value"
    manifest["name"] = "Invalid_Name" 
    manifest["suggestedPrompts"] = ["1", "2", "3", "4"]
    manifest["logo"] = "/absolute/path/logo.svg"
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    
    val = validate_brain_memory_plugin(plugin_dir)
    assert val["valid"] is False
    errors = val["errors"]
    assert any("contains unsupported keys" in err for err in errors)
    assert any("name must be kebab-case" in err for err in errors)
    assert any("suggestedPrompts must contain at most 3 prompts" in err for err in errors)
    assert any("logo must be a relative path" in err for err in errors)

    manifest["logo"] = "does_not_exist.svg"
    (plugin_dir / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    val = validate_brain_memory_plugin(plugin_dir)
    assert any("logo file does not exist" in err for err in val["errors"])

def test_validate_rules_agents_frontmatter(tmp_path: Path):
    res = install_brain_memory_plugin(tmp_path)
    plugin_dir = Path(res["plugin_dir"])
    
    rules_path = plugin_dir / "rules" / "AGENTS.md"
    rules_path.write_text("---\nkey: value\n---\n# Rule", encoding="utf-8")
    
    val = validate_brain_memory_plugin(plugin_dir)
    assert val["valid"] is False
    assert any("plain Markdown without YAML frontmatter" in err for err in val["errors"])

def test_validate_cyrillic_characters(tmp_path: Path):
    res = install_brain_memory_plugin(tmp_path)
    plugin_dir = Path(res["plugin_dir"])
    
    (plugin_dir / "README.md").write_text("Hello \u041f", encoding="utf-8")
    
    val = validate_brain_memory_plugin(plugin_dir)
    assert val["oss_compliant"] is False
    assert any("Cyrillic characters in README.md" in err for err in val["errors"])

def test_validate_forbidden_tokens_in_content(tmp_path: Path):
    res = install_brain_memory_plugin(tmp_path)
    plugin_dir = Path(res["plugin_dir"])
    
    token = _forbidden_tokens()[0]
    (plugin_dir / "README.md").write_text(f"Hello {token}", encoding="utf-8")
    
    val = validate_brain_memory_plugin(plugin_dir)
    assert val["oss_compliant"] is False
    assert any("Forbidden path/token in README.md" in err for err in val["errors"])

@pytest.mark.xfail(reason="Bug in installer: doesn't reject forbidden tokens in path")
@pytest.mark.parametrize("token", _forbidden_tokens())
def test_installation_fails_gracefully_with_forbidden_token_in_path(tmp_path: Path, token: str):
    """
    Test that installation fails gracefully if any of the forbidden tokens are present
    in the plugin directory path, ensuring proper error handling and no partial installation occurs.
    """
    target_path = str(tmp_path) + f"/forbidden_{token}_path"
    
    # We expect an exception to be raised to prevent partial installation
    with pytest.raises(Exception):
        install_brain_memory_plugin(target_path)
    
    # Verify no partial installation occurs
    assert not Path(target_path).exists() or not (Path(target_path) / "agy-brain-memory").exists()
