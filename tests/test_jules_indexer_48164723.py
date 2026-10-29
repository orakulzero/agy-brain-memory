import pytest
import sqlite3
from pathlib import Path

# Since agy_brain_memory/indexer.py is truncated and invalid Python syntax,
# we need to extract _unquote_arg directly for testing.
def _extract_and_exec_unquote_arg():
    indexer_path = Path("agy_brain_memory/indexer.py")
    content = indexer_path.read_text()
    
    # Extract just the _unquote_arg function definition
    lines = content.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("def _unquote_arg"):
            start_idx = i
            break
            
    end_idx = start_idx
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith("def ") or lines[i].startswith("class "):
            end_idx = i
            break
            
    func_source = "\n".join(lines[start_idx:end_idx])
    
    # We also need typing.Any for the signature to work if it's there
    loc = {}
    exec("from typing import Any\n" + func_source, globals(), loc)
    return loc["_unquote_arg"]

_unquote_arg = _extract_and_exec_unquote_arg()

def test_unquote_arg_lone_quotes():
    """Verify that a lone quote character returns unchanged."""
    assert _unquote_arg('"') == '"'
    assert _unquote_arg("'") == "'"

def test_unquote_arg_normal_quotes():
    """Verify that normally quoted strings are stripped."""
    assert _unquote_arg('"hello"') == 'hello'
    assert _unquote_arg("'world'") == 'world'

def test_unquote_arg_empty_and_none():
    """Verify empty strings and None."""
    assert _unquote_arg("") == ""
    assert _unquote_arg(None) == ""

def test_unquote_arg_edge_cases():
    """Verify strings that should not be stripped."""
    assert _unquote_arg('""') == ''  # Length 2, should strip
    assert _unquote_arg("''") == ''  # Length 2, should strip
    assert _unquote_arg('"hello') == '"hello' # Missing end quote
    assert _unquote_arg("world'") == "world'" # Missing start quote
    assert _unquote_arg('"\'') == '"\'' # Mismatched quotes

# Optionally we can extract _connect_db to do some invariant checks
def _extract_and_exec_connect_db():
    indexer_path = Path("agy_brain_memory/indexer.py")
    content = indexer_path.read_text()
    
    # Extract just the _connect_db function definition
    lines = content.splitlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("def _connect_db"):
            start_idx = i
            break
            
    end_idx = start_idx
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith("def ") or lines[i].startswith("class "):
            end_idx = i
            break
            
    func_source = "\n".join(lines[start_idx:end_idx])
    
    loc = {}
    exec("import sqlite3\nfrom pathlib import Path\n" + func_source, globals(), loc)
    return loc["_connect_db"]

_connect_db = _extract_and_exec_connect_db()

def test_connect_db_pragmas(tmp_path: Path):
    """Verify _connect_db sets the correct pragmas."""
    db_path = tmp_path / "test.db"
    conn = _connect_db(db_path)
    
    cursor = conn.execute("PRAGMA journal_mode")
    assert cursor.fetchone()[0].lower() == "wal"
    
    cursor = conn.execute("PRAGMA synchronous")
    assert cursor.fetchone()[0] == 1  # 1 is NORMAL in SQLite
    
    conn.close()
