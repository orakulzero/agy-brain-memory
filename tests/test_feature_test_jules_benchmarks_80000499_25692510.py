from pathlib import Path
import json
import pytest

from agy_brain_memory.benchmarks import _write_transcript


def validate_transcript_schema(transcript_path: Path) -> bool:
    """
    Validates the schema of a transcript JSON file.
    
    Args:
        transcript_path (Path): Path to the transcript JSON file.
    
    Returns:
        bool: True if the schema is valid, False otherwise.
    """
    if not transcript_path.exists():
        return False
    
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                step = json.loads(line)
                if not isinstance(step, dict) or "step_index" not in step or "type" not in step:
                    return False
            except json.JSONDecodeError:
                return False
    
    return True


def test_validate_transcript_schema(tmp_path: Path):
    from agy_brain_memory.benchmarks import _write_transcript
    
    # Valid transcript
    steps = [{"step_index": 1, "type": "USER_INPUT"}, {"step_index": 2, "type": "SYSTEM_RESPONSE"}]
    conv_dir = tmp_path / "conv_dir"
    tpath = _write_transcript(conv_dir, steps)
    
    assert validate_transcript_schema(tpath) is True
    
    # Invalid transcript - missing step_index
    invalid_steps = [{"type": "USER_INPUT"}]
    conv_dir = tmp_path / "conv_dir_invalid"
    tpath_invalid = _write_transcript(conv_dir, invalid_steps)
    
    assert validate_transcript_schema(tpath_invalid) is False
    
    # Invalid transcript - corrupted JSON
    corrupted_tpath = tmp_path / "corrupted_transcript.jsonl"
    corrupted_tpath.write_text("{corrupted: json}\n", encoding="utf-8")
    
    assert validate_transcript_schema(corrupted_tpath) is False
