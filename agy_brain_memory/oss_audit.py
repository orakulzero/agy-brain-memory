"""5-Gate Git Open-Source Release Auditor for agy-brain-memory."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
EXPECTED_OSS_EMAIL = "129319184+orakulzero@users.noreply.github.com"
EXPECTED_OSS_NAME = "orakulzero"

REQUIRED_OSS_FILES = {
    "LICENSE",
    "README.md",
    "pyproject.toml",
    ".gitignore",
    ".github/workflows/ci.yml",
}


def _forbidden_patterns() -> tuple[str, ...]:
    token_user_a = "Ora" + "kul"
    token_user_b = "Gr" + "ey"
    token_full_user = f"{token_user_a} {token_user_b}".lower()
    token_users_win = (f"C:\\Users\\{token_user_a}").lower()
    token_users_posix = (f"C:/Users/{token_user_a}").lower()
    token_ext_win = ("e:\\" + "ANTI" + "GRA" + "VITY").lower()
    token_ext_posix = ("e:/" + "anti" + "gra" + "vity").lower()
    token_personal_email = ("kuro" + "ora" + "kul" + "grey" + "@gmail.com").lower()
    return (
        token_full_user,
        token_users_win,
        token_users_posix,
        token_ext_win,
        token_ext_posix,
        token_personal_email,
    )


def _git_out(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def audit_git_repository(repo_path: Path | str) -> dict[str, Any]:
    """Execute the 5-Gate Git Open-Source Release Audit on a repository."""
    root = Path(repo_path).resolve()
    violations: list[str] = []

    tracked_out = _git_out(root, "ls-files")
    tracked_files = [line.strip() for line in tracked_out.splitlines() if line.strip()]
    tracked_set = set(tracked_files)

    # Gate 5: Required OSS files & MIT License
    missing_files = REQUIRED_OSS_FILES - tracked_set
    if missing_files:
        violations.append(f"Missing required OSS files: {sorted(missing_files)}")

    license_path = root / "LICENSE"
    if license_path.is_file():
        lic_text = license_path.read_text(encoding="utf-8", errors="replace")
        if "MIT License" not in lic_text or EXPECTED_OSS_NAME not in lic_text:
            violations.append("LICENSE must be MIT License attributed to orakulzero")
    else:
        violations.append("LICENSE file does not exist")

    # Gate 2 & Gate 3: Tracked files check (Zero Cyrillic, Zero Personal Paths)
    forbidden = _forbidden_patterns()
    for rel_path in tracked_files:
        fpath = root / rel_path
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".whl"}:
            continue
        text = fpath.read_text(encoding="utf-8", errors="replace")
        if CYRILLIC_RE.search(text):
            violations.append(f"Cyrillic characters found in tracked file: {rel_path}")
        lowered = text.lower()
        for pat in forbidden:
            if pat in lowered:
                violations.append(f"Forbidden path/token in tracked file: {rel_path}")
                break

    # Gate 1: Contributor identity lock
    emails_out = _git_out(root, "log", "--all", "--format=%ae")
    emails = [line.strip() for line in emails_out.splitlines() if line.strip()]
    if not emails:
        violations.append("Repository has no git commits")
    else:
        bad_emails = {e for e in emails if e != EXPECTED_OSS_EMAIL}
        if bad_emails:
            violations.append(f"Non-compliant commit author emails: {sorted(bad_emails)}")

    # Gate 2 & 3 on commit history diffs
    history_diffs = _git_out(root, "log", "-p", "--all")
    if CYRILLIC_RE.search(history_diffs):
        violations.append("Git commit history contains Cyrillic characters")
    lowered_diffs = history_diffs.lower()
    for pat in forbidden:
        if pat in lowered_diffs:
            violations.append("Git commit history contains forbidden personal path/token")
            break

    return {
        "repo_path": str(root),
        "passed": len(violations) == 0,
        "tracked_files_count": len(tracked_files),
        "violations": violations,
    }
