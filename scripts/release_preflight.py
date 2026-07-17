"""Release preflight checks for forbidden tracked artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_SUFFIXES = (
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".onnx",
)
ALLOWED_MODELS_FILES = frozenset(
    {
        "models/xdecomposer/README.md",
        "models/xdecomposer/manifest.example.yaml",
    }
)
INTERNAL_PATH_RE = re.compile(
    r"(/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|[A-Z]:\\\\Users\\\\)"
)
PRIVATE_SECRET_RE = re.compile(
    r"(?i)(private[_-]?token|api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
)


@dataclass(frozen=True)
class PreflightIssue:
    """One release preflight issue."""

    path: str
    reason: str


def run_preflight(repo_root: Path) -> list[PreflightIssue]:
    """Return release-blocking issues for tracked files."""
    tracked_files = _git_ls_files(repo_root)
    issues: list[PreflightIssue] = []
    for relative_path in tracked_files:
        issues.extend(_path_issues(relative_path))
        path = repo_root / relative_path
        if path.is_file() and _is_probably_text(path):
            issues.extend(_content_issues(relative_path, path.read_text(encoding="utf-8")))
    return issues


def main(argv: list[str] | None = None) -> int:
    """Run release preflight checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(argv)

    issues = run_preflight(args.repo_root)
    payload = {
        "status": "ok" if not issues else "failed",
        "issue_count": len(issues),
        "issues": [issue.__dict__ for issue in issues],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"status: {payload['status']}")
        for issue in issues:
            print(f"- {issue.path}: {issue.reason}")
    return 0 if not issues else 1


def _git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _path_issues(relative_path: str) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    path = Path(relative_path)
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        issues.append(PreflightIssue(relative_path, "tracked model/checkpoint artifact"))
    if relative_path.startswith("models/") and relative_path not in ALLOWED_MODELS_FILES:
        issues.append(PreflightIssue(relative_path, "tracked models/ asset outside allowlist"))
    if relative_path.startswith("runs/"):
        issues.append(PreflightIssue(relative_path, "tracked run output"))
    if path.name == ".env" or path.suffix == ".env":
        issues.append(PreflightIssue(relative_path, "tracked environment file"))
    return issues


def _content_issues(relative_path: str, text: str) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    if INTERNAL_PATH_RE.search(text):
        issues.append(PreflightIssue(relative_path, "contains internal absolute user path"))
    if PRIVATE_SECRET_RE.search(text):
        issues.append(PreflightIssue(relative_path, "contains private token/key-like assignment"))
    return issues


def _is_probably_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
