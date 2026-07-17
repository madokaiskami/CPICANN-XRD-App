from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.release_preflight import run_preflight


def test_release_preflight_accepts_allowed_xdecomposer_docs(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    _tracked_file(tmp_path, "models/xdecomposer/README.md", "local asset notes\n")
    _tracked_file(tmp_path, "models/xdecomposer/manifest.example.yaml", "model_id: example\n")

    assert run_preflight(tmp_path) == []


def test_release_preflight_rejects_weights_runs_env_and_internal_paths(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    _tracked_file(tmp_path, "models/real/checkpoint.pt", b"weights")
    _tracked_file(tmp_path, "runs/output.csv", "sample_id,status\n")
    _tracked_file(tmp_path, ".env", "PRIVATE_TOKEN=abcdefghijklmnop\n")
    _tracked_file(tmp_path, "docs/local.md", "used /home/example/private locally\n")

    issues = run_preflight(tmp_path)
    reasons = {issue.reason for issue in issues}

    assert "tracked model/checkpoint artifact" in reasons
    assert "tracked models/ asset outside allowlist" in reasons
    assert "tracked run output" in reasons
    assert "tracked environment file" in reasons
    assert "contains internal absolute user path" in reasons
    assert "contains private token/key-like assignment" in reasons


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)


def _tracked_file(repo: Path, relative_path: str, content: str | bytes) -> None:
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", relative_path], cwd=repo, check=True, capture_output=True)
