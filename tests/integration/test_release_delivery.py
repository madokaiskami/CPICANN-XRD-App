"""Release delivery contract tests for v0.2.0-rc1."""

from __future__ import annotations

from pathlib import Path

from scripts.release_preflight import run_preflight


def test_release_documents_exist_and_mark_xdecomposer_experimental() -> None:
    required_paths = [
        Path("CHANGELOG.md"),
        Path("docs/release_notes_v0.2.0.md"),
        Path("docs/deployment_runbook.md"),
        Path("docs/rollback.md"),
        Path("docs/third_party_licenses.md"),
        Path("docs/xdecomposer_scientific_validation.md"),
    ]

    for path in required_paths:
        assert path.is_file(), path

    release_notes = Path("docs/release_notes_v0.2.0.md").read_text(encoding="utf-8")
    validation = Path("docs/xdecomposer_scientific_validation.md").read_text(encoding="utf-8")
    assert "experimental release candidate" in release_notes
    assert "Status: experimental" in validation
    assert "It does not claim" in validation
    assert "status = experimental" in validation


def test_ci_keeps_real_model_and_xdecomposer_smoke_out_of_default_path() -> None:
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    model_smoke = Path(".github/workflows/model-smoke.yml").read_text(encoding="utf-8")

    assert "not model and not xdecomposer_model and not xdecomposer_cpicann_model" in ci
    assert "scripts/release_preflight.py" in ci
    assert "workflow_dispatch" in model_smoke
    assert "secrets.CPICANN_MODEL_URL" in model_smoke


def test_docker_and_release_workflows_cover_two_images_and_digests() -> None:
    docker = Path(".github/workflows/docker.yml").read_text(encoding="utf-8")
    release = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "docker/Dockerfile.cpu" in docker
    assert "docker/Dockerfile.xdecomposer" in docker
    assert "compose.xdecomposer.yaml" in docker
    assert "docker/Dockerfile.cpu" in release
    assert "docker/Dockerfile.xdecomposer" in release
    assert "IMAGE_DIGESTS" in release


def test_release_preflight_current_repo_passes() -> None:
    assert run_preflight(Path.cwd()) == []
