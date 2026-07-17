from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

SERVICE_SRC = Path(__file__).resolve().parents[2] / "services" / "xdecomposer_service" / "src"
if str(SERVICE_SRC) not in sys.path:
    sys.path.insert(0, str(SERVICE_SRC))

from xdecomposer_service.health import build_health, build_info, build_ready  # noqa: E402
from xdecomposer_service.settings import Settings  # noqa: E402


def test_healthz_is_process_liveness() -> None:
    response = build_health()

    assert response.status == "ok"
    assert response.service == "xdecomposer-service"


def test_readyz_reports_not_ready_when_manifest_is_missing(tmp_path: Path) -> None:
    response = build_ready(
        Settings(
            manifest_path=tmp_path / "missing.yaml",
            expected_python_minor=_current_python_minor(),
        )
    )

    assert response.status == "not_ready"
    assert response.assets.assets_present is False
    assert response.assets.manifest_valid is False
    assert "assets_not_ready" in response.warnings


def test_readyz_reports_ready_with_hash_verified_assets(tmp_path: Path) -> None:
    manifest_path = _write_runtime_manifest(tmp_path)

    response = build_ready(
        Settings(
            manifest_path=manifest_path,
            expected_python_minor=_current_python_minor(),
        )
    )

    assert response.status == "ready"
    assert response.assets.assets_present is True
    assert response.assets.manifest_valid is True
    assert response.assets.model_id == "xdecomposer-runtime-test"
    assert response.assets.xrd_length == 3500
    assert response.assets.num_sources == 4


def test_info_reports_optional_upstream_import(tmp_path: Path) -> None:
    manifest_path = _write_runtime_manifest(tmp_path)
    source_dir = tmp_path / "upstream"
    module_dir = source_dir / "src" / "models"
    module_dir.mkdir(parents=True)
    (module_dir / "xdecomposer.py").write_text("class XDecomposer: pass\n", encoding="utf-8")

    response = build_info(
        Settings(
            manifest_path=manifest_path,
            upstream_source_dir=source_dir,
            expected_python_minor=_current_python_minor(),
        )
    )

    assert response.upstream.importable is True
    assert response.upstream.module == "src.models.xdecomposer"
    json.loads(response.model_dump_json())


def test_require_upstream_import_blocks_readiness_when_missing(tmp_path: Path) -> None:
    manifest_path = _write_runtime_manifest(tmp_path)

    response = build_ready(
        Settings(
            manifest_path=manifest_path,
            require_upstream_import=True,
            expected_python_minor=_current_python_minor(),
        )
    )

    assert response.status == "not_ready"
    assert "upstream_not_importable" in response.warnings


def _write_runtime_manifest(tmp_path: Path) -> Path:
    separator = tmp_path / "checkpoints" / "xdecomposer" / "latest.pt"
    mae = tmp_path / "checkpoints" / "pretrain" / "checkpoint_latest.pt"
    reference = tmp_path / "reference_bank.pt"
    separator.parent.mkdir(parents=True)
    mae.parent.mkdir(parents=True)
    separator.write_bytes(b"separator")
    mae.write_bytes(b"mae")
    reference.write_bytes(b"reference")
    manifest = {
        "model_id": "xdecomposer-runtime-test",
        "xrd_length": 3500,
        "num_sources": 4,
        "separator_checkpoint": {
            "path": "checkpoints/xdecomposer/latest.pt",
            "sha256": _sha256(separator),
        },
        "mae_checkpoint": {
            "path": "checkpoints/pretrain/checkpoint_latest.pt",
            "sha256": _sha256(mae),
        },
        "reference_bank": {
            "path": "reference_bank.pt",
            "sha256": _sha256(reference),
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_python_minor() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"
