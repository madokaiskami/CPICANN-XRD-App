from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cpicann_xrd.cli import app
from cpicann_xrd.decomposition.assets import (
    load_xdecomposer_manifest,
    verify_xdecomposer_assets,
)
from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.hash import sha256_file


def test_valid_manifest_loads_required_contract(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)

    manifest = load_xdecomposer_manifest(manifest_path)

    assert manifest.model_id == "xdecomposer-test"
    assert manifest.upstream_commit == "48c4efe3681256cd16d3b8d2f56d665cf58fd129"
    assert manifest.source_license == "MIT"
    assert manifest.checkpoint_license == "Research-only"
    assert manifest.dataset_license == "Research-only"
    assert manifest.python == "3.10"
    assert manifest.torch == "2.10.0"
    assert manifest.cuda == "12.8"
    assert manifest.xrd_length == 3500
    assert manifest.num_sources == 4
    assert manifest.unconfirmed_license_fields() == []


def test_missing_required_field_is_structured_error(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    del data["model_id"]
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(CpicannXrdError) as exc_info:
        load_xdecomposer_manifest(manifest_path)

    assert exc_info.value.error_code == ErrorCode.XDECOMPOSER_ASSET_MANIFEST_INVALID
    assert exc_info.value.details["errors"][0]["loc"] == ("model_id",)


def test_invalid_sha256_is_structured_error(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["separator_checkpoint"]["sha256"] = "not-a-sha"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(CpicannXrdError) as exc_info:
        load_xdecomposer_manifest(manifest_path)

    assert exc_info.value.error_code == ErrorCode.XDECOMPOSER_ASSET_MANIFEST_INVALID


def test_empty_required_value_is_structured_error(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["model_id"] = ""
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(CpicannXrdError) as exc_info:
        load_xdecomposer_manifest(manifest_path)

    assert exc_info.value.error_code == ErrorCode.XDECOMPOSER_ASSET_MANIFEST_INVALID


def test_unknown_license_rejects_production_mode(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, checkpoint_license="UNKNOWN")

    with pytest.raises(CpicannXrdError) as exc_info:
        verify_xdecomposer_assets(manifest_path, production=True)

    assert exc_info.value.error_code == ErrorCode.XDECOMPOSER_ASSET_LICENSE_UNCONFIRMED
    assert exc_info.value.details["fields"] == ["checkpoint_license"]


def test_example_manifest_parses_but_is_not_production_ready() -> None:
    manifest_path = Path("models/xdecomposer/manifest.example.yaml")

    manifest = load_xdecomposer_manifest(manifest_path)

    assert manifest.model_id == "xdecomposer-mp20-example"
    assert "checkpoint_license" in manifest.unconfirmed_license_fields()
    with pytest.raises(CpicannXrdError) as exc_info:
        verify_xdecomposer_assets(manifest_path, production=True)
    assert exc_info.value.error_code == ErrorCode.XDECOMPOSER_ASSET_LICENSE_UNCONFIRMED


def test_cli_verify_assets_json_success(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["xdecomposer", "verify-assets", "--manifest", str(manifest_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["model_id"] == "xdecomposer-test"
    assert len(payload["assets"]) == 3


def _write_manifest(
    tmp_path: Path,
    *,
    checkpoint_license: str = "Research-only",
    dataset_license: str = "Research-only",
) -> Path:
    separator = tmp_path / "checkpoints" / "xdecomposer" / "latest.pt"
    mae = tmp_path / "checkpoints" / "pretrain" / "checkpoint_latest.pt"
    reference = tmp_path / "reference_bank.pt"
    separator.parent.mkdir(parents=True)
    mae.parent.mkdir(parents=True)
    separator.write_bytes(b"separator")
    mae.write_bytes(b"mae")
    reference.write_bytes(b"reference")

    manifest = {
        "schema_version": "1.0",
        "model_id": "xdecomposer-test",
        "upstream_repo": "https://github.com/Licht0812/XDecomposer",
        "upstream_commit": "48c4efe3681256cd16d3b8d2f56d665cf58fd129",
        "source_license": "MIT",
        "checkpoint_license": checkpoint_license,
        "dataset_license": dataset_license,
        "python": "3.10",
        "torch": "2.10.0",
        "cuda": "12.8",
        "xrd_length": 3500,
        "num_sources": 4,
        "separator_checkpoint": {
            "path": "checkpoints/xdecomposer/latest.pt",
            "sha256": sha256_file(separator),
            "size_bytes": separator.stat().st_size,
        },
        "mae_checkpoint": {
            "path": "checkpoints/pretrain/checkpoint_latest.pt",
            "sha256": sha256_file(mae),
            "size_bytes": mae.stat().st_size,
        },
        "reference_bank": {
            "path": "reference_bank.pt",
            "sha256": sha256_file(reference),
            "size_bytes": reference.stat().st_size,
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path
