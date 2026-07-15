from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cpicann_xrd.decomposition.assets import verify_xdecomposer_assets
from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.hash import sha256_file


def test_verify_assets_returns_hashes_for_all_assets(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)

    result = verify_xdecomposer_assets(manifest_path)

    assert result.status == "ok"
    assert result.model_id == "xdecomposer-test"
    assert [asset.name for asset in result.assets] == [
        "separator_checkpoint",
        "mae_checkpoint",
        "reference_bank",
    ]
    assert all(asset.expected_sha256 == asset.actual_sha256 for asset in result.assets)


def test_missing_asset_is_structured_error(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    (tmp_path / "reference_bank.pt").unlink()

    with pytest.raises(CpicannXrdError) as exc_info:
        verify_xdecomposer_assets(manifest_path)

    assert exc_info.value.error_code == ErrorCode.XDECOMPOSER_ASSET_MISSING
    assert exc_info.value.details["asset"] == "reference_bank"


def test_hash_mismatch_is_structured_error(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    (tmp_path / "checkpoints" / "xdecomposer" / "latest.pt").write_bytes(b"changed")

    with pytest.raises(CpicannXrdError) as exc_info:
        verify_xdecomposer_assets(manifest_path)

    assert exc_info.value.error_code == ErrorCode.XDECOMPOSER_ASSET_HASH_MISMATCH
    assert exc_info.value.details["asset"] == "separator_checkpoint"


def test_size_mismatch_is_structured_hash_error(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["mae_checkpoint"]["size_bytes"] = 999
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(CpicannXrdError) as exc_info:
        verify_xdecomposer_assets(manifest_path)

    assert exc_info.value.error_code == ErrorCode.XDECOMPOSER_ASSET_HASH_MISMATCH
    assert exc_info.value.details["asset"] == "mae_checkpoint"


def _write_manifest(tmp_path: Path) -> Path:
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
        "checkpoint_license": "Research-only",
        "dataset_license": "Research-only",
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
