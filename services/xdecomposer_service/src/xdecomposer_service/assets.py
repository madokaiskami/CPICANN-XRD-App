"""Local manifest and asset checks for the isolated service."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from xdecomposer_service.schemas import AssetStatus


def inspect_assets(manifest_path: Path) -> AssetStatus:
    """Inspect manifest presence and declared asset hashes without loading weights."""
    if not manifest_path.is_file():
        return AssetStatus(
            manifest_path=str(manifest_path),
            assets_present=False,
            manifest_valid=False,
            message="manifest_not_found",
        )

    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return AssetStatus(
            manifest_path=str(manifest_path),
            assets_present=False,
            manifest_valid=False,
            message=f"manifest_unreadable: {exc}",
        )
    if not isinstance(raw, dict):
        return AssetStatus(
            manifest_path=str(manifest_path),
            assets_present=False,
            manifest_valid=False,
            message="manifest_not_mapping",
        )

    missing_fields = [
        field
        for field in ("model_id", "xrd_length", "num_sources")
        if field not in raw or raw[field] in (None, "")
    ]
    if missing_fields:
        return AssetStatus(
            manifest_path=str(manifest_path),
            assets_present=False,
            manifest_valid=False,
            message=f"manifest_missing_fields: {', '.join(missing_fields)}",
        )

    asset_errors = []
    reference_bank_required = bool(raw.get("reference_bank_required", True))
    asset_names = ["separator_checkpoint", "mae_checkpoint"]
    if reference_bank_required or raw.get("reference_bank") is not None:
        asset_names.append("reference_bank")

    for name in asset_names:
        error = _check_asset(manifest_path, name, raw.get(name))
        if error is not None:
            asset_errors.append(error)
    if asset_errors:
        return AssetStatus(
            manifest_path=str(manifest_path),
            assets_present=False,
            manifest_valid=True,
            model_id=str(raw["model_id"]),
            xrd_length=_int_or_none(raw.get("xrd_length")),
            num_sources=_int_or_none(raw.get("num_sources")),
            message="; ".join(asset_errors),
        )

    return AssetStatus(
        manifest_path=str(manifest_path),
        assets_present=True,
        manifest_valid=True,
        model_id=str(raw["model_id"]),
        xrd_length=_int_or_none(raw.get("xrd_length")),
        num_sources=_int_or_none(raw.get("num_sources")),
        message="ok",
    )


def _check_asset(manifest_path: Path, name: str, value: Any) -> str | None:
    if not isinstance(value, dict):
        return f"{name}: missing_or_invalid"
    path_value = value.get("path")
    expected_sha = value.get("sha256")
    if not path_value or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        return f"{name}: path_or_sha_invalid"
    path = Path(path_value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        return f"{name}: file_not_found"
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        return f"{name}: sha256_mismatch"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
