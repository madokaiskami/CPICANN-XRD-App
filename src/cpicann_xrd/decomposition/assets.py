"""XDecomposer asset manifest schemas and closed verification."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.hash import sha256_file
from cpicann_xrd.schemas import StrictBaseModel

UNKNOWN_LICENSE = "UNKNOWN"
HASH_RE = r"^[0-9a-f]{64}$"


class XDecomposerAssetRef(StrictBaseModel):
    """One required XDecomposer asset reference."""

    path: Path
    sha256: str = Field(pattern=HASH_RE)
    size_bytes: int | None = Field(default=None, gt=0)

    @field_validator("path", mode="before")
    @classmethod
    def normalize_path(cls, value: str | Path) -> Path:
        if str(value).strip() == "":
            raise ValueError("asset path must not be empty")
        return Path(value)


class XDecomposerAssetManifest(StrictBaseModel):
    """Versioned XDecomposer asset manifest.

    This manifest only authorizes local asset discovery and hash verification.
    It does not load checkpoints or start a decomposition backend.
    """

    schema_version: Literal["1.0"] = "1.0"
    model_id: str = Field(min_length=1)
    upstream_repo: str = Field(min_length=1)
    upstream_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_license: str = Field(min_length=1)
    checkpoint_license: str = Field(min_length=1)
    dataset_license: str = Field(min_length=1)
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    cuda: str = Field(min_length=1)
    xrd_length: int = Field(gt=0)
    num_sources: int = Field(gt=0)
    reference_bank_required: bool = True
    separator_checkpoint: XDecomposerAssetRef
    mae_checkpoint: XDecomposerAssetRef
    reference_bank: XDecomposerAssetRef | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def reference_bank_must_be_present_when_required(self) -> XDecomposerAssetManifest:
        if self.reference_bank_required and self.reference_bank is None:
            raise ValueError("reference_bank is required when reference_bank_required is true")
        return self

    def unconfirmed_license_fields(self) -> list[str]:
        """Return license fields still set to UNKNOWN."""
        fields = {
            "source_license": self.source_license,
            "checkpoint_license": self.checkpoint_license,
        }
        if self.reference_bank_required or self.reference_bank is not None:
            fields["dataset_license"] = self.dataset_license
        return [name for name, value in fields.items() if value.strip().upper() == UNKNOWN_LICENSE]


class VerifiedAsset(StrictBaseModel):
    """Verification result for one concrete asset file."""

    name: Literal["separator_checkpoint", "mae_checkpoint", "reference_bank"]
    path: Path
    expected_sha256: str
    actual_sha256: str
    size_bytes: int


class XDecomposerAssetVerification(StrictBaseModel):
    """Successful asset verification payload."""

    status: Literal["ok"] = "ok"
    manifest_path: Path
    model_id: str
    production: bool
    assets: list[VerifiedAsset]
    unconfirmed_license_fields: list[str]


def load_xdecomposer_manifest(path: Path) -> XDecomposerAssetManifest:
    """Load and validate an XDecomposer asset manifest from YAML."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CpicannXrdError(
            ErrorCode.XDECOMPOSER_ASSET_MANIFEST_INVALID,
            "XDecomposer manifest 无法读取",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if not isinstance(data, dict):
        raise CpicannXrdError(
            ErrorCode.XDECOMPOSER_ASSET_MANIFEST_INVALID,
            "XDecomposer manifest 必须是 YAML mapping",
            details={"path": str(path)},
        )
    try:
        return XDecomposerAssetManifest.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False)
        raise CpicannXrdError(
            ErrorCode.XDECOMPOSER_ASSET_MANIFEST_INVALID,
            "XDecomposer manifest 字段无效",
            details={"path": str(path), "errors": errors},
        ) from exc


def verify_xdecomposer_assets(
    manifest_path: Path,
    *,
    production: bool = False,
) -> XDecomposerAssetVerification:
    """Verify that all manifest assets exist and match their SHA-256 values."""
    manifest = load_xdecomposer_manifest(manifest_path)
    unconfirmed = manifest.unconfirmed_license_fields()
    if production and unconfirmed:
        raise CpicannXrdError(
            ErrorCode.XDECOMPOSER_ASSET_LICENSE_UNCONFIRMED,
            "XDecomposer manifest 仍包含未确认许可字段，不能用于生产模式",
            details={
                "manifest_path": str(manifest_path),
                "fields": unconfirmed,
            },
        )

    verified_assets = [
        _verify_asset(manifest_path, "separator_checkpoint", manifest.separator_checkpoint),
        _verify_asset(manifest_path, "mae_checkpoint", manifest.mae_checkpoint),
    ]
    if manifest.reference_bank is not None:
        verified_assets.append(
            _verify_asset(manifest_path, "reference_bank", manifest.reference_bank)
        )
    return XDecomposerAssetVerification(
        manifest_path=manifest_path,
        model_id=manifest.model_id,
        production=production,
        assets=verified_assets,
        unconfirmed_license_fields=unconfirmed,
    )


def resolve_manifest_asset_path(manifest_path: Path, asset: XDecomposerAssetRef) -> Path:
    """Resolve relative asset paths against the manifest directory."""
    if asset.path.is_absolute():
        return asset.path
    return manifest_path.parent / asset.path


def _verify_asset(
    manifest_path: Path,
    name: Literal["separator_checkpoint", "mae_checkpoint", "reference_bank"],
    asset: XDecomposerAssetRef,
) -> VerifiedAsset:
    path = resolve_manifest_asset_path(manifest_path, asset)
    if not path.is_file():
        raise CpicannXrdError(
            ErrorCode.XDECOMPOSER_ASSET_MISSING,
            "XDecomposer 资产文件不存在",
            details={
                "manifest_path": str(manifest_path),
                "asset": name,
                "path": str(path),
            },
        )
    if asset.size_bytes is not None:
        actual_size = path.stat().st_size
        if actual_size != asset.size_bytes:
            raise CpicannXrdError(
                ErrorCode.XDECOMPOSER_ASSET_HASH_MISMATCH,
                "XDecomposer 资产文件大小不匹配",
                details={
                    "manifest_path": str(manifest_path),
                    "asset": name,
                    "path": str(path),
                    "expected_size_bytes": asset.size_bytes,
                    "actual_size_bytes": actual_size,
                },
            )
    else:
        actual_size = path.stat().st_size

    actual_sha256 = sha256_file(path)
    if actual_sha256 != asset.sha256:
        raise CpicannXrdError(
            ErrorCode.XDECOMPOSER_ASSET_HASH_MISMATCH,
            "XDecomposer 资产 SHA-256 校验失败",
            details={
                "manifest_path": str(manifest_path),
                "asset": name,
                "path": str(path),
                "expected_sha256": asset.sha256,
                "actual_sha256": actual_sha256,
            },
        )
    return VerifiedAsset(
        name=name,
        path=path,
        expected_sha256=asset.sha256,
        actual_sha256=actual_sha256,
        size_bytes=actual_size,
    )
