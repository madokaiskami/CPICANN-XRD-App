"""Environment-driven settings for the isolated XDecomposer service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BUNDLED_SOURCE_DIR = Path(__file__).resolve().parents[2] / "vendor" / "XDecomposer"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for health and asset checks."""

    manifest_path: Path = Path("/app/models/xdecomposer/manifest.yaml")
    upstream_source_dir: Path | None = DEFAULT_BUNDLED_SOURCE_DIR
    require_upstream_import: bool = False
    expected_python_minor: str = "3.10"
    device: str = "auto"
    preload_model: bool = False


def load_settings() -> Settings:
    """Load settings from environment variables."""
    source_dir = os.environ.get("XDECOMPOSER_SOURCE_DIR")
    return Settings(
        manifest_path=Path(
            os.environ.get(
                "XDECOMPOSER_MANIFEST_PATH",
                "/app/models/xdecomposer/manifest.yaml",
            )
        ),
        upstream_source_dir=DEFAULT_BUNDLED_SOURCE_DIR if not source_dir else Path(source_dir),
        require_upstream_import=_env_bool("XDECOMPOSER_REQUIRE_UPSTREAM_IMPORT", default=False),
        expected_python_minor=os.environ.get("XDECOMPOSER_EXPECTED_PYTHON", "3.10"),
        device=os.environ.get("XDECOMPOSER_DEVICE", "auto"),
        preload_model=_env_bool("XDECOMPOSER_PRELOAD_MODEL", default=False),
    )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
