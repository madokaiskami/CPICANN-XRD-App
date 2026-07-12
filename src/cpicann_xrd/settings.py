"""Application settings and configuration loading."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AppSettings(BaseModel):
    """Validated application settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["fake", "cpicann"] = "fake"
    model_dir: Path = Path("models")
    model_id: str = "cpicann-single-d1"
    device: str = "cpu"
    output_dir: Path = Path("runs")
    top_k: int = Field(default=5, gt=0)
    allow_model_download: bool = False

    @field_validator("model_dir", "output_dir", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()


ENV_TO_FIELD = {
    "CPICANN_BACKEND": "backend",
    "CPICANN_MODEL_DIR": "model_dir",
    "CPICANN_MODEL_ID": "model_id",
    "CPICANN_DEVICE": "device",
    "CPICANN_OUTPUT_DIR": "output_dir",
    "CPICANN_TOP_K": "top_k",
    "CPICANN_ALLOW_MODEL_DOWNLOAD": "allow_model_download",
}


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _coerce_env_value(field: str, value: str) -> Any:
    if field == "top_k":
        return int(value)
    if field == "allow_model_download":
        return _parse_bool(value)
    return value


def read_config_file(path: Path | None) -> dict[str, Any]:
    """Read a YAML config file, returning an empty mapping when absent."""
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("configuration file must contain a mapping")
    return dict(data)


def read_env_overrides(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Read CPICANN_* environment variable overrides."""
    source = os.environ if env is None else env
    values: dict[str, Any] = {}
    for env_name, field in ENV_TO_FIELD.items():
        if env_name in source:
            values[field] = _coerce_env_value(field, source[env_name])
    return values


def load_settings(
    *,
    config_path: Path | None = Path("configs/default.yaml"),
    env: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> AppSettings:
    """Load settings with precedence: CLI > env > config file > defaults."""
    merged: dict[str, Any] = {}
    merged.update(read_config_file(config_path))
    merged.update(read_env_overrides(env))
    if cli_overrides:
        merged.update({key: value for key, value in cli_overrides.items() if value is not None})
    return AppSettings.model_validate(merged)
