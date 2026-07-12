from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cpicann_xrd.model.registry import create_default_registry
from cpicann_xrd.settings import AppSettings, load_settings


def test_load_settings_uses_defaults_when_no_inputs() -> None:
    settings = load_settings(config_path=None, env={})

    assert settings == AppSettings()
    assert settings.backend == "fake"
    assert settings.top_k == 5


def test_load_settings_precedence_cli_over_env_over_file(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "backend: fake",
                "model_dir: file-models",
                "model_id: file-model",
                "device: cpu",
                "top_k: 3",
                "allow_model_download: false",
            ]
        ),
        encoding="utf-8",
    )
    env = {
        "CPICANN_MODEL_ID": "env-model",
        "CPICANN_TOP_K": "7",
        "CPICANN_ALLOW_MODEL_DOWNLOAD": "true",
    }

    settings = load_settings(
        config_path=config,
        env=env,
        cli_overrides={"top_k": 9, "model_dir": "cli-models"},
    )

    assert settings.model_id == "env-model"
    assert settings.top_k == 9
    assert settings.model_dir == Path("cli-models")
    assert settings.allow_model_download is True


def test_load_settings_rejects_invalid_values(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("top_k: 0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(config_path=config, env={})

    with pytest.raises(ValueError, match="Invalid boolean"):
        load_settings(config_path=None, env={"CPICANN_ALLOW_MODEL_DOWNLOAD": "maybe"})


def test_default_registry_contains_fake_and_planned_cpicann_manifest() -> None:
    registry = create_default_registry()
    manifests = registry.list()

    assert [manifest.model_id for manifest in manifests] == ["cpicann-single-d1", "fake-cpicann"]
    assert registry.get("fake-cpicann").backend == "fake"
    assert registry.get("cpicann-single-d1").num_classes == 23073
