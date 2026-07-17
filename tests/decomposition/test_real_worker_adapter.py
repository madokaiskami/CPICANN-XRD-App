from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

SERVICE_SRC = Path(__file__).resolve().parents[2] / "services" / "xdecomposer_service" / "src"
if str(SERVICE_SRC) not in sys.path:
    sys.path.insert(0, str(SERVICE_SRC))

from xdecomposer_service.backend import (  # noqa: E402
    RealXDecomposerBackend,
    XDecomposerBackendError,
)
from xdecomposer_service.main import decompose  # noqa: E402
from xdecomposer_service.schemas import XDecomposerRequest  # noqa: E402
from xdecomposer_service.settings import Settings  # noqa: E402


def test_real_adapter_runs_with_fake_upstream_module(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    manifest_path = _write_manifest(tmp_path, torch=torch)
    source_dir = _write_fake_upstream(tmp_path)
    backend = RealXDecomposerBackend(
        Settings(
            manifest_path=manifest_path,
            upstream_source_dir=source_dir,
            expected_python_minor=_current_python_minor(),
            device="cpu",
        )
    )

    result = backend.decompose(
        XDecomposerRequest(
            sample_id="adapter-smoke",
            two_theta=[10.0, 45.0, 80.0],
            intensity=[0.0, 10.0, 0.0],
            max_sources=2,
            return_component_patterns=True,
        )
    )

    assert result.model_id == "xdecomposer-adapter-test"
    assert result.preprocessing.output_points == 3500
    assert len(result.components) == 2
    assert result.components[0].pattern is not None
    assert result.reconstruction_error >= 0


def test_real_adapter_fails_closed_when_upstream_is_missing(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    manifest_path = _write_manifest(tmp_path, torch=torch)
    backend = RealXDecomposerBackend(
        Settings(
            manifest_path=manifest_path,
            upstream_source_dir=tmp_path / "missing-upstream",
            expected_python_minor=_current_python_minor(),
            device="cpu",
        )
    )

    with pytest.raises(XDecomposerBackendError) as exc_info:
        backend.decompose(
            XDecomposerRequest(
                sample_id="adapter-smoke",
                two_theta=[10.0, 45.0, 80.0],
                intensity=[0.0, 10.0, 0.0],
            )
        )

    assert exc_info.value.code == "upstream_not_importable"


def test_decompose_endpoint_uses_real_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    manifest_path = _write_manifest(tmp_path, torch=torch)
    source_dir = _write_fake_upstream(tmp_path)
    monkeypatch.setenv("XDECOMPOSER_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setenv("XDECOMPOSER_SOURCE_DIR", str(source_dir))
    monkeypatch.setenv("XDECOMPOSER_EXPECTED_PYTHON", _current_python_minor())
    monkeypatch.setenv("XDECOMPOSER_DEVICE", "cpu")

    result = decompose(
        XDecomposerRequest(
            sample_id="endpoint-smoke",
            two_theta=[10.0, 45.0, 80.0],
            intensity=[0.0, 10.0, 0.0],
            max_sources=2,
        )
    )

    assert result.model_id == "xdecomposer-adapter-test"
    assert len(result.components) == 2


def _write_fake_upstream(tmp_path: Path) -> Path:
    module_dir = tmp_path / "upstream" / "src" / "models"
    module_dir.mkdir(parents=True)
    (module_dir / "xdecomposer.py").write_text(
        "\n".join(
            [
                "import torch",
                "",
                "class XDecomposer(torch.nn.Module):",
                "    def __init__(self, num_sources=4):",
                "        super().__init__()",
                "        self.num_sources = num_sources",
                "",
                "    def forward(self, x):",
                "        scales = torch.linspace(1.0, 0.25, self.num_sources, device=x.device)",
                "        patterns = x.repeat(1, self.num_sources, 1) * scales.view(1, -1, 1)",
                "        logits = torch.linspace(2.0, -2.0, self.num_sources, device=x.device)",
                "        return {'component_patterns': patterns, 'activity_logits': logits}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path / "upstream"


def _write_manifest(tmp_path: Path, *, torch: object) -> Path:
    separator = tmp_path / "checkpoints" / "xdecomposer" / "latest.pt"
    mae = tmp_path / "checkpoints" / "pretrain" / "checkpoint_latest.pt"
    separator.parent.mkdir(parents=True)
    mae.parent.mkdir(parents=True)
    torch.save({"model": {}}, separator)
    torch.save({"config": {}}, mae)
    manifest = {
        "model_id": "xdecomposer-adapter-test",
        "xrd_length": 3500,
        "num_sources": 4,
        "reference_bank_required": False,
        "separator_checkpoint": {
            "path": "checkpoints/xdecomposer/latest.pt",
            "sha256": _sha256(separator),
        },
        "mae_checkpoint": {
            "path": "checkpoints/pretrain/checkpoint_latest.pt",
            "sha256": _sha256(mae),
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_python_minor() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"
