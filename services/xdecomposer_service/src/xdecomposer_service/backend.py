"""Hash-verified real XDecomposer adapter for the isolated worker."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from xdecomposer_service.preprocessing import preprocess_for_xdecomposer, sha256_array
from xdecomposer_service.schemas import DecomposedComponent, XDecomposerRequest, XDecomposerResult
from xdecomposer_service.settings import Settings
from xdecomposer_service.upstream_adapter import import_upstream_module


class XDecomposerBackendError(RuntimeError):
    """Structured worker-side decomposition error."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class VerifiedAsset:
    """One verified asset path from the manifest."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RuntimeManifest:
    """Runtime subset of the XDecomposer manifest."""

    model_id: str
    xrd_length: int
    num_sources: int
    separator_checkpoint: VerifiedAsset
    mae_checkpoint: VerifiedAsset
    reference_bank: VerifiedAsset | None


class RealXDecomposerBackend:
    """Load the upstream XDecomposer model and run single-sample decomposition."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._manifest = _load_verified_manifest(settings.manifest_path)
        self._model: Any | None = None
        self._torch: Any | None = None

    def decompose(self, request: XDecomposerRequest) -> XDecomposerResult:
        """Run one hash-verified XDecomposer decomposition."""
        preprocessed = preprocess_for_xdecomposer(request.two_theta, request.intensity)
        torch = self._load_torch()
        model = self._load_model(torch)
        input_tensor = torch.from_numpy(preprocessed.tensor).to(self._device(torch))

        with torch.inference_mode():
            output = model(input_tensor)

        patterns, activity_logits = _extract_model_output(output, torch=torch)
        patterns = _normalize_patterns(patterns, expected_length=self._manifest.xrd_length)
        if activity_logits is None:
            activity_logits = _activity_logits_from_patterns(patterns)
            warnings = ["activity_logits_missing; probabilities derived from component mass"]
        else:
            activity_logits = _normalize_logits(activity_logits, expected_sources=patterns.shape[0])
            warnings = []

        source = preprocessed.intensity.astype(np.float32, copy=False)
        max_sources = min(request.max_sources, patterns.shape[0])
        patterns = patterns[:max_sources]
        activity_logits = activity_logits[:max_sources]
        probabilities = _sigmoid(activity_logits)
        masses = np.maximum(patterns, 0.0).sum(axis=1)
        total_mass = float(masses.sum())
        weights = masses / total_mass if total_mass > 0 else np.zeros_like(masses)
        reconstruction = np.maximum(patterns, 0.0).sum(axis=0).astype(np.float32, copy=False)
        residual = (source - reconstruction).astype(np.float32, copy=False)
        reconstruction_error = float(np.mean(np.abs(residual)))

        order = sorted(
            range(patterns.shape[0]),
            key=lambda index: (
                probabilities[index] >= request.activity_threshold,
                float(weights[index]),
                -index,
            ),
            reverse=True,
        )
        components = [
            DecomposedComponent(
                component_index=rank,
                original_slot_index=slot,
                active_probability=float(probabilities[slot]),
                is_active=bool(probabilities[slot] >= request.activity_threshold),
                estimated_weight=float(weights[slot]),
                pattern_sha256=sha256_array(patterns[slot].astype(np.float32, copy=False)),
                pattern=(
                    patterns[slot].astype(np.float32, copy=False).tolist()
                    if request.return_component_patterns
                    else None
                ),
                reference_matches=[],
                warnings=[]
                if self._manifest.reference_bank is not None
                else ["reference_bank_not_used"],
            )
            for rank, slot in enumerate(order)
        ]
        return XDecomposerResult(
            sample_id=request.sample_id,
            model_id=self._manifest.model_id,
            components=components,
            reconstruction_error=reconstruction_error,
            residual_sha256=sha256_array(residual),
            reconstruction_sha256=sha256_array(reconstruction),
            preprocessing=preprocessed.metadata,
            warnings=warnings,
            artifacts={},
        )

    def _load_torch(self) -> Any:
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on runtime image
            raise XDecomposerBackendError(
                "torch_not_importable",
                "PyTorch is required for real XDecomposer inference",
                details={"reason": str(exc)},
            ) from exc
        self._torch = torch
        return torch

    def _device(self, torch: Any) -> Any:
        if self._settings.device == "cuda" and not torch.cuda.is_available():
            raise XDecomposerBackendError(
                "cuda_unavailable",
                "XDecomposer requested CUDA but CUDA is unavailable",
            )
        if self._settings.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self._settings.device)

    def _load_model(self, torch: Any) -> Any:
        if self._model is not None:
            return self._model

        try:
            module = import_upstream_module(self._settings.upstream_source_dir)
        except RuntimeError as exc:
            raise XDecomposerBackendError(
                "upstream_not_importable",
                "Upstream XDecomposer source is not importable",
                details={
                    "source_dir": (
                        None
                        if self._settings.upstream_source_dir is None
                        else str(self._settings.upstream_source_dir)
                    ),
                    "reason": str(exc),
                },
            ) from exc

        checkpoint = torch.load(
            self._manifest.separator_checkpoint.path,
            map_location=self._device(torch),
            weights_only=False,
        )
        mae_checkpoint = torch.load(
            self._manifest.mae_checkpoint.path,
            map_location="cpu",
            weights_only=False,
        )
        model = _construct_model_from_checkpoints(
            module,
            checkpoint=checkpoint,
            mae_checkpoint=mae_checkpoint,
            fallback_num_sources=self._manifest.num_sources,
        )
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
        model.to(self._device(torch))
        model.eval()
        self._model = model
        return model


def _load_verified_manifest(path: Path) -> RuntimeManifest:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise XDecomposerBackendError(
            "manifest_unreadable",
            "XDecomposer manifest cannot be read",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if not isinstance(raw, dict):
        raise XDecomposerBackendError("manifest_invalid", "XDecomposer manifest must be a mapping")
    try:
        reference_bank_required = bool(raw.get("reference_bank_required", True))
        reference_bank_raw = raw.get("reference_bank")
        if reference_bank_required and reference_bank_raw is None:
            raise ValueError("reference_bank is required")
        return RuntimeManifest(
            model_id=str(raw["model_id"]),
            xrd_length=int(raw["xrd_length"]),
            num_sources=int(raw["num_sources"]),
            separator_checkpoint=_verify_manifest_asset(path, raw["separator_checkpoint"]),
            mae_checkpoint=_verify_manifest_asset(path, raw["mae_checkpoint"]),
            reference_bank=(
                _verify_manifest_asset(path, reference_bank_raw)
                if reference_bank_raw is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise XDecomposerBackendError(
            "manifest_invalid",
            "XDecomposer manifest fields are invalid",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def _verify_manifest_asset(manifest_path: Path, value: Any) -> VerifiedAsset:
    if not isinstance(value, dict):
        raise ValueError("asset entry must be a mapping")
    asset_path = Path(str(value["path"]))
    if not asset_path.is_absolute():
        asset_path = manifest_path.parent / asset_path
    expected_sha = str(value["sha256"])
    if not asset_path.is_file():
        raise ValueError(f"asset does not exist: {asset_path}")
    actual_size = asset_path.stat().st_size
    expected_size = value.get("size_bytes")
    if expected_size is not None and int(expected_size) != actual_size:
        raise ValueError(f"asset size mismatch: {asset_path}")
    actual_sha = _sha256_file(asset_path)
    if actual_sha != expected_sha:
        raise ValueError(f"asset sha256 mismatch: {asset_path}")
    return VerifiedAsset(path=asset_path, sha256=actual_sha, size_bytes=actual_size)


def _construct_model_from_checkpoints(
    module: Any,
    *,
    checkpoint: Any,
    mae_checkpoint: Any,
    fallback_num_sources: int,
) -> Any:
    build_xdecomposer = getattr(module, "build_xdecomposer", None)
    if build_xdecomposer is not None and isinstance(checkpoint, dict):
        config = checkpoint.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        mae_config = mae_checkpoint.get("config", {}) if isinstance(mae_checkpoint, dict) else {}
        if not isinstance(mae_config, dict):
            mae_config = {}
        mae = _construct_mae(module, config=config, mae_config=mae_config)
        return build_xdecomposer(
            mae,
            num_sources=int(config.get("num_phases", fallback_num_sources)),
            cnn_channels=config.get("cnn_channels", [64, 128, 256, 512]),
            cnn_kernels=config.get("cnn_kernels"),
            cnn_strides=config.get("cnn_strides"),
            use_transformer=not bool(config.get("no_transformer", False)),
            use_film=not bool(config.get("no_film", False)),
            use_skip_connections=not bool(config.get("no_skip_connections", False)),
            mask_type=config.get("mask_type", "soft"),
        )

    model_class = getattr(module, "XDecomposer", None)
    if model_class is None:
        raise XDecomposerBackendError(
            "upstream_model_missing",
            "Upstream module does not expose XDecomposer or build_xdecomposer",
        )
    return _construct_model_direct(model_class, num_sources=fallback_num_sources)


def _construct_mae(module: Any, *, config: dict[str, Any], mae_config: dict[str, Any]) -> Any:
    package_name = str(getattr(module, "__package__", ""))
    transformer_module_name = (
        f"{package_name}.xrd_transformer" if package_name else "src.models.xrd_transformer"
    )
    try:
        transformer_module = importlib.import_module(transformer_module_name)
    except Exception as exc:
        raise XDecomposerBackendError(
            "upstream_model_missing",
            "Upstream XRDMaskedAutoencoder module is not importable",
            details={"module": transformer_module_name, "reason": str(exc)},
        ) from exc
    mae_class = getattr(transformer_module, "XRDMaskedAutoencoder", None)
    if mae_class is None:
        raise XDecomposerBackendError(
            "upstream_model_missing",
            "Upstream module does not expose XRDMaskedAutoencoder",
            details={"module": transformer_module_name},
        )
    return mae_class(
        xrd_length=int(config.get("xrd_length", 3500)),
        d_model=int(mae_config.get("d_model", 768)),
        n_layers=int(mae_config.get("n_layers", 4)),
        n_heads=int(mae_config.get("n_heads", 12)),
        decoder_d_model=int(mae_config.get("decoder_dim", 512)),
        decoder_n_layers=int(mae_config.get("decoder_layers", 4)),
    )


def _construct_model_direct(model_class: Any, *, num_sources: int) -> Any:
    attempts = (
        {"num_sources": num_sources},
        {"num_source": num_sources},
        {"n_sources": num_sources},
        {},
    )
    errors: list[str] = []
    for kwargs in attempts:
        try:
            return model_class(**kwargs)
        except TypeError as exc:
            errors.append(f"{kwargs}: {exc}")
    raise XDecomposerBackendError(
        "model_construction_failed",
        "Cannot construct upstream XDecomposer model with known argument sets",
        details={"attempts": errors},
    )


def _extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _strip_module_prefix(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict):
        return state_dict
    stripped = {}
    for key, value in state_dict.items():
        stripped[key[7:] if isinstance(key, str) and key.startswith("module.") else key] = value
    return stripped


def _extract_model_output(
    output: Any, *, torch: Any
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
    if isinstance(output, dict):
        patterns = _first_present(
            output,
            ("component_patterns", "components", "separated_patterns", "pred_patterns", "patterns"),
        )
        logits = _first_present(output, ("activity_logits", "logits", "presence_logits"))
    elif isinstance(output, (tuple, list)):
        patterns = output[0]
        logits = output[1] if len(output) > 1 else None
    else:
        patterns = output
        logits = None
    if patterns is None:
        raise XDecomposerBackendError(
            "output_shape_mismatch",
            "XDecomposer output does not contain component patterns",
        )
    return _to_numpy(patterns, torch=torch), None if logits is None else _to_numpy(
        logits, torch=torch
    )


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _to_numpy(value: Any, *, torch: Any) -> npt.NDArray[np.float32]:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _normalize_patterns(
    patterns: npt.NDArray[np.float32],
    *,
    expected_length: int,
) -> npt.NDArray[np.float32]:
    array = np.asarray(patterns, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise XDecomposerBackendError(
            "output_shape_mismatch",
            "XDecomposer component patterns must be rank 2 after squeezing",
            details={"shape": list(array.shape)},
        )
    if array.shape[-1] != expected_length:
        raise XDecomposerBackendError(
            "output_shape_mismatch",
            "XDecomposer component pattern length does not match manifest",
            details={"shape": list(array.shape), "expected_length": expected_length},
        )
    return np.clip(array.astype(np.float32, copy=False), a_min=0.0, a_max=None)


def _normalize_logits(
    logits: npt.NDArray[np.float32],
    *,
    expected_sources: int,
) -> npt.NDArray[np.float32]:
    array = np.asarray(logits, dtype=np.float32).reshape(-1)
    if array.size < expected_sources:
        raise XDecomposerBackendError(
            "output_shape_mismatch",
            "XDecomposer activity logits shorter than component slots",
            details={"shape": list(array.shape), "expected_sources": expected_sources},
        )
    return array[:expected_sources]


def _activity_logits_from_patterns(patterns: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    masses = np.maximum(patterns, 0.0).sum(axis=1)
    max_mass = float(masses.max()) if masses.size else 0.0
    if max_mass <= 0:
        return np.full(patterns.shape[0], -20.0, dtype=np.float32)
    probabilities = np.clip(masses / max_mass, 1e-6, 1.0 - 1e-6)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def _sigmoid(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
