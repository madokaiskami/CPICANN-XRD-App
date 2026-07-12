"""Convert a trusted CPICANN checkpoint into a pure state_dict artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from cpicann_xrd.exceptions import ErrorCode
from cpicann_xrd.model.hash import sha256_file
from cpicann_xrd.model.loader import load_manifest, verify_file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CPICANN checkpoint to pure state_dict.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/models/cpicann-single-d1.yaml"),
    )
    parser.add_argument("--force", action="store_true", help="Overwrite destination if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if manifest.checkpoint_path is None:
        raise SystemExit("manifest has no checkpoint_path")
    if manifest.weight_path is None:
        raise SystemExit("manifest has no weight_path")

    source = args.model_dir / manifest.model_id / manifest.checkpoint_path
    destination = args.model_dir / manifest.model_id / manifest.weight_path
    if not source.exists():
        raise SystemExit(f"MODEL_NOT_INSTALLED: expected {source}")
    if destination.exists() and not args.force:
        raise SystemExit(f"destination exists; use --force: {destination}")

    verify_file_sha256(
        source,
        manifest.checkpoint_sha256,
        error_code=ErrorCode.MODEL_HASH_MISMATCH,
    )
    checkpoint = torch.load(source, map_location=torch.device("cpu"), weights_only=False)
    state_dict = _extract_model_state_dict(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, destination)
    print(f"source={source}")
    print(f"destination={destination}")
    print(f"state_dict_keys={len(state_dict)}")
    print(f"state_dict_sha256={sha256_file(destination)}")


def _extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise SystemExit("checkpoint must be a mapping")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise SystemExit("checkpoint does not contain a model state_dict")
    if not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in model_state.items()
    ):
        raise SystemExit("model state_dict must contain only string tensor items")
    return dict(model_state)


if __name__ == "__main__":
    main()
