"""Verify local model artifacts against the Phase 4 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.loader import load_manifest, resolve_weight_path, verify_file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a local CPICANN model bundle.")
    parser.add_argument("--model-id", default="cpicann-single-d1")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/models/cpicann-single-d1.yaml")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if manifest.model_id != args.model_id:
        raise SystemExit(f"manifest model_id mismatch: {manifest.model_id} != {args.model_id}")
    if manifest.checkpoint_path is not None:
        checkpoint_path = args.model_dir / manifest.model_id / manifest.checkpoint_path
        if not checkpoint_path.exists():
            raise SystemExit(f"MODEL_NOT_INSTALLED: expected {checkpoint_path}")
        verify_file_sha256(
            checkpoint_path,
            manifest.checkpoint_sha256,
            error_code=ErrorCode.MODEL_HASH_MISMATCH,
        )
    weight_path = resolve_weight_path(manifest, args.model_dir)
    if not weight_path.exists():
        raise SystemExit(f"MODEL_NOT_INSTALLED: expected {weight_path}")
    try:
        actual_sha256 = verify_file_sha256(
            weight_path,
            manifest.weight_sha256,
            error_code=ErrorCode.MODEL_HASH_MISMATCH,
        )
    except CpicannXrdError as exc:
        raise SystemExit(f"{exc.error_code.value}: {exc.message}") from exc
    print(f"model_id={manifest.model_id}")
    print(f"weight_path={weight_path}")
    print(f"weight_sha256={actual_sha256}")
    print("status=ok")


if __name__ == "__main__":
    main()
