"""Explicit model download helper.

The CPICANN pretrained repository is gated. This script intentionally requires a
user-provided URL and opt-in flag instead of silently downloading weights.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

from cpicann_xrd.exceptions import ErrorCode
from cpicann_xrd.model.loader import load_manifest, verify_file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a CPICANN model artifact explicitly.")
    parser.add_argument("--url", required=True, help="Authorized direct download URL.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/models/cpicann-single-d1.yaml")
    )
    parser.add_argument(
        "--allow", action="store_true", help="Confirm that download is explicitly allowed."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_allowed = os.environ.get("CPICANN_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}
    if not args.allow and not env_allowed:
        raise SystemExit("Set --allow or CPICANN_ALLOW_MODEL_DOWNLOAD=true to download weights.")

    manifest = load_manifest(args.manifest)
    if manifest.weight_path is None:
        raise SystemExit("manifest has no weight_path")
    destination_dir = args.model_dir / manifest.model_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / manifest.weight_path.name
    with urllib.request.urlopen(args.url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    verify_file_sha256(
        destination, manifest.weight_sha256, error_code=ErrorCode.MODEL_HASH_MISMATCH
    )
    print(f"downloaded={destination}")


if __name__ == "__main__":
    main()
