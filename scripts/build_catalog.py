"""Build or verify a version-locked CPICANN phase catalog."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cpicann_xrd.catalog.catalog import (
    load_catalog,
    load_catalog_from_manifest,
    phase_record_from_row,
)
from cpicann_xrd.model.hash import sha256_file

DEFAULT_SOURCE = Path("data/catalog/CPICANN_strucs_catalog.csv")
DEFAULT_CATALOG = Path("data/catalog/cpicann_single_phase_d1_catalog.csv")
DEFAULT_MANIFEST = Path("data/catalog/catalog_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or verify a CPICANN phase catalog.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Raw source CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-id", default="cpicann-single-d1")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify an existing catalog and manifest without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        catalog, manifest = load_catalog_from_manifest(args.manifest)
        print(f"model_id={manifest.model_id}")
        print(f"catalog_path={args.manifest.parent / manifest.catalog_path}")
        print(f"catalog_sha256={manifest.catalog_sha256}")
        print(f"record_count={len(catalog)}")
        print("status=ok")
        return

    source = args.source
    records = [phase_record_from_row(row) for row in _read_csv_rows(source)]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "class_index",
            "cod_id",
            "formula",
            "reduced_formula",
            "elements",
            "space_group",
            "space_group_number",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.class_index):
            writer.writerow(
                {
                    "class_index": record.class_index,
                    "cod_id": record.cod_id,
                    "formula": record.formula,
                    "reduced_formula": record.reduced_formula,
                    "elements": ",".join(sorted(record.elements)),
                    "space_group": record.space_group or "",
                    "space_group_number": record.space_group_number or "",
                }
            )
    catalog = load_catalog(output)
    manifest_path = args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(source) if source != output else None
    manifest_payload = {
        "schema_version": "1.0",
        "model_id": args.model_id,
        "catalog_path": str(output.relative_to(manifest_path.parent)),
        "catalog_sha256": sha256_file(output),
        "record_count": len(catalog),
        "source_path": str(source) if source != output else None,
        "source_sha256": source_sha256,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"catalog_path={output}")
    print(f"catalog_sha256={manifest_payload['catalog_sha256']}")
    print(f"record_count={len(catalog)}")
    print("status=ok")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
