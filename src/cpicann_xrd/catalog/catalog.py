"""Versioned phase catalog loading and validation."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from cpicann_xrd.catalog.elements import validate_element_symbols
from cpicann_xrd.catalog.formula import parse_formula_elements, reduce_formula
from cpicann_xrd.model.hash import sha256_file
from cpicann_xrd.schemas import PhaseRecord, StrictBaseModel


class CatalogManifest(StrictBaseModel):
    """Version lock metadata for a phase catalog."""

    schema_version: str = "1.0"
    model_id: str = Field(min_length=1)
    catalog_path: Path
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_count: int = Field(gt=0)
    source_path: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("catalog_path", mode="before")
    @classmethod
    def normalize_catalog_path(cls, value: str | Path) -> Path:
        return Path(value)


class PhaseCatalog:
    """Contiguous class-index-to-phase catalog."""

    def __init__(self, records: list[PhaseRecord]) -> None:
        _validate_records(records)
        self._records = tuple(sorted(records, key=lambda record: record.class_index))
        self._by_index = {record.class_index: record for record in self._records}

    @property
    def records(self) -> tuple[PhaseRecord, ...]:
        """Return catalog records sorted by class index."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[PhaseRecord]:
        return iter(self._records)

    def get(self, class_index: int) -> PhaseRecord:
        """Return a record by class index."""
        return self._by_index[class_index]

    @classmethod
    def from_csv(cls, path: str | Path) -> PhaseCatalog:
        """Load a normalized catalog CSV."""
        catalog_path = Path(path)
        records: list[PhaseRecord] = []
        with catalog_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(phase_record_from_row(row))
        return cls(records)

    @classmethod
    def from_records(cls, records: list[PhaseRecord]) -> PhaseCatalog:
        """Create a catalog from records."""
        return cls(records)


def load_catalog(path: str | Path) -> PhaseCatalog:
    """Load and validate a normalized catalog CSV."""
    return PhaseCatalog.from_csv(path)


def load_catalog_manifest(path: str | Path) -> CatalogManifest:
    """Load catalog manifest JSON."""
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("catalog manifest must contain an object")
    return CatalogManifest.model_validate(data)


def load_catalog_from_manifest(
    manifest_path: str | Path,
) -> tuple[PhaseCatalog, CatalogManifest]:
    """Load a catalog and verify it against a manifest."""
    resolved_manifest_path = Path(manifest_path)
    manifest = load_catalog_manifest(resolved_manifest_path)
    catalog_path = manifest.catalog_path
    if not catalog_path.is_absolute():
        catalog_path = resolved_manifest_path.parent / catalog_path
    actual_sha256 = sha256_file(catalog_path)
    if actual_sha256 != manifest.catalog_sha256:
        raise ValueError(
            f"catalog SHA-256 mismatch: expected {manifest.catalog_sha256}, got {actual_sha256}"
        )
    catalog = load_catalog(catalog_path)
    if len(catalog) != manifest.record_count:
        raise ValueError(
            f"catalog record count mismatch: expected {manifest.record_count}, got {len(catalog)}"
        )
    return catalog, manifest


def phase_record_from_row(row: dict[str, Any]) -> PhaseRecord:
    """Build a phase record from a CSV row."""
    formula = _required(row, "formula")
    raw_elements = str(row.get("elements") or "").strip()
    if not raw_elements:
        raw_elements = str(row.get("symbolSet") or "").strip()
    elements = (
        frozenset(
            _normalize_catalog_element(item) for item in re_split_elements(raw_elements) if item
        )
        if raw_elements
        else parse_formula_elements(formula)
    )
    validate_element_symbols(elements)
    reduced_formula = str(row.get("reduced_formula") or "").strip() or reduce_formula(formula)
    space_group_number = _optional_int(row.get("space_group_number"))
    if space_group_number is None:
        space_group_number = _optional_int(row.get("spaceGroupNo"))
    return PhaseRecord(
        class_index=int(_required_alias(row, "class_index", "No")),
        cod_id=_normalize_cod_id(_required_alias(row, "cod_id", "dataId")),
        formula=formula,
        reduced_formula=reduced_formula,
        elements=elements,
        space_group=str(row.get("space_group") or row.get("spaceGroup") or "").strip() or None,
        space_group_number=space_group_number,
    )


def _validate_records(records: list[PhaseRecord]) -> None:
    if not records:
        raise ValueError("catalog must not be empty")
    sorted_indexes = sorted(record.class_index for record in records)
    expected = list(range(len(records)))
    if sorted_indexes != expected:
        raise ValueError("catalog class_index values must be unique and contiguous from 0")


def re_split_elements(value: str) -> list[str]:
    """Split serialized element sets."""
    return [item.strip() for item in re.split(r"[\s,;|]+", value) if item.strip()]


def _normalize_catalog_element(value: str) -> str:
    if value == "D":
        return "H"
    return value


def _required(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"catalog row is missing required field: {key}")
    return value


def _required_alias(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    raise ValueError(f"catalog row is missing required field: {'/'.join(keys)}")


def _optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    return int(text)


def _normalize_cod_id(value: str) -> str:
    if value.endswith(".0"):
        return value[:-2]
    return value
