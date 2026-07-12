from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cpicann_xrd.catalog.catalog import (
    PhaseCatalog,
    load_catalog_from_manifest,
    phase_record_from_row,
)
from cpicann_xrd.schemas import PhaseRecord


def test_catalog_manifest_loads_real_cpicann_catalog() -> None:
    catalog, manifest = load_catalog_from_manifest(Path("data/catalog/catalog_manifest.json"))

    assert manifest.model_id == "cpicann-single-d1"
    assert manifest.record_count == 23073
    assert len(catalog) == 23073
    assert catalog.get(0).cod_id == "1522982"
    assert catalog.get(0).formula == "Mn4 Ni8 Sn4"
    assert catalog.get(0).elements == frozenset({"Mn", "Ni", "Sn"})
    assert catalog.get(23072).cod_id == "4031642"
    assert catalog.get(23072).formula == "Zr8 V4 Ni12"


def test_catalog_manifest_loads_fixture_catalog() -> None:
    catalog, manifest = load_catalog_from_manifest(
        Path("data/catalog/phase5_fixture_catalog_manifest.json")
    )

    assert manifest.model_id == "phase5-fixture-catalog"
    assert manifest.record_count == 5
    assert len(catalog) == 5
    assert catalog.get(0).cod_id == "FIXTURE-00000"
    assert catalog.get(0).elements == frozenset({"Li", "Zr", "O"})
    assert catalog.get(1).space_group_number == 137


def test_catalog_rejects_non_contiguous_or_duplicate_indexes() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        PhaseCatalog.from_records(
            [
                _record(0, "Li2O"),
                _record(2, "ZrO2"),
            ]
        )
    with pytest.raises(ValueError, match="contiguous"):
        PhaseCatalog.from_records(
            [
                _record(0, "Li2O"),
                _record(0, "ZrO2"),
            ]
        )


def test_catalog_parses_formula_when_elements_are_missing(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_index",
                "cod_id",
                "formula",
                "reduced_formula",
                "elements",
                "space_group",
                "space_group_number",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "class_index": "0",
                "cod_id": "COD-1",
                "formula": "Li2ZrO3",
                "reduced_formula": "",
                "elements": "",
                "space_group": "P2_1/c",
                "space_group_number": "14",
            }
        )

    catalog = PhaseCatalog.from_csv(catalog_path)

    assert catalog.get(0).elements == frozenset({"Li", "Zr", "O"})
    assert catalog.get(0).reduced_formula == "Li2ZrO3"


def test_catalog_rejects_invalid_element_symbols() -> None:
    with pytest.raises(ValueError, match="Invalid element symbols"):
        phase_record_from_row(
            {
                "class_index": "0",
                "cod_id": "bad",
                "formula": "Li2XxO3",
            }
        )


def _record(class_index: int, formula: str) -> PhaseRecord:
    return PhaseRecord(
        class_index=class_index,
        cod_id=f"COD-{class_index}",
        formula=formula,
        reduced_formula=formula,
        elements={"Li", "O"},
    )
