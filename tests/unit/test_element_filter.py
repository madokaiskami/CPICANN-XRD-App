from __future__ import annotations

import pytest
from pydantic import ValidationError

from cpicann_xrd.catalog.catalog import load_catalog
from cpicann_xrd.catalog.element_filter import filter_records
from cpicann_xrd.schemas import FilterSpec


def test_element_filter_truth_table() -> None:
    catalog = load_catalog("data/catalog/phase5_fixture_catalog.csv")

    assert _formulas(filter_records(catalog.records, FilterSpec(include_must={"Zr", "O"}))) == [
        "Li2ZrO3",
        "ZrO2",
        "LiZrTiO4",
        "HfZrO4",
    ]
    assert _formulas(
        filter_records(catalog.records, FilterSpec(allowed_elements={"Li", "Zr", "O"}))
    ) == [
        "Li2ZrO3",
        "ZrO2",
        "Li2O",
    ]
    assert _formulas(
        filter_records(
            catalog.records,
            FilterSpec(include_must={"Zr", "O"}, allowed_elements={"Li", "Zr", "O"}),
        )
    ) == [
        "Li2ZrO3",
        "ZrO2",
    ]
    assert _formulas(
        filter_records(
            catalog.records,
            FilterSpec(include_must={"Li", "Zr", "O"}, allowed_elements={"Li", "Zr", "O"}),
        )
    ) == [
        "Li2ZrO3",
    ]
    assert filter_records(catalog.records, FilterSpec(include_must={"Pb"})) == ()


def test_filter_spec_rejects_illegal_element_symbol() -> None:
    with pytest.raises(ValidationError, match="Invalid element symbols"):
        FilterSpec(include_must={"Xx"})


def _formulas(records: tuple) -> list[str]:
    return [record.formula for record in records]
