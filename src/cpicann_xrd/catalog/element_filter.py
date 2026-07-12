"""Element constraint filtering."""

from __future__ import annotations

from cpicann_xrd.catalog.elements import validate_element_symbols
from cpicann_xrd.schemas import FilterSpec, PhaseRecord


def candidate_matches_filter(record: PhaseRecord, filter_spec: FilterSpec) -> bool:
    """Return whether a candidate phase satisfies element constraints."""
    candidate_elements = validate_element_symbols(record.elements)
    include_must = validate_element_symbols(filter_spec.include_must)
    allowed_elements = (
        None
        if filter_spec.allowed_elements is None
        else validate_element_symbols(filter_spec.allowed_elements)
    )
    must_ok = include_must.issubset(candidate_elements)
    allowed_ok = allowed_elements is None or candidate_elements.issubset(allowed_elements)
    return must_ok and allowed_ok


def filter_records(
    records: tuple[PhaseRecord, ...],
    filter_spec: FilterSpec,
) -> tuple[PhaseRecord, ...]:
    """Filter records using the immutable Phase 5 truth-table semantics."""
    return tuple(record for record in records if candidate_matches_filter(record, filter_spec))
