"""Chemical formula parsing used by catalog construction."""

from __future__ import annotations

import re
import warnings

from pymatgen.core.composition import Composition

from cpicann_xrd.catalog.elements import validate_element_symbols

D_TOKEN_RE = re.compile(r"(?<![A-Za-z])D(?=\d|\s|$)")


def parse_formula_elements(formula: str) -> frozenset[str]:
    """Return validated element symbols parsed from a chemical formula."""
    composition = _parse_composition(formula)
    elements = frozenset(element.symbol for element in composition.elements)
    return validate_element_symbols(elements)


def reduce_formula(formula: str) -> str:
    """Return pymatgen's reduced formula."""
    composition = _parse_composition(formula)
    validate_element_symbols(frozenset(element.symbol for element in composition.elements))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No Pauling electronegativity.*",
            category=UserWarning,
        )
        return composition.reduced_formula


def _parse_composition(formula: str) -> Composition:
    compact = formula.strip()
    if not compact:
        raise ValueError("formula must not be empty")
    normalized = D_TOKEN_RE.sub("H", compact)
    return Composition(normalized)
