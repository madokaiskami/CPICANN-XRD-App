"""Chemical formula parsing used by catalog construction."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping

from cpicann_xrd.catalog.elements import validate_element_symbols

FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")


def parse_formula_elements(formula: str) -> frozenset[str]:
    """Return validated element symbols parsed from a chemical formula."""
    return frozenset(_parse_formula_counts(formula))


def reduce_formula(formula: str) -> str:
    """Return a deterministic integer reduced formula when possible."""
    counts = _parse_formula_counts(formula)
    if not counts:
        raise ValueError("formula does not contain element symbols")
    integer_counts: dict[str, int] = {}
    for element, count in counts.items():
        rounded = round(count)
        if not math.isclose(count, rounded, rel_tol=0.0, abs_tol=1e-9):
            return _format_formula(counts)
        integer_counts[element] = int(rounded)
    divisor = math.gcd(*integer_counts.values())
    reduced = {element: count // divisor for element, count in integer_counts.items()}
    return _format_formula(reduced)


def _parse_formula_counts(formula: str) -> dict[str, float]:
    compact = re.sub(r"[\s_]+", "", formula)
    if not compact:
        raise ValueError("formula must not be empty")
    counts: dict[str, float] = defaultdict(float)
    position = 0
    for match in FORMULA_TOKEN_RE.finditer(compact):
        if match.start() != position:
            raise ValueError(f"unsupported formula syntax near: {compact[position:]}")
        element = _normalize_formula_element(match.group(1))
        amount_text = match.group(2)
        amount = float(amount_text) if amount_text else 1.0
        if amount <= 0:
            raise ValueError("formula counts must be positive")
        counts[element] += amount
        position = match.end()
    if position != len(compact):
        raise ValueError(f"unsupported formula syntax near: {compact[position:]}")
    validate_element_symbols(frozenset(counts))
    return dict(counts)


def _format_formula(counts: Mapping[str, int | float]) -> str:
    parts: list[str] = []
    for element in sorted(counts):
        amount = counts[element]
        if isinstance(amount, int) or amount.is_integer():
            integer = int(amount)
            suffix = "" if integer == 1 else str(integer)
        else:
            suffix = f"{amount:g}"
        parts.append(f"{element}{suffix}")
    return "".join(parts)


def _normalize_formula_element(value: str) -> str:
    if value == "D":
        return "H"
    return value
