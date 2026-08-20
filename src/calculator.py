"""Deterministic weighted-average calculation over extracted fund fields.

Phase 6 of the pipeline: once Phase 3 (`fund_filter`) has picked the
qualifying funds and Phase 4 (`field_extraction`) has pulled `expense_ratio`
and `aum` out of each one, this module does the actual arithmetic the
original example question asks for - "weighted average expense ratio,
weighted by AUM" - with plain Python. No LLM call happens here, and none
ever should: the math is exact and reproducible, so letting a model do it
would trade a guaranteed-correct computation for a plausible-looking one.
"""

from __future__ import annotations

from typing import Iterable, Mapping


def weighted_average_expense_ratio(funds: Iterable[Mapping[str, object]]) -> float:
    """Return the AUM-weighted average expense ratio across *funds*.

    weighted_average = sum(expense_ratio_i * aum_i) / sum(aum_i)

    Args:
        funds: Each mapping must contain numeric 'expense_ratio' and 'aum'
            fields, e.g. plain dicts or `ExtractedFields.__dict__`-shaped
            records from `field_extraction.py`. Mapping (not a fixed
            dataclass) is used so any {expense_ratio, aum}-shaped record
            works, matching the convention in `fund_filter.py`.

    Returns:
        The AUM-weighted average expense ratio, in the same units as the
        input (e.g. 0.45 for an expense ratio of 0.45%).

    Raises:
        ValueError: If `funds` is empty, if any fund is missing
            'expense_ratio' or 'aum' or has a non-numeric value for either,
            or if total AUM across all funds is zero (the weighted average
            is undefined - there is nothing to weight by).
    """
    total_weighted_ratio = 0.0
    total_aum = 0.0
    fund_count = 0

    for fund in funds:
        expense_ratio = fund.get("expense_ratio")
        aum = fund.get("aum")

        if not isinstance(expense_ratio, (int, float)):
            raise ValueError(
                f"Each fund must contain a numeric 'expense_ratio' field, got: {expense_ratio!r}"
            )
        if not isinstance(aum, (int, float)):
            raise ValueError(f"Each fund must contain a numeric 'aum' field, got: {aum!r}")

        total_weighted_ratio += expense_ratio * aum
        total_aum += aum
        fund_count += 1

    if fund_count == 0:
        raise ValueError("Cannot compute a weighted average over zero funds")
    if total_aum == 0:
        raise ValueError(
            "Cannot compute a weighted average: total AUM across all funds is zero"
        )

    return total_weighted_ratio / total_aum
