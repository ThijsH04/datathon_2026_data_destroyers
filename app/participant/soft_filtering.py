"""Soft filtering layer.

Kept conservative: the ranker is the right place to penalise weak matches.
Hard-pruning here risks empty result sets when the user phrases preferences
strongly ("very quiet, very modern").

Two things we DO drop at this stage:
1. Parking-only object categories when the user clearly wants a home.
2. Top-quartile-priced listings when the user asked for the CHEAPEST option
   (`soft_budget_hint == "lowest"`) AND we still have a healthy pool left —
   this focuses the ranker on the value end of the market.
"""

from __future__ import annotations

from typing import Any

from app.participant import _price_stats


_PARKING_LIKE = {"Parkplatz", "Parkplatz, Garage", "Tiefgarage", "Einzelgarage"}

# Only trim for budget when the pool is large enough that losing a quartile
# still leaves the ranker plenty of variety.
_MIN_POOL_FOR_BUDGET_TRIM = 80


def filter_soft_facts(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates

    hints = set(soft_facts.get("object_category_hint") or [])
    user_wants_parking_object = bool(
        hints and hints.issubset({"Parkplatz", "Tiefgarage"})
    )

    step1: list[dict[str, Any]] = []
    for candidate in candidates:
        category = candidate.get("object_category")
        if category in _PARKING_LIKE and not user_wants_parking_object:
            continue
        step1.append(candidate)

    # If the parking cull emptied the list, fall back to the original pool.
    if not step1:
        step1 = list(candidates)

    soft_budget_hint = soft_facts.get("soft_budget_hint")
    if soft_budget_hint == "lowest" and len(step1) >= _MIN_POOL_FOR_BUDGET_TRIM:
        trimmed = _drop_top_price_quartile(step1)
        if len(trimmed) >= _MIN_POOL_FOR_BUDGET_TRIM // 2:
            return trimmed

    return step1


def _drop_top_price_quartile(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop listings in the top price quartile of their (city, rooms) cohort.

    Uses the same percentile buckets as the price component in ranking — a
    listing with `pct == 1.0` is in the top quartile of comparable homes.
    Listings we can't percentile (missing price or data) are kept.
    """
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        pct = _price_stats.price_percentile(
            candidate.get("price"),
            candidate.get("city"),
            candidate.get("rooms"),
        )
        if pct is not None and pct >= 1.0:
            continue
        out.append(candidate)
    return out
