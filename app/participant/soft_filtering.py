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

import math
from typing import Any

from app.participant import _price_stats


_PARKING_LIKE = {"Parkplatz", "Parkplatz, Garage", "Tiefgarage", "Einzelgarage"}
_HOME_LIKE = {
    "Attikawohnung",
    "Dachwohnung",
    "Duplex",
    "Einliegerwohnung",
    "Loft",
    "Maisonette",
    "Studio",
    "Terrassenwohnung",
    "Wohnung",
    "Zimmer",
}
_HOME_HINTS = {"Dachwohnung", "Loft", "Maisonette", "Studio", "Wohnung"}

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

    if hints & _HOME_HINTS:
        home_trimmed = [
            candidate
            for candidate in step1
            if not candidate.get("object_category") or candidate.get("object_category") in _HOME_LIKE
        ]
        if len(home_trimmed) >= max(10, len(step1) // 5):
            step1 = home_trimmed

    anchor_trimmed = _trim_to_anchor_area(step1, soft_facts)
    if anchor_trimmed:
        step1 = anchor_trimmed

    soft_budget_hint = soft_facts.get("soft_budget_hint")
    if soft_budget_hint == "lowest" and len(step1) >= _MIN_POOL_FOR_BUDGET_TRIM:
        trimmed = _drop_top_price_quartile(step1)
        if len(trimmed) >= _MIN_POOL_FOR_BUDGET_TRIM // 2:
            return trimmed

    return step1


def _trim_to_anchor_area(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[dict[str, Any]]:
    anchors = list(soft_facts.get("anchors") or [])
    if not anchors or len(candidates) < 20:
        return []

    trimmed: list[dict[str, Any]] = []
    for candidate in candidates:
        lat = candidate.get("latitude")
        lon = candidate.get("longitude")
        if lat is None or lon is None:
            continue

        for anchor in anchors:
            try:
                distance = _distance_km(
                    float(lat),
                    float(lon),
                    float(anchor["lat"]),
                    float(anchor["lon"]),
                )
            except (TypeError, ValueError, KeyError):
                continue
            if distance <= _anchor_radius_km(anchor):
                trimmed.append(candidate)
                break

    # This is a soft trim. If the radius would starve the ranker, keep the
    # broader pool and let the anchor-distance component order it.
    if len(trimmed) >= max(10, len(candidates) // 20):
        return trimmed
    return []


def _anchor_radius_km(anchor: dict[str, Any]) -> float:
    minutes = anchor.get("max_minutes")
    if minutes:
        try:
            target_km = float(minutes) / 60.0 * 18.0
        except (TypeError, ValueError):
            target_km = 0.0
        return max(8.0, target_km * 2.0)
    return 25.0


def _distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return earth_radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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
