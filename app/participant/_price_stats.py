"""City-level price statistics used for context-relative price scoring.

The challenge's "not too expensive" signal is meaningless as an absolute
number (CHF 2000 is cheap in Zürich, expensive in Delémont). We compute
per-city price distributions once, lazily, and expose helpers the ranker
uses to score a listing relative to its own city.

Also groups stats by city+rooms because a CHF 2500 1-room flat is
expensive while a CHF 2500 4-room flat is cheap in the same city.
"""

from __future__ import annotations

import logging
import sqlite3
import statistics
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_stats() -> dict[str, Any]:
    """Compute price stats once on first use. Returns a nested dict:

        {
            "city": {city_lower: {median, p25, p75, n}},
            "city_rooms": {(city_lower, rooms_bucket): {median, p25, p75, n}},
            "global": {median, p25, p75, n},
        }

    `rooms_bucket` is the rounded-down integer (so 3, 3.5, 3.8 all bucket to 3).
    """
    settings = get_settings()
    db_path = settings.db_path
    if not db_path.exists():
        logger.warning("price stats: DB not found at %s, returning empty stats", db_path)
        return {"city": {}, "city_rooms": {}, "global": None}

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT LOWER(city) AS city_key, rooms, price
                FROM listings
                WHERE price IS NOT NULL AND price > 0
                """
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("price stats: SQL error %s", exc)
        return {"city": {}, "city_rooms": {}, "global": None}

    by_city: dict[str, list[int]] = {}
    by_city_rooms: dict[tuple[str, int], list[int]] = {}
    all_prices: list[int] = []

    for city_key, rooms, price in rows:
        if not city_key:
            continue
        try:
            price_int = int(price)
        except (TypeError, ValueError):
            continue
        all_prices.append(price_int)
        by_city.setdefault(city_key, []).append(price_int)

        if rooms is not None:
            try:
                rooms_bucket = int(float(rooms))
                by_city_rooms.setdefault((city_key, rooms_bucket), []).append(price_int)
            except (TypeError, ValueError):
                pass

    def summarize(values: list[int]) -> dict[str, float]:
        values = sorted(values)
        n = len(values)
        if n < 5:
            # Too few samples — return None-shaped dict so callers treat as missing.
            return {"median": values[n // 2] if values else 0, "p25": 0, "p75": 0, "n": n}
        return {
            "median": statistics.median(values),
            "p25": values[n // 4],
            "p75": values[(3 * n) // 4],
            "n": n,
        }

    city_stats = {k: summarize(v) for k, v in by_city.items() if len(v) >= 5}
    city_rooms_stats = {k: summarize(v) for k, v in by_city_rooms.items() if len(v) >= 5}
    global_stats = summarize(all_prices) if all_prices else None

    return {"city": city_stats, "city_rooms": city_rooms_stats, "global": global_stats}


def price_percentile(
    price: float | int | None,
    city: str | None,
    rooms: float | None,
) -> float | None:
    """Return a rough percentile-rank [0, 1] for this listing's price within
    its (city, room-count) cohort. 0 = very cheap for its cohort, 1 = very
    expensive. Returns None when we don't have enough data to judge.

    This is intentionally coarse — we just return 0.0/0.25/0.5/0.75/1.0 based
    on p25/median/p75 breakpoints. Good enough for soft ranking.
    """
    if price is None or price <= 0:
        return None

    stats = _load_stats()
    city_key = (city or "").lower() if city else None

    buckets = None
    if city_key and rooms is not None:
        try:
            rooms_bucket = int(float(rooms))
            buckets = stats["city_rooms"].get((city_key, rooms_bucket))
        except (TypeError, ValueError):
            buckets = None
    if not buckets and city_key:
        buckets = stats["city"].get(city_key)
    if not buckets:
        buckets = stats["global"]
    if not buckets or buckets.get("n", 0) < 5:
        return None

    p25, median, p75 = buckets["p25"], buckets["median"], buckets["p75"]
    if price <= p25:
        return 0.0
    if price <= median:
        return 0.33
    if price <= p75:
        return 0.66
    return 1.0


def city_median(city: str | None) -> float | None:
    if not city:
        return None
    stats = _load_stats()
    entry = stats["city"].get(city.lower())
    if not entry or entry.get("n", 0) < 5:
        return None
    return entry["median"]
