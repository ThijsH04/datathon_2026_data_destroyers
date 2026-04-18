"""Multi-signal listing ranker.

Core idea: the LLM already decided how important each soft signal is (via the
`weight` field in `soft_facts["preferences"]`). The ranker's job is to evaluate
each signal against the listing's structured fields + description text, then
combine the per-signal scores using those LLM-assigned weights.

Design choices
--------------
- Component scorers (text / features / anchors / structured distances / price
  / category / image / data completeness) each emit a score in [0, 1] and a
  list of human-readable reasons.
- The combiner uses LLM-assigned preference weights directly — no hard-coded
  "text is 3x feature" constant. Component magnitudes are shaped only by how
  strongly the user asked for each thing.
- When the LLM reports `conflicts` (e.g. "cheap BUT central"), the combiner
  switches from a weighted arithmetic mean to a weighted geometric mean so
  listings that score high on only one conflicting axis are penalised — this
  matches "Strategy B: balance all signals" from the challenge slides.
- Context-relative pricing via `_price_stats`: "not too expensive" is scored
  against the per-(city, rooms) price cohort, not an absolute CHF number.
- Reasons are derived from the same components that drove the score, so the
  number on the card always matches the words next to it.
- All signals degrade gracefully when fields are missing — listings with
  partial data are not auto-penalised below those with complete data.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from app.models.schemas import ListingData, RankedListingResult
from app.participant import _price_stats
from app.participant._lexicon import SOFT_KEYWORDS


# Per-component base weight (applied AFTER preference weights). These shape
# how much a full-strength component can influence the final score; the
# LLM-assigned preference weights still decide which components actually fire.
COMPONENT_BASE = {
    "text_match": 3.0,
    "feature_match": 2.5,
    "anchor_distance": 2.5,
    "structured_distance": 1.5,
    "price": 1.5,
    "object_category": 1.0,
    "image_availability": 0.4,
    "data_completeness": 0.3,
}


def rank_listings(
    candidates: list[dict[str, Any]],
    soft_facts: dict[str, Any],
) -> list[RankedListingResult]:
    if not candidates:
        return []

    preferences: dict[str, float] = dict(soft_facts.get("preferences") or {})
    feature_boosts: dict[str, float] = dict(soft_facts.get("feature_boosts") or {})
    anchors: list[dict[str, Any]] = list(soft_facts.get("anchors") or [])
    object_hints: list[str] = list(soft_facts.get("object_category_hint") or [])
    soft_budget_hint: str = soft_facts.get("soft_budget_hint") or "none"
    conflicts: list = list(soft_facts.get("conflicts") or [])
    evidence: dict[str, str] = dict(soft_facts.get("evidence") or {})
    dominant: str | None = soft_facts.get("dominant_signal")

    use_geometric = bool(conflicts)

    scored: list[tuple[float, dict[str, list[str]], dict[str, Any]]] = []
    for candidate in candidates:
        components: dict[str, float] = {}
        explanations: dict[str, list[str]] = {}

        text_score, text_reasons = _score_text(candidate, preferences)
        if text_score:
            components["text_match"] = text_score
            explanations["text_match"] = text_reasons

        feature_score, feature_reasons = _score_features(candidate, feature_boosts)
        if feature_score:
            components["feature_match"] = feature_score
            explanations["feature_match"] = feature_reasons

        anchor_score, anchor_reasons = _score_anchor_distance(candidate, anchors)
        if anchor_score:
            components["anchor_distance"] = anchor_score
            explanations["anchor_distance"] = anchor_reasons

        structured_score, structured_reasons = _score_structured_distances(candidate, preferences)
        if structured_score:
            components["structured_distance"] = structured_score
            explanations["structured_distance"] = structured_reasons

        price_score, price_reasons = _score_price(
            candidate, preferences, soft_budget_hint,
        )
        if price_score:
            components["price"] = price_score
            explanations["price"] = price_reasons

        category_score, category_reasons = _score_object_category(candidate, object_hints)
        if category_score:
            components["object_category"] = category_score
            explanations["object_category"] = category_reasons

        image_score = _score_image_availability(candidate)
        if image_score:
            components["image_availability"] = image_score
            explanations["image_availability"] = ["has photos"]

        completeness_score = _score_data_completeness(candidate)
        if completeness_score:
            components["data_completeness"] = completeness_score

        # Combine components: weighted arithmetic by default, geometric when
        # the LLM flagged conflicts (rewards balance over single-axis spikes).
        if use_geometric:
            raw_score = _weighted_geometric(components)
        else:
            raw_score = sum(
                COMPONENT_BASE[name] * value for name, value in components.items()
            )
        scored.append((raw_score, explanations, candidate))

    # Min-max normalise so scores fall in [0, 1] within this query's pool.
    raw_scores = [item[0] for item in scored]
    lo, hi = (min(raw_scores), max(raw_scores)) if raw_scores else (0.0, 0.0)
    span = max(hi - lo, 1e-6)

    ranked: list[RankedListingResult] = []
    for raw_score, explanations, candidate in scored:
        normalized = (raw_score - lo) / span if span > 0 else 0.5
        # Lift the absolute floor a bit so the worst result still reads as
        # "matches your filters" rather than "score 0.00".
        score = round(0.05 + 0.95 * normalized, 4)
        reason = _format_reason(explanations, dominant, evidence)
        ranked.append(
            RankedListingResult(
                listing_id=str(candidate["listing_id"]),
                score=score,
                reason=reason,
                listing=_to_listing_data(candidate),
            )
        )

    ranked.sort(key=lambda r: (-r.score, r.listing_id))
    return ranked


# ---------------------------------------------------------------------------
# Individual signal scorers
# ---------------------------------------------------------------------------


def _score_text(
    candidate: dict[str, Any], preferences: dict[str, float],
) -> tuple[float, list[str]]:
    if not preferences:
        return 0.0, []
    haystack = _candidate_text(candidate)
    if not haystack:
        return 0.0, []

    total = 0.0
    matches: list[str] = []
    for concept, weight in preferences.items():
        synonyms = SOFT_KEYWORDS.get(concept, [])
        if not synonyms:
            continue
        hits = sum(1 for syn in synonyms if syn in haystack)
        if hits == 0:
            continue
        # Diminishing returns: 1 hit = 1.0, 2 hits = 1.4, 3+ ≈ 1.6
        concept_score = weight * (1.0 + 0.4 * math.log1p(max(hits - 1, 0)))
        total += concept_score
        matches.append(concept.replace("_", " "))

    # Normalise by the number of preferences (so many-concept queries don't get
    # artificially hot scores) and by max possible weight (1.5) so the output
    # lives in ~[0, 1.5].
    norm = total / max(len(preferences), 1)
    return norm, matches


def _candidate_text(candidate: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "description", "street", "city"):
        value = candidate.get(key)
        if value:
            parts.append(str(value))
    return _strip_html(" ".join(parts).lower())


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _score_features(
    candidate: dict[str, Any],
    feature_boosts: dict[str, float],
) -> tuple[float, list[str]]:
    if not feature_boosts:
        return 0.0, []

    candidate_features = candidate.get("features") or []
    feature_set = {str(f).lower() for f in candidate_features}

    total = 0.0
    reasons: list[str] = []
    for column, weight in feature_boosts.items():
        feature_key = column.replace("feature_", "")
        if feature_key in feature_set:
            total += weight
            reasons.append(feature_key.replace("_", " "))
    if total == 0:
        return 0.0, []
    return total / max(len(feature_boosts), 1), reasons


def _score_anchor_distance(
    candidate: dict[str, Any], anchors: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    if not anchors:
        return 0.0, []
    lat = candidate.get("latitude")
    lon = candidate.get("longitude")
    if lat is None or lon is None:
        return 0.0, []

    best_score = 0.0
    best_reason = ""
    for anchor in anchors:
        try:
            dist_km = _haversine(
                float(lat), float(lon), float(anchor["lat"]), float(anchor["lon"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
        # Default decay: 0 km → 1.0, 8 km → ~0.37, 15 km → ~0.15, 30 km+ ≈ 0
        component = math.exp(-dist_km / 8.0)
        commute = anchor.get("max_minutes")
        if commute:
            # Rough Swiss public-transport door-to-door pace ≈ 18 km/h. If the
            # user gave a commute window we sharpen the curve so listings
            # outside the window decay faster.
            assumed_speed_kmh = 18
            target_km = max(float(commute) / 60.0 * assumed_speed_kmh, 1.0)
            ratio = dist_km / target_km
            component = math.exp(-2.0 * max(ratio - 1, 0))
        if component > best_score:
            best_score = component
            best_reason = f"~{dist_km:.1f} km from {anchor['label']}"
    if best_score == 0:
        return 0.0, []
    return best_score, [best_reason]


def _score_structured_distances(
    candidate: dict[str, Any], preferences: dict[str, float],
) -> tuple[float, list[str]]:
    """Use the listing's precomputed distance fields when relevant preferences fire."""
    parts: list[float] = []
    reasons: list[str] = []

    def _meters_to_score(meters: Any) -> float | None:
        if meters is None:
            return None
        try:
            m = float(meters)
        except (TypeError, ValueError):
            return None
        if m <= 0:
            return 1.0
        # 0 m → 1.0, 500 m → ~0.53, 1000 m → ~0.29, 2000+ m ≈ 0.08
        return math.exp(-m / 800.0)

    if "near_transport" in preferences:
        score = _meters_to_score(candidate.get("distance_public_transport"))
        if score:
            parts.append(score * preferences["near_transport"])
            reasons.append("close to public transport")

    if "near_school" in preferences or "family_friendly" in preferences:
        weight = max(
            preferences.get("near_school", 0.0),
            preferences.get("family_friendly", 0.0),
        )
        distances = [
            candidate.get("distance_school_1"),
            candidate.get("distance_school_2"),
            candidate.get("distance_kindergarten"),
        ]
        best = max(
            (_meters_to_score(value) or 0.0 for value in distances), default=0.0,
        )
        if best:
            parts.append(best * weight)
            reasons.append("schools nearby")

    if "central" in preferences:
        score = _meters_to_score(candidate.get("distance_shop"))
        if score:
            parts.append(score * preferences["central"])
            reasons.append("amenities nearby")

    if "quiet" in preferences:
        # Quietness proxy: NOT on top of a transit line. Sweet spot 300-1500m.
        dpt = candidate.get("distance_public_transport")
        if isinstance(dpt, (int, float)) and dpt > 0:
            if 300 <= dpt <= 1500:
                parts.append(0.7 * preferences["quiet"])
                reasons.append("off main transport lines")
            elif dpt < 100:
                parts.append(0.1 * preferences["quiet"])  # right on a tram line

    if not parts:
        return 0.0, []
    return sum(parts) / len(parts), reasons


def _score_price(
    candidate: dict[str, Any],
    preferences: dict[str, float],
    soft_budget_hint: str,
) -> tuple[float, list[str]]:
    """Score the price against the (city, rooms) cohort distribution.

    Fires when the user asked for "cheap" / "affordable" (preferences) or used
    a fuzzy budget hint like "not too expensive" (soft_budget_hint).
    """
    price = candidate.get("price")
    if not isinstance(price, (int, float)) or price <= 0:
        return 0.0, []

    cheap_weight = preferences.get("cheap", 0.0)
    cares = cheap_weight > 0 or soft_budget_hint in {"value_sensitive", "lowest"}
    if not cares:
        return 0.0, []

    pct = _price_stats.price_percentile(
        price, candidate.get("city"), candidate.get("rooms"),
    )
    if pct is None:
        return 0.0, []

    # pct: 0.0 = bottom quartile (great value), 1.0 = top quartile (expensive).
    score = max(0.0, 1.0 - pct)
    if score <= 0:
        return 0.0, []

    # Effective weight: honour explicit "cheap" pref, or use 0.7 for the fuzzy
    # "not too expensive" hint, or 0.9 for "lowest".
    weight = cheap_weight
    if soft_budget_hint == "lowest":
        weight = max(weight, 0.9)
    elif soft_budget_hint == "value_sensitive":
        weight = max(weight, 0.7)

    city = candidate.get("city") or "the area"
    phrase = (
        f"below the median for {city}" if pct <= 0.33
        else f"on par with {city} cohort"
    )
    return score * weight, [phrase]


def _score_object_category(
    candidate: dict[str, Any], hints: list[str],
) -> tuple[float, list[str]]:
    if not hints:
        return 0.0, []
    category = candidate.get("object_category") or ""
    object_type = candidate.get("object_type") or ""
    if category in hints or object_type in hints:
        return 1.0, [f"matches {category or object_type}"]
    if "Wohnung" in hints and candidate.get("rooms"):
        return 0.5, []
    return 0.0, []


def _score_image_availability(candidate: dict[str, Any]) -> float:
    images = candidate.get("image_urls") or []
    if not images:
        return 0.0
    if len(images) >= 5:
        return 1.0
    return 0.5 + 0.1 * (len(images) - 1)


def _score_data_completeness(candidate: dict[str, Any]) -> float:
    fields = ["price", "rooms", "area", "available_from", "street", "latitude"]
    filled = sum(1 for f in fields if candidate.get(f) not in (None, "", 0))
    return filled / len(fields)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _weighted_geometric(components: dict[str, float]) -> float:
    """Weighted geometric mean of non-zero components.

    Used when the LLM flagged conflicting soft signals. A listing that scores
    high on only one axis (e.g. very cheap but not at all central) will get a
    lower combined score than one that scores moderately on every axis.
    """
    if not components:
        return 0.0
    total_w = sum(COMPONENT_BASE[name] for name in components)
    if total_w <= 0:
        return 0.0
    log_sum = 0.0
    for name, value in components.items():
        w = COMPONENT_BASE[name]
        # Clamp to 0.01 to avoid log(0) collapsing the product to zero.
        v = max(0.01, value)
        log_sum += w * math.log(v)
    return math.exp(log_sum / total_w)


# ---------------------------------------------------------------------------
# Reason / formatting helpers
# ---------------------------------------------------------------------------


def _format_reason(
    explanations: dict[str, list[str]],
    dominant: str | None,
    evidence: dict[str, str],
) -> str:
    """Turn the per-component reason lists into a one-line human explanation."""
    bits: list[str] = []
    # Order signals by perceived value to the user, not by score weight.
    priority = [
        "anchor_distance",      # "3.2 km from ETH" is very tangible
        "text_match",           # "bright, modern" — the soft ask verbatim
        "feature_match",        # concrete features the user wanted
        "structured_distance",  # "close to transport" with a distance
        "price",                # "below the Zürich median"
        "object_category",
        "image_availability",
    ]
    for name in priority:
        for item in (explanations.get(name) or []):
            if item and item not in bits:
                bits.append(item)
            if len(bits) >= 4:
                break
        if len(bits) >= 4:
            break

    # If we still have nothing but the LLM told us what dominates, fall back
    # to the LLM's own one-line justification.
    if not bits and dominant and evidence.get(dominant):
        bits.append(evidence[dominant])

    if not bits:
        return "Matches the hard filters; no extra soft signals fired."
    return "; ".join(bits[:4])


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return earth_radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Listing -> Pydantic conversion
# ---------------------------------------------------------------------------


def _to_listing_data(candidate: dict[str, Any]) -> ListingData:
    return ListingData(
        id=str(candidate["listing_id"]),
        title=candidate["title"],
        description=candidate.get("description"),
        street=candidate.get("street"),
        city=candidate.get("city"),
        postal_code=candidate.get("postal_code"),
        canton=candidate.get("canton"),
        latitude=candidate.get("latitude"),
        longitude=candidate.get("longitude"),
        price_chf=candidate.get("price"),
        rooms=candidate.get("rooms"),
        living_area_sqm=_coerce_int(candidate.get("area")),
        available_from=candidate.get("available_from"),
        image_urls=_coerce_image_urls(candidate.get("image_urls")),
        hero_image_url=candidate.get("hero_image_url"),
        original_listing_url=candidate.get("original_url"),
        features=candidate.get("features") or [],
        offer_type=candidate.get("offer_type"),
        object_category=candidate.get("object_category"),
        object_type=candidate.get("object_type"),
    )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _coerce_image_urls(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return None
