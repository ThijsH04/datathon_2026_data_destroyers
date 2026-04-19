"""Soft-preference extractor.

Primary path: reads the shared Claude extraction (same call the hard extractor
makes — deduped via lru_cache). This gives us LLM-assigned importance weights,
conflict detection, and POI anchors that rule-based parsing can't match.

Fallback path: rule-based concept matching with intensifier/softener scaling.

Output contract (stable regardless of path):
    {
        "raw_query": str,
        "preferences": {signal: weight in [0, 1.5]},
        "feature_boosts": {feature_column: weight},
        "anchors": [{label, lat, lon, max_minutes?}],
        "object_category_hint": list[str],
        "soft_max_price": int | None,       # sentinel 1 = "user cares about value"
        "soft_budget_hint": "value_sensitive"|"lowest"|"none",
        "conflicts": [[signal_a, signal_b], ...],
        "dominant_signal": str | None,
        "evidence": {signal: one-line justification from LLM},
        "negations": list[str],
        "tokens": list[str],
        "source": "llm" | "rules",
    }
"""

from __future__ import annotations

import re
from typing import Any

from app.participant import _llm_extractor
from app.participant._lexicon import (
    LOCATION_ANCHORS,
    SOFT_FEATURE_BOOST_MAP,
    SOFT_KEYWORDS,
)


_INTENSIFIERS = re.compile(
    r"\b(very|really|super|extremely|absolutely|sehr|wirklich|absolut|äusserst)\b",
    re.IGNORECASE,
)
_PRIORITY_PREFIX = re.compile(
    r"\b(above all|most importantly)\b",
    re.IGNORECASE,
)
_SOFTENERS = re.compile(
    r"\b(ideally|if possible|nice to have|preferably|hopefully|wenn möglich|"
    r"falls möglich|gerne|am liebsten|möglichst|optional|bonus|maybe|perhaps|"
    r"would be nice|idéalement|idealement|si possible|de préférence)\b",
    re.IGNORECASE,
)
_FOLLOWING_SOFTENERS = re.compile(
    r"^\s*(?:is\s+)?(?:a\s+)?(bonus|optional|nice to have|would be nice|"
    r"if possible|si possible)\b",
    re.IGNORECASE,
)
_NEGATIONS = re.compile(
    r"\b(no|not|without|kein|keine|ohne|nicht|never)\b",
    re.IGNORECASE,
)
_VALUE_BUDGET_HINT = re.compile(
    r"\b(not too expensive|nothing too expensive|not expensive|affordable|cheap|budget|"
    r"nicht zu teuer|günstig|preiswert|erschwinglich|bezahlbar|"
    r"pas trop cher|pas cher|abordable|bon marché)\b",
    re.IGNORECASE,
)
_LOWEST_BUDGET_HINT = re.compile(
    r"\b(cheapest|lowest|least expensive|as cheap as possible|"
    r"günstigste|billigste|am günstigsten|le moins cher)\b",
    re.IGNORECASE,
)
_COMMUTE_PATTERN = re.compile(
    r"(?:max(?:imum)?|under|less than|within|in|bis zu)?\s*"
    r"(\d{1,3})\s*(?:min|minutes?|mins|minuten)\s*"
    r"(?:by|with|per|mit)?\s*(?:public transport|train|tram|bus|öv|sbb)?",
    re.IGNORECASE,
)
_DOOR_TO_DOOR = re.compile(
    r"(\d{1,3})\s*(?:min|minutes?|mins|minuten)?\s*door[\s-]?to[\s-]?door",
    re.IGNORECASE,
)
_HALF_HOUR = re.compile(
    r"\b(half\s+an?\s+hour|halbe\s+stunde|30\s*min(?:utes|uten)?)\b",
    re.IGNORECASE,
)
_TRANSPORT_INTENT = re.compile(
    r"\b(transport|public transport|tram|bus|train|station|bahnhof|hauptbahnhof|"
    r"hb|sbb|öv|oev|öffi|oeffi|haltestelle|stop|door[\s-]?to[\s-]?door)\b",
    re.IGNORECASE,
)
_EXCESSIVE_BRIGHT = re.compile(r"\btoo\s+bright\b", re.IGNORECASE)

_HARD_FEATURE_TO_SOFT_SIGNAL: dict[str, str] = {
    "balcony": "balcony_pref",
    "parking": "parking_pref",
    "garage": "parking_pref",
    "elevator": "elevator_pref",
    "child_friendly": "family_friendly",
    "pets_allowed": "pet_pref",
    "new_build": "new_pref",
    "fireplace": "cozy",
}


def extract_soft_facts(query: str) -> dict[str, Any]:
    raw = (query or "").strip()

    # Primary: use the shared LLM extraction (cached, reused by hard extractor)
    llm_result = _llm_extractor.extract_combined(query)
    if llm_result is not None:
        soft = llm_result.get("soft", {})
        hard = llm_result.get("hard", {})
        return _build_payload_from_llm(raw, soft, hard)

    return _build_payload_from_rules(raw)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


def _build_payload_from_llm(
    raw: str,
    soft: dict[str, Any],
    hard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = raw.lower()
    preferences: dict[str, float] = dict(soft.get("preferences") or {})
    anchors = list(soft.get("anchors") or [])
    if not anchors:
        anchors = _extract_anchors(text)

    _apply_llm_preference_guards(text, preferences)
    _add_hard_feature_preferences(preferences, hard or {})
    _add_negated_noise_preference(text, preferences)

    # Derive structured feature boosts from the LLM-assigned soft weights so the
    # ranker can use "feature_balcony" directly when scoring listings.
    feature_boosts: dict[str, float] = {}
    for signal, weight in preferences.items():
        feature_col = SOFT_FEATURE_BOOST_MAP.get(signal)
        if feature_col:
            feature_boosts[feature_col] = round(
                max(feature_boosts.get(feature_col, 0.0), float(weight)), 3,
            )

    soft_budget_hint = soft.get("soft_budget_hint") or "none"
    fallback_budget_hint = _extract_budget_hint(text)
    if soft_budget_hint == "none" and fallback_budget_hint != "none":
        soft_budget_hint = fallback_budget_hint
    soft_max_price: int | None = 1 if soft_budget_hint in {"value_sensitive", "lowest"} else None

    conflicts = _normalize_conflicts(soft.get("conflicts") or [])
    for pair in _extract_conflicts(preferences):
        if pair not in conflicts and [pair[1], pair[0]] not in conflicts:
            conflicts.append(pair)

    return {
        "raw_query": raw,
        "preferences": preferences,
        "feature_boosts": feature_boosts,
        "anchors": anchors,
        "object_category_hint": _extract_object_category_hint(text),
        "soft_max_price": soft_max_price,
        "soft_budget_hint": soft_budget_hint,
        "conflicts": conflicts,
        "dominant_signal": soft.get("dominant_signal"),
        "evidence": soft.get("evidence") or {},
        "negations": _merge_negations(soft.get("negations") or [], _extract_negations(text)),
        "tokens": _tokenize(text),
        "source": "llm",
    }


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _build_payload_from_rules(raw: str) -> dict[str, Any]:
    text = raw.lower()
    preferences: dict[str, float] = {}
    feature_boosts: dict[str, float] = {}

    for concept, synonyms in SOFT_KEYWORDS.items():
        weight = _match_concept(text, synonyms)
        if weight <= 0:
            continue
        preferences[concept] = round(weight, 3)

        feature_col = SOFT_FEATURE_BOOST_MAP.get(concept)
        if feature_col:
            feature_boosts[feature_col] = round(
                max(feature_boosts.get(feature_col, 0.0), weight), 3,
            )

    _apply_llm_preference_guards(text, preferences)
    _add_negated_noise_preference(text, preferences)

    anchors = _extract_anchors(text)
    soft_budget_hint = _extract_budget_hint(text)
    soft_max_price = 1 if soft_budget_hint in {"value_sensitive", "lowest"} else None
    negations = _extract_negations(text)
    object_category_hint = _extract_object_category_hint(text)
    conflicts = _extract_conflicts(preferences)

    dominant = max(preferences.items(), key=lambda kv: kv[1])[0] if preferences else None

    return {
        "raw_query": raw,
        "preferences": preferences,
        "feature_boosts": feature_boosts,
        "anchors": anchors,
        "object_category_hint": object_category_hint,
        "soft_max_price": soft_max_price,
        "soft_budget_hint": soft_budget_hint,
        "conflicts": conflicts,
        "dominant_signal": dominant,
        "evidence": {},
        "negations": negations,
        "tokens": _tokenize(text),
        "source": "rules",
    }


def _match_concept(text: str, synonyms: list[str]) -> float:
    best = 0.0
    for synonym in synonyms:
        pattern = rf"(?<![\w]){re.escape(synonym)}(?![\w])"
        for match in re.finditer(pattern, text):
            window_start = max(0, match.start() - 25)
            window_end = min(len(text), match.end() + 25)
            window = text[window_start:window_end]
            preceding = text[window_start : match.start()]
            local_preceding = text[max(0, match.start() - 18) : match.start()]
            local_following = text[match.end() : min(len(text), match.end() + 22)]
            clause_preceding = re.split(r"[,;.]|\bbut\b|\bund\b|\band\b", preceding)[-1]

            if _NEGATIONS.search(clause_preceding):
                continue

            weight = 1.0
            if _INTENSIFIERS.search(local_preceding):
                weight = 1.4
            elif _PRIORITY_PREFIX.search(window) and match.start() <= 45:
                # Query-opening emphasis like "above all" usually applies to
                # the first stated preference, not to every later clause.
                weight = 1.4
            if _SOFTENERS.search(local_preceding) or _FOLLOWING_SOFTENERS.search(local_following):
                weight = max(weight - 0.4, 0.4)
            best = max(best, weight)
    return best


def _apply_llm_preference_guards(text: str, preferences: dict[str, float]) -> None:
    if "near_transport" in preferences and not _TRANSPORT_INTENT.search(text):
        preferences.pop("near_transport", None)
    if _EXCESSIVE_BRIGHT.search(text):
        preferences.pop("bright", None)


def _add_hard_feature_preferences(
    preferences: dict[str, float],
    hard: dict[str, Any],
) -> None:
    for feature in hard.get("features") or []:
        signal = _HARD_FEATURE_TO_SOFT_SIGNAL.get(str(feature))
        if signal:
            preferences[signal] = max(float(preferences.get(signal, 0.0)), 0.8)


def _add_negated_noise_preference(text: str, preferences: dict[str, float]) -> None:
    if re.search(
        r"\b(no|without|kein|keine|ohne)\s+(?:street\s+)?(?:noise|lärm|laerm)\b",
        text,
        re.IGNORECASE,
    ):
        preferences["quiet"] = max(preferences.get("quiet", 0.0), 1.0)


def _extract_anchors(text: str) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    for alias, (label, lat, lon) in sorted(
        LOCATION_ANCHORS.items(), key=lambda item: -len(item[0]),
    ):
        pattern = rf"(?<![\w]){re.escape(alias)}(?![\w])"
        if not re.search(pattern, text):
            continue
        if label in seen_labels:
            continue
        seen_labels.add(label)
        anchors.append({
            "label": label,
            "alias": alias,
            "lat": lat,
            "lon": lon,
            "max_minutes": _extract_commute_window(text),
        })
    return anchors


def _extract_commute_window(text: str) -> int | None:
    if _HALF_HOUR.search(text):
        return 30
    door = _DOOR_TO_DOOR.search(text)
    if door:
        try:
            return int(door.group(1))
        except (TypeError, ValueError):
            pass
    commute = _COMMUTE_PATTERN.search(text)
    if commute:
        try:
            value = int(commute.group(1))
            if 5 <= value <= 180:
                return value
        except (TypeError, ValueError):
            pass
    return None


def _extract_budget_hint(text: str) -> str:
    if _LOWEST_BUDGET_HINT.search(text):
        return "lowest"
    if _VALUE_BUDGET_HINT.search(text):
        return "value_sensitive"
    return "none"


def _extract_conflicts(preferences: dict[str, float]) -> list[list[str]]:
    conflict_pairs = [
        ("cheap", "central"),
        ("quiet", "near_transport"),
        ("quiet", "nightlife"),
        ("spacious", "cheap"),
        ("green", "central"),
        ("cozy", "modern"),
        ("cozy", "new_pref"),
        ("luxury", "cheap"),
    ]
    return [
        [left, right]
        for left, right in conflict_pairs
        if preferences.get(left, 0.0) > 0 and preferences.get(right, 0.0) > 0
    ]


def _normalize_conflicts(conflicts: list[Any]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for pair in conflicts:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left, right = pair
        if not isinstance(left, str) or not isinstance(right, str):
            continue
        item = [left, right]
        if item not in normalized and [right, left] not in normalized:
            normalized.append(item)
    return normalized


def _merge_negations(raw: list[Any], inferred: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*raw, *inferred]:
        if not isinstance(item, str) or not item:
            continue
        if item not in merged:
            merged.append(item)
    return merged


def _extract_negations(text: str) -> list[str]:
    negations: set[str] = set()
    if _EXCESSIVE_BRIGHT.search(text):
        negations.add("too_bright")
    if re.search(r"\b(no|without|kein|keine|ohne)\s+(?:ground\s+floor|erdgeschoss)\b", text):
        negations.add("ground_floor")
    if re.search(r"\b(no|without|kein|keine|ohne)\s+(?:street\s+)?(?:noise|lärm|laerm)\b", text):
        negations.add("noise")
    if not negations:
        negations.update(m.lower() for m in _NEGATIONS.findall(text))
    return sorted(negations)


def _extract_object_category_hint(text: str) -> list[str]:
    hints: list[str] = []
    if re.search(r"\b(apartment|flat|wohnung|appartement)\b", text):
        hints.append("Wohnung")
    if re.search(r"\b(studio)\b", text):
        hints.append("Studio")
    if re.search(r"\b(loft)\b", text):
        hints.append("Loft")
    if re.search(r"\b(maisonette|duplex)\b", text):
        hints.append("Maisonette")
    if re.search(r"\b(attic|dachwohnung|attika)\b", text):
        hints.append("Dachwohnung")
    return hints


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())
