"""Hard-fact extractor.

Primary path: a single Claude call (see `_llm_extractor.extract_combined`)
returns both hard filters and soft preferences. The cached result is reused
by the soft extractor in the same request, so we only pay for one LLM call
per unique query.

Fallback path: rule-based bilingual parser. Triggers whenever the LLM path
is unavailable (no API key, SDK missing, network error, malformed output).
The fallback is conservative — it only marks something as a hard constraint
when the wording is unambiguous. Soft hints stay for the ranker.

Hard-filter precision is the make-or-break automated metric, so we err on
the side of NOT over-constraining.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from app.models.schemas import HardFilters
from app.participant import _llm_extractor
from app.participant._lexicon import (
    CITY_ALIASES,
    HARD_FEATURE_KEYWORDS,
    LOCATION_ANCHORS,
    PRICE_LOWER_WORDS,
    PRICE_UPPER_WORDS,
)


_NUM_PATTERN = re.compile(r"\d[\d'.,]*")
_ROOM_NUM_PATTERN = re.compile(
    r"""
    (?P<num>\d+(?:[.,]\d)?)        # number, optionally fractional (3, 3.5, 3,5)
    [\s\-]*                         # optional whitespace or hyphen ("3-room", "3 Zi")
    (?:                             # followed by an apartment-size word
        \+?\s*(?:zi|zim|zimmer|zimmerwohnung|room|rooms|rm|pieces?|pi[eè]ces?|locali)
        |\.5\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_ROOM_RANGE_PATTERN = re.compile(
    r"""
    (?P<lo>\d+(?:[.,]\d)?)\s*
    (?:-|to|bis|and|und)\s*
    (?P<hi>\d+(?:[.,]\d)?)\s*
    (?:zi|zim|zimmer|zimmerwohnung|room|rooms|rm|pieces?|pi[eè]ces?|locali)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_PRICE_RANGE_PATTERN = re.compile(
    r"between\s+(\d[\d'.,]*)\s+(?:and|to|-)\s+(\d[\d'.,]*)",
    re.IGNORECASE,
)


def extract_hard_facts(query: str) -> HardFilters:
    """Parse the query into structured hard filters.

    Uses Claude when available; falls back to a conservative rule-based
    parser when the LLM is unavailable or returns nothing useful.
    """
    if not query or not query.strip():
        return HardFilters()

    # Primary: shared Claude extraction (cached, so the soft extractor reuses it).
    llm_result = _llm_extractor.extract_combined(query)
    if llm_result is not None:
        try:
            hard = _apply_rule_guards(query, dict(llm_result.get("hard") or {}))
            return HardFilters(**hard)
        except Exception:
            # Schema mismatch — fall back silently.
            pass

    return _rule_based(query)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------


def _apply_rule_guards(query: str, hard: dict[str, object]) -> dict[str, object]:
    """Make exact hard facts stable even when Claude varies slightly."""
    norm = _normalize(query)

    cities = _extract_cities(norm)
    if cities:
        hard["city"] = cities
    elif hard.get("city"):
        # If the only city-looking text is inside a landmark like "ETH Zürich",
        # keep it as an anchor rather than a hard city filter.
        hard.pop("city", None)

    min_rooms, max_rooms = _extract_room_bounds(norm)
    if min_rooms is not None or max_rooms is not None:
        if min_rooms is None:
            hard.pop("min_rooms", None)
        else:
            hard["min_rooms"] = min_rooms

        if max_rooms is None:
            hard.pop("max_rooms", None)
        else:
            hard["max_rooms"] = max_rooms

    return hard


def _rule_based(query: str) -> HardFilters:
    raw = query.strip()
    norm = _normalize(raw)

    cities = _extract_cities(norm)
    min_rooms, max_rooms = _extract_room_bounds(norm)
    min_price, max_price = _extract_price_bounds(raw, norm)
    features = _extract_hard_features(norm)
    offer_type = _extract_offer_type(norm)

    return HardFilters(
        city=cities or None,
        min_rooms=min_rooms,
        max_rooms=max_rooms,
        min_price=min_price,
        max_price=max_price,
        features=features or None,
        offer_type=offer_type,
    )


def _normalize(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("–", "-").replace("—", "-").replace("\u00a0", " ")
    return lowered


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def _extract_cities(text: str) -> list[str]:
    matched: list[str] = []
    seen: set[str] = set()
    text_ascii = _strip_accents(text)
    landmark_spans = _extract_landmark_spans(text, text_ascii)
    for alias in sorted(CITY_ALIASES.keys(), key=len, reverse=True):
        alias_ascii = _strip_accents(alias)
        pattern = rf"(?<![\w]){re.escape(alias)}(?![\w])"
        pattern_ascii = rf"(?<![\w]){re.escape(alias_ascii)}(?![\w])"
        matches = list(re.finditer(pattern, text)) or list(
            re.finditer(pattern_ascii, text_ascii)
        )
        if not matches:
            continue
        if all(_inside_any_span(match.start(), match.end(), landmark_spans) for match in matches):
            continue
        canonical = CITY_ALIASES[alias]
        if canonical not in seen:
            seen.add(canonical)
            matched.append(canonical)
    return matched


def _extract_landmark_spans(text: str, text_ascii: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for alias in LOCATION_ANCHORS:
        # Only suppress a city match when it appears inside a more specific
        # named place like "ETH Zürich" or "Geneva airport".
        if not any(city_alias in alias for city_alias in CITY_ALIASES):
            continue
        alias_ascii = _strip_accents(alias)
        pattern = rf"(?<![\w]){re.escape(alias)}(?![\w])"
        pattern_ascii = rf"(?<![\w]){re.escape(alias_ascii)}(?![\w])"
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end()))
        for match in re.finditer(pattern_ascii, text_ascii):
            spans.append((match.start(), match.end()))
    return spans


def _inside_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def _extract_room_bounds(text: str) -> tuple[float | None, float | None]:
    if re.search(r"\bstudio\b", text):
        return None, 1.5

    range_match = _ROOM_RANGE_PATTERN.search(text)
    if range_match:
        try:
            lo = float(range_match.group("lo").replace(",", "."))
            hi = float(range_match.group("hi").replace(",", "."))
        except ValueError:
            return None, None
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    matches = list(_ROOM_NUM_PATTERN.finditer(text))
    if not matches:
        return None, None

    raw_value = matches[0].group("num").replace(",", ".")
    try:
        value = float(raw_value)
    except ValueError:
        return None, None

    prefix = text[max(0, matches[0].start() - 18) : matches[0].start()]
    suffix = text[matches[0].end() : matches[0].end() + 8]
    is_lower_only = bool(
        re.search(r"\b(at least|min|minimum|ab|mindestens)\s*$", prefix)
        or "+" in matches[0].group(0)
    )
    is_upper_only = bool(
        re.search(r"\b(at most|max|maximum|bis|höchstens|maximal|up to)\s*$", prefix)
        or re.search(r"^\s*(or fewer|or less|max)\b", suffix)
    )

    if is_lower_only:
        return value, None
    if is_upper_only:
        return None, value
    return value - 0.5, value + 0.5


def _extract_price_bounds(raw: str, text: str) -> tuple[int | None, int | None]:
    range_match = _PRICE_RANGE_PATTERN.search(raw.lower())
    if range_match:
        lo = _to_int(range_match.group(1))
        hi = _to_int(range_match.group(2))
        if lo and hi and lo > hi:
            lo, hi = hi, lo
        return lo, hi

    numbers: list[tuple[int, int, int]] = []
    for match in _NUM_PATTERN.finditer(text):
        value = _to_int(match.group(0))
        if value is None or value < 200 or value > 50_000_000:
            continue
        if 1900 <= value <= 2100 and not _looks_like_price_context(text, match.start()):
            continue
        numbers.append((match.start(), match.end(), value))

    if not numbers:
        return None, None

    min_price: int | None = None
    max_price: int | None = None
    for start, end, value in numbers:
        before = text[max(0, start - 25) : start]
        after = text[end : end + 25]
        context = f"{before} {after}"

        if any(word in before for word in PRICE_UPPER_WORDS):
            max_price = _min_or(max_price, value)
        elif any(word in before for word in PRICE_LOWER_WORDS):
            min_price = _max_or(min_price, value)
        elif "chf" in context or "fr." in context or "franken" in context:
            max_price = _min_or(max_price, value)

    return min_price, max_price


def _looks_like_price_context(text: str, idx: int) -> bool:
    window = text[max(0, idx - 25) : idx + 10]
    return any(token in window for token in ("chf", "fr.", "franken", "$", "€"))


def _to_int(value: str) -> int | None:
    cleaned = value.replace("'", "").replace(" ", "")
    if "." in cleaned and cleaned.rsplit(".", 1)[-1].isdigit() and len(cleaned.rsplit(".", 1)[-1]) <= 2:
        cleaned = cleaned.split(".", 1)[0]
    cleaned = cleaned.replace(",", "").replace(".", "")
    try:
        return int(cleaned)
    except ValueError:
        return None


def _min_or(current: int | None, candidate: int) -> int:
    return candidate if current is None else min(current, candidate)


def _max_or(current: int | None, candidate: int) -> int:
    return candidate if current is None else max(current, candidate)


def _extract_hard_features(text: str) -> list[str]:
    softener = re.compile(
        r"\b(ideally|if possible|nice to have|preferably|hopefully|wenn möglich|"
        r"falls möglich|gerne|am liebsten|möglichst|ideal|optional|bonus|"
        r"would be nice|idéalement|idealement|si possible|de préférence)\b",
        re.IGNORECASE,
    )
    following_softener = re.compile(
        r"^\s*(?:if possible|would be nice|nice to have|optional|bonus|si possible)\b",
        re.IGNORECASE,
    )

    matched: list[str] = []
    seen: set[str] = set()
    for feature_key, synonyms in HARD_FEATURE_KEYWORDS.items():
        for synonym in synonyms:
            pattern = rf"(?<![\w]){re.escape(synonym)}(?![\w])"
            for match in re.finditer(pattern, text):
                window_start = max(0, match.start() - 30)
                window_end = min(len(text), match.end() + 30)
                preceding = text[window_start : match.start()]
                following = text[match.end() : window_end]
                if softener.search(preceding) or following_softener.search(following):
                    continue
                if feature_key not in seen:
                    seen.add(feature_key)
                    matched.append(feature_key)
                break
    return matched


def _extract_offer_type(text: str) -> str | None:
    if re.search(r"\b(buy|kaufen|for sale|to buy|kauf)\b", text):
        return "SALE"
    if re.search(r"\b(rent|miete|to rent|mieten|for rent)\b", text):
        return "RENT"
    return None


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
