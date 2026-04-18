"""Single-call Claude extractor that returns BOTH hard filters and soft
preferences (with dynamically assigned importance weights) in one round-trip.

Design notes:
- One Claude call per user query. `hard_fact_extraction` and
  `soft_fact_extraction` both call `extract_combined(query)`; `functools.lru_cache`
  dedupes so we only actually hit the API once per unique query string.
- Tool-use with `tool_choice=any` forces Claude to return structured JSON that
  matches our schema — no free-text parsing, no JSON-mode failure modes.
- A strict "hedge detection" rule in the system prompt prevents soft hints like
  "ideally with parking" from leaking into hard filters (hard-filter precision
  is the make-or-break automated metric per the challenge brief).
- Prompt caching (`cache_control: ephemeral`) keeps the long system prompt
  amortised across queries within a 5-minute window.
- If the API call fails OR `ANTHROPIC_API_KEY` is unset, we return `None` and
  the rule-based extractors kick in as graceful fallback.

Expected output schema (see `_TOOL_SCHEMA` below):
    {
      "hard": {
          "city": [str],
          "canton": str | None,
          "min_price": int | None,
          "max_price": int | None,
          "min_rooms": float | None,
          "max_rooms": float | None,
          "features": [str],           # strictly required features only
          "offer_type": "RENT"|"SALE"|None,
          "latitude": float | None,
          "longitude": float | None,
          "radius_km": float | None,
      },
      "soft": {
          "preferences": [
              {"signal": str, "weight": float, "evidence": str},
              ...
          ],
          "feature_boosts": {feature_col: weight},
          "anchors": [{"label": str, "lat": float, "lon": float,
                       "max_minutes": int | None}],
          "soft_budget_hint": "value_sensitive"|"lowest"|None,
          "conflicts": [[str, str], ...],
          "dominant_signal": str | None,
          "negations": [str],
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_local_env() -> None:
    """Load simple KEY=VALUE pairs from a local .env file when present."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("could not read %s: %s", env_path, exc)
        return

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()

# The set of soft signals the ranker knows how to score. Claude is instructed
# to only emit signals from this list so we never get orphan weights.
SOFT_SIGNAL_VOCAB: list[str] = [
    "bright",
    "modern",
    "quiet",
    "spacious",
    "cozy",
    "central",
    "view",
    "family_friendly",
    "near_transport",
    "near_school",
    "luxury",
    "cheap",
    "balcony_pref",
    "parking_pref",
    "elevator_pref",
    "garden",
    "pet_pref",
    "new_pref",
    "furnished",
    "student",
]

HARD_FEATURE_VOCAB: list[str] = [
    "balcony",
    "elevator",
    "parking",
    "garage",
    "fireplace",
    "child_friendly",
    "pets_allowed",
    "new_build",
    "wheelchair_accessible",
    "minergie_certified",
]


_TOOL_SCHEMA: dict[str, Any] = {
    "name": "extract_query",
    "description": (
        "Decompose a real-estate search query into HARD constraints (strict "
        "filters) and SOFT preferences (ranking hints with importance weights)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hard": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Swiss city names, canonical form. Normalize: "
                            "Zurich->Zürich, Geneva->Genève, Lucerne->Luzern, "
                            "Berne->Bern. Leave empty if no city is explicitly "
                            "required."
                        ),
                    },
                    "canton": {
                        "type": "string",
                        "description": "Two-letter Swiss canton code (ZH, BE, GE, VD, VS, TI, ...)",
                    },
                    "min_price": {"type": "integer"},
                    "max_price": {"type": "integer"},
                    "min_rooms": {"type": "number"},
                    "max_rooms": {"type": "number"},
                    "features": {
                        "type": "array",
                        "items": {"type": "string", "enum": HARD_FEATURE_VOCAB},
                        "description": (
                            "ONLY features the user explicitly REQUIRES. Never "
                            "include features that are hedged with 'ideally', "
                            "'if possible', 'preferably', 'would be nice', "
                            "'bonus', 'optional', 'am liebsten', 'wenn möglich'."
                        ),
                    },
                    "offer_type": {"type": "string", "enum": ["RENT", "SALE"]},
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "radius_km": {"type": "number"},
                },
                "required": [],
            },
            "soft": {
                "type": "object",
                "properties": {
                    "preferences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "signal": {"type": "string", "enum": SOFT_SIGNAL_VOCAB},
                                "weight": {
                                    "type": "number",
                                    "description": "Importance in [0.0, 1.5]. See system prompt for scale.",
                                },
                                "evidence": {
                                    "type": "string",
                                    "description": "One short phrase from the query that justifies this weight.",
                                },
                            },
                            "required": ["signal", "weight"],
                        },
                    },
                    "anchors": {
                        "type": "array",
                        "description": (
                            "Well-known POIs the user wants to be near (ETH, HB Zürich, "
                            "EPFL, university, airport, etc.). Use your geographic "
                            "knowledge to supply reasonable lat/lon for Swiss landmarks."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "lat": {"type": "number"},
                                "lon": {"type": "number"},
                                "max_minutes": {
                                    "type": "integer",
                                    "description": "Commute time the user mentioned, or null.",
                                },
                            },
                            "required": ["label", "lat", "lon"],
                        },
                    },
                    "soft_budget_hint": {
                        "type": "string",
                        "enum": ["value_sensitive", "lowest", "none"],
                        "description": (
                            "'value_sensitive' when user uses fuzzy words like "
                            "'not too expensive', 'affordable', 'cheap'. 'lowest' "
                            "when user wants the cheapest option. 'none' otherwise."
                        ),
                    },
                    "conflicts": {
                        "type": "array",
                        "description": "Pairs of soft signals the user flagged as in tension (e.g. ['cheap', 'central']).",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "dominant_signal": {
                        "type": "string",
                        "description": "The single most-important soft signal in the query, or null.",
                    },
                    "negations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Soft signals the user explicitly DOES NOT want ('not too loud', 'no ground floor').",
                    },
                },
                "required": ["preferences"],
            },
        },
        "required": ["hard", "soft"],
    },
}


_SYSTEM_PROMPT = """You are a real-estate query understanding component for a Swiss listing search engine. \
You parse natural-language queries (German, English, French, mixed) into structured filters.

# Two layers
1. HARD constraints — must be respected, must NOT include anything hedged.
2. SOFT preferences — ranking hints with importance weights.

# HEDGE DETECTION (critical)
A feature, price, or room count is HARD only when the user expresses it as a requirement.
Hedged language makes it SOFT:
- "ideally", "if possible", "preferably", "would be nice", "bonus", "optional", "maybe"
- "am liebsten", "wenn möglich", "falls möglich", "gerne", "idealerweise", "möglichst"
- "I prefer", "I'd like", "I want" → SOFT unless paired with absolute language

Examples:
- "with balcony" → hard feature: balcony
- "ideally with parking" → soft preference: parking_pref (NOT hard)
- "under 2800 CHF" → max_price=2800 (hard)
- "not too expensive" → soft_budget_hint=value_sensitive (NOT max_price)
- "3-room apartment" → min_rooms=2.5, max_rooms=3.5 (hard)
- "3+ rooms" / "at least 3 rooms" → min_rooms=3 only (hard, no upper)
- "studio" → max_rooms=1.5 (hard)
- "family flat" → min_rooms=3 (hard — strong implication, no hedge)

# CITIES & LANDMARKS
- Normalize to canonical Swiss names: Zurich→Zürich, Geneva→Genève, Lucerne→Luzern.
- If the user mentions a Swiss landmark (ETH Zurich, HB Zürich, EPFL, CERN, university XYZ, airport), \
emit it in `soft.anchors` with your best lat/lon. Do NOT put landmark radius in `hard.latitude/longitude/radius_km` \
unless the user strictly requires a geographic proximity.
- "Kanton Zürich" / "Canton of Zurich" → canton="ZH"
- If they only mention a landmark ("near ETH"), DO NOT add city="Zürich" — the anchor alone handles it, \
because forcing city=Zürich might drop valid listings in neighbouring municipalities.

# SOFT WEIGHTS (0.0 to 1.5)
Assign weights based on prominence, emphasis, hedging:
- 1.2-1.5: emphatic ("very bright", "really modern", "most importantly", "above all")
- 0.8-1.1: natural mention, not hedged ("bright apartment", "close to transport")
- 0.5-0.7: hedged but present ("ideally with parking", "nice views if possible")
- 0.2-0.4: weakly hedged or very optional ("bonus: pet friendly", "maybe a garden")
- 0.0: do not emit (user did not mention it)

Do NOT treat "too" as a positive intensifier except inside fixed budget phrases
like "not too expensive". "too bright" means excessive brightness, not a strong
positive preference for bright.

Mentioning something FIRST in a query usually signals priority — weight it higher.
When two signals conflict (e.g. "cheap BUT central"), list them in `conflicts` and weight both similarly.

# SOFT SIGNAL VOCABULARY (use these exact keys)
bright, modern, quiet, spacious, cozy, central, view, family_friendly, near_transport, \
near_school, luxury, cheap, balcony_pref, parking_pref, elevator_pref, garden, pet_pref, new_pref, furnished, student

# HARD FEATURE VOCABULARY (use these exact keys)
balcony, elevator, parking, garage, fireplace, child_friendly, pets_allowed, \
new_build, wheelchair_accessible, minergie_certified

# Output
Always call the `extract_query` tool. Emit ONLY fields you are confident about. \
Leave fields absent if the user didn't mention them — never guess.
"""


def _build_client():
    """Lazy client construction so module import doesn't fail when the SDK or
    key is missing. Returns None if we can't use Claude."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic  # local import keeps the module importable without the SDK
    except ImportError:
        logger.warning("anthropic SDK not installed; LLM extraction disabled")
        return None
    try:
        return anthropic.Anthropic()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("failed to construct Anthropic client: %s", exc)
        return None


_CLIENT = _build_client()
_MODEL = os.environ.get("ROBIN_LLM_MODEL", "claude-haiku-4-5-20251001")


def _empty_result() -> dict[str, Any]:
    return {
        "hard": {},
        "soft": {
            "preferences": [],
            "anchors": [],
            "soft_budget_hint": "none",
            "conflicts": [],
            "dominant_signal": None,
            "negations": [],
        },
    }


@lru_cache(maxsize=512)
def extract_combined(query: str) -> dict[str, Any] | None:
    """Call Claude once per unique query. Cached via `lru_cache` so the hard
    and soft extractors can both call this without double-billing.

    Returns `None` on any failure — callers should fall back to rule-based
    extraction. Never raises.
    """
    if _CLIENT is None:
        return None
    if not query or not query.strip():
        return _empty_result()

    try:
        response = _CLIENT.messages.create(
            model=_MODEL,
            max_tokens=1024,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "extract_query"},
            messages=[{"role": "user", "content": query.strip()}],
        )
    except Exception as exc:
        logger.warning("Claude extraction failed for query=%r: %s", query, exc)
        return None

    for block in response.content:
        # anthropic SDK returns tool_use blocks with .type == "tool_use"
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "extract_query":
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return _sanitize(payload)

    logger.warning("Claude returned no tool_use block for query=%r", query)
    return None


def _sanitize(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the raw LLM payload so downstream code can trust the shape."""
    hard = payload.get("hard") or {}
    soft = payload.get("soft") or {}

    # Hard: drop None/empty, keep only allowed keys
    clean_hard: dict[str, Any] = {}
    for key in (
        "city",
        "canton",
        "min_price",
        "max_price",
        "min_rooms",
        "max_rooms",
        "features",
        "offer_type",
        "latitude",
        "longitude",
        "radius_km",
    ):
        value = hard.get(key)
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        clean_hard[key] = value

    # Features: filter to the known vocabulary so unknown strings don't break SQL
    if "features" in clean_hard:
        clean_hard["features"] = [
            f for f in clean_hard["features"] if f in HARD_FEATURE_VOCAB
        ]
        if not clean_hard["features"]:
            clean_hard.pop("features")

    # Soft: normalize preferences into a flat dict + keep the raw list for UI/reasons
    preferences = soft.get("preferences") or []
    pref_map: dict[str, float] = {}
    pref_evidence: dict[str, str] = {}
    for item in preferences:
        if not isinstance(item, dict):
            continue
        signal = item.get("signal")
        weight = item.get("weight")
        if signal not in SOFT_SIGNAL_VOCAB:
            continue
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        w = max(0.0, min(1.5, w))
        # Keep the max weight if the LLM emits duplicates
        if w > pref_map.get(signal, 0.0):
            pref_map[signal] = round(w, 3)
            if item.get("evidence"):
                pref_evidence[signal] = str(item["evidence"])[:120]

    anchors = []
    for anchor in soft.get("anchors") or []:
        if not isinstance(anchor, dict):
            continue
        try:
            lat = float(anchor["lat"])
            lon = float(anchor["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        anchors.append(
            {
                "label": str(anchor.get("label") or "anchor"),
                "lat": lat,
                "lon": lon,
                "max_minutes": _coerce_optional_int(anchor.get("max_minutes")),
            }
        )

    conflicts_raw = soft.get("conflicts") or []
    conflicts: list[list[str]] = []
    for pair in conflicts_raw:
        if isinstance(pair, list) and len(pair) == 2:
            a, b = pair
            if isinstance(a, str) and isinstance(b, str):
                conflicts.append([a, b])

    clean_soft = {
        "preferences": pref_map,
        "evidence": pref_evidence,
        "anchors": anchors,
        "soft_budget_hint": (
            soft.get("soft_budget_hint")
            if soft.get("soft_budget_hint") in {"value_sensitive", "lowest", "none"}
            else "none"
        ),
        "conflicts": conflicts,
        "dominant_signal": (
            soft.get("dominant_signal")
            if isinstance(soft.get("dominant_signal"), str)
            else None
        ),
        "negations": [n for n in (soft.get("negations") or []) if isinstance(n, str)],
    }

    return {"hard": clean_hard, "soft": clean_soft}


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def available() -> bool:
    """Return True when the LLM extractor is usable (API key + SDK present)."""
    return _CLIENT is not None
