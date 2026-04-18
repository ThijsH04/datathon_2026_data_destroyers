"""End-to-end smoke test for the LLM extraction pipeline.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/llm_smoke.py

What it checks:
1. Extractor is available (key + SDK present).
2. A handful of representative Swiss-German/English queries yield sane
   hard filters and soft preferences with weights.
3. The cache really dedupes (second call to the same query doesn't hit the API).

Intentionally uses the public `extract_hard_facts` / `extract_soft_facts`
surface so it exercises the same code paths the FastAPI handler does.
"""

from __future__ import annotations

import json
import os
import sys
import time

from app.participant import _llm_extractor
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.soft_fact_extraction import extract_soft_facts


QUERIES = [
    "affordable apartment right in the city centre of Zürich, ideally under 1800 CHF",
    "very quiet flat but still within 5 minutes of a tram stop, Bern",
    "large family apartment with 5+ rooms, not too expensive, Lausanne",
    "cozy rustic apartment but with a modern kitchen and new build, Basel",
    "above all I need a very bright apartment, modern is a bonus, Geneva",
    "extremely quiet, really bright, very spacious flat in Luzern",
    "3-room flat in Zürich, ideally with a balcony, parking would be nice, elevator if possible",
    "I'd really love a garden apartment, preferably pet-friendly, maybe near a school",
    "4-Zimmer-Wohnung in Winterthur, nicht zu teuer, gerne mit Balkon",
    "Wohnung in der Nähe vom Hauptbahnhof Zürich, max 20 Minuten mit dem ÖV",
    "helle Wohnung ohne Erdgeschoss, kein Lärm, Bern",
    "quiet flat near ETH Zürich, max 15 min walk, 3 rooms",
    "somewhere between EPFL and Geneva airport, 3-4 rooms, not too expensive",
    "nice apartment in Zürich, nothing too expensive, bright and modern",
    "cheapest available studio in Basel I can find",
    "no ground floor, no street noise, modern flat in Zürich, at least 3 rooms",
    "appartement lumineux à Genève, pas trop cher, idéalement avec balcon",
    "flat in Bern",
    "very bright modern quiet luxury apartment near ETH Zürich with balcony, parking, garden, elevator, 4 rooms max 4000 CHF",
    
]

"""



    "bright, modern apartment near ETH Zürich with balcony, 4 rooms, max 4000 CHF",
    "too bright, modern apartment near ETH Zürich with balcony, 4 rooms, max 4000 CHF",

"""


def _print_header(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — LLM extractor will fall back to rules.")
        print("Set the key to exercise the Claude path:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("Proceeding with rule-based fallback so you can still sanity-check shape.")

    print(f"LLM extractor available: {_llm_extractor.available()}")

    for query in QUERIES:
        _print_header(query)

        t0 = time.perf_counter()
        hard = extract_hard_facts(query)
        soft = extract_soft_facts(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"elapsed: {elapsed_ms:.0f} ms  source: {soft.get('source')}")
        print("hard:", json.dumps(hard.model_dump(exclude_none=True), ensure_ascii=False))
        print("soft.preferences:", json.dumps(soft.get("preferences"), ensure_ascii=False))
        print("soft.feature_boosts:", json.dumps(soft.get("feature_boosts"), ensure_ascii=False))
        print("soft.anchors:", json.dumps(soft.get("anchors"), ensure_ascii=False))
        print("soft.conflicts:", soft.get("conflicts"))
        print("soft.dominant_signal:", soft.get("dominant_signal"))
        print("soft.soft_budget_hint:", soft.get("soft_budget_hint"))

    # Cache check: same query should be near-instant the second time.
    _print_header("cache dedup check")
    repeat = QUERIES[0]
    t0 = time.perf_counter()
    extract_hard_facts(repeat)
    extract_soft_facts(repeat)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"repeat roundtrip: {elapsed_ms:.1f} ms (should be << first call)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
