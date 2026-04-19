"""Quick CLI to query the running API and print a compact, readable ranking.

Usage:
    uv run python scripts/try_query.py "3-room bright apartment in Zurich under 2800 CHF"
    uv run python scripts/try_query.py "quiet studio in Geneva with nice views" --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the listings API")
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--url", default="http://127.0.0.1:8000/listings")
    args = parser.parse_args()

    req = urllib.request.Request(
        args.url,
        data=json.dumps({"query": args.query, "limit": args.limit}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())

    meta = body["meta"]
    print("QUERY:", meta.get("query"))
    print("HARD FILTERS:", meta.get("hard_facts"))
    soft = meta.get("soft_facts", {})
    if soft.get("source"):
        print("EXTRACTOR:", soft["source"])
    if soft.get("preferences"):
        print("SOFT PREFS:", soft["preferences"])
    if soft.get("feature_boosts"):
        print("SOFT FEATURES:", soft["feature_boosts"])
    if soft.get("anchors"):
        print("ANCHORS:", [a["label"] for a in soft["anchors"]])
    print(f"CANDIDATES: {meta.get('candidates_considered')} -> "
          f"showing top {len(body['listings'])}")
    print("-" * 90)

    for item in body["listings"]:
        l = item["listing"]
        price = f"CHF {l['price_chf']}" if l.get("price_chf") else "price?"
        rooms = f"{l['rooms']} rm" if l.get("rooms") is not None else "rooms?"
        area = f"{l['living_area_sqm']} m²" if l.get("living_area_sqm") else ""
        features = ", ".join(l.get("features") or []) or "-"
        print(f"#{item['score']:.3f}  {l['title'][:75]}")
        print(f"        {l.get('city')} · {rooms} · {price} · {area}  [{l.get('object_category') or ''}]")
        print(f"        features: {features}")
        print(f"        why    : {item['reason']}")
        if l.get("hero_image_url"):
            print(f"        photo  : {l['hero_image_url']}")
        if l.get("original_listing_url"):
            print(f"        url    : {l['original_listing_url']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
