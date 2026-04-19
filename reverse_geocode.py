"""
Reverse geocode missing addresses using Nominatim (nominatim.org).
Targets:
  - sred_data_withmontageimages_latlong.csv   (all 11,105 rows missing addresses)
  - structured_data_withimages*.csv           (554 rows missing object_street)

Nominatim ToS: max 1 req/sec, must set a descriptive User-Agent.
Progress is saved to a cache file so the script can be interrupted and resumed.
"""

import csv
import json
import time
import sys
import os
import requests
from pathlib import Path

RAW = Path("raw_data")
CACHE_FILE = Path("geocode_cache.json")
# Switch between public API and local Docker instance:
#   Local (no rate limit): NOMINATIM_BASE = "http://localhost:8080"
#   Public (1 req/s):      NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
NOMINATIM_BASE = "http://localhost:8080"
USER_AGENT = "datathon2026-data-destroyers/1.0 (berkaysekeroglu1@gmail.com)"
DELAY = 0.0  # no rate limit when running locally

TARGETS = [
    RAW / "sred_data_withmontageimages_latlong.csv",
    # structured with images — only rows with missing object_street
    *sorted(RAW.glob("structured_data_withimages*.csv")),
]


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False))


def nominatim_reverse(lat: float, lon: float, session: requests.Session) -> dict:
    url = f"{NOMINATIM_BASE}/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1}
    r = session.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def extract_address(result: dict) -> tuple[str, str, str]:
    """Return (street, zip, city) from a Nominatim response."""
    addr = result.get("address", {})
    house = addr.get("house_number", "")
    road = addr.get("road") or addr.get("pedestrian") or addr.get("path") or ""
    street = f"{road} {house}".strip() if house else road
    postcode = addr.get("postcode", "")
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or addr.get("municipality")
        or ""
    )
    return street, postcode, city


def process_file(csv_path: Path, cache: dict, session: requests.Session):
    # Write to a temp file, then atomically replace the original so the
    # harness always picks up exactly one copy of each CSV.
    out_path = csv_path.with_suffix(".tmp.csv")
    print(f"\n{'='*60}")
    print(f"File  : {csv_path.name}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    needs_geocoding = [
        r for r in rows
        if not r.get("object_street") and r.get("geo_lat") and r.get("geo_lng")
    ]
    print(f"Rows needing geocoding: {len(needs_geocoding)} / {len(rows)}")

    api_calls = 0
    cache_hits = 0
    errors = 0

    for i, row in enumerate(needs_geocoding):
        lat = row["geo_lat"].strip()
        lon = row["geo_lng"].strip()
        key = f"{lat},{lon}"

        if key in cache:
            street, postcode, city = cache[key]
            cache_hits += 1
        else:
            try:
                result = nominatim_reverse(float(lat), float(lon), session)
                street, postcode, city = extract_address(result)
                cache[key] = [street, postcode, city]
                save_cache(cache)
                api_calls += 1
                time.sleep(DELAY)
            except Exception as e:
                print(f"  ERROR at ({lat},{lon}): {e}", file=sys.stderr)
                errors += 1
                street, postcode, city = "", "", ""
                time.sleep(DELAY)

        row["object_street"] = row.get("object_street") or street
        row["object_zip"] = row.get("object_zip") or postcode
        row["object_city"] = row.get("object_city") or city

        if (i + 1) % 100 == 0 or (i + 1) == len(needs_geocoding):
            pct = (i + 1) / len(needs_geocoding) * 100
            print(
                f"  [{i+1}/{len(needs_geocoding)}] {pct:.1f}%  "
                f"api={api_calls}  cache={cache_hits}  errors={errors}"
            )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_path.replace(csv_path)  # atomic overwrite of original
    print(f"Updated in-place: {csv_path.name}")
    return api_calls, cache_hits, errors


def main():
    cache = load_cache()
    print(f"Loaded {len(cache)} cached coordinates.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    total_api = total_cache = total_errors = 0

    for target in TARGETS:
        if not target.exists():
            print(f"Skipping (not found): {target}")
            continue
        a, c, e = process_file(target, cache, session)
        total_api += a
        total_cache += c
        total_errors += e

    print(f"\nDone. Total API calls: {total_api}, cache hits: {total_cache}, errors: {total_errors}")
    print(f"Cache saved to: {CACHE_FILE}")


if __name__ == "__main__":
    main()
