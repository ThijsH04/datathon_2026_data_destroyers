"""
Normalize address columns in structured_data CSVs.

These files store address data in location_address JSON rather than
object_street/object_city/object_zip. This script:
  1. Extracts street/city/zip/state from location_address JSON (free, instant)
  2. Falls back to Nominatim reverse geocoding for rows still missing a street
"""

import csv
import json
import os
import sys
import time
import requests
from pathlib import Path

RAW = Path("raw_data")
CACHE_FILE = Path("geocode_cache.json")
NOMINATIM_BASE = os.getenv("NOMINATIM_BASE_URL", "http://localhost:8080").rstrip("/")
DELAY = 0.0 if NOMINATIM_BASE != "https://nominatim.openstreetmap.org" else 1.1
USER_AGENT = "datathon2026-data-destroyers/1.0 (berkaysekeroglu1@gmail.com)"

TARGETS = sorted([
    *RAW.glob("structured_data_with*.csv"),
    *RAW.glob("structured_data_without*.csv"),
])


def _clean(v: str | None) -> str:
    if not v:
        return ""
    s = v.strip()
    return "" if s.upper() == "NULL" else s


def _from_location_address(row: dict) -> tuple[str, str, str, str]:
    try:
        loc = json.loads(row.get("location_address") or "{}")
    except Exception:
        loc = {}
    street_name = _clean(loc.get("Street", ""))
    street_number = _clean(loc.get("StreetNumber", ""))
    street = f"{street_name} {street_number}".strip() if street_number else street_name
    city = _clean(loc.get("City", ""))
    postcode = _clean(loc.get("PostalCode", ""))
    state = _clean(loc.get("canton", ""))
    return street, city, postcode, state


def _nominatim_reverse(lat: float, lon: float, session: requests.Session) -> dict:
    r = session.get(
        f"{NOMINATIM_BASE}/reverse",
        params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _from_nominatim(result: dict) -> tuple[str, str, str]:
    addr = result.get("address", {})
    road = addr.get("road") or addr.get("pedestrian") or addr.get("path") or ""
    house = addr.get("house_number", "")
    street = f"{road} {house}".strip() if house else road
    postcode = addr.get("postcode", "")
    city = (
        addr.get("city") or addr.get("town")
        or addr.get("village") or addr.get("municipality") or ""
    )
    return street, postcode, city


def process(csv_path: Path, cache: dict, session: requests.Session):
    print(f"\n{'='*60}\nFile: {csv_path.name}")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    from_json = from_geo = api_calls = errors = 0

    for i, row in enumerate(rows):
        cur_street = _clean(row.get("object_street", ""))
        cur_city   = _clean(row.get("object_city", ""))
        cur_zip    = _clean(row.get("object_zip", ""))
        cur_state  = _clean(row.get("object_state", ""))

        # Step 1: fill from location_address JSON
        if not cur_street or not cur_city or not cur_zip:
            js, jc, jz, jst = _from_location_address(row)
            if js or jc or jz:
                from_json += 1
            cur_street = cur_street or js
            cur_city   = cur_city   or jc
            cur_zip    = cur_zip    or jz
            cur_state  = cur_state  or jst

        # Step 2: still missing street → geocode
        if not cur_street:
            lat = (row.get("geo_lat") or "").strip()
            lon = (row.get("geo_lng") or "").strip()
            if lat and lon:
                key = f"{lat},{lon}"
                if key not in cache:
                    try:
                        result = _nominatim_reverse(float(lat), float(lon), session)
                        gs, gz, gc = _from_nominatim(result)
                        cache[key] = [gs, gz, gc]
                        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False))
                        api_calls += 1
                        time.sleep(DELAY)
                    except Exception as e:
                        print(f"  geocode error ({lat},{lon}): {e}", file=sys.stderr)
                        cache[key] = ["", "", ""]
                        errors += 1
                        time.sleep(DELAY)
                gs, gz, gc = cache[key]
                if gs or gz or gc:
                    from_geo += 1
                cur_street = cur_street or gs
                cur_zip    = cur_zip    or gz
                cur_city   = cur_city   or gc

        row["object_street"] = cur_street
        row["object_city"]   = cur_city
        row["object_zip"]    = cur_zip
        row["object_state"]  = cur_state.upper() if cur_state else cur_state

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(rows)}] json={from_json} geo={from_geo} api={api_calls} errors={errors}")

    tmp = csv_path.with_suffix(".tmp.csv")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)

    print(f"  filled from location_address JSON : {from_json}")
    print(f"  filled from geocoding             : {from_geo}")
    print(f"  nominatim API calls               : {api_calls}")
    print(f"  errors                            : {errors}")
    print(f"  updated in-place: {csv_path.name}")


def main():
    cache: dict = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
    print(f"Loaded {len(cache)} cached coordinates.")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    for target in TARGETS:
        if not target.exists():
            print(f"Skipping (not found): {target}")
            continue
        process(target, cache, session)

    print("\nDone.")


if __name__ == "__main__":
    main()
