from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Override with NOMINATIM_BASE_URL env var to point at a local instance.
# Local instance has no rate limit; public nominatim.org requires 1 req/s.
_DEFAULT_BASE = "https://nominatim.openstreetmap.org"
_USER_AGENT = "datathon2026-data-destroyers/1.0 (berkaysekeroglu1@gmail.com)"


def _base_url() -> str:
    return os.getenv("NOMINATIM_BASE_URL", _DEFAULT_BASE).rstrip("/")


def _delay() -> float:
    # No delay for local instances; honour Nominatim ToS for the public API.
    if _base_url() == _DEFAULT_BASE:
        return 1.1
    return 0.0


def _reverse(lat: float, lon: float, session: requests.Session) -> dict:
    url = f"{_base_url()}/reverse"
    r = session.get(
        url,
        params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def _extract(result: dict) -> tuple[str, str, str]:
    addr = result.get("address", {})
    road = (
        addr.get("road")
        or addr.get("pedestrian")
        or addr.get("path")
        or ""
    )
    house = addr.get("house_number", "")
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


def geocode_missing_addresses(csv_paths: list[Path], cache_path: Path) -> None:
    """Fill in object_street/object_zip/object_city for rows that have lat/long
    but no street address. Updates CSVs in-place. Safe to re-run (cached)."""

    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    session = requests.Session()
    session.headers["User-Agent"] = _USER_AGENT
    delay = _delay()

    total_api = 0

    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        needs = [
            r for r in rows
            if not (r.get("object_street") or "").strip()
            and (r.get("geo_lat") or "").strip()
            and (r.get("geo_lng") or "").strip()
        ]
        if not needs:
            continue

        logger.info("Geocoding %d rows in %s", len(needs), csv_path.name)

        for row in needs:
            lat = row["geo_lat"].strip()
            lon = row["geo_lng"].strip()
            key = f"{lat},{lon}"

            if key not in cache:
                try:
                    result = _reverse(float(lat), float(lon), session)
                    cache[key] = list(_extract(result))
                    cache_path.write_text(json.dumps(cache, ensure_ascii=False))
                    total_api += 1
                    if delay:
                        time.sleep(delay)
                except Exception as exc:
                    logger.warning("Geocode failed (%s, %s): %s", lat, lon, exc)
                    cache[key] = ["", "", ""]
                    if delay:
                        time.sleep(delay)

            street, postcode, city = cache[key]
            row["object_street"] = row.get("object_street") or street
            row["object_zip"] = row.get("object_zip") or postcode
            row["object_city"] = row.get("object_city") or city

        tmp = csv_path.with_suffix(".tmp.csv")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(csv_path)
        logger.info("Updated %s", csv_path.name)

    if total_api:
        logger.info("Geocoding complete — %d API calls made", total_api)
