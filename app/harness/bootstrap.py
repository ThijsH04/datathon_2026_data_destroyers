from __future__ import annotations

import logging
from pathlib import Path

from app.db import get_connection
from app.harness.csv_import import create_indexes, create_schema, import_csvs
from app.harness.geocode import geocode_missing_addresses
from app.harness.sred_transform import ensure_sred_normalized_csv


logger = logging.getLogger(__name__)


def bootstrap_database(*, db_path: Path, raw_data_dir: Path) -> None:
    ensure_sred_normalized_csv(raw_data_dir)

    if db_path.exists():
        if not _schema_matches(db_path):
            logger.error(
                "\033[31mListings DB schema mismatch at %s. The harness will not overwrite the existing database. "
                "Remove or migrate it manually if you need the newer schema.\033[0m",
                db_path,
            )
            return
        return

    csv_paths = _csv_paths(raw_data_dir)
    # geocode_missing_addresses(csv_paths, cache_path=db_path.parent / "geocode_cache.json")

    with get_connection(db_path) as connection:
        create_schema(connection)
        import_csvs(connection, csv_paths)
        create_indexes(connection)


def _csv_paths(raw_data_dir: Path) -> list[Path]:
    if not raw_data_dir.exists() or not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    csv_paths = sorted(path for path in raw_data_dir.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in raw data directory: {raw_data_dir}")
    return csv_paths


def _schema_matches(db_path: Path) -> bool:
    required_columns = {
        "latitude",
        "longitude",
        "features_json",
        "platform_id",
        "scrape_source",
        "street",
        "object_type",
        "feature_wheelchair_accessible",
        "feature_private_laundry",
        "feature_minergie_certified",
        "transit_count_500m",
        "shops_count_500m",
        "parks_count_500m",
        "schools_count_500m",
        "nightlife_count_500m",
        "dist_to_transit_m",
        "dist_to_shops_m",
        "dist_to_parks_m",
        "dist_to_schools_m",
        "dist_to_noisy_roads_m",
        "dist_to_noisy_trains_m",
        "dist_to_lakes_m",
        "dist_to_rivers_m",
        "weighted_crime_per_1000",
    }

    with get_connection(db_path) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'listings'"
        ).fetchone()
        if table is None:
            return False

        columns = {
            column[1]
            for column in connection.execute("PRAGMA table_info(listings)").fetchall()
        }

    return required_columns <= columns
