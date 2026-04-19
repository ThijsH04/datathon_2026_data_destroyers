import logging
import sqlite3
from pathlib import Path
import csv
import json

import pytest

from app.harness.bootstrap import bootstrap_database
from app.participant.listing_row_parser import _prepare_listing_row


def test_bootstrap_creates_sqlite_database(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_data_dir = repo_root / "raw_data"
    db_path = tmp_path / "listings.db"

    bootstrap_database(db_path=db_path, raw_data_dir=raw_data_dir)

    assert db_path.exists()

    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) FROM listings").fetchone()
        columns = {
            column[1]
            for column in connection.execute("PRAGMA table_info(listings)").fetchall()
        }

    assert row is not None
    assert row[0] > 0
    assert {
        "latitude",
        "longitude",
        "features_json",
        "platform_id",
        "scrape_source",
        "street",
        "object_type",
    } <= columns


def test_bootstrap_preserves_existing_db_on_schema_mismatch(
    tmp_path: Path,
    caplog,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_data_dir = repo_root / "raw_data"
    db_path = tmp_path / "listings.db"

    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE listings (listing_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO listings (listing_id) VALUES ('kept')")
        connection.commit()

    caplog.set_level(logging.ERROR)

    bootstrap_database(db_path=db_path, raw_data_dir=raw_data_dir)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute("SELECT listing_id FROM listings").fetchone()

    assert row == ("kept",)
    assert "schema mismatch" in caplog.text.lower()


def test_bootstrap_imports_all_csvs_from_raw_data_directory(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_data_dir = repo_root / "raw_data"
    db_path = tmp_path / "listings.db"

    bootstrap_database(db_path=db_path, raw_data_dir=raw_data_dir)

    expected_rows = 0
    for csv_path in sorted(raw_data_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as handle:
            expected_rows += sum(1 for _ in csv.DictReader(handle))

    with sqlite3.connect(db_path) as connection:
        total_rows = connection.execute("SELECT COUNT(*) FROM listings").fetchone()[0]
        scrape_sources = {
            row[0]
            for row in connection.execute("SELECT DISTINCT scrape_source FROM listings").fetchall()
        }

    assert total_rows == expected_rows
    assert scrape_sources
    assert all(source for source in scrape_sources)


def test_bootstrap_generates_normalized_sred_csv(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_data_dir = repo_root / "raw_data"
    db_path = tmp_path / "listings.db"

    bootstrap_database(db_path=db_path, raw_data_dir=raw_data_dir)

    sred_csv_paths = sorted(raw_data_dir.glob("sred*.csv"))
    assert sred_csv_paths
    sred_csv_path = sred_csv_paths[0]

    with sred_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader)

    assert first_row["scrape_source"] == "SRED"
    assert first_row["platform_id"]
    assert first_row["id"]
    assert first_row["title"] != ""
    assert isinstance(json.loads(first_row["images"]), dict)


def test_bootstrap_imports_sred_rows_when_bundle_is_available(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_data_dir = repo_root / "raw_data"
    db_path = tmp_path / "listings.db"

    bootstrap_database(db_path=db_path, raw_data_dir=raw_data_dir)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT listing_id, title, price, rooms, latitude, longitude
            FROM listings
            WHERE scrape_source = 'SRED'
            LIMIT 1
            """
        ).fetchone()

    assert row is not None
    assert row[0]
    assert row[1]
    assert row[2] is not None
    assert row[3] is not None
    assert row[4] is not None
    assert row[5] is not None


def test_bootstrap_makes_local_sred_images_available(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_data_dir = repo_root / "raw_data"
    if not (raw_data_dir / "sred_images").exists() and not (raw_data_dir / "SRED_data(1)").exists():
        pytest.skip("local SRED image bundle is not present in this raw_data layout")

    db_path = tmp_path / "listings.db"

    bootstrap_database(db_path=db_path, raw_data_dir=raw_data_dir)

    sred_images_dir = raw_data_dir / "sred_images"
    assert sred_images_dir.exists()
    assert any(path.is_file() for path in sred_images_dir.iterdir())


def test_prepare_listing_row_extracts_comparis_style_features() -> None:
    row = {
        "id": "comparis-1",
        "platform_id": "comparis-1",
        "scrape_source": "COMPARIS",
        "title": "Comparis listing",
        "object_description": "Bright apartment",
        "location_address": (
            '{"PostalCode":"8001","City":"Zurich","Street":"Main Street","StreetNumber":"10","canton":"zh"}'
        ),
        "orig_data": (
            '{"Features":['
            '{"Key":"HasBalconies","Value":true},'
            '{"Key":"HasLift","Value":true},'
            '{"Key":"HasParkingIndoor","Value":true},'
            '{"Key":"HasWashingmachine","Value":true},'
            '{"Key":"HasDryer","Value":true}'
            '],"MainData":['
            '{"Key":"IsWheelchairAccessible","Value":"true"},'
            '{"Key":"IsMinergieCertified","Value":"true"},'
            '{"Key":"PetsAllowed","Value":"true"},'
            '{"Key":"IsNewBuilding","Value":"true"}'
            "]}"
        ),
        "images": '{"images":[{"url":"https://example.com/1.jpg"}]}',
        "offer_type": "rent",
        "object_category": "Wohnung",
        "object_type": "Wohnung",
        "platform_url": "https://example.com/listing",
        "rent_net": "2100",
        "rent_extra": "150",
        "number_of_rooms": "3.5",
        "area": "85",
        "available_from": "2026-05-01",
        "geo_lat": "47.37",
        "geo_lng": "8.54",
        "prop_child_friendly": "true",
        "transit_count_500m": "12",
        "dist_to_transit_m": "999999",
        "nearest_transit_name": "Main Stop",
        "dist_to_shops_m": "135",
        "weighted_crime_per_1000": "248.4",
    }

    prepared = _prepare_listing_row(row)
    assert prepared[0] == "comparis-1"
    assert prepared[1] == "comparis-1"
    assert prepared[2] == "COMPARIS"
    assert prepared[3] == "Comparis listing"
    assert prepared[4] == "Bright apartment"
    assert prepared[5] == "Main Street 10"
    assert prepared[6] == "Zurich"
    assert prepared[7] == "8001"
    assert prepared[8] == "ZH"
    assert prepared[9] == 2250
    assert prepared[20] == 12
    assert prepared[27] is None
    assert prepared[28] == "Main Stop"
    assert prepared[29] == 135
    assert prepared[40] == 248.4
    assert prepared[41] == 1
    assert prepared[42] == 1
    assert prepared[43] == 1
    assert prepared[44] == 1
    assert prepared[46] == 1
    assert prepared[47] == 1
    assert prepared[49] == 1
    assert prepared[50] == 1
    assert prepared[51] == 1
    assert prepared[52] == 1
    assert prepared[54] == "RENT"
    assert prepared[55] == "Wohnung"
    assert prepared[56] == "Wohnung"
    assert prepared[57] == "https://example.com/listing"
    assert {"balcony", "elevator", "parking"} <= set(json.loads(prepared[53]))
    assert isinstance(json.loads(prepared[58]), dict)


def test_prepare_listing_row_uses_robinreal_flat_flags_without_fabricating_unknowns() -> None:
    row = {
        "id": "robinreal-1",
        "platform_id": "robinreal-1",
        "scrape_source": "ROBINREAL",
        "title": "Robinreal listing",
        "object_description": "Well connected apartment",
        "location_address": (
            '{"PostalCode":"9015","City":"St. Gallen","Street":"Herisauerstrasse 15","StreetNumber":"","canton":"","Country":"CH"}'
        ),
        "orig_data": (
            '{"source":"robinreal","listingId":"robinreal-1","organizationId":"org-1",'
            '"building_type":"Wohnung","kategorietype":"Wohnung","object_type":"Wohnung",'
            '"offer_type":"RENT","type":"Rent","state":"inactive","price_unit":"MONTHLY"}'
        ),
        "images": '{"images":[{"url":"https://example.com/1.jpg"}]}',
        "offer_type": "RENT",
        "object_category": "Wohnung",
        "object_type": "Terrassenwohnung",
        "platform_url": "https://example.com/robinreal",
        "price": "2250",
        "number_of_rooms": "3",
        "area": "85",
        "available_from": "1970-01-01",
        "geo_lat": "47.4063918",
        "geo_lng": "9.3052015",
        "prop_balcony": "false",
        "prop_elevator": "false",
        "prop_parking": "true",
        "prop_garage": "false",
        "prop_fireplace": "false",
        "prop_child_friendly": "false",
        "animal_allowed": "true",
        "object_street": "Herisauerstrasse 15",
        "object_zip": "9015",
        "object_city": "St. Gallen",
        "object_state": "",
        "rent_gross": "2250",
        "is_new_building": "false",
    }

    prepared = _prepare_listing_row(row)
    assert prepared[5] == "Herisauerstrasse 15"
    assert prepared[6] == "St. Gallen"
    assert prepared[7] == "9015"
    assert prepared[8] is None
    assert prepared[9] == 2250
    assert prepared[41] == 0
    assert prepared[42] == 0
    assert prepared[43] == 1
    assert prepared[44] == 0
    assert prepared[45] == 0
    assert prepared[46] == 0
    assert prepared[47] == 1
    assert prepared[49] == 0
    assert prepared[50] is None
    assert prepared[51] is None
    assert prepared[52] is None
    assert prepared[54] == "RENT"
    assert "parking" in json.loads(prepared[53])
