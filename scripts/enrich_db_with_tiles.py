import argparse
import gzip
import json
import math
import sqlite3

import mapbox_vector_tile
from pmtiles.reader import MmapSource, Reader
from pmtiles.tile import Compression


METRIC_COLUMNS = [
    "grocery",
    "dining",
    "cafes",
    "nightlife",
    "healthcare",
    "early_education",
    "education",
    "parks",
    "playgrounds",
    "sports",
    "cycling",
    "transit",
    "car_infra",
    "culture",
    "pet_friendly",
    "financial",
    "safety",
    "shopping",
    "personal_care",
    "accommodation",
    "coworking",
    "beaches",
    "total",
    "_livability",
    "_activity",
    "_liv_grade",
    "_act_grade",
    "_missing",
]


def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def latlon_to_tile_point(lat, lon, zoom, tile_x, tile_y, extent=4096):
    n = 2 ** zoom
    xf = (lon + 180.0) / 360.0 * n
    yf = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    local_x = (xf - tile_x) * extent
    local_y = (yf - tile_y) * extent
    return local_x, local_y


def point_in_ring(point, ring):
    px, py = point
    inside = False
    if not ring:
        return False

    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]

        intersects = (yi > py) != (yj > py)
        if intersects:
            denom = yj - yi
            if denom != 0:
                x_intersection = (xj - xi) * (py - yi) / denom + xi
                if px < x_intersection:
                    inside = not inside
        j = i

    return inside


def point_in_polygon_geometry(point, geometry):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        if not coords:
            return False
        if not point_in_ring(point, coords[0]):
            return False
        for hole in coords[1:]:
            if point_in_ring(point, hole):
                return False
        return True

    if gtype == "MultiPolygon":
        for polygon in coords:
            if not polygon:
                continue
            if not point_in_ring(point, polygon[0]):
                continue
            in_hole = any(point_in_ring(point, hole) for hole in polygon[1:])
            if not in_hole:
                return True

    return False


def decode_tile(reader, header, z, x, y):
    tile_bytes = reader.get(z, x, y)
    if tile_bytes is None:
        return None
    if header.get("tile_compression") == Compression.GZIP:
        tile_bytes = gzip.decompress(tile_bytes)
    return mapbox_vector_tile.decode(tile_bytes)


def find_feature(decoded, layer_name, point):
    layer = decoded.get(layer_name, {})
    for feature in layer.get("features", []):
        if point_in_polygon_geometry(point, feature.get("geometry", {})):
            return feature
    return None


def ensure_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS listing_hex_metrics (
            listing_id TEXT PRIMARY KEY,
            tile_zoom INTEGER,
            tile_x INTEGER,
            tile_y INTEGER,
            tile_feature_id TEXT,
            tile_properties_json TEXT,
            grocery INTEGER,
            dining INTEGER,
            cafes INTEGER,
            nightlife INTEGER,
            healthcare INTEGER,
            early_education INTEGER,
            education INTEGER,
            parks INTEGER,
            playgrounds INTEGER,
            sports INTEGER,
            cycling INTEGER,
            transit INTEGER,
            car_infra INTEGER,
            culture INTEGER,
            pet_friendly INTEGER,
            financial INTEGER,
            safety INTEGER,
            shopping INTEGER,
            personal_care INTEGER,
            accommodation INTEGER,
            coworking INTEGER,
            beaches INTEGER,
            total INTEGER,
            livability INTEGER,
            activity INTEGER,
            liv_grade TEXT,
            act_grade TEXT,
            missing TEXT,
            FOREIGN KEY (listing_id) REFERENCES listings(listing_id)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_listing_hex_metrics_tile ON listing_hex_metrics(tile_zoom, tile_x, tile_y)"
    )
    connection.commit()


def upsert_metric(connection, listing_id, zoom, x, y, feature_id, props):
    values = {
        "listing_id": listing_id,
        "tile_zoom": zoom,
        "tile_x": x,
        "tile_y": y,
        "tile_feature_id": "" if feature_id is None else str(feature_id),
        "tile_properties_json": json.dumps(props, ensure_ascii=True),
        "grocery": int(props.get("grocery", 0)),
        "dining": int(props.get("dining", 0)),
        "cafes": int(props.get("cafes", 0)),
        "nightlife": int(props.get("nightlife", 0)),
        "healthcare": int(props.get("healthcare", 0)),
        "early_education": int(props.get("early_education", 0)),
        "education": int(props.get("education", 0)),
        "parks": int(props.get("parks", 0)),
        "playgrounds": int(props.get("playgrounds", 0)),
        "sports": int(props.get("sports", 0)),
        "cycling": int(props.get("cycling", 0)),
        "transit": int(props.get("transit", 0)),
        "car_infra": int(props.get("car_infra", 0)),
        "culture": int(props.get("culture", 0)),
        "pet_friendly": int(props.get("pet_friendly", 0)),
        "financial": int(props.get("financial", 0)),
        "safety": int(props.get("safety", 0)),
        "shopping": int(props.get("shopping", 0)),
        "personal_care": int(props.get("personal_care", 0)),
        "accommodation": int(props.get("accommodation", 0)),
        "coworking": int(props.get("coworking", 0)),
        "beaches": int(props.get("beaches", 0)),
        "total": int(props.get("total", 0)),
        "livability": int(props.get("_livability", 0)),
        "activity": int(props.get("_activity", 0)),
        "liv_grade": str(props.get("_liv_grade", "")),
        "act_grade": str(props.get("_act_grade", "")),
        "missing": str(props.get("_missing", "")),
    }

    columns = list(values.keys())
    placeholders = ",".join(["?"] * len(columns))
    updates = ",".join([f"{c}=excluded.{c}" for c in columns if c != "listing_id"])

    connection.execute(
        f"INSERT INTO listing_hex_metrics ({','.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(listing_id) DO UPDATE SET {updates}",
        [values[c] for c in columns],
    )


def main():
    parser = argparse.ArgumentParser(description="Enrich listings DB with hex metrics from PMTiles")
    parser.add_argument("--db", required=True, help="Path to listings SQLite DB")
    parser.add_argument("--pmtiles", required=True, help="Path to PMTiles file")
    parser.add_argument("--zoom", type=int, default=14)
    parser.add_argument("--layer", default="hex")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    with sqlite3.connect(args.db) as connection:
        ensure_table(connection)

        rows = connection.execute(
            "SELECT listing_id, latitude, longitude FROM listings WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
        ).fetchall()

        processed = 0
        matched = 0

        with open(args.pmtiles, "rb") as f:
            reader = Reader(MmapSource(f))
            header = reader.header()
            cache = {}

            for listing_id, lat, lng in rows:
                processed += 1

                try:
                    lat = float(lat)
                    lng = float(lng)
                except (TypeError, ValueError):
                    continue

                x, y = latlon_to_tile(lat, lng, args.zoom)
                point = latlon_to_tile_point(lat, lng, args.zoom, x, y)

                key = (args.zoom, x, y)
                decoded = cache.get(key)
                if decoded is None:
                    decoded = decode_tile(reader, header, args.zoom, x, y)
                    cache[key] = decoded

                if decoded is None:
                    continue

                feature = find_feature(decoded, args.layer, point)
                if feature is None:
                    continue

                matched += 1
                upsert_metric(
                    connection,
                    listing_id=listing_id,
                    zoom=args.zoom,
                    x=x,
                    y=y,
                    feature_id=feature.get("id"),
                    props=feature.get("properties", {}),
                )

                if processed % args.batch_size == 0:
                    connection.commit()
                    print(f"processed={processed} matched={matched}")

        connection.commit()
        print(f"done processed={processed} matched={matched}")


if __name__ == "__main__":
    main()
