import argparse
import csv
import glob
import gzip
import json
import math
import os
from pathlib import Path

import mapbox_vector_tile
from pmtiles.reader import MmapSource, Reader
from pmtiles.tile import Compression


def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def latlon_to_tile_point(lat, lon, zoom, tile_x, tile_y, extent=4096):
    n = 2 ** zoom
    xf = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    yf = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

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


def match_feature(decoded, layer_name, point):
    layer = decoded.get(layer_name, {})
    for feature in layer.get("features", []):
        geometry = feature.get("geometry", {})
        if point_in_polygon_geometry(point, geometry):
            return feature
    return None


def parse_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def enrich_csv(file_path, reader, header, zoom, lat_col, lng_col, layer_name, backup_suffix):
    path = Path(file_path)
    backup_path = path.with_suffix(path.suffix + backup_suffix)

    with path.open(newline="", encoding="utf-8") as f:
        csv_reader = csv.DictReader(f)
        fieldnames = list(csv_reader.fieldnames or [])

        extra_cols = [
            "tile_match",
            "tile_layer",
            "tile_feature_id",
            "tile_zoom",
            "tile_x",
            "tile_y",
            "tile_properties_json",
        ]
        for c in extra_cols:
            if c not in fieldnames:
                fieldnames.append(c)

        rows = []
        cache = {}
        matched = 0
        processed = 0

        for row in csv_reader:
            processed += 1
            lat = parse_float(row.get(lat_col))
            lng = parse_float(row.get(lng_col))

            row["tile_match"] = "0"
            row["tile_layer"] = layer_name
            row["tile_feature_id"] = ""
            row["tile_zoom"] = str(zoom)
            row["tile_x"] = ""
            row["tile_y"] = ""
            row["tile_properties_json"] = ""

            if lat is None or lng is None:
                rows.append(row)
                continue

            x, y = latlon_to_tile(lat, lng, zoom)
            point = latlon_to_tile_point(lat, lng, zoom, x, y)
            row["tile_x"] = str(x)
            row["tile_y"] = str(y)

            key = (zoom, x, y)
            decoded = cache.get(key)
            if decoded is None:
                decoded = decode_tile(reader, header, zoom, x, y)
                cache[key] = decoded

            if decoded is not None:
                feature = match_feature(decoded, layer_name, point)
                if feature is not None:
                    matched += 1
                    row["tile_match"] = "1"
                    feature_id = feature.get("id")
                    row["tile_feature_id"] = "" if feature_id is None else str(feature_id)
                    row["tile_properties_json"] = json.dumps(feature.get("properties", {}), ensure_ascii=True)

            rows.append(row)

    if backup_suffix:
        with path.open("rb") as src, backup_path.open("wb") as dst:
            dst.write(src.read())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return processed, matched


def main():
    parser = argparse.ArgumentParser(description="Enrich CSV rows with containing hex feature from PMTiles")
    parser.add_argument("--glob", dest="glob_pattern", default="raw_data/*.csv")
    parser.add_argument("--pmtiles", required=True)
    parser.add_argument("--zoom", type=int, default=14)
    parser.add_argument("--lat-col", default="geo_lat")
    parser.add_argument("--lng-col", default="geo_lng")
    parser.add_argument("--layer", default="hex")
    parser.add_argument("--backup-suffix", default=".bak")

    args = parser.parse_args()

    files = sorted(glob.glob(args.glob_pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched: {args.glob_pattern}")

    with open(args.pmtiles, "rb") as f:
        reader = Reader(MmapSource(f))
        header = reader.header()

        for csv_file in files:
            processed, matched = enrich_csv(
                file_path=csv_file,
                reader=reader,
                header=header,
                zoom=args.zoom,
                lat_col=args.lat_col,
                lng_col=args.lng_col,
                layer_name=args.layer,
                backup_suffix=args.backup_suffix,
            )
            print(f"{os.path.basename(csv_file)}: processed={processed}, matched={matched}")


if __name__ == "__main__":
    main()
