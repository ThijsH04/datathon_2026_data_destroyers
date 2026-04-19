import argparse
import gzip
import math
from pmtiles.reader import MmapSource, Reader
from pmtiles.tile import Compression
import mapbox_vector_tile
import pandas as pd


# ---------------------------
# Web Mercator conversion
# ---------------------------
def latlon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)

    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    )

    return x, y


def latlon_to_tile_point(lat, lon, zoom, tile_x, tile_y, extent=4096):
    n = 2 ** zoom
    xf = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    yf = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

    local_x = (xf - tile_x) * extent
    local_y = extent-(yf - tile_y) * extent
    return local_x, local_y


def tile_point_to_latlon(local_x, local_y, zoom, tile_x, tile_y, extent=4096):
    n = 2 ** zoom

    xf = tile_x + (local_x / extent)
    yf = tile_y + (local_y / extent)

    lon = xf / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * yf / n)))
    lat = math.degrees(lat_rad)

    return lat, lon


def tile_bbox_latlon(zoom, tile_x, tile_y):
    nw_lat, nw_lon = tile_point_to_latlon(0, 0, zoom, tile_x, tile_y)
    se_lat, se_lon = tile_point_to_latlon(4096, 4096, zoom, tile_x, tile_y)
    return {
        "min_lat": round(se_lat, 6),
        "min_lon": round(nw_lon, 6),
        "max_lat": round(nw_lat, 6),
        "max_lon": round(se_lon, 6),
    }


def first_ring_vertices_as_latlon(geometry, zoom, tile_x, tile_y, limit=6):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    # print(gtype, coords)
    ring = None
    if gtype == "Polygon" and coords:
        ring = coords[0]
    elif gtype == "MultiPolygon" and coords and coords[0]:
        ring = coords[0][0]

    if not ring:
        return []

    vertices = []
    for local_x, local_y in ring[:limit]:
        lat, lon = tile_point_to_latlon(local_x, local_y, zoom, tile_x, tile_y)
        vertices.append({"lat": round(lat, 6), "lon": round(lon, 6)})

    return vertices


def geometry_bbox_latlon(geometry, zoom, tile_x, tile_y):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    lats = []
    lons = []

    def add_point(local_x, local_y):
        lat, lon = tile_point_to_latlon(local_x, local_y, zoom, tile_x, tile_y)
        lats.append(lat)
        lons.append(lon)

    if gtype == "Polygon":
        for ring in coords:
            for local_x, local_y in ring:
                add_point(local_x, local_y)
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                for local_x, local_y in ring:
                    add_point(local_x, local_y)

    if not lats or not lons:
        return None

    return {
        "min_lat": round(min(lats), 6),
        "min_lon": round(min(lons), 6),
        "max_lat": round(max(lats), 6),
        "max_lon": round(max(lons), 6),
    }


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
    # print(gtype)
    if gtype == "Polygon":
        if not coords:
            return False

        if not point_in_ring(point, coords[0]):
            return False

        for hole in coords[1:]:
            if point_in_ring(point, hole):
                return False 
        
        # print(len(coords[0]))
        return True

    if gtype == "MultiPolygon":
        for polygon in coords:

            # print(len(coords[0]))

            if not polygon:
                continue

            if not point_in_ring(point, polygon[0]):
                continue

            in_hole = any(point_in_ring(point, hole) for hole in polygon[1:])

            # print(len(coords[0]))

            return True

    return False


# ---------------------------
# Decode MVT
# ---------------------------
def decode(tile_bytes):
    if tile_bytes is None:
        return None
    return mapbox_vector_tile.decode(tile_bytes)


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)

    args = parser.parse_args()

    # All expected property columns
    cols = [
        "grocery", "dining", "cafes", "nightlife", "healthcare",
        "early_education", "education", "parks", "playgrounds",
        "sports", "cycling", "transit", "car_infra", "culture",
        "pet_friendly", "financial", "safety", "shopping",
        "personal_care", "accommodation", "coworking", "beaches",
        "total", "_livability", "_activity", "_liv_grade",
        "_act_grade"
    ]

    pandas_df = pd.read_csv(args.csv)
    rows = []

    # Open tile file ONCE (big performance win)
    with open(args.file, "rb") as f:
        reader = Reader(MmapSource(f))
        header = reader.header()

        for i, row in pandas_df.iterrows():
            row_data = row.to_dict()

            # Initialize all columns as None
            for col in cols:
                row_data[col] = None

            # Skip invalid coordinates
            if pd.isna(row.get("geo_lat")) or pd.isna(row.get("geo_lng")):
                rows.append(row_data)
                continue

            z = 13
            x, y = latlon_to_tile(row["geo_lat"], row["geo_lng"], z)
            point = latlon_to_tile_point(row["geo_lat"], row["geo_lng"], z, x, y)

            tile_bytes = reader.get(z, x, y)

            if tile_bytes and header.get("tile_compression") == Compression.GZIP:
                tile_bytes = gzip.decompress(tile_bytes)

            if tile_bytes is None:
                print("No tile found here")
                rows.append(row_data)
                continue

            decoded = decode(tile_bytes)

            if not decoded:
                print("Failed to decode tile")
                rows.append(row_data)
                continue

            # Find matches across all layers
            matches = []

            for layer_name, layer in decoded.items():
                for feature in layer.get("features", []):
                    geometry = feature.get("geometry", {})
                    if point_in_polygon_geometry(point, geometry):
                        matches.append(feature)

            if not matches:
                print("No polygon match for this point")
            else:
                # Take first match (you can change to aggregation if needed)
                props = matches[0].get("properties", {})

                for col in cols:
                    row_data[col] = props.get(col)

            rows.append(row_data)

    # Create final dataframe
    output_df = pd.DataFrame(rows)

    # Save to CSV
    output_df.to_csv(args.csv, index=False)
    print(f"Saved output to {args.csv}")


if __name__ == "__main__":
    main()