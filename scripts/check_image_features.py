"""
Print 10 listings from image_features.db that have real scores from Claude.
Run: uv run scripts/check_image_features.py
"""
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("IMAGE_FEATURES_DB_PATH", PROJECT_ROOT / "data" / "image_features_sred.db"))

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT
        listing_id, scrape_source,
        brightness_score, natural_light_score, modernity_score,
        spaciousness_score, renovation_quality,
        kitchen_age_class, bathroom_age_class,
        view_type, balcony_visible,
        living_room_size, kitchen_size, bedroom_size, bathroom_size,
        is_furnished, visible_areas, error
    FROM listing_images
    WHERE error IS NULL
      AND brightness_score IS NOT NULL
      AND modernity_score IS NOT NULL
    LIMIT 10
""").fetchall()

print(f"Found {len(rows)} listings with scores\n")
for r in rows:
    print(f"listing_id : {r['listing_id']}  ({r['scrape_source']})")
    print(f"  brightness       : {r['brightness_score']}")
    print(f"  natural_light    : {r['natural_light_score']}")
    print(f"  modernity        : {r['modernity_score']}")
    print(f"  spaciousness     : {r['spaciousness_score']}")
    print(f"  renovation       : {r['renovation_quality']}")
    print(f"  kitchen_age      : {r['kitchen_age_class']}")
    print(f"  bathroom_age     : {r['bathroom_age_class']}")
    print(f"  view_type        : {r['view_type']}")
    print(f"  balcony_visible  : {r['balcony_visible']}")
    print(f"  living_room_size : {r['living_room_size']}")
    print(f"  kitchen_size     : {r['kitchen_size']}")
    print(f"  bedroom_size     : {r['bedroom_size']}")
    print(f"  bathroom_size    : {r['bathroom_size']}")
    print(f"  is_furnished     : {r['is_furnished']}")
    print(f"  visible_areas    : {r['visible_areas']}")
    print()

total_ok  = conn.execute("SELECT COUNT(*) FROM listing_images WHERE error IS NULL AND brightness_score IS NOT NULL").fetchone()[0]
total_err = conn.execute("SELECT COUNT(*) FROM listing_images WHERE error IS NOT NULL").fetchone()[0]
print(f"DB summary — ok: {total_ok}  errors: {total_err}")
conn.close()
