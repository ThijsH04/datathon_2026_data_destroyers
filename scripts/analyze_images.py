"""
Image feature extraction for listing photos using Claude Haiku vision.

For each listing, ALL image URLs are fetched and analysed individually.
Results are stored in two tables:
  - listing_images   : one row per (listing_id, image_index)
  - image_features   : one aggregated row per listing_id (populated by aggregate_features())

Broken image URLs are skipped gracefully. Non-interior images (exterior shots,
floor plans, etc.) are classified and excluded from interior-specific scores.

Usage:
    uv run scripts/analyze_images.py [--concurrency 5] [--limit 50] [--dry-run]
    uv run scripts/analyze_images.py --aggregate-only   # re-aggregate without re-fetching

Prerequisites:
    uv add anthropic tqdm
    ANTHROPIC_API_KEY must be set in the environment or in a .env file at the project root.
    Bootstrap the main DB first (start the FastAPI app once).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import sqlite3

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import open_clip
    import torch
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import anthropic

try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm as sync_tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env from project root if present (before reading any os.getenv calls below)
_dotenv = PROJECT_ROOT / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
import boto3
from botocore.exceptions import BotoCoreError, ClientError

MAIN_DB_PATH = Path(os.getenv("LISTINGS_DB_PATH", PROJECT_ROOT / "data" / "listings.db"))
FEATURES_DB_PATH = Path(os.getenv("IMAGE_FEATURES_DB_PATH", PROJECT_ROOT / "data" / "image_features.db"))
SRED_IMAGES_DIR = Path(os.getenv("LISTINGS_RAW_DATA_DIR", PROJECT_ROOT / "raw_data")) / "sred_images"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Property type tiers
# ---------------------------------------------------------------------------
#
# Tier 1 – full residential analysis (room sizes, kitchen, bedroom, bathroom)
# Tier 2 – commercial / special use (brightness, modernity, spaciousness only)
# Tier 3 – skip entirely (parking, garages, storage, bare land)
#
# Unknown / None object_type defaults to Tier 1 (SRED has no object_type).
# All strings must match the UTF-8 values stored in the SQLite DB exactly.

SKIP_TYPES: frozenset[str] = frozenset({
    # Parking
    "Tiefgarage",
    "Einzelgarage",
    "Doppelgarage",
    "Garage",
    "Parkplatz",
    "Offener Parkplatz",
    "Moto Hallenplatz",
    # Storage / land
    "Kellerabteil",
    "Grundstück",
    "Lager",
})

COMMERCIAL_TYPES: frozenset[str] = frozenset({
    "Gewerbeobjekt",
    "Gewerbe",
    "Ladenfläche",
    "Büro",
    "Praxis",
    "Restaurant",
    "Hobbyraum",
    "Bastelraum",
    "Diverses",
})


def get_analysis_tier(object_type: str | None) -> int:
    """Return 1 (full), 2 (commercial), or 3 (skip)."""
    if not object_type:
        return 1  # SRED and unknowns get full analysis
    if object_type in SKIP_TYPES:
        return 3
    if object_type in COMMERCIAL_TYPES:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Image location categories
# ---------------------------------------------------------------------------

INTERIOR_LOCATIONS = {
    "living_room", "kitchen", "bedroom", "bathroom",
    "balcony_terrace", "dining_room", "hallway", "other_interior",
}

KITCHEN_LOCATIONS  = {"kitchen"}
BATHROOM_LOCATIONS = {"bathroom"}

# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

CREATE_LISTING_IMAGES = """
CREATE TABLE IF NOT EXISTS listing_images (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    listing_id       TEXT    NOT NULL,
    scrape_source    TEXT    NOT NULL,
    image_index      INTEGER NOT NULL,   -- 0-based position in the listing's image array
    image_url        TEXT,

    -- What the image shows (may be a collage with multiple rooms)
    is_collage       INTEGER,  -- 1 if the image contains multiple panels/rooms
    visible_areas    TEXT,     -- JSON array of areas visible, e.g. ["kitchen","living_room","exterior"]
                               -- values: "living_room"|"kitchen"|"bedroom"|"bathroom"|
                               --   "balcony_terrace"|"dining_room"|"hallway"|"other_interior"|
                               --   "exterior"|"building_facade"|"floor_plan"|"garden"|"unknown"

    -- Brightness & Light  (interior only)
    brightness_score        REAL,
    natural_light_score     REAL,
    south_facing_likelihood REAL,

    -- Interior Modernity  (interior only)
    modernity_score         REAL,
    kitchen_age_class       TEXT,   -- "modern"|"dated"|"not_visible"|"unknown"
    bathroom_age_class      TEXT,   -- "modern"|"dated"|"not_visible"|"unknown"
    renovation_quality      TEXT,   -- "excellent"|"good"|"average"|"poor"|"unknown"

    -- Space & Layout  (interior only)
    spaciousness_score      REAL,
    open_floor_plan         INTEGER,  -- 1/0/NULL
    visual_clutter_score    REAL,

    -- Views & Surroundings
    view_type               TEXT,   -- "greenery"|"cityscape"|"mountain"|"courtyard"|"street"|"none"|"unknown"
    balcony_visible         INTEGER,

    -- Perceived room sizes (filled only for the relevant room type)
    living_room_size        TEXT,   -- "large"|"medium"|"small"|"not_visible"
    kitchen_size            TEXT,
    bedroom_size            TEXT,
    bathroom_size           TEXT,
    balcony_size            TEXT,
    garden_size             TEXT,
    is_furnished            TEXT,   -- "furnished"|"partially_furnished"|"unfurnished"|"unknown"
    visual_description      TEXT,   -- 2-3 sentence description of visible items for embedding

    raw_vlm_json     TEXT,
    processed_at     TEXT,
    error            TEXT,

    UNIQUE(listing_id, image_index)
)
"""

CREATE_IMAGE_FEATURES = """
CREATE TABLE IF NOT EXISTS image_features (
    listing_id              TEXT PRIMARY KEY,
    scrape_source           TEXT NOT NULL,
    total_images            INTEGER,  -- number of URLs found
    analysed_images         INTEGER,  -- successfully analysed
    interior_images         INTEGER,  -- subset that are interior shots

    -- Aggregated scores (averages / best across interior images)
    brightness_score        REAL,
    natural_light_score     REAL,
    south_facing_likelihood REAL,
    modernity_score         REAL,
    kitchen_age_class       TEXT,
    bathroom_age_class      TEXT,
    renovation_quality      TEXT,
    spaciousness_score      REAL,
    open_floor_plan         INTEGER,
    visual_clutter_score    REAL,
    view_type               TEXT,
    balcony_visible         INTEGER,
    living_room_size        TEXT,
    kitchen_size            TEXT,
    bedroom_size            TEXT,
    bathroom_size           TEXT,
    balcony_size            TEXT,
    garden_size             TEXT,
    is_furnished            TEXT,   -- "furnished"|"partially_furnished"|"unfurnished"|"unknown"
    visual_description      TEXT,   -- 2-3 sentence description of visible items for embedding

    aggregated_at           TEXT
)
"""

UPSERT_IMAGE = """
INSERT INTO listing_images (
    listing_id, scrape_source, image_index, image_url,
    is_collage, visible_areas,
    brightness_score, natural_light_score, south_facing_likelihood,
    modernity_score, kitchen_age_class, bathroom_age_class, renovation_quality,
    spaciousness_score, open_floor_plan, visual_clutter_score,
    view_type, balcony_visible,
    living_room_size, kitchen_size, bedroom_size, bathroom_size, balcony_size, garden_size, is_furnished, visual_description,
    raw_vlm_json, processed_at, error
) VALUES (
    :listing_id, :scrape_source, :image_index, :image_url,
    :is_collage, :visible_areas,
    :brightness_score, :natural_light_score, :south_facing_likelihood,
    :modernity_score, :kitchen_age_class, :bathroom_age_class, :renovation_quality,
    :spaciousness_score, :open_floor_plan, :visual_clutter_score,
    :view_type, :balcony_visible,
    :living_room_size, :kitchen_size, :bedroom_size, :bathroom_size, :balcony_size, :garden_size, :is_furnished, :visual_description,
    :raw_vlm_json, :processed_at, :error
)
ON CONFLICT(listing_id, image_index) DO UPDATE SET
    image_url               = excluded.image_url,
    is_collage              = excluded.is_collage,
    visible_areas           = excluded.visible_areas,
    brightness_score        = excluded.brightness_score,
    natural_light_score     = excluded.natural_light_score,
    south_facing_likelihood = excluded.south_facing_likelihood,
    modernity_score         = excluded.modernity_score,
    kitchen_age_class       = excluded.kitchen_age_class,
    bathroom_age_class      = excluded.bathroom_age_class,
    renovation_quality      = excluded.renovation_quality,
    spaciousness_score      = excluded.spaciousness_score,
    open_floor_plan         = excluded.open_floor_plan,
    visual_clutter_score    = excluded.visual_clutter_score,
    view_type               = excluded.view_type,
    balcony_visible         = excluded.balcony_visible,
    living_room_size        = excluded.living_room_size,
    kitchen_size            = excluded.kitchen_size,
    bedroom_size            = excluded.bedroom_size,
    bathroom_size           = excluded.bathroom_size,
    balcony_size            = excluded.balcony_size,
    garden_size             = excluded.garden_size,
    is_furnished            = excluded.is_furnished,
    visual_description      = excluded.visual_description,
    raw_vlm_json            = excluded.raw_vlm_json,
    processed_at            = excluded.processed_at,
    error                   = excluded.error
"""

UPSERT_FEATURES = """
INSERT INTO image_features (
    listing_id, scrape_source,
    total_images, analysed_images, interior_images,
    brightness_score, natural_light_score, south_facing_likelihood,
    modernity_score, kitchen_age_class, bathroom_age_class, renovation_quality,
    spaciousness_score, open_floor_plan, visual_clutter_score,
    view_type, balcony_visible,
    living_room_size, kitchen_size, bedroom_size, bathroom_size, balcony_size, garden_size, is_furnished, visual_description,
    aggregated_at
) VALUES (
    :listing_id, :scrape_source,
    :total_images, :analysed_images, :interior_images,
    :brightness_score, :natural_light_score, :south_facing_likelihood,
    :modernity_score, :kitchen_age_class, :bathroom_age_class, :renovation_quality,
    :spaciousness_score, :open_floor_plan, :visual_clutter_score,
    :view_type, :balcony_visible,
    :living_room_size, :kitchen_size, :bedroom_size, :bathroom_size, :balcony_size, :garden_size, :is_furnished, :visual_description,
    :aggregated_at
)
ON CONFLICT(listing_id) DO UPDATE SET
    scrape_source           = excluded.scrape_source,
    total_images            = excluded.total_images,
    analysed_images         = excluded.analysed_images,
    interior_images         = excluded.interior_images,
    brightness_score        = excluded.brightness_score,
    natural_light_score     = excluded.natural_light_score,
    south_facing_likelihood = excluded.south_facing_likelihood,
    modernity_score         = excluded.modernity_score,
    kitchen_age_class       = excluded.kitchen_age_class,
    bathroom_age_class      = excluded.bathroom_age_class,
    renovation_quality      = excluded.renovation_quality,
    spaciousness_score      = excluded.spaciousness_score,
    open_floor_plan         = excluded.open_floor_plan,
    visual_clutter_score    = excluded.visual_clutter_score,
    view_type               = excluded.view_type,
    balcony_visible         = excluded.balcony_visible,
    living_room_size        = excluded.living_room_size,
    kitchen_size            = excluded.kitchen_size,
    bedroom_size            = excluded.bedroom_size,
    bathroom_size           = excluded.bathroom_size,
    balcony_size            = excluded.balcony_size,
    garden_size             = excluded.garden_size,
    is_furnished            = excluded.is_furnished,
    visual_description      = excluded.visual_description,
    aggregated_at           = excluded.aggregated_at
"""


def open_features_db() -> sqlite3.Connection:
    FEATURES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEATURES_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(CREATE_LISTING_IMAGES)
    conn.execute(CREATE_IMAGE_FEATURES)
    conn.commit()
    return conn


def already_processed_images(conn: sqlite3.Connection) -> set[tuple[str, int]]:
    """Return (listing_id, image_index) pairs that are already done (success or permanent error)."""
    rows = conn.execute(
        "SELECT listing_id, image_index FROM listing_images WHERE error IS NULL OR error NOT LIKE 'http_%'"
    ).fetchall()
    return {(r["listing_id"], r["image_index"]) for r in rows}


# ---------------------------------------------------------------------------
# Load listings from main DB
# ---------------------------------------------------------------------------

def load_listings(limit: int | None, exclude_sources: list[str] | None = None) -> list[dict[str, Any]]:
    if not MAIN_DB_PATH.exists():
        raise FileNotFoundError(
            f"Main DB not found at {MAIN_DB_PATH}. "
            "Bootstrap it first: start the FastAPI app or run `uv run -m app.harness.bootstrap`."
        )
    conn = sqlite3.connect(MAIN_DB_PATH)
    conn.row_factory = sqlite3.Row
    q = "SELECT listing_id, scrape_source, images_json, object_type FROM listings"
    params: list[Any] = []
    if exclude_sources:
        placeholders = ",".join("?" * len(exclude_sources))
        q += f" WHERE scrape_source NOT IN ({placeholders})"
        params.extend(s.upper() for s in exclude_sources)
    if limit:
        q += f" LIMIT {limit}"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Image URL extraction
# ---------------------------------------------------------------------------

def extract_image_urls(images_json: str | None) -> list[str]:
    """Return all valid image URLs from images_json, in order."""
    if not images_json:
        return []
    try:
        data = json.loads(images_json)
    except (json.JSONDecodeError, TypeError):
        return []
    images = data.get("images", []) if isinstance(data, dict) else []
    urls = []
    for item in images:
        if not isinstance(item, dict):
            continue
        url = item.get("url", "").strip()
        if url:
            urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# Image fetching
# ---------------------------------------------------------------------------

# Matches virtual-hosted S3 URLs:
# https://{bucket}.s3.{region}.amazonaws.com/{key}
_S3_URL_RE = re.compile(
    r"https://(?P<bucket>[^.]+)\.s3\.(?P<region>[^.]+)\.amazonaws\.com/(?P<key>.+)"
)


def _fetch_s3_bytes(url: str) -> bytes:
    """Fetch a private S3 object using boto3 (reads credentials from env / .env)."""
    m = _S3_URL_RE.match(url)
    if not m:
        raise ValueError(f"Cannot parse S3 URL: {url}")
    bucket = m.group("bucket")
    region = m.group("region")
    key = m.group("key")
    s3 = boto3.client("s3", region_name=region)
    buf = io.BytesIO()
    try:
        s3.download_fileobj(bucket, key, buf)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 download failed ({bucket}/{key}): {exc}") from exc
    return buf.getvalue()


def _resize_image(data: bytes, max_px: int) -> Any:
    """Open and resize image so its longest side is at most max_px."""
    img = PILImage.open(io.BytesIO(data)).convert("RGB")
    if max_px > 0 and max(img.size) > max_px:
        img.thumbnail((max_px, max_px), PILImage.LANCZOS)
    return img


def _build_grid(images: list[bytes], tile_px: int = 300) -> bytes:
    """
    Stitch multiple images into a grid and return JPEG bytes.
    Layout: up to 3 columns, rows as needed.
    Each tile is resized to tile_px on its longest side.
    Falls back to returning the first image as-is if PIL unavailable.
    """
    if not HAS_PIL:
        return images[0] if images else b""

    tiles = [_resize_image(img, tile_px) for img in images]
    n = len(tiles)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    tile_w = max(t.width  for t in tiles)
    tile_h = max(t.height for t in tiles)

    grid = PILImage.new("RGB", (cols * tile_w, rows * tile_h), color=(30, 30, 30))
    for i, tile in enumerate(tiles):
        col = i % cols
        row = i // cols
        # centre each tile within its cell
        x = col * tile_w + (tile_w - tile.width)  // 2
        y = row * tile_h + (tile_h - tile.height) // 2
        grid.paste(tile, (x, y))

    buf = io.BytesIO()
    grid.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pre-filtering: near-duplicate removal + interior classification
# ---------------------------------------------------------------------------

_clip_model: Any = None
_clip_preprocess: Any = None
_clip_tokenizer: Any = None


def _get_device() -> str:
    if not HAS_CLIP:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_clip() -> None:
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return
    device = _get_device()
    log.info("Loading CLIP model (ViT-B-32) on %s — first run only …", device)
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    _clip_model = _clip_model.to(device).eval()
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    log.info("CLIP model loaded on %s.", device)


_INTERIOR_PROMPTS = [
    "a photo of an interior room",
    "a photo of a living room",
    "a photo of a bedroom",
    "a photo of a kitchen",
    "a photo of a bathroom",
    "a photo of a balcony or terrace",
]
_EXTERIOR_PROMPTS = [
    "a photo of a building exterior or facade",
    "a floor plan or architectural blueprint",
    "a photo of a street or outdoor area",
    "an aerial or neighbourhood photo",
]


def filter_images(
    raw_images: list[bytes],
    dedup_threshold: int = 8,
    interior_threshold: float = 0.5,
) -> list[bytes]:
    """
    1. Drop near-duplicates via perceptual hash (requires imagehash).
    2. Drop non-interior images via CLIP zero-shot (requires open_clip + torch).
    Always returns at least 1 image (the first) to avoid empty listings.
    """
    if not raw_images:
        return raw_images

    # --- Step 1: near-duplicate removal ---
    if HAS_IMAGEHASH and HAS_PIL:
        seen: list[Any] = []
        deduped: list[bytes] = []
        for data in raw_images:
            try:
                h = imagehash.phash(PILImage.open(io.BytesIO(data)))
                if not any(abs(h - s) < dedup_threshold for s in seen):
                    seen.append(h)
                    deduped.append(data)
            except Exception:
                deduped.append(data)  # keep on error
        raw_images = deduped or raw_images
        log.debug("After dedup: %d images", len(raw_images))

    # --- Step 2: interior classification via CLIP ---
    if HAS_CLIP and HAS_PIL and len(raw_images) > 1:
        _load_clip()
        try:
            text_tokens = _clip_tokenizer(_INTERIOR_PROMPTS + _EXTERIOR_PROMPTS)
            device = _get_device()
            with torch.no_grad():
                text_feat = _clip_model.encode_text(text_tokens.to(device))
                text_feat /= text_feat.norm(dim=-1, keepdim=True)

            n_int = len(_INTERIOR_PROMPTS)
            interior: list[bytes] = []
            for data in raw_images:
                img = _clip_preprocess(
                    PILImage.open(io.BytesIO(data)).convert("RGB")
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_feat = _clip_model.encode_image(img)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)
                    sims = (img_feat @ text_feat.T).squeeze(0).softmax(dim=0)

                int_score = sims[:n_int].sum().item()
                if int_score >= interior_threshold:
                    interior.append(data)

            # keep all if CLIP filtered everything (e.g. all exterior → let Claude judge)
            raw_images = interior if interior else raw_images
            log.debug("After CLIP interior filter: %d images", len(raw_images))
        except Exception as exc:
            log.warning("CLIP filtering failed, skipping: %s", exc)

    return raw_images


async def fetch_image_b64(url: str, client: httpx.AsyncClient, max_px: int = 0) -> tuple[str, str]:
    """Return (base64_data, media_type). Raises on network/file failure."""
    if url.startswith("/raw-data-images/"):
        # SRED local file
        filename = url.split("/")[-1]
        path = SRED_IMAGES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"SRED image not found: {path}")
        data = path.read_bytes()
        media_type = "image/jpeg"

    elif ".amazonaws.com/" in url:
        # Private S3 object — use boto3 with env credentials
        # boto3 is synchronous; run in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _fetch_s3_bytes, url)
        suffix = url.rsplit(".", 1)[-1].lower()
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "png": "image/png", "webp": "image/webp"}.get(suffix, "image/jpeg")

    else:
        # Public CDN URL (COMPARIS etc.)
        resp = await client.get(url, timeout=20.0, follow_redirects=True)
        resp.raise_for_status()
        data = resp.content
        ct = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        media_type = ct if ct.startswith("image/") else "image/jpeg"

    if max_px > 0:
        data = _resize_image(data, max_px)
    return base64.standard_b64encode(data).decode(), media_type


# ---------------------------------------------------------------------------
# VLM prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert real-estate image analyst.
Analyse the provided property listing photo and return a JSON object only — no prose, no markdown fences.
All scores are floats between 0.0 and 1.0 (inclusive). Use null for fields that cannot be assessed."""

_USER_PROMPT_BASE = """Analyse this real-estate listing photo and return ONLY a valid JSON object with exactly these fields.

IMPORTANT: The image may be a single room photo, a street/exterior shot, OR a collage/montage showing multiple rooms at once (e.g. 4 panels: bathroom top-left, living room top-right, kitchen bottom-left, exterior bottom-right). Examine ALL panels carefully before answering.

{
  "is_collage": true if the image contains multiple panels/rooms stitched together, else false,
  "visible_areas": array of ALL areas visible anywhere in the image — include every panel.
    Each element is one of: "living_room"|"kitchen"|"bedroom"|"bathroom"|"balcony_terrace"|
    "dining_room"|"hallway"|"other_interior"|"exterior"|"building_facade"|"floor_plan"|"garden"|"unknown",

  // Brightness & Light — average across all visible interior panels; null if no interior visible
  "brightness_score": float 0-1 (0=very dark, 1=very bright/sunny),
  "natural_light_score": float 0-1 (proportion that appears natural/daylight),
  "south_facing_likelihood": float 0-1 (likelihood of direct sunlight from shadows/intensity),

  // Modernity — assess across all visible interior panels; null if no interior visible
  "modernity_score": float 0-1 (0=very dated, 1=ultra-modern),
  "kitchen_age_class": "modern"|"dated"|"not_visible"|"unknown",
  "bathroom_age_class": "modern"|"dated"|"not_visible"|"unknown",
  "renovation_quality": "excellent"|"good"|"average"|"poor"|"unknown",

  // Space & Layout — assess across all visible interior panels; null if no interior visible
  "spaciousness_score": float 0-1 (0=very cramped, 1=very spacious),
  "open_floor_plan": true|false|null,
  "visual_clutter_score": float 0-1 (0=very tidy, 1=very cluttered),

  // Views — assess from any panel showing windows or outdoors
  "view_type": "greenery"|"cityscape"|"mountain"|"courtyard"|"street"|"none"|"unknown",
  "balcony_visible": true|false|null,

  // Room sizes — assess each room if it appears ANYWHERE in the image (any panel).
  // Use "not_visible" only if that room type does not appear at all in the image.
  "living_room_size": "large"|"medium"|"small"|"not_visible",
  "kitchen_size": "large"|"medium"|"small"|"not_visible",
  "bedroom_size": "large"|"medium"|"small"|"not_visible",
  "bathroom_size": "large"|"medium"|"small"|"not_visible",
  "balcony_size": "large"|"medium"|"small"|"not_visible",
  "garden_size": "large"|"medium"|"small"|"not_visible",

  // Furnishing — assess from visible furniture across all panels
  "is_furnished": "furnished"|"partially_furnished"|"unfurnished"|"unknown",

  // Visual description — 2-3 sentences describing what is concretely visible:
  // mention furniture, materials, fixtures, window size, flooring, notable features.
  // Write as if describing to someone searching for a flat. Do not repeat scores.
  "visual_description": "string"    // omit this field entirely if not requested
}

Do not include any text outside the JSON object."""

_USER_PROMPT_BASE_NO_DESC = _USER_PROMPT_BASE.replace(
    '\n\n  // Visual description — 2-3 sentences describing what is concretely visible:\n'
    '  // mention furniture, materials, fixtures, window size, flooring, notable features.\n'
    '  // Write as if describing to someone searching for a flat. Do not repeat scores.\n'
    '  "visual_description": "string"    // omit this field entirely if not requested',
    ""
)

_USER_PROMPT_COMMERCIAL = """Analyse this real-estate listing photo (commercial/special-use property) and return ONLY a valid JSON object with exactly these fields.

IMPORTANT: The image may be a single room photo, a street/exterior shot, OR a collage/montage showing multiple areas at once. Examine ALL panels carefully before answering.

{
  "is_collage": true if the image contains multiple panels stitched together, else false,
  "visible_areas": array of ALL areas visible anywhere in the image.
    Each element is one of: "office_space"|"retail_floor"|"storage"|"reception"|"common_area"|
    "other_interior"|"exterior"|"building_facade"|"floor_plan"|"garden"|"unknown",

  // Brightness & Light
  "brightness_score": float 0-1 (0=very dark, 1=very bright/sunny),
  "natural_light_score": float 0-1,
  "south_facing_likelihood": float 0-1,

  // Modernity
  "modernity_score": float 0-1 (0=very dated, 1=ultra-modern),
  "kitchen_age_class": "not_visible",
  "bathroom_age_class": "not_visible",
  "renovation_quality": "excellent"|"good"|"average"|"poor"|"unknown",

  // Space & Layout
  "spaciousness_score": float 0-1,
  "open_floor_plan": true|false|null,
  "visual_clutter_score": float 0-1,

  // Views
  "view_type": "greenery"|"cityscape"|"mountain"|"courtyard"|"street"|"none"|"unknown",
  "balcony_visible": true|false|null,

  // Room sizes — always "not_visible" for commercial properties
  "living_room_size": "not_visible",
  "kitchen_size": "not_visible",
  "bedroom_size": "not_visible",
  "bathroom_size": "not_visible",
  "balcony_size": "not_visible",
  "garden_size": "not_visible",
  "is_furnished": "unknown",

  // Visual description — 1-2 sentences describing the visible space, materials, and layout.
  "visual_description": "string"    // omit this field entirely if not requested
}

Do not include any text outside the JSON object."""

_USER_PROMPT_COMMERCIAL_NO_DESC = _USER_PROMPT_COMMERCIAL.replace(
    '\n\n  // Visual description — 1-2 sentences describing the visible space, materials, and layout.\n'
    '  "visual_description": "string"    // omit this field entirely if not requested',
    ""
)


def build_user_prompt(tier: int, include_description: bool = True) -> str:
    if tier == 1:
        return _USER_PROMPT_BASE if include_description else _USER_PROMPT_BASE_NO_DESC
    return _USER_PROMPT_COMMERCIAL if include_description else _USER_PROMPT_COMMERCIAL_NO_DESC


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

async def analyse_image(
    image_b64: str,
    media_type: str,
    claude: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    tier: int = 1,
    include_description: bool = True,
) -> dict[str, Any]:
    async with semaphore:
        response = await claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512 if not include_description else 1024,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache repeated system prompt
            }],
            messages=[
                {
                    "role": "user",
                    "content": [
                        # Text prompt first so it can be cached; image is variable
                        {
                            "type": "text",
                            "text": build_user_prompt(tier, include_description),
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                        },
                    ],
                }
            ],
        )
    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(raw_text), raw_text


def _bool_to_int(v: Any) -> int | None:
    if v is None:
        return None
    return 1 if v else 0


def _build_image_row(
    listing_id: str,
    scrape_source: str,
    image_index: int,
    image_url: str,
    features: dict[str, Any],
    raw_text: str,
) -> dict[str, Any]:
    visible_areas = features.get("visible_areas") or []
    if not isinstance(visible_areas, list):
        visible_areas = []
    return {
        "listing_id": listing_id,
        "scrape_source": scrape_source,
        "image_index": image_index,
        "image_url": image_url,
        "is_collage": _bool_to_int(features.get("is_collage")),
        "visible_areas": json.dumps(visible_areas),
        "brightness_score": features.get("brightness_score"),
        "natural_light_score": features.get("natural_light_score"),
        "south_facing_likelihood": features.get("south_facing_likelihood"),
        "modernity_score": features.get("modernity_score"),
        "kitchen_age_class": features.get("kitchen_age_class"),
        "bathroom_age_class": features.get("bathroom_age_class"),
        "renovation_quality": features.get("renovation_quality"),
        "spaciousness_score": features.get("spaciousness_score"),
        "open_floor_plan": _bool_to_int(features.get("open_floor_plan")),
        "visual_clutter_score": features.get("visual_clutter_score"),
        "view_type": features.get("view_type"),
        "balcony_visible": _bool_to_int(features.get("balcony_visible")),
        "living_room_size": features.get("living_room_size"),
        "kitchen_size": features.get("kitchen_size"),
        "bedroom_size": features.get("bedroom_size"),
        "bathroom_size": features.get("bathroom_size"),
        "balcony_size": features.get("balcony_size"),
        "garden_size": features.get("garden_size"),
        "is_furnished": features.get("is_furnished"),
        "visual_description": features.get("visual_description"),
        "raw_vlm_json": raw_text,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }


def _error_image_row(
    listing_id: str,
    scrape_source: str,
    image_index: int,
    image_url: str,
    error: str,
) -> dict[str, Any]:
    null_fields = {
        k: None for k in [
            "is_collage", "visible_areas",
            "brightness_score", "natural_light_score", "south_facing_likelihood",
            "modernity_score", "kitchen_age_class", "bathroom_age_class", "renovation_quality",
            "spaciousness_score", "open_floor_plan", "visual_clutter_score",
            "view_type", "balcony_visible",
            "living_room_size", "kitchen_size", "bedroom_size", "bathroom_size", "balcony_size", "garden_size",
            "is_furnished", "visual_description", "raw_vlm_json",
        ]
    }
    return {
        "listing_id": listing_id,
        "scrape_source": scrape_source,
        "image_index": image_index,
        "image_url": image_url,
        **null_fields,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "error": error[:500],
    }


# ---------------------------------------------------------------------------
# Per-listing worker
# ---------------------------------------------------------------------------

async def process_listing(
    listing: dict[str, Any],
    http_client: httpx.AsyncClient,
    claude: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    done_pairs: set[tuple[str, int]],
    dry_run: bool,
    max_images: int = 0,
    resize_px: int = 0,
    dedup_threshold: int = 8,
    interior_threshold: float = 0.5,
    include_description: bool = True,
) -> list[dict[str, Any]]:
    listing_id = listing["listing_id"]
    scrape_source = listing["scrape_source"] or "UNKNOWN"
    object_type = listing.get("object_type")
    tier = get_analysis_tier(object_type)

    if tier == 3:
        log.debug("Skipping %s (object_type=%r, tier 3)", listing_id, object_type)
        return []

    urls = extract_image_urls(listing.get("images_json"))
    if not urls:
        return []
    if max_images > 0:
        urls = urls[:max_images]

    # Already processed this listing's grid (image_index=0 is the stitched grid)
    if (listing_id, 0) in done_pairs:
        return []

    if dry_run:
        log.info("[dry-run] %s tier=%d type=%r — %d image(s) → 1 grid call",
                 listing_id, tier, object_type, len(urls))
        return []

    # Fetch all images, skip individual failures
    raw_images: list[bytes] = []
    for idx, url in enumerate(urls):
        try:
            resp = await http_client.get(url, timeout=20.0, follow_redirects=True) \
                if not url.startswith("/raw-data-images/") and ".amazonaws.com/" not in url \
                else None

            if url.startswith("/raw-data-images/"):
                filename = url.split("/")[-1]
                path = SRED_IMAGES_DIR / filename
                if not path.exists():
                    raise FileNotFoundError(f"SRED image not found: {path}")
                raw_images.append(path.read_bytes())
            elif ".amazonaws.com/" in url:
                loop = asyncio.get_running_loop()
                raw_images.append(await loop.run_in_executor(None, _fetch_s3_bytes, url))
            else:
                resp.raise_for_status()
                raw_images.append(resp.content)
        except Exception as exc:
            log.debug("Fetch failed %s image[%d]: %s", listing_id, idx, exc)

    if not raw_images:
        return [_error_image_row(listing_id, scrape_source, 0,
                                 urls[0] if urls else "", "all_images_failed")]

    # Filter: drop near-duplicates and non-interior images
    raw_images = filter_images(raw_images, dedup_threshold, interior_threshold)

    # Stitch into a single grid, encode as base64
    try:
        grid_bytes = _build_grid(raw_images, tile_px=resize_px or 300)
        grid_b64 = base64.standard_b64encode(grid_bytes).decode()
    except Exception as exc:
        return [_error_image_row(listing_id, scrape_source, 0, urls[0], f"grid_{exc}")]

    # One API call for the whole listing
    try:
        features, raw_text = await analyse_image(grid_b64, "image/jpeg", claude, semaphore, tier, include_description)
        row = _build_image_row(listing_id, scrape_source, 0, "|".join(urls), features, raw_text)
        return [row]
    except Exception as exc:
        error_tag = f"vlm_{type(exc).__name__}: {exc}"
        log.warning("VLM failed %s: %s", listing_id, exc)
        return [_error_image_row(listing_id, scrape_source, 0, urls[0], error_tag)]


# ---------------------------------------------------------------------------
# Aggregation: listing_images → image_features
# ---------------------------------------------------------------------------

SIZE_RANK = {"large": 3, "medium": 2, "small": 1, "not_visible": 0}
FURNISHED_RANK = {"furnished": 3, "partially_furnished": 2, "unfurnished": 1, "unknown": 0}
QUALITY_RANK = {"excellent": 4, "good": 3, "average": 2, "poor": 1, "unknown": 0}
AGE_RANK = {"modern": 2, "dated": 1, "not_visible": 0, "unknown": 0}
VIEW_RANK = {"mountain": 5, "greenery": 4, "cityscape": 3, "courtyard": 2, "street": 1, "none": 0, "unknown": -1}


def _avg(vals: list[float]) -> float | None:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _best_text(vals: list[str | None], rank: dict[str, int]) -> str | None:
    clean = [v for v in vals if v and v in rank]
    if not clean:
        return None
    return max(clean, key=lambda v: rank[v])


def aggregate_features(conn: sqlite3.Connection) -> int:
    """Aggregate listing_images → image_features. Returns number of listings updated."""
    listing_ids = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT listing_id FROM listing_images"
        ).fetchall()
    ]

    rows_out: list[dict[str, Any]] = []
    for listing_id in listing_ids:
        imgs = conn.execute(
            "SELECT * FROM listing_images WHERE listing_id = ? ORDER BY image_index",
            (listing_id,),
        ).fetchall()
        imgs = [dict(r) for r in imgs]

        scrape_source = imgs[0]["scrape_source"]
        total = len(imgs)
        ok = [i for i in imgs if not i["error"]]

        def _areas(img: dict) -> set[str]:
            try:
                return set(json.loads(img.get("visible_areas") or "[]"))
            except (json.JSONDecodeError, TypeError):
                return set()

        interior     = [i for i in ok if _areas(i) & INTERIOR_LOCATIONS]
        kitchen_imgs = [i for i in ok if _areas(i) & KITCHEN_LOCATIONS]
        bath_imgs    = [i for i in ok if _areas(i) & BATHROOM_LOCATIONS]

        rows_out.append({
            "listing_id": listing_id,
            "scrape_source": scrape_source,
            "total_images": total,
            "analysed_images": len(ok),
            "interior_images": len(interior),

            # Average brightness/light across all interior images
            "brightness_score":        _avg([i["brightness_score"] for i in interior]),
            "natural_light_score":     _avg([i["natural_light_score"] for i in interior]),
            # Best (highest) south-facing likelihood across interior images
            "south_facing_likelihood": max(
                (i["south_facing_likelihood"] for i in interior if i["south_facing_likelihood"] is not None),
                default=None,
            ),

            # Average modernity across interior images
            "modernity_score":   _avg([i["modernity_score"] for i in interior]),
            # Kitchen/bathroom class from the most informative image of that room
            "kitchen_age_class": _best_text([i["kitchen_age_class"] for i in kitchen_imgs or interior], AGE_RANK),
            "bathroom_age_class": _best_text([i["bathroom_age_class"] for i in bath_imgs or interior], AGE_RANK),
            # Best renovation quality seen across any interior image
            "renovation_quality": _best_text([i["renovation_quality"] for i in interior], QUALITY_RANK),

            # Max spaciousness (most spacious room wins for listing appeal)
            "spaciousness_score": max(
                (i["spaciousness_score"] for i in interior if i["spaciousness_score"] is not None),
                default=None,
            ),
            # Open floor plan if ANY interior image shows it
            "open_floor_plan": 1 if any(i["open_floor_plan"] == 1 for i in interior) else (
                0 if any(i["open_floor_plan"] == 0 for i in interior) else None
            ),
            # Average clutter across interior images
            "visual_clutter_score": _avg([i["visual_clutter_score"] for i in interior]),

            # Best view type across all images (not just interior)
            "view_type": _best_text([i["view_type"] for i in ok], VIEW_RANK),
            # Balcony visible in any image
            "balcony_visible": 1 if any(i["balcony_visible"] == 1 for i in ok) else (
                0 if any(i["balcony_visible"] == 0 for i in ok) else None
            ),

            # Room sizes: Claude assessed each room directly from whatever was visible
            # in the image (including collage panels), so aggregate across all ok images.
            "living_room_size": _best_text([i["living_room_size"] for i in ok], SIZE_RANK),
            "kitchen_size":     _best_text([i["kitchen_size"]     for i in ok], SIZE_RANK),
            "bedroom_size":     _best_text([i["bedroom_size"]     for i in ok], SIZE_RANK),
            "bathroom_size":    _best_text([i["bathroom_size"]    for i in ok], SIZE_RANK),
            "balcony_size":     _best_text([i["balcony_size"]     for i in ok], SIZE_RANK),
            "garden_size":      _best_text([i["garden_size"]      for i in ok], SIZE_RANK),
            "is_furnished":     _best_text([i["is_furnished"]     for i in ok], FURNISHED_RANK),
            # Concatenate all per-image descriptions — embed this at the listing level
            "visual_description": " ".join(
                i["visual_description"] for i in ok
                if i.get("visual_description")
            ) or None,

            "aggregated_at": datetime.now(timezone.utc).isoformat(),
        })

    with conn:
        conn.executemany(UPSERT_FEATURES, rows_out)

    return len(rows_out)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run(concurrency: int, limit: int | None, dry_run: bool, aggregate_only: bool, exclude_sources: list[str] | None = None, max_images: int = 0, resize_px: int = 0, dedup_threshold: int = 8, interior_threshold: float = 0.5, include_description: bool = True) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not dry_run and not aggregate_only:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    features_conn = open_features_db()

    if aggregate_only:
        log.info("Aggregating listing_images → image_features …")
        n = aggregate_features(features_conn)
        log.info("Aggregated %d listings.", n)
        features_conn.close()
        return

    log.info("Loading listings from %s", MAIN_DB_PATH)
    if exclude_sources:
        log.info("Excluding sources: %s", exclude_sources)
    listings = load_listings(limit, exclude_sources)
    log.info("Loaded %d listings", len(listings))

    done_pairs = already_processed_images(features_conn)
    log.info("%d image slots already processed", len(done_pairs))

    semaphore = asyncio.Semaphore(concurrency)
    claude = anthropic.AsyncAnthropic(api_key=api_key)

    total_ok = 0
    total_errors = 0
    flush_buffer: list[dict[str, Any]] = []
    FLUSH_EVERY = 100  # write to DB every N completed listings

    def _flush(buffer: list[dict[str, Any]]) -> None:
        if not buffer:
            return
        with features_conn:
            features_conn.executemany(UPSERT_IMAGE, buffer)
        log.info("Flushed %d rows to disk (%d ok so far)", len(buffer), total_ok)
        buffer.clear()

    async with httpx.AsyncClient(
        headers={"User-Agent": "datathon-image-analyzer/1.0"},
        limits=httpx.Limits(max_connections=concurrency * 3),
    ) as http_client:
        tasks = [
            process_listing(l, http_client, claude, semaphore, done_pairs, dry_run,
                            max_images, resize_px, dedup_threshold, interior_threshold, include_description)
            for l in listings
        ]

        pbar = sync_tqdm(total=len(tasks), desc="Listings") if HAS_TQDM else None
        completed = 0

        for coro in asyncio.as_completed(tasks):
            rows = await coro
            completed += 1

            for r in rows:
                if r.get("error"):
                    total_errors += 1
                else:
                    total_ok += 1
                flush_buffer.append(r)

            if pbar:
                pbar.set_postfix(ok=total_ok, err=total_errors)
                pbar.update(1)

            # Flush to disk periodically so a disconnect doesn't lose everything
            if not dry_run and completed % FLUSH_EVERY == 0:
                _flush(flush_buffer)

        if pbar:
            pbar.close()

        # Final flush for any remainder
        if not dry_run:
            _flush(flush_buffer)

    if not dry_run:
        log.info("Aggregating listing_images → image_features …")
        n = aggregate_features(features_conn)
        log.info("Aggregated %d listings", n)

    log.info("Done — images ok: %d  errors: %d", total_ok, total_errors)

    by_type: dict[str, int] = {}
    for r in errors:
        tag = r["error"].split("_")[0] if r["error"] else "other"
        by_type[tag] = by_type.get(tag, 0) + 1
    if by_type:
        log.info("Error breakdown: %s", by_type)

    features_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse listing images with Claude Haiku.")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max parallel Claude requests (default: 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N listings (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls; just log what would be processed")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Re-aggregate listing_images → image_features without re-fetching")
    parser.add_argument("--exclude-sources", nargs="+", metavar="SOURCE",
                        help="Skip listings from these scrape_source values (e.g. ROBINREAL COMPARIS)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to include in the grid per listing (default: 0 = all)")
    parser.add_argument("--resize-px", type=int, default=300,
                        help="Tile size in pixels for each image in the grid (default: 300)")
    parser.add_argument("--dedup-threshold", type=int, default=8,
                        help="Perceptual hash distance for near-duplicate detection (default: 8, requires imagehash)")
    parser.add_argument("--interior-threshold", type=float, default=0.5,
                        help="CLIP interior score threshold 0-1 (default: 0.5, requires open_clip)")
    parser.add_argument("--no-description", action="store_true",
                        help="Skip visual_description field — saves ~$9 on full run")
    args = parser.parse_args()

    asyncio.run(run(args.concurrency, args.limit, args.dry_run, args.aggregate_only,
                    args.exclude_sources, args.max_images, args.resize_px,
                    args.dedup_threshold, args.interior_threshold,
                    include_description=not args.no_description))


if __name__ == "__main__":
    main()
