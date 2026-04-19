## Data Destroyers - Datathon 2026 Challenge Summary

### Problem Statement
This repository targets the Datathon 2026 listing-search challenge: convert natural-language housing queries into high-quality ranked listing results.

The core requirement is hybrid retrieval:
- hard constraints must be satisfied (city, price, rooms, offer type, distance, features)
- soft preferences should guide ranking (for example: family friendly, bright, transport access, neighborhood quality)

### Solution Shape In This Repo
The project is a starter harness extended with a practical search pipeline:

1. Query understanding
- Hard-fact extraction produces structured constraints.
- Soft-fact extraction captures ranking hints/preferences.

2. Candidate retrieval (hard filtering)
- SQLite-backed filtering over structured listing fields.
- Geospatial radius filtering and sortable candidate retrieval.

3. Soft filtering and ranking
- Soft-filter pass for preference-aware candidate pruning.
- Ranking stage combines relevance signals and emits explanations.

4. API and app delivery
- FastAPI endpoints for both text-based and structured search.
- MCP Apps SDK server + web widget support for interactive experiences.

### Data Pipeline
The ingestion flow reads CSV listing exports and builds a local SQLite search database.

Current import behavior:
- bootstrap prefers `*_enriched.csv` files when available
- avoids accidental duplicate imports from parallel base/enriched exports
- creates indexes for fast structured retrieval

### Enriched Feature Coverage
The database includes both core listing fields and enriched context features such as:
- proximity/count metrics (transit, shops, parks, schools, hospitals, nightlife)
- distance-to-POI/noise/water metrics
- quality scores (`_livability`, `_activity`, etc.)
- risk metrics (`weighted_crime_per_1000`, `crime_index_0_100`)

Note: some enriched columns can be null for individual rows when source enrichment is missing for those listings.

### Image Analysis
The pipeline also uses image-aware signals as a complementary relevance layer.

- Listings carry image URLs and hero images that are exposed through the API for downstream scoring and UI display.
- Visual information can support soft preferences that are hard to capture with metadata alone (for example: brightness, modern interior style, kitchen quality, balcony/outdoor feel).
- In ranking terms, image-derived signals are treated as soft evidence: they improve ordering among already valid candidates, while hard constraints remain enforced by structured filtering.

In short, image analysis in this repo is designed to strengthen the relevance ranking layer, not to replace hard filtering.

### Current Technical Status
- Import tuple, INSERT columns, and schema column counts are aligned.
- Rebuilds from enriched sources complete successfully.
- Deduplication integrity check on rebuilt DB is clean (`count(*) == count(distinct listing_id)`).
- `uv` environment and lockfile are repaired and runnable.

### Key Endpoints
- `GET /health`: service health check
- `POST /listings`: natural-language query flow
- `POST /listings/search/filter`: direct structured hard-filter search

### Team Value
This repo now functions as a robust baseline for:
- experimenting with better hard/soft query parsing
- improving ranking quality with richer signals
- shipping a demonstrable, API-first prototype for challenge evaluation

