from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.hard_filters import HardFilterParams, search_listings
from app.models.schemas import HardFilters, ListingsResponse
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.ranking import rank_listings
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts


# Size of the candidate pool we hand to the ranker. The API caller still gets
# back at most `limit` listings; a wider pool just gives the ranker more room
# to reorder by soft signals. 500 is a safe ceiling for SQLite locally.
_CANDIDATE_POOL_SIZE = 500


def filter_hard_facts(db_path: Path, hard_facts: HardFilters) -> list[dict[str, Any]]:
    return search_listings(db_path, to_hard_filter_params(hard_facts))


def query_from_text(
    *,
    db_path: Path,
    query: str,
    limit: int,
    offset: int,
    image_features_db_path: Path | None = None,
) -> ListingsResponse:
    hard_facts = extract_hard_facts(query)
    soft_facts = extract_soft_facts(query)

    # Fetch a wide candidate pool so the ranker can reorder by soft signals,
    # then slice the ranked result down to the caller's window.
    pool_filters = hard_facts.model_copy(update={
        "limit": _CANDIDATE_POOL_SIZE,
        "offset": 0,
    })
    candidates = filter_hard_facts(db_path, pool_filters)
    candidates = filter_soft_facts(candidates, soft_facts)

    ranked = rank_listings(
        candidates,
        soft_facts,
        preserve_order=hard_facts.sort_by is not None,
        image_features_db_path=image_features_db_path,
    )
    window = ranked[offset : offset + limit]

    return ListingsResponse(
        listings=window,
        meta={
            "query": query,
            "hard_facts": hard_facts.model_dump(exclude_none=True),
            "soft_facts": _meta_soft_facts(soft_facts),
            "candidates_considered": len(candidates),
            "total_ranked": len(ranked),
            "limit": limit,
            "offset": offset,
        },
    )


def query_from_filters(
    *,
    db_path: Path,
    hard_facts: HardFilters | None,
) -> ListingsResponse:
    structured_hard_facts = hard_facts or HardFilters()
    soft_facts = extract_soft_facts("")
    candidates = filter_hard_facts(db_path, structured_hard_facts)
    candidates = filter_soft_facts(candidates, soft_facts)
    return ListingsResponse(
        listings=rank_listings(
            candidates,
            soft_facts,
            preserve_order=structured_hard_facts.sort_by is not None,
        ),
        meta={
            "hard_facts": structured_hard_facts.model_dump(exclude_none=True),
            "candidates_considered": len(candidates),
        },
    )


def to_hard_filter_params(hard_facts: HardFilters) -> HardFilterParams:
    return HardFilterParams(
        city=hard_facts.city,
        excluded_city=hard_facts.excluded_city,
        postal_code=hard_facts.postal_code,
        excluded_postal_code=hard_facts.excluded_postal_code,
        canton=hard_facts.canton,
        min_price=hard_facts.min_price,
        max_price=hard_facts.max_price,
        min_rooms=hard_facts.min_rooms,
        max_rooms=hard_facts.max_rooms,
        latitude=hard_facts.latitude,
        longitude=hard_facts.longitude,
        radius_km=hard_facts.radius_km,
        features=hard_facts.features,
        offer_type=hard_facts.offer_type,
        object_category=hard_facts.object_category,
        limit=hard_facts.limit,
        offset=hard_facts.offset,
        sort_by=hard_facts.sort_by,
    )


def _meta_soft_facts(soft_facts: dict[str, Any]) -> dict[str, Any]:
    """Strip noisy fields from the soft-facts meta payload."""
    meta = dict(soft_facts)
    meta.pop("tokens", None)
    return meta
