from app.models.schemas import HardFilters
from app.participant.hard_fact_extraction import extract_hard_facts
from app.participant.ranking import rank_listings
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts
from app.harness.search_service import to_hard_filter_params


def test_extract_hard_facts_returns_stub_structure() -> None:
    result = extract_hard_facts("3 room flat in zurich")

    assert isinstance(result, HardFilters)


def test_landmark_city_is_not_promoted_to_hard_city_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.hard_fact_extraction._llm_extractor.extract_combined",
        lambda query: None,
    )
    result = extract_hard_facts("quiet flat near ETH Zürich, max 15 min walk, 3 rooms")

    assert result.city is None
    assert result.min_rooms == 2.5
    assert result.max_rooms == 3.5


def test_hedged_features_do_not_become_hard_filters(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.hard_fact_extraction._llm_extractor.extract_combined",
        lambda query: None,
    )
    result = extract_hard_facts(
        "3-room flat in Zürich, ideally with a balcony, parking would be nice, elevator if possible"
    )

    assert result.city == ["Zürich"]
    assert result.features is None


def test_rule_fallback_extracts_explicit_sort_by(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.hard_fact_extraction._llm_extractor.extract_combined",
        lambda query: None,
    )
    result = extract_hard_facts("cheapest 3-room flat in Zürich")

    assert result.city == ["Zürich"]
    assert result.sort_by == "price_asc"


def test_llm_hard_guards_normalize_exact_room_bounds_and_landmark_city(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.hard_fact_extraction._llm_extractor.extract_combined",
        lambda query: {
            "hard": {"city": ["Zürich"], "min_rooms": 4.0, "max_rooms": 4.5},
            "soft": {},
        },
    )

    result = extract_hard_facts("quiet flat near ETH Zürich, 4 rooms")

    assert result.city is None
    assert result.min_rooms == 3.5
    assert result.max_rooms == 4.5


def test_participant_soft_fact_modules_are_importable() -> None:
    candidates = [{"listing_id": "1", "title": "Example"}]

    soft_facts = extract_soft_facts("bright flat")
    filtered = filter_soft_facts(candidates, soft_facts)
    ranked = rank_listings(filtered, soft_facts)

    assert isinstance(soft_facts, dict)
    assert isinstance(filtered, list)
    assert all(item["listing_id"] in {"1"} for item in filtered)
    assert isinstance(ranked, list)
    assert ranked
    assert all(item.listing_id for item in ranked)
    assert all(isinstance(item.score, float) for item in ranked)


def test_harness_service_converts_hard_filters_to_search_params() -> None:
    filters = HardFilters(
        city=["Zurich"],
        features=["balcony"],
        limit=5,
        offset=2,
        sort_by="price_asc",
    )

    params = to_hard_filter_params(filters)

    assert params.city == ["Zurich"]
    assert params.features == ["balcony"]
    assert params.limit == 5
    assert params.offset == 2
    assert params.sort_by == "price_asc"
