from app.participant.ranking import rank_listings
from app.participant.soft_fact_extraction import extract_soft_facts
from app.participant.soft_filtering import filter_soft_facts


def test_extract_soft_facts_returns_stub_structure() -> None:
    result = extract_soft_facts("bright flat near transport")

    assert isinstance(result, dict)


def test_rule_fallback_detects_conflicting_soft_preferences(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.soft_fact_extraction._llm_extractor.extract_combined",
        lambda query: None,
    )
    result = extract_soft_facts(
        "affordable apartment right in the city centre of Zürich, ideally under 1800 CHF"
    )

    assert result["source"] == "rules"
    assert result["preferences"]["cheap"] == result["preferences"]["central"]
    assert ["cheap", "central"] in result["conflicts"]
    assert result["soft_budget_hint"] == "value_sensitive"


def test_rule_fallback_keeps_hedged_features_soft(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.soft_fact_extraction._llm_extractor.extract_combined",
        lambda query: None,
    )
    result = extract_soft_facts(
        "3-room flat in Zürich, ideally with a balcony, parking would be nice, elevator if possible"
    )

    assert result["feature_boosts"]["feature_balcony"] == 0.6
    assert result["feature_boosts"]["feature_parking"] == 0.6
    assert result["feature_boosts"]["feature_elevator"] == 0.6


def test_rule_fallback_budget_hint_lowest(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.soft_fact_extraction._llm_extractor.extract_combined",
        lambda query: None,
    )
    result = extract_soft_facts("cheapest available studio in Basel I can find")

    assert result["soft_budget_hint"] == "lowest"
    assert result["soft_max_price"] == 1


def test_rule_fallback_detects_enriched_lifestyle_preferences(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.soft_fact_extraction._llm_extractor.extract_combined",
        lambda query: None,
    )
    result = extract_soft_facts(
        "safe green apartment near the lake with lively nightlife nearby"
    )

    assert result["preferences"]["safe"] == 1.0
    assert result["preferences"]["green"] == 1.0
    assert result["preferences"]["near_water"] == 1.0
    assert result["preferences"]["nightlife"] == 1.0


def test_llm_soft_guards_remove_unrelated_transport_and_too_bright(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.soft_fact_extraction._llm_extractor.extract_combined",
        lambda query: {
            "hard": {"features": ["balcony"]},
            "soft": {
                "preferences": {
                    "bright": 1.3,
                    "modern": 1.1,
                    "near_transport": 0.9,
                },
                "anchors": [{"label": "ETH Zürich", "lat": 47.3768, "lon": 8.5492}],
                "soft_budget_hint": "none",
                "conflicts": [],
                "dominant_signal": None,
                "negations": [],
            },
        },
    )

    result = extract_soft_facts(
        "too bright, modern apartment near ETH Zürich with balcony, 4 rooms, max 4000 CHF"
    )

    assert result["source"] == "llm"
    assert "bright" not in result["preferences"]
    assert "near_transport" not in result["preferences"]
    assert result["preferences"]["modern"] == 1.1
    assert result["preferences"]["balcony_pref"] == 0.8
    assert result["feature_boosts"]["feature_balcony"] == 0.8
    assert "too_bright" in result["negations"]


def test_llm_soft_guard_keeps_explicit_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.participant.soft_fact_extraction._llm_extractor.extract_combined",
        lambda query: {
            "hard": {},
            "soft": {
                "preferences": {"near_transport": 1.2},
                "anchors": [],
                "soft_budget_hint": "none",
                "conflicts": [],
                "dominant_signal": None,
                "negations": [],
            },
        },
    )

    result = extract_soft_facts("Wohnung in der Nähe vom Hauptbahnhof Zürich")

    assert result["preferences"]["near_transport"] == 1.2


def test_filter_soft_facts_returns_candidate_subset() -> None:
    candidates = [{"listing_id": "1"}, {"listing_id": "2"}]

    filtered = filter_soft_facts(candidates, {"raw_query": "quiet"})

    assert isinstance(filtered, list)
    assert {item["listing_id"] for item in filtered} <= {"1", "2"}


def test_filter_soft_facts_trims_far_anchor_and_non_home_results() -> None:
    near_homes = [
        {
            "listing_id": f"near-{idx}",
            "object_category": "Wohnung",
            "latitude": 47.3763 + idx * 0.0001,
            "longitude": 8.5483,
        }
        for idx in range(25)
    ]
    candidates = [
        *near_homes,
        {
            "listing_id": "far-home",
            "object_category": "Wohnung",
            "latitude": 46.011,
            "longitude": 8.955,
        },
        {
            "listing_id": "near-commercial",
            "object_category": "Gewerbeobjekt",
            "latitude": 47.3763,
            "longitude": 8.5483,
        },
        {
            "listing_id": "missing-coordinates",
            "object_category": "Wohnung",
        },
    ]

    filtered = filter_soft_facts(
        candidates,
        {
            "object_category_hint": ["Wohnung"],
            "anchors": [
                {
                    "label": "ETH Zürich",
                    "lat": 47.3763,
                    "lon": 8.5483,
                    "max_minutes": 20,
                }
            ],
        },
    )

    ids = {item["listing_id"] for item in filtered}
    assert "far-home" not in ids
    assert "near-commercial" not in ids
    assert "missing-coordinates" not in ids
    assert {item["listing_id"] for item in near_homes} <= ids


def test_rank_listings_returns_ranked_shape() -> None:
    ranked = rank_listings(
        candidates=[
            {
                "listing_id": "abc",
                "title": "Example",
                "city": "Zurich",
                "price": 2500,
                "rooms": 3.0,
                "latitude": 47.37,
                "longitude": 8.54,
                "street": "Main 1",
                "postal_code": "8000",
                "canton": "ZH",
                "area": 75.0,
                "available_from": "2026-06-01",
                "image_urls": ["https://example.com/1.jpg"],
                "hero_image_url": "https://example.com/1.jpg",
                "original_url": "https://example.com/listing",
                "features": ["balcony", "elevator"],
                "offer_type": "RENT",
                "object_category": "Wohnung",
                "object_type": "Apartment",
            }
        ],
        soft_facts={"raw_query": "bright"},
    )

    assert len(ranked) == 1
    assert ranked[0].listing_id == "abc"
    assert isinstance(ranked[0].score, float)
    assert isinstance(ranked[0].reason, str)
    assert ranked[0].listing.id == "abc"
    assert ranked[0].listing.title == "Example"
    assert ranked[0].listing.city == "Zurich"
    assert ranked[0].listing.image_urls
    assert "has photos" not in ranked[0].reason


def test_rank_listings_uses_enriched_transport_and_shop_signals() -> None:
    ranked = rank_listings(
        candidates=[
            {
                "listing_id": "near",
                "title": "Well connected flat",
                "city": "Zurich",
                "dist_to_transit_m": 60,
                "nearest_transit_name": "Central Stop",
                "transit_count_500m": 14,
                "dist_to_shops_m": 90,
                "shops_count_500m": 5,
                "pedestrian_zones_count_500m": 8,
            },
            {
                "listing_id": "far",
                "title": "Remote flat",
                "city": "Zurich",
                "dist_to_transit_m": 1800,
                "transit_count_500m": 0,
                "dist_to_shops_m": 2200,
                "shops_count_500m": 0,
                "pedestrian_zones_count_500m": 0,
            },
        ],
        soft_facts={"preferences": {"near_transport": 1.0, "central": 1.0}},
    )

    assert [item.listing_id for item in ranked] == ["near", "far"]
    assert "Central Stop nearby" in ranked[0].reason


def test_rank_listings_uses_enriched_environment_quiet_and_safety_signals() -> None:
    ranked = rank_listings(
        candidates=[
            {
                "listing_id": "calm",
                "title": "Calm green home",
                "city": "Zurich",
                "dist_to_parks_m": 120,
                "parks_count_500m": 4,
                "dist_to_lakes_m": 350,
                "dist_to_noisy_roads_m": 1600,
                "dist_to_noisy_trains_m": 1400,
                "nightlife_count_500m": 1,
                "weighted_crime_per_1000": 220,
            },
            {
                "listing_id": "busy",
                "title": "Busy urban home",
                "city": "Zurich",
                "dist_to_parks_m": 1800,
                "parks_count_500m": 0,
                "dist_to_lakes_m": 2500,
                "dist_to_noisy_roads_m": 40,
                "dist_to_noisy_trains_m": 50,
                "nightlife_count_500m": 35,
                "weighted_crime_per_1000": 3200,
            },
        ],
        soft_facts={"preferences": {"green": 1.0, "near_water": 0.8, "quiet": 1.0, "safe": 1.0}},
    )

    assert [item.listing_id for item in ranked] == ["calm", "busy"]
    assert "parks and green space nearby" in ranked[0].reason
