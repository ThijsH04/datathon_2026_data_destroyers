"""Shared lexicon for hard/soft fact extraction and ranking.

Bilingual (German + English, plus a few French/Italian fallbacks) keyword sets
that map natural-language terms to structured categories. Centralized here so
the extractor and the ranker stay in lock-step on what each term means.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Cities — canonical English/German names mapped to the dataset's city column.
# Values are returned exactly as they appear in the listings DB (lowercased
# match in the SQL hard filter handles casing).
# ---------------------------------------------------------------------------
CITY_ALIASES: dict[str, str] = {
    "zurich": "Zürich",
    "zürich": "Zürich",
    "zuerich": "Zürich",
    "geneva": "Genève",
    "genève": "Genève",
    "geneve": "Genève",
    "genf": "Genève",
    "bern": "Bern",
    "berne": "Bern",
    "basel": "Basel",
    "bâle": "Basel",
    "basle": "Basel",
    "lausanne": "Lausanne",
    "winterthur": "Winterthur",
    "lucerne": "Luzern",
    "luzern": "Luzern",
    "st. gallen": "St. Gallen",
    "st gallen": "St. Gallen",
    "saint gallen": "St. Gallen",
    "sankt gallen": "St. Gallen",
    "lugano": "Lugano",
    "biel": "Biel/Bienne",
    "bienne": "Biel/Bienne",
    "biel/bienne": "Biel/Bienne",
    "thun": "Thun",
    "schaffhausen": "Schaffhausen",
    "fribourg": "Fribourg",
    "freiburg": "Fribourg",
    "neuchatel": "Neuchâtel",
    "neuchâtel": "Neuchâtel",
    "sion": "Sion",
    "sitten": "Sion",
    "chur": "Chur",
    "köniz": "Köniz",
    "koniz": "Köniz",
    "uster": "Uster",
    "zug": "Zug",
    "olten": "Olten",
    "solothurn": "Solothurn",
    "soleure": "Solothurn",
    "burgdorf": "Burgdorf",
    "bulle": "Bulle",
    "delémont": "Delémont",
    "delemont": "Delémont",
    "grenchen": "Grenchen",
    "la chaux-de-fonds": "La Chaux-de-Fonds",
    "chaux-de-fonds": "La Chaux-de-Fonds",
    "crans-montana": "Crans-Montana",
}


# ---------------------------------------------------------------------------
# Hard feature flags — natural-language synonyms → DB feature key.
# (Only tokens we are very confident about are treated as HARD constraints.
#  Looser hints like "with parking ideally" are handled as soft boosts.)
# ---------------------------------------------------------------------------
HARD_FEATURE_KEYWORDS: dict[str, list[str]] = {
    "balcony": [
        "balcony", "balconies", "balkon", "balkone",
        "terrasse", "terrace", "loggia",
    ],
    "elevator": ["elevator", "lift", "aufzug"],
    "parking": ["parking", "parkplatz", "parkieren", "stellplatz"],
    "garage": ["garage", "tiefgarage"],
    "fireplace": ["fireplace", "cheminée", "kamin", "cheminee"],
    "child_friendly": [
        "child-friendly", "child friendly", "kid friendly", "kid-friendly",
        "kinderfreundlich", "familienfreundlich", "family friendly", "family-friendly",
    ],
    "pets_allowed": [
        "pets allowed", "pet friendly", "pet-friendly",
        "haustiere erlaubt", "haustiere", "hund erlaubt",
    ],
    "wheelchair_accessible": [
        "wheelchair", "wheelchair-accessible", "wheelchair accessible",
        "rollstuhl", "rollstuhlgängig", "barrierefrei",
    ],
    "minergie_certified": ["minergie"],
    "new_build": ["new build", "neubau", "newly built", "new construction"],
}


# ---------------------------------------------------------------------------
# Soft preferences — fuzzy/subjective concepts the ranker scores against
# listing text + structured fields.
# ---------------------------------------------------------------------------
SOFT_KEYWORDS: dict[str, list[str]] = {
    "bright": [
        "bright", "sunny", "light-flooded", "lightflooded", "lots of light",
        "luminous", "sunlit",
        "hell", "helle", "heller", "lichtdurchflutet", "sonnig", "sonnendurchflutet",
        "lumineux", "lumineuse",
    ],
    "modern": [
        "modern", "modernes", "modernized", "modernised", "renovated", "renoviert",
        "newly renovated", "freshly renovated", "neu renoviert", "kernsaniert",
        "sanier", "saniert", "neuwertig",
    ],
    "quiet": [
        "quiet", "calm", "peaceful", "ruhig", "ruhige", "ruhiger",
        "abgelegen", "tranquille", "street noise", "lärm", "laerm",
    ],
    "safe": [
        "safe", "safety", "secure", "low crime", "crime", "safer",
        "sicher", "sicherheit", "sichere", "sicheres", "kriminalität", "kriminalitaet",
        "sécurisé", "securise", "sécurité", "securite",
    ],
    "green": [
        "green", "park", "parks", "nature", "nature nearby", "green area",
        "grün", "gruen", "park", "parks", "natur", "naherholung", "erholung",
        "verdure", "parc", "nature",
    ],
    "near_water": [
        "lake", "river", "water", "waterfront", "lakeside", "riverside",
        "see", "fluss", "ufer", "wasser", "seesicht", "seeufer",
        "lac", "rivière", "riviere", "eau",
    ],
    "nightlife": [
        "nightlife", "bars", "bar", "restaurants", "lively", "vibrant",
        "going out", "ausgehen", "bars", "restaurants", "lebendig", "belebt",
        "sortir", "animé", "anime",
    ],
    "central": [
        "central", "zentral", "in the center", "city center", "stadtmitte",
        "mitten in", "innenstadt", "centre", "downtown",
    ],
    "view": [
        "view", "views", "aussicht", "ausblick", "blick",
        "panorama", "panoramic", "lake view", "seesicht",
        "mountain view", "bergsicht",
    ],
    "spacious": [
        "spacious", "large", "big", "geräumig", "grosszügig", "grosse",
        "weitläufig", "spaziös",
    ],
    "cozy": [
        "cozy", "cosy", "rustic", "gemütlich", "charming", "charmant",
    ],
    "family_friendly": [
        "family", "familie", "familien", "kids", "kinder", "child", "school",
        "schule", "kindergarten", "playground", "spielplatz", "for kids",
        "family-friendly", "family friendly", "kinderfreundlich",
        "familienfreundlich",
    ],
    "near_transport": [
        "transport", "public transport", "tram", "bus", "train",
        "öv", "oeffentliche", "öffentliche verkehrsmittel", "öffi",
        "bahnhof", "train station", "sbb", "haltestelle", "stop",
    ],
    "near_school": [
        "school", "schule", "schulen", "kindergarten", "near schools",
        "education", "uni", "university", "universität", "hochschule",
    ],
    "luxury": [
        "luxury", "luxurious", "premium", "high-end", "exclusive",
        "luxuriös", "edel", "noble", "exklusiv", "hochwertig",
    ],
    "cheap": [
        "cheap", "affordable", "budget", "low-cost", "inexpensive",
        "günstig", "preiswert", "erschwinglich", "bezahlbar",
        "not too expensive", "nothing too expensive", "nicht zu teuer",
        "not expensive", "cheapest", "lowest", "pas trop cher", "pas cher",
        "abordable", "bon marché",
    ],
    "balcony_pref": [
        "balcony", "balkon", "balcon", "terrasse", "terrace", "loggia",
    ],
    "parking_pref": [
        "parking", "parkplatz", "garage", "stellplatz",
    ],
    "elevator_pref": [
        "elevator", "lift", "aufzug",
    ],
    "garden": [
        "garden", "garten", "gardens", "yard",
    ],
    "pet_pref": [
        "pet", "pets", "dog", "cat", "haustier", "hund", "katze",
    ],
    "new_pref": [
        "new", "neu", "neubau", "brand new", "new construction",
    ],
    "furnished": [
        "furnished", "möbliert", "moebliert", "fully furnished",
    ],
    "student": [
        "student", "studenten", "studi", "studierende", "wg",
    ],
}


# Mapping from soft key to which structured feature flag (if any) gives a
# stronger signal when present in the listing.
SOFT_FEATURE_BOOST_MAP: dict[str, str] = {
    "balcony_pref": "feature_balcony",
    "parking_pref": "feature_parking",
    "elevator_pref": "feature_elevator",
    "new_pref": "feature_new_build",
    "pet_pref": "feature_pets_allowed",
    "family_friendly": "feature_child_friendly",
    "luxury": "feature_minergie_certified",
}


# ---------------------------------------------------------------------------
# Geocoded anchors — places users commonly reference for "near X" / commute.
# Coords are approximate centroids; good enough for distance scoring.
# ---------------------------------------------------------------------------
LOCATION_ANCHORS: dict[str, tuple[str, float, float]] = {
    "eth": ("ETH Zürich", 47.3763, 8.5483),
    "eth zurich": ("ETH Zürich", 47.3763, 8.5483),
    "eth zürich": ("ETH Zürich", 47.3763, 8.5483),
    "epfl": ("EPFL", 46.5191, 6.5668),
    "uzh": ("University of Zürich", 47.3744, 8.5510),
    "university of zurich": ("University of Zürich", 47.3744, 8.5510),
    "hb zurich": ("Zürich HB", 47.3779, 8.5403),
    "hauptbahnhof zurich": ("Zürich HB", 47.3779, 8.5403),
    "hauptbahnhof zürich": ("Zürich HB", 47.3779, 8.5403),
    "zurich hb": ("Zürich HB", 47.3779, 8.5403),
    "zürich hb": ("Zürich HB", 47.3779, 8.5403),
    "bern hb": ("Bern HB", 46.9489, 7.4400),
    "geneva airport": ("Geneva Airport", 46.2381, 6.1090),
    "zurich airport": ("Zürich Airport", 47.4647, 8.5492),
    "cern": ("CERN", 46.2333, 6.0530),
    "unil": ("UNIL Lausanne", 46.5217, 6.5790),
    "ehl": ("EHL Lausanne", 46.5469, 6.5837),
    "usi": ("USI Lugano", 46.0107, 8.9576),
    "uni basel": ("Uni Basel", 47.5587, 7.5837),
    "basel sbb": ("Basel SBB", 47.5476, 7.5896),
}


# Words that modify intent — useful when picking which side of "X under N" or
# "N to N" the parser should treat as a budget vs a hint.
PRICE_UPPER_WORDS = {
    "under", "below", "less than", "max", "maximum", "up to", "at most",
    "bis", "höchstens", "maximal", "<=",
}
PRICE_LOWER_WORDS = {
    "over", "above", "more than", "min", "minimum", "at least", "ab",
    "mindestens", ">=",
}
