"""Hard-fact extractor.

Primary path: a single Claude call (see `_llm_extractor.extract_combined`)
returns both hard filters and soft preferences. The cached result is reused
by the soft extractor in the same request, so we only pay for one LLM call
per unique query.

Fallback path: rule-based bilingual parser. Triggers whenever the LLM path
is unavailable (no API key, SDK missing, network error, malformed output).
The fallback is conservative — it only marks something as a hard constraint
when the wording is unambiguous. Soft hints stay for the ranker.

Hard-filter precision is the make-or-break automated metric, so we err on
the side of NOT over-constraining.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from app.models.schemas import HardFilters
import boto3



def extract_hard_facts(query: str) -> HardFilters:
    # Intentionally stubbed. Teams are expected to replace this with
    # their own query understanding or structured extraction logic.

    # prompt =     """
    # You are a housing listings assistant. When a user describes what they're looking for in a home, parse their request and provide hard constraints:

    # **Hard constraints** (must-haves): price ceiling, minimum bedrooms/bathrooms, location/city, property type, pet policy, etc.

    # Output a json file containing the hard and soft constraints. The list of required constraints can be found on https://institutions-interactive-size-apnic.trycloudflare.com/openapi.json. Use this URL for future api calls as well. For features, there are the following options: balcony, elevator, parking, garage, fireplace, child_friendly, pets_allowed, temporary, new_build, wheelchair_accessible, private_laundry,minergie_certified 
    # Then call the POST /listings/search/filter endpoint with the parsed constraints. Being unable to output anything is fine, give the user suggestions on how they could get an output in that case
    # """
    # client = boto3.client("bedrock-runtime", region_name="us-west-2")
    # try:
    #     response = client.converse(
    #         modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [{"text": prompt}]
    #             }
    #         ]
    #     )

    #     return i, response["output"]["message"]["content"][0]["text"].strip()

    # except Exception as e:
    #     return i, f"Error: {str(e)}"
    return HardFilters()
