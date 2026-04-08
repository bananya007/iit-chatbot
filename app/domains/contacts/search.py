"""
domains/contacts/search.py
Slot-driven search for the iit_contacts index.

When entity_type is known:
  "office"  → filter category to Administrative / Center / Institute / Academic
  "person"  → filter category to Faculty / Staff

When entity_type is null, no category filter is applied (fuzzy fallback).
The extracted name slot is used as the search text when available, which
strips filler phrases ("how do I contact…") for cleaner ES matching.
"""

import logging

from app.common.es_client import es

logger = logging.getLogger(__name__)

INDEX = "iit_contacts"
SOURCE_FIELDS = [
    "name", "department", "category", "description",
    "phone", "fax", "email", "building", "address",
    "city", "state", "zip", "source_url",
]

_OFFICE_CATEGORIES = ["Administrative", "Center", "Institute", "Academic"]
_PERSON_CATEGORIES = ["Faculty", "Staff"]


def search_contacts(
    query: str,
    entity_type: str = None,
    name: str = None,
    top_k: int = 3,
    exact_name: bool = False,
) -> list:
    """
    Args:
        query:       raw user query (used as fallback search text)
        entity_type: "office" | "person" | None
        name:        extracted office/person name from slot extraction
        top_k:       max results before score-gating
        exact_name:  True when a specific name was extracted — tightens score-gate
                     and caps results at 1

    Returns list of ES hit dicts.
    """
    # Use the cleaner extracted name when available, else the raw query
    search_text = name if name else query

    text_clause = {
        "multi_match": {
            "query":     search_text,
            "fields":    ["name^3", "department^2", "description"],
            "type":      "best_fields",
            "fuzziness": "AUTO",
        }
    }

    filter_clauses = []
    if entity_type == "office":
        filter_clauses.append({"terms": {"category": _OFFICE_CATEGORIES}})
    elif entity_type == "person":
        filter_clauses.append({"terms": {"category": _PERSON_CATEGORIES}})
    # entity_type=None → no category filter, search all records

    es_query = {
        "bool": {
            "must":   [text_clause],
            "filter": filter_clauses,
        }
    }

    try:
        resp = es.search(
            index=INDEX,
            body={
                "size":    top_k,
                "query":   es_query,
                "_source": SOURCE_FIELDS,
            },
        )
        hits = resp["hits"]["hits"]
    except Exception as exc:
        logger.error(f"Contacts search failed: {exc}")
        return []

    if not hits:
        return []

    # When a specific name was extracted, return only the top hit.
    if exact_name:
        return hits[:1]

    # Score-gate: tighter (80%) for office queries, looser (60%) for fuzzy.
    gate = 0.80 if entity_type == "office" else 0.60
    threshold = hits[0]["_score"] * gate
    return [h for h in hits if h["_score"] >= threshold]
