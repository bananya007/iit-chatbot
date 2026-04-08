"""
domains/tuition/search.py
Pure ES filter search for the tuition_fees_v2 index.

No BM25, no kNN — every fee record has structured keyword fields and a float
amount_value, making filter queries exact and zero-hallucination.
"""

from app.common.es_client import es

INDEX  = "tuition_fees_v2"
TOP_K  = 5

SOURCE_FIELDS = [
    "school", "level", "fee_name", "enrollment", "unit",
    "amount_value", "academic_year", "term", "program", "chunk_text",
]


def load_filter_values() -> dict:
    """
    Aggregate distinct keyword values from the index.
    Called once at startup; builds school_fees mapping for Case 2 clarification.
    """
    resp = es.search(
        index=INDEX,
        body={
            "size": 0,
            "aggs": {
                "schools":   {"terms": {"field": "school",        "size": 50}},
                "fee_names": {"terms": {"field": "fee_name",      "size": 50}},
                "programs":  {"terms": {"field": "program",       "size": 20}},
                "years":     {"terms": {"field": "academic_year", "size": 10}},
                "by_school": {
                    "terms": {"field": "school", "size": 20},
                    "aggs":  {"fees": {"terms": {"field": "fee_name", "size": 50}}},
                },
            },
        },
    )
    aggs = resp["aggregations"]

    school_fees = {
        bucket["key"]: sorted(b["key"] for b in bucket["fees"]["buckets"] if b["key"])
        for bucket in aggs["by_school"]["buckets"]
    }

    return {
        "schools":     [b["key"] for b in aggs["schools"]["buckets"]],
        "fee_names":   [b["key"] for b in aggs["fee_names"]["buckets"] if b["key"]],
        "programs":    [b["key"] for b in aggs["programs"]["buckets"] if b["key"]],
        "years":       [b["key"] for b in aggs["years"]["buckets"]    if b["key"]],
        "school_fees": school_fees,
    }


def _filter_query(known: dict, top_k: int) -> list:
    """
    Build and run a pure ES filter query.
    Returns [] immediately if no filterable fields are present.

    Fields with high index coverage are hard-filtered:
      school (100%), level (100%), fee_name (65%), academic_year (90%).
    Sparse fields (enrollment, program) use softer logic.
    """
    filters = []

    if known.get("school"):
        filters.append({"term": {"school": known["school"]}})
    if known.get("level"):
        filters.append({"term": {"level": known["level"]}})
    if known.get("fee_name"):
        filters.append({"term": {"fee_name": known["fee_name"]}})
    if known.get("academic_year"):
        filters.append({"term": {"academic_year": known["academic_year"]}})
    if known.get("enrollment"):
        # Soft filter: match requested enrollment OR records with no enrollment set
        filters.append({"bool": {"should": [
            {"term": {"enrollment": known["enrollment"]}},
            {"bool": {"must_not": {"exists": {"field": "enrollment"}}}},
        ], "minimum_should_match": 1}})
    if known.get("term"):
        # Soft filter: match requested term OR records with no term set
        filters.append({"bool": {"should": [
            {"term": {"term": known["term"]}},
            {"bool": {"must_not": {"exists": {"field": "term"}}}},
        ], "minimum_should_match": 1}})
    if known.get("unit"):
        filters.append({"term": {"unit": known["unit"]}})

    if not filters:
        return []

    resp = es.search(
        index=INDEX,
        body={
            "size":     top_k,
            "query":    {"bool": {"filter": filters}},
            "_source":  SOURCE_FIELDS,
        },
    )
    return resp["hits"]["hits"]


def structured_search(known: dict, top_k: int = TOP_K) -> dict:
    """
    Primary entry point. Wraps _filter_query with automatic year fallback.

    If the requested academic_year isn't in the index, retries without it
    and flags the caller so it can inform the user.

    Returns:
        {"hits": [...], "year_fallback": bool, "requested_year": str}
    """
    hits = _filter_query(known, top_k)

    # Fallback 1: drop unit filter if it returned no results.
    # Some schools (e.g. Mies) don't use per_credit billing — the regex may
    # extract unit=per_credit from "per credit hour" even when that unit
    # doesn't exist in the index for that school.
    if not hits and known.get("unit"):
        known_no_unit = {k: v for k, v in known.items() if k != "unit"}
        hits = _filter_query(known_no_unit, top_k)

    # Fallback 2: drop academic_year if still no hits.
    if not hits and known.get("academic_year"):
        known_no_year = {k: v for k, v in known.items() if k != "academic_year"}
        fallback = _filter_query(known_no_year, top_k)
        if fallback:
            return {
                "hits":           fallback,
                "year_fallback":  True,
                "requested_year": known["academic_year"],
            }

    return {"hits": hits, "year_fallback": False, "requested_year": ""}
