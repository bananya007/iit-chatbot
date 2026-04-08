"""
domains/calendar/search.py
Calendar search: BM25 on semantic_text + term filter + date range filter.

Replaces the previous hybrid lexical + semantic + RRF design.
With 118 structured documents and pre-built synonym expansion in semantic_text,
a single BM25 query outperforms the added complexity of vector search + RRF.

No vector search. No RRF. No Painless scripts.
"""

from app.common.es_client import es

INDEX  = "iit_calendar"
SOURCE = ["event_name", "term", "start_date", "end_date", "source_urls", "semantic_text"]

_MONTH_END = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}


def _build_query(query: str, slots: dict) -> dict:
    filters = []

    # Term filter — exclude Coursera terms unless explicitly requested
    if slots.get("terms"):
        # Multiple terms in query: OR filter so events from any mentioned term are included
        filters.append({"bool": {
            "should": [{"match_phrase": {"term": t}} for t in slots["terms"]],
            "minimum_should_match": 1,
        }})
    elif slots.get("term"):
        filters.append({"match_phrase": {"term": slots["term"]}})
    if not slots.get("include_coursera"):
        filters.append({
            "bool": {"must_not": {"match": {"term": "Coursera"}}}
        })

    # Month filter only — specific dates are passed to the LLM as context for reasoning,
    # not used as hard ES filters. This lets the LLM compare "April 10 vs March 7 deadline"
    # regardless of how the question is worded.
    if slots.get("month"):
        # Month-only: events that overlap with any day in the month
        year = slots.get("year", 2026)
        m    = slots["month"]
        last = _MONTH_END.get(m, 31)
        filters.append({"range": {"start_date": {"lte": f"{year}-{m:02d}-{last}"}}})
        filters.append({"range": {"end_date":   {"gte": f"{year}-{m:02d}-01"}}})

    return {
        "bool": {
            "filter": filters,
            "must":   [{"match": {"semantic_text": {"query": query, "operator": "or"}}}],
        }
    }


def calendar_search(query: str, slots: dict, top_k: int = 5) -> list:
    """
    BM25 on semantic_text with optional term/date filters.

    When a term filter is specified, fetches up to 100 events for that term
    so the LLM sees the full semester calendar rather than just the top-5
    BM25 matches (which may miss less-common events).

    Falls back without term filter if initial search returns no results.
    Deduplicates by event_name before returning.

    Returns list of ES hit dicts.
    """
    has_term = bool(slots.get("term") or slots.get("terms"))
    fetch_size = 30 if has_term else top_k * 3

    def _run(s: dict) -> list:
        try:
            resp = es.search(
                index=INDEX,
                body={
                    "size":    fetch_size,
                    "query":   _build_query(query, s),
                    "_source": SOURCE,
                },
            )
            return resp["hits"]["hits"]
        except Exception:
            return []

    hits = _run(slots)

    # Fallback: retry without term filter if no results
    if not hits and has_term:
        slots_no_term = {k: v for k, v in slots.items() if k not in ("term", "terms")}
        hits = _run(slots_no_term)

    # Deduplicate by event_name (same event can appear in multiple terms)
    seen:   set  = set()
    unique: list = []
    for h in hits:
        name = h["_source"].get("event_name", "")
        if name not in seen:
            seen.add(name)
            unique.append(h)

    if not unique:
        return []

    # For term queries return all unique events; otherwise honour top_k
    return unique if has_term else unique[:top_k]
