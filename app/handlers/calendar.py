"""
handlers/calendar.py
CALENDAR domain handler.

Pipeline:
  1. extract_slots()       — regex slot extraction (no LLM)
  2. term_from_history()   — inherit term from previous turn if missing
  3. needs_clarification() — fire only when event keyword but no time context
  4. calendar_search()     — BM25 + term filter + month filter + reranker
  5. generate_answer()     — LLM synthesises a direct answer from retrieved events

State persisted across turns:
  intent_query  — original query stored when clarification is triggered
  clarify_count — number of clarification turns taken
"""

import logging

from app.domains.calendar.pipeline import (
    extract_slots,
    term_from_history,
    needs_clarification,
    generate_answer,
)
from app.domains.calendar.search import calendar_search

try:
    from app.common.clarification_options import options_cache
    _OPTIONS_AVAILABLE = True
except Exception:
    _OPTIONS_AVAILABLE = False
    options_cache      = None

logger = logging.getLogger(__name__)


def _calendar_options() -> list:
    if _OPTIONS_AVAILABLE and options_cache:
        return options_cache.calendar_terms
    return []


def _source_footnote(hits: list) -> str:
    seen: list = []
    for h in hits:
        for url in (h["_source"].get("source_urls") or []):
            if url and url not in seen:
                seen.append(url)
    if not seen:
        return ""
    return "\n\nSource: " + seen[0] if len(seen) == 1 else (
        "\n\nSources:\n" + "\n".join(f"• {u}" for u in seen[:2])
    )


def _resolve_slots(query: str, history: list, state: dict) -> tuple[dict, str, dict]:
    """
    Resolve slots for the current turn, handling clarification follow-ups
    and term carryover from history.

    Returns: (slots, search_query, debug)
    """
    debug: dict = {}

    if state["intent_query"] and state["clarify_count"] > 0:
        intent_slots   = extract_slots(state["intent_query"])
        followup_slots = extract_slots(query)
        slots = {**intent_slots, **followup_slots}
        if not followup_slots.get("has_event"):
            slots["has_event"]  = intent_slots.get("has_event",  False)
            slots["is_holiday"] = intent_slots.get("is_holiday", False)
        search_query           = f"{state['intent_query']} {query}"
        state["intent_query"]  = ""
        state["clarify_count"] = 0
    else:
        slots        = extract_slots(query)
        search_query = query

        if (not slots.get("term")
                and not slots.get("terms")
                and not slots.get("date_str")
                and not slots.get("month")
                and slots.get("has_event")
                and not slots.get("is_holiday")):
            inherited = term_from_history(history)
            if inherited:
                slots["term"] = inherited
                debug["term_from_history"] = inherited

    debug["slots"] = {k: v for k, v in slots.items() if v}
    return slots, search_query, debug


def retrieve(query: str, history: list, state: dict = None) -> tuple:
    """
    Run retrieval only — no answer generation.
    Used by the multi-domain cross-reranking path in chat.py.

    Returns:
        hits (list), debug (dict), clarify_reply (str | None), updated state (dict)
        clarify_reply is None when retrieval succeeded, or a clarification string.
    """
    if state is None:
        state = {}
    state.setdefault("intent_query",  "")
    state.setdefault("clarify_count", 0)

    slots, search_query, debug = _resolve_slots(query, history, state)

    if needs_clarification(slots):
        options = _calendar_options()
        reply   = "Which semester or year are you referring to?"
        if options:
            reply += f" ({', '.join(options[:6])})"
        debug["clarification"]  = True
        state["intent_query"]   = query
        state["clarify_count"] += 1
        return [], debug, reply, state

    hits = calendar_search(search_query, slots)
    debug["events"]       = len(hits)
    debug["search_query"] = search_query
    return hits, debug, None, state


def handle(query: str, history: list, state: dict = None, stream: bool = False) -> tuple:
    """
    Run the full CALENDAR pipeline for one turn.

    Returns:
        reply (str | Generator), debug (dict), is_clarification (bool), updated state (dict)
    """
    hits, debug, clarify_reply, state = retrieve(query, history, state)

    if clarify_reply is not None:
        return clarify_reply, debug, True, state

    search_query = debug.get("search_query", query)
    reply = generate_answer(search_query, hits, history, stream=stream)

    # Append source URLs if reply is a plain string (not streaming)
    if not stream and isinstance(reply, str):
        reply += _source_footnote(hits)

    return reply, debug, False, state
