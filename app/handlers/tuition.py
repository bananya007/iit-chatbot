"""
handlers/tuition.py
TUITION domain handler.

Pipeline:
  1. extract_known()          — regex field extraction (no LLM)
  2. structured_search()      — pure ES filter query (no BM25/kNN)
  3. _needs_clarification()   — result-count-based clarification guard
  4. _classify_confirmation() — regex confirmation classifier (no LLM)
  5. generate_answer()        — 1 LLM call to format fee records

State is passed in/out by the orchestrator (chat.py) so no global mutable state.
"""

from app.domains.tuition.pipeline import (
    extract_known,
    _needs_clarification,
    _classify_confirmation,
    generate_answer,
    MAX_CLARIFY,
)
from app.domains.tuition.search import load_filter_values, structured_search

# Filter values loaded once at import time
_fv = load_filter_values()


_MULTI_DOMAIN_MAX_HITS = 8  # if unfiltered results exceed this, tuition steps aside


def retrieve(query: str, history: list, state: dict, multi_domain: bool = False) -> tuple:
    """
    Run retrieval only — no answer generation.
    Used by the multi-domain cross-reranking path in chat.py.

    Args:
        multi_domain: when True, skip all clarification logic and return [] if the
                      result set is too large (ambiguous query, no school specified).
                      Single-domain behaviour is unchanged when False.

    Returns:
        hits (list), debug (dict), clarify_reply (str | None), updated state (dict)
    """
    debug: dict = {}

    state.setdefault("accumulated_known",  {})
    state.setdefault("clarify_count",      0)
    state.setdefault("intent_query",       "")
    state.setdefault("pending_suggestion", {})
    state.setdefault("session_school",     "")  # Fix 7: persist school across turns

    confirmed_suggestion: dict = {}
    if state["pending_suggestion"]:
        classification = _classify_confirmation(query)
        if classification == "confirm":
            confirmed_suggestion        = state["pending_suggestion"]
            state["pending_suggestion"] = {}
        elif classification == "deny":
            state["pending_suggestion"] = {}
            state["clarify_count"]     += 1
            debug["clarification"]      = True
            return [], debug, "Could you clarify which fee you're looking for?", state
        else:
            state["pending_suggestion"] = {}
            state["clarify_count"]      = 0

    if state["clarify_count"] == 0:
        state["intent_query"]       = query
        state["accumulated_known"]  = {}
        state["pending_suggestion"] = {}

    new_known                  = extract_known(query, _fv)
    state["accumulated_known"] = {**state["accumulated_known"], **new_known}
    if confirmed_suggestion:
        state["accumulated_known"].update(confirmed_suggestion)
    known = state["accumulated_known"]

    # Fix 7: apply or update session-level school memory
    if known.get("school"):
        state["session_school"] = known["school"]
    elif state["session_school"] and not known.get("school"):
        known["school"] = state["session_school"]
        state["accumulated_known"]["school"] = state["session_school"]

    debug["filters"] = {k: v for k, v in known.items() if v}

    result   = structured_search(known)
    hits     = result["hits"]
    year_fb  = result["year_fallback"]

    debug["records"]       = len(hits)
    debug["year_fallback"] = year_fb

    # Multi-domain mode: skip clarification entirely.
    # If result set is too large (unfiltered/ambiguous), step aside and let
    # the other domain answer instead of asking for more slots.
    if multi_domain:
        if len(hits) > _MULTI_DOMAIN_MAX_HITS:
            debug["multi_domain_skip"] = True
            return [], debug, None, state
        state["clarify_count"] = 0
        return hits, debug, None, state

    if state["clarify_count"] < MAX_CLARIFY:
        clarify_q, suggestion = _needs_clarification(known, hits, _fv, history)
        if clarify_q:
            state["pending_suggestion"] = suggestion
            state["clarify_count"]     += 1
            debug["clarification"]      = True
            return [], debug, clarify_q, state

    state["clarify_count"] = 0
    return hits, debug, None, state


def handle(query: str, history: list, state: dict, stream: bool = False) -> tuple:
    """
    Run the TUITION pipeline for one turn.

    Args:
        query:   current user message
        history: conversation history
        state:   mutable dict persisted across turns by the orchestrator
                 keys: accumulated_known, clarify_count, intent_query, pending_suggestion
        stream:  If True, the answer reply is a generator (for Streamlit st.write_stream).
                 If False (default), returns a buffered string (for CLI).

    Returns:
        reply (str | Generator), debug (dict), is_clarification (bool), updated state (dict)
    """
    debug: dict = {}

    state.setdefault("accumulated_known",   {})
    state.setdefault("clarify_count",       0)
    state.setdefault("intent_query",        "")
    state.setdefault("pending_suggestion",  {})
    state.setdefault("session_school",      "")  # Fix 7: persist school across turns

    # ── Step 0: Resolve pending fuzzy-match suggestion ────────────────────────
    confirmed_suggestion: dict = {}
    if state["pending_suggestion"]:
        classification = _classify_confirmation(query)
        if classification == "confirm":
            confirmed_suggestion        = state["pending_suggestion"]
            state["pending_suggestion"] = {}
        elif classification == "deny":
            state["pending_suggestion"] = {}
            reply = "Could you clarify which fee you're looking for?"
            history += [{"role": "user", "content": query}, {"role": "assistant", "content": reply}]
            state["clarify_count"] += 1
            debug["clarification"] = True
            return reply, debug, True, state
        else:
            state["pending_suggestion"] = {}
            state["clarify_count"]      = 0

    # ── Step 1: Extract + merge fields ────────────────────────────────────────
    if state["clarify_count"] == 0:
        state["intent_query"]      = query
        state["accumulated_known"] = {}
        state["pending_suggestion"] = {}

    new_known                  = extract_known(query, _fv)
    state["accumulated_known"] = {**state["accumulated_known"], **new_known}
    if confirmed_suggestion:
        state["accumulated_known"].update(confirmed_suggestion)
    known = state["accumulated_known"]

    # Fix 7: apply or update session-level school memory
    if known.get("school"):
        state["session_school"] = known["school"]
    elif state["session_school"] and not known.get("school"):
        known["school"] = state["session_school"]
        state["accumulated_known"]["school"] = state["session_school"]

    debug["filters"] = {k: v for k, v in known.items() if v}

    # ── Step 2: Structured ES filter search ───────────────────────────────────
    result   = structured_search(known)
    hits     = result["hits"]
    year_fb  = result["year_fallback"]
    req_year = result["requested_year"]

    debug["records"]       = len(hits)
    debug["year_fallback"] = year_fb

    # ── Step 3: Clarification guard ───────────────────────────────────────────
    if state["clarify_count"] < MAX_CLARIFY:
        clarify_q, suggestion = _needs_clarification(known, hits, _fv, history)
        if clarify_q:
            state["pending_suggestion"] = suggestion
            state["clarify_count"]     += 1
            debug["clarification"]      = True
            return clarify_q, debug, True, state

    # ── Step 4: Generate answer ───────────────────────────────────────────────
    state["clarify_count"] = 0
    reply = generate_answer(
        state["intent_query"], hits, history,
        year_fallback=year_fb, requested_year=req_year,
        stream=stream,
    )
    return reply, debug, False, state
