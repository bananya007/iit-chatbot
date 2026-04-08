"""
core/orchestrator.py
Shared turn handler — single source of routing dispatch logic.

Both chat.py (CLI) and streamlit_app.py (web) call handle_turn().
Neither entry point contains domain dispatch logic directly.

Fix C: eliminates duplicated routing code between the two entry points.
Fix F: when tuition clarify_count > 0, bypass the LLM router and route
       directly to TUITION so single-word clarification replies work.
Fix G: clear session_school when TUITION is not in the router's domains
       (entity-based slot expiry — industry standard pattern).
"""

from app.router.router import (
    get_routing_intent,
    DOMAIN_DOCUMENTS,
    DOMAIN_TUITION,
    DOMAIN_CALENDAR,
    DOMAIN_CONTACTS,
)
from app.handlers.documents import handle as handle_documents, retrieve as retrieve_documents
from app.handlers.tuition   import handle as handle_tuition,   retrieve as retrieve_tuition
from app.handlers.calendar  import handle as handle_calendar,  retrieve as retrieve_calendar
from app.handlers.contacts  import handle as handle_contacts,  retrieve as retrieve_contacts
from app.common.multi_domain import cross_domain_answer

_NO_DOMAIN_REPLY = (
    "I'm not sure which area of IIT information that relates to. "
    "I can help with academic policies, tuition & fees, the academic "
    "calendar, or department contact information."
)
_NO_HITS_REPLY = "I couldn't find an answer for that. Please check iit.edu."


def handle_turn(
    query: str,
    history: list,
    tuition_state: dict,
    calendar_state: dict,
    stream: bool = False,
) -> tuple:
    """
    Route a single user turn and return the reply.

    Fix F: if tuition is mid-clarification (clarify_count > 0), skip the
    LLM router and send the reply directly back to TUITION.

    Fix G: if TUITION is not in the router domains, clear session_school
    so it doesn't leak into unrelated topics.

    Args:
        query:          user's message
        history:        conversation history (list of {role, content})
        tuition_state:  mutable tuition slot state (modified in place)
        calendar_state: mutable calendar slot state (modified in place)
        stream:         if True, reply may be a generator (for Streamlit)

    Returns:
        (reply, domains, tuition_state, calendar_state, is_clarification)
        reply is str or Generator depending on stream flag.
    """

    # Fix F: bypass router when tuition is mid-clarification
    if tuition_state.get("clarify_count", 0) > 0:
        domains = [DOMAIN_TUITION]
    else:
        routing = get_routing_intent(query)
        domains = routing.get("domains", [])

        # Fix G: clear session_school when domain is no longer TUITION
        if DOMAIN_TUITION not in domains:
            tuition_state["session_school"] = ""

    if not domains:
        return _NO_DOMAIN_REPLY, domains, tuition_state, calendar_state, False

    if len(domains) == 1:
        domain = domains[0]
        if domain == DOMAIN_DOCUMENTS:
            reply, debug, is_clar = handle_documents(query, history, stream=stream)
        elif domain == DOMAIN_TUITION:
            reply, debug, is_clar, tuition_state = handle_tuition(
                query, history, tuition_state, stream=stream
            )
        elif domain == DOMAIN_CALENDAR:
            reply, debug, is_clar, calendar_state = handle_calendar(
                query, history, calendar_state, stream=stream
            )
        elif domain == DOMAIN_CONTACTS:
            reply, debug, is_clar = handle_contacts(query, history)
        else:
            reply, is_clar = _NO_HITS_REPLY, False

        return reply, domains, tuition_state, calendar_state, is_clar

    # Multi-domain: parallel retrieve → rerank → single answer
    # Clarification only fires when NO domain produced hits.
    hits_by_domain: dict = {}
    clarifications: list = []

    for domain in domains:
        if domain == DOMAIN_DOCUMENTS:
            hits, debug, clarify = retrieve_documents(query, history)
        elif domain == DOMAIN_TUITION:
            hits, debug, clarify, tuition_state = retrieve_tuition(
                query, history, tuition_state, multi_domain=True
            )
        elif domain == DOMAIN_CALENDAR:
            hits, debug, clarify, calendar_state = retrieve_calendar(
                query, history, calendar_state
            )
        elif domain == DOMAIN_CONTACTS:
            hits, debug, clarify = retrieve_contacts(query, history)
        else:
            hits, clarify = [], None

        if hits:
            hits_by_domain[domain] = hits
        elif clarify:
            clarifications.append(clarify)

    if hits_by_domain:
        reply = cross_domain_answer(query, hits_by_domain, history, stream=stream)
        return reply, domains, tuition_state, calendar_state, False
    elif clarifications:
        return clarifications[0], domains, tuition_state, calendar_state, True
    else:
        return _NO_HITS_REPLY, domains, tuition_state, calendar_state, False
