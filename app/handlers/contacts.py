"""
handlers/contacts.py
CONTACTS domain handler.

Pipeline:
  1. extract_slots()    — GPT extracts entity_type + name (one call)
  2. search_contacts()  — filtered ES query using slots
  3. _format_records()  — structured template render (no LLM)
"""

import logging

from app.domains.contacts.pipeline import extract_slots
from app.domains.contacts.search import search_contacts

logger = logging.getLogger(__name__)


def _format_records(hits: list) -> str:
    """Format contact records as a readable response. No LLM call."""
    if not hits:
        return (
            "I couldn't find contact information for that department or person. "
            "Please check the IIT directory at iit.edu/directory."
        )

    parts = []
    for h in hits:
        s     = h["_source"]
        lines = [s.get("name") or s.get("department", "Unknown")]

        if s.get("department") and s.get("department") != s.get("name"):
            lines.append(f"  Department: {s['department']}")
        if s.get("phone"):
            lines.append(f"  Phone: {s['phone']}")
        if s.get("fax"):
            lines.append(f"  Fax: {s['fax']}")
        if s.get("email"):
            lines.append(f"  Email: {s['email']}")
        if s.get("building"):
            addr = s["building"]
            if s.get("address"):
                addr += f", {s['address']}"
            lines.append(f"  Location: {addr}")
        if s.get("description"):
            desc = s["description"]
            lines.append(f"  About: {desc[:120]}…" if len(desc) > 120 else f"  About: {desc}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _run(query: str) -> tuple:
    """Shared slot extraction + search — called by both retrieve() and handle()."""
    slots      = extract_slots(query)
    exact_name = bool(slots["name"])
    hits       = search_contacts(
        query,
        entity_type=slots["entity_type"],
        name=slots["name"],
        exact_name=exact_name,
    )
    debug = {"slots": slots, "matched": len(hits)}
    if hits:
        debug["top_match"] = hits[0]["_source"].get("name", "")
    return hits, debug


def retrieve(query: str, history: list) -> tuple:
    """
    Run retrieval only — returns raw hits for cross-domain reranking.

    Returns:
        hits (list), debug (dict), clarify_reply (None — contacts never clarify)
    """
    hits, debug = _run(query)
    return hits, debug, None


def handle(query: str, history: list) -> tuple:
    """
    Run the CONTACTS pipeline for one turn.

    Returns:
        reply (str), debug (dict), is_clarification (bool)
    """
    hits, debug = _run(query)
    reply = _format_records(hits)
    return reply, debug, False
