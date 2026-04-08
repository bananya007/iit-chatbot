"""
domains/calendar/pipeline.py
Calendar slot extraction, clarification logic, and term carryover from history.

Slots extracted from query (all regex, zero LLM calls):
  term           — semester label e.g. "Fall 2026", "Spring 2026"
  date_str       — specific date e.g. "2026-03-16"
  month          — month number (1-12) for month-only queries
  year           — year for month filter (defaults to 2026)
  is_holiday     — True when query mentions a named holiday
  has_event      — True when query contains an academic event keyword
  include_coursera — True when query explicitly mentions Coursera

Clarification fires only when:
  has_event=True AND no time context (term/date/month) AND not is_holiday
"""

import re
from typing import Generator, Union

from app.common.llm_client import call_gpt, stream_gpt

MONTHS = {
    "january": 1, "february": 2, "march": 3,  "april": 4,
    "may":     5, "june":     6, "july":   7,  "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_RE_TERM = re.compile(
    r"\b(spring|fall|summer)\s*(20\d{2})?\b", re.IGNORECASE
)
_RE_DATE_FULL = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2})\s*[,]?\s*(20\d{2})\b",
    re.IGNORECASE,
)
_RE_DATE_MONTH_DAY = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2})\b",
    re.IGNORECASE,
)
_RE_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_RE_EVENT = re.compile(
    r"\b(exam|exams|final|finals|midterm|midterms|break|graduation|commencement|"
    r"drop|withdraw|withdrawal|deadline|start|end|begin|close|class|classes|"
    r"register|registration|enroll|recess|grades?|schedule|orientation|incomplete)\b",
    re.IGNORECASE,
)
_RE_HOLIDAY = re.compile(
    r"\b(thanksgiving|christmas|labor\s+day|memorial\s+day|juneteenth|"
    r"independence\s+day|martin\s+luther\s+king|mlk|new\s+year|"
    r"spring\s+break|fall\s+break|winter\s+break|floating\s+holiday|"
    r"university\s+holiday)\b",
    re.IGNORECASE,
)


def extract_slots(query: str) -> dict:
    """
    Extract all calendar slots from a single query string.
    Zero LLM calls — deterministic regex only.
    """
    slots: dict = {}
    q = query.strip()

    slots["include_coursera"] = bool(re.search(r"\bcoursera\b", q, re.IGNORECASE))
    slots["is_holiday"]       = bool(_RE_HOLIDAY.search(q))
    slots["has_event"]        = bool(_RE_EVENT.search(q))

    # Term: "fall 2026", "spring", "summer 2026" etc.
    # Collect ALL term mentions; multi-term queries use OR filtering in search
    all_terms = [
        f"{s.title()} {y}".strip()
        for s, y in _RE_TERM.findall(q)
    ]
    if len(all_terms) == 1:
        slots["term"] = all_terms[0]
    elif len(all_terms) > 1:
        slots["terms"] = all_terms   # multi-term: handled by _build_query OR filter

    # Date — most specific match wins
    m = _RE_DATE_FULL.search(q)
    if m:
        month = MONTHS[m.group(1).lower()]
        day, year = int(m.group(2)), int(m.group(3))
        slots["date_str"] = f"{year}-{month:02d}-{day:02d}"
    else:
        m = _RE_DATE_MONTH_DAY.search(q)
        if m:
            month = MONTHS[m.group(1).lower()]
            day   = int(m.group(2))
            slots["date_str"] = f"2026-{month:02d}-{day:02d}"
        else:
            m = _RE_MONTH.search(q)
            if m:
                slots["month"] = MONTHS[m.group(1).lower()]
                slots["year"]  = 2026

    return slots


def term_from_history(history: list) -> str | None:
    """
    Scan the last 4 messages (2 turns) in history for a term mention.
    Used to carry over semester context so students don't re-specify it.
    Returns the term string or None.
    """
    for msg in reversed(history[-4:]):
        m = _RE_TERM.search(msg.get("content", ""))
        if m:
            season = m.group(1).title()
            year   = m.group(2) or ""
            return f"{season} {year}".strip()
    return None


_CALENDAR_SYSTEM = (
    "You are an IIT student assistant with access to the academic calendar. "
    "Answer the student's question directly using ONLY the calendar events provided. "
    "If the question mentions a specific date, use it to reason explicitly — for example, "
    "whether that date is before or after a deadline, or whether a break overlaps with it. "
    "Do not just list all events. Give a direct, concise answer. "
    "If no events are relevant, say you don't have that information and direct them to "
    "iit.edu/registrar/academic-calendar."
)

_NO_INFO = (
    "I don't have that calendar information. "
    "Please check the IIT academic calendar at iit.edu/registrar/academic-calendar."
)


def generate_answer(
    query: str, hits: list, history: list, stream: bool = False
) -> Union[str, Generator]:
    """Generate a direct natural-language answer from retrieved calendar events."""
    if not hits:
        return _NO_INFO

    events_ctx = []
    for h in hits:
        src   = h["_source"]
        name  = src.get("event_name", "")
        term  = src.get("term", "")
        start = src.get("start_date", "")
        end   = src.get("end_date", "")
        date_part = f"{start} to {end}" if (end and end != start) else start
        events_ctx.append(f"- {name} ({term}): {date_part}")

    context  = "\n".join(events_ctx)
    hist_ctx = history[-6:] if history else []

    messages = [
        {"role": "system", "content": _CALENDAR_SYSTEM},
        *hist_ctx,
        {"role": "user", "content": f"Calendar events:\n{context}\n\nQuestion: {query}"},
    ]

    if stream:
        return stream_gpt(messages, max_tokens=256, temperature=0.1)
    return call_gpt(messages, max_tokens=256, temperature=0.1)


def needs_clarification(slots: dict) -> bool:
    """
    Clarify only when the query has an event keyword but no time context.
    Holidays and date queries never need clarification.
    """
    has_time = slots.get("term") or slots.get("terms") or slots.get("date_str") or slots.get("month")
    return (
        slots.get("has_event")
        and not has_time
        and not slots.get("is_holiday")
    )
