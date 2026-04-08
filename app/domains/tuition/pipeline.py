"""
domains/tuition/pipeline.py
IIT Tuition structured lookup pipeline.

LLM calls per turn (reduced from 1-2 to 1):
  - _classify_confirmation: replaced with regex (was 1 LLM call on fuzzy-match turns)
  - generate_answer:        1 LLM call (kept — formats exact fee records into prose)
  - extract_known:          0 LLM calls (regex-based, unchanged)
"""

import re
from typing import Generator, Union

from app.common.llm_client import call_gpt, stream_gpt

MAX_HISTORY = 8
MAX_CLARIFY = 2  # max clarifying questions before forcing an answer

# ── Field extraction (regex, no LLM) ──────────────────────────────────────────

_SCHOOL_MAP = [
    ("institute of design",    "Institute of Design"),
    ("intensive english",      "Intensive English Program"),
    ("chicago-kent",           "Chicago-Kent"),
    ("chicago kent",           "Chicago-Kent"),
    ("stuart",                 "Stuart School of Business"),
    ("kent",                   "Chicago-Kent"),
    ("mies",                   "Mies"),
    ("iep",                    "Intensive English Program"),
]

_PROG_NORM = {
    "ll.m.": "LLM", "llm": "LLM", "phd": "PHD",
    "mdes": "MDES", "mdess": "MDES", "mdm": "MDM",
    "mba": "MBA", "mpa": "MPA",
}

# Fix 8: program → school inference
_PROG_TO_SCHOOL = {
    "MDES": "Institute of Design",
    "MDM":  "Institute of Design",
    "LLM":  "Chicago-Kent",
    "MBA":  "Stuart School of Business",
    "MPA":  "Stuart School of Business",
}


def extract_known(query: str, fv: dict) -> dict:
    """
    Regex-based extraction of filter fields from a single user utterance.
    Zero LLM calls — deterministic, instant.
    """
    text  = query.lower()
    known: dict = {}

    for key, val in _SCHOOL_MAP:
        if key in text:
            known["school"] = val
            break

    if re.search(r"\bundergrad\w*|\bbachelor\w*", text):
        known["level"] = "undergrad"
    elif re.search(r"\bgraduate\b|\bgrad\b|\bmaster\w*|\bphd\b|\bdoctoral?\b", text):
        known["level"] = "graduate"

    if re.search(r"\bfull[\s-]?time\b", text):
        known["enrollment"] = "full_time"
    elif re.search(r"\bpart[\s-]?time\b", text):
        known["enrollment"] = "part_time"

    m = re.search(r"(\d{4})[-–](\d{2,4})", query)
    if m:
        y1, y2 = m.group(1), m.group(2)
        if len(y2) == 2:
            y2 = y1[:2] + y2
        known["academic_year"] = f"{y1}-{y2}"

    pm = re.search(r"\b(MDES[S]?|MDM|PHD|LLM|LL\.M\.|MBA|MPA)\b", query, re.IGNORECASE)
    if pm:
        prog = _PROG_NORM.get(pm.group().lower(), pm.group().upper())
        known["program"] = prog
        # Fix 8: infer school from program when not explicitly stated
        if "school" not in known and prog in _PROG_TO_SCHOOL:
            known["school"] = _PROG_TO_SCHOOL[prog]

    if re.search(r"\bfall\b", text):
        known["term"] = "Fall"
    elif re.search(r"\bspring\b", text):
        known["term"] = "Spring"
    elif re.search(r"\bsummer\b", text):
        known["term"] = "Summer"

    if re.search(r"\bper[\s-]credit\b|\bper[\s-]credit[\s-]hour\b|/credit\b", text):
        known["unit"] = "per_credit"
    elif re.search(r"\bper[\s-]course\b|/course\b", text):
        known["unit"] = "per_course"
    elif re.search(r"\bper[\s-]semester\b|/semester\b", text):
        known["unit"] = "per_semester"

    for fee in sorted(fv["fee_names"], key=len, reverse=True):
        if fee.lower() in text:
            known["fee_name"] = fee
            break

    return known


# ── Clarification guard ───────────────────────────────────────────────────────

def _needs_clarification(
    known: dict, hits: list, fv: dict, history: list
) -> tuple:
    """
    Three-case slot-filling clarification (school → fee → level).
    Returns (question_str | None, pending_suggestion_dict).
    """
    last_assistant = ""
    for msg in reversed(history):
        if msg["role"] == "assistant":
            last_assistant = msg["content"].lower()
            break

    school = known.get("school")
    fee    = known.get("fee_name")

    # Case 1: School unknown
    if not school:
        if "which iit school" not in last_assistant and "which school" not in last_assistant:
            school_list = ", ".join(fv["schools"])
            fee_ctx = f" for the {fee}" if fee else ""
            return f"Which IIT school are you asking about{fee_ctx}? ({school_list})", {}

    # Case 2: School known but zero results
    if school and not hits:
        already_asked = (
            "which fee" in last_assistant
            or "did you mean" in last_assistant
            or "couldn't find" in last_assistant
            or "double-check" in last_assistant
        )
        if not already_asked:
            fees_at_school = fv["school_fees"].get(school, fv["fee_names"])
            if fee:
                stop     = {"fee", "the", "a", "an", "for", "of", "at"}
                keywords = {w for w in fee.lower().split() if w not in stop}
                matches  = [f for f in fees_at_school if any(kw in f.lower() for kw in keywords)]
                if len(matches) == 1:
                    return f"Did you mean the {matches[0]}?", {"fee_name": matches[0]}
                elif 2 <= len(matches) <= 3:
                    return f"Which fee did you mean? ({', '.join(matches)})", {}
                else:
                    return (
                        f"I couldn't find a '{fee}' at {school}. Could you double-check the fee name?",
                        {},
                    )
            else:
                return f"Which fee are you looking for at {school}? ({', '.join(fees_at_school)})", {}

    # Case 3: Level unknown — ask before searching, don't depend on hits
    # Mies and some other schools have both undergrad and grad records.
    # With TOP_K=5, only one level may appear in hits even when both exist.
    if school and not known.get("level"):
        if "graduate or undergraduate" not in last_assistant and \
           "graduate or undergrad" not in last_assistant:
            return "Is this for a graduate or undergraduate student?", {}

    return None, {}


# ── Confirmation classifier (regex, replaces LLM call) ───────────────────────

_CONFIRM_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "correct", "sure", "ok", "okay",
    "right", "exactly", "that's it", "that's right", "sounds good",
    "that one", "perfect",
})
_DENY_WORDS = frozenset({
    "no", "nope", "nah", "wrong", "not that", "different", "other",
    "something else", "not right", "incorrect",
})


def _classify_confirmation(user_reply: str) -> str:
    """
    Classify the user's reply to a fuzzy-match suggestion as confirm / deny / new.
    Regex-based — eliminates the LLM call previously used for this simple task.
    """
    text = user_reply.lower().strip()
    if any(w in text for w in _CONFIRM_WORDS):
        return "confirm"
    if any(w in text for w in _DENY_WORDS):
        return "deny"
    return "new"


# ── Answer generation ─────────────────────────────────────────────────────────

_ANSWER_PROMPT = """You are an IIT tuition and fees assistant.

Answer using ONLY the fee records provided below. Rules:
- Use the exact dollar amounts — never estimate, round, or invent figures.
- Always state each amount with its billing unit (per credit, per semester, per course, etc.).
- Multiple records may be present — only include rates that directly answer the question.
- Write in natural prose sentences, not lists or metadata labels.
- If the user asked about a specific term, only mention that term's rate.
- If records differ by enrollment status (full-time/part-time), state both amounts.
- Keep the answer concise — 2-3 sentences maximum.
- If the student asks where to find, view, or access tuition information, direct them to iit.edu/student-accounting.
- If no dollar amounts are in the records, say: "The exact amount is not available. Please check iit.edu/student-accounting."
- Do not use markdown. Plain text only."""


def _format_records(hits: list) -> str:
    rows = []
    for h in hits:
        src = h["_source"]
        if src.get("amount_value") is None:
            continue
        amount     = f"${src['amount_value']:,.0f}"
        unit       = (src.get("unit") or "").replace("_", " ")
        amount_str = f"{amount} {unit}".strip()
        parts = []
        for field, label in [
            ("school", "school"), ("level", "level"), ("fee_name", "fee"),
            ("enrollment", "enrollment"), ("term", "term"),
            ("academic_year", "year"), ("program", "program"),
        ]:
            val = src.get(field)
            if not val:
                continue
            val = "all students" if (field == "level" and val == "all") else val.replace("_", " ")
            parts.append(f"{label}: {val}")
        parts.append(f"amount: {amount_str}")
        rows.append("- " + " | ".join(parts))
    return "\n".join(rows)


def generate_answer(
    query: str,
    hits: list,
    history: list,
    year_fallback: bool = False,
    requested_year: str = "",
    stream: bool = False,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a natural language response from structured fee records.

    Args:
        stream: If True, returns a generator yielding token pieces (for Streamlit).
                If False (default), returns the full accumulated string (for CLI).
    """
    if not hits:
        msg = (
            "I couldn't find a matching fee record for that query. "
            "Please check the IIT website at iit.edu."
        )
        if stream:
            return (t for t in [msg])
        return msg

    context = _format_records(hits)

    year_note = ""
    if year_fallback and requested_year:
        years_in_hits = {h["_source"].get("academic_year") for h in hits if h["_source"].get("academic_year")}
        available     = next(iter(years_in_hits), "the most recent year available")
        year_note = (
            f"NOTE TO ASSISTANT: The user asked about {requested_year} but that year is not "
            f"yet in the database. The records below are the most recent available ({available}). "
            f"You MUST still provide the dollar amounts. End your answer with: "
            f"'Note: {requested_year} data is not yet available; these figures are from {available}.'\n\n"
        )

    user_msg = f"{year_note}Fee records:\n{context}\n\nQuestion: {query}"
    messages = [{"role": "system", "content": _ANSWER_PROMPT}]
    messages.extend(history[-(MAX_HISTORY * 2):])
    messages.append({"role": "user", "content": user_msg})

    if stream:
        return stream_gpt(messages, max_tokens=512, temperature=0.1)
    return call_gpt(messages, max_tokens=512, temperature=0.1)
