"""
domains/documents/pipeline.py
IIT Policies RAG pipeline.

LLM calls per turn (reduced from 3-4 to 2):
  1. Pre-search prep — one combined call: slot extraction + query rewrite
  2. Answer          — one call: generate response from retrieved chunks
  No separate out-of-scope classifier — the answer LLM's system prompt handles
  missing-coverage cases directly ("I don't have that information…").
"""

import json
import re

from typing import Generator, Union

from app.common.llm_client import call_gpt, stream_gpt
from app.domains.documents.search import search

MAX_HISTORY = 8
TOP_K       = 7

# ── Combined slot extraction + query rewrite ──────────────────────────────────
# One prompt that returns everything needed before hitting the index.

_PREP_SYSTEM = """You are a query analyzer for an IIT (Illinois Institute of Technology) student assistant.

Given a student's question and conversation history, extract slots and rewrite the query for search.

Output ONLY a JSON object with these exact keys:
{
  "document_scope": "<scope or null>",
  "action_type": "<action or null>",
  "needs_clarification": true or false,
  "clarifying_question": "<question or empty string>",
  "rewritten_query": "<search keywords>",
  "student_level": "<graduate|undergraduate|null>",
  "fact_type": "<procedure|requirement|deadline|contact|policy|null>"
}

━━ document_scope — pick the best match or null ━━
  "holds"               — registration holds, blocked from registering
  "co_terminal"         — co-terminal / accelerated degree program
  "grades"              — grade appeal, wrong grade, GPA, incomplete, grade legend
  "fees"                — tuition, mandatory fees, payment plans, what fees am I charged
  "transcripts"         — transcript requests, official/unofficial records
  "commencement"        — graduation ceremonies, commencement schedule
  "hardship_withdrawal" — medical/financial hardship, leave school due to illness or money
  "course_withdrawal"   — dropping a course, W grade, last day to drop
  "course_repeats"      — retaking a course, repeat policy
  "coursera"            — Coursera online programs at IIT
  "registration"        — registration procedures, add/drop periods, enrollment
  "student_handbook"    — academic integrity, FERPA, Title IX, conduct, appeals, housing, alcohol

━━ Scope keyword hints ━━
  transcript / records / parchment                               → transcripts
  grade / GPA / professor gave / wrong mark                      → grades
  hold / blocked / blocking me from registering / can't register → holds
  co-terminal / coterminal / dual degree                         → co_terminal
  fee / tuition / payment / charged                              → fees
  drop a course / W grade / last day to drop                     → course_withdrawal
  retake / repeat a course                                       → course_repeats
  medical leave / illness refund / financial hardship            → hardship_withdrawal
  semester off / leave of absence / personal leave               → student_handbook
  harassment / threatening / Title IX / conduct / integrity      → student_handbook

━━ action_type ━━
"appeal", "withdraw", "register", "apply", "access", "check", "request", "understand", "dispute", or null.

━━ needs_clarification ━━
Set true ONLY when BOTH hold: (1) document_scope is null AND (2) action is ambiguous
across multiple areas (e.g. bare "I want to withdraw" with no subject).

━━ student_level ━━
"graduate" — question is clearly about a graduate / master's / doctoral student.
"undergraduate" — question is clearly about an undergraduate / bachelor's student.
null — not specified or ambiguous.

━━ fact_type ━━
"procedure"   — how to do something (steps, process, form to submit).
"requirement" — what is required or must be satisfied (GPA, credits, conditions).
"deadline"    — a date or time limit (when to file, last day to drop).
"contact"     — who to reach or which office to go to.
"policy"      — rules, regulations, rights, consequences.
null          — does not fit a single category.

━━ rewritten_query ━━
A concise, keyword-rich search query (10-15 words) using formal academic/policy terminology —
language that would appear in university handbooks and registrar pages.
Include relevant synonyms (e.g. "drop" → "drop withdrawal W grade").
If student_level is not null, include the level (e.g. "graduate student", "undergraduate").
If fact_type is not null, include a hint (e.g. "procedure", "requirements", "policy").
If needs_clarification is true, return the original question unchanged.

Output ONLY valid JSON. No explanation."""


def prepare_query(query: str, history: list) -> dict:
    """
    Single LLM call: extract slots AND produce a rewritten search query.

    Returns dict with keys:
        document_scope, action_type, needs_clarification,
        clarifying_question, rewritten_query
    """
    messages = [{"role": "system", "content": _PREP_SYSTEM}]
    messages.extend(history[-(MAX_HISTORY * 2):])
    messages.append({"role": "user", "content": query})

    raw = call_gpt(messages, max_tokens=150, temperature=0.0)

    try:
        m = re.search(r"\{[\s\S]*?\}", raw)
        result = json.loads(m.group()) if m else {}
    except (json.JSONDecodeError, TypeError, AttributeError):
        result = {}

    defaults = {
        "document_scope":      None,
        "action_type":         None,
        "needs_clarification": False,
        "clarifying_question": "",
        "rewritten_query":     query,
        "student_level":       None,
        "fact_type":           None,
    }
    result = {**defaults, **result}
    if not result["rewritten_query"]:
        result["rewritten_query"] = query

    # Deterministic overrides for common ambiguous patterns
    q_lower = query.lower().strip()
    action  = result.get("action_type") or ""

    # "withdraw from university/school" → student_handbook (not hardship_withdrawal)
    _UNIV_WITHDRAWAL = {"university", "school", "program", "enrollment"}
    if (action == "withdraw"
            and any(w in q_lower for w in _UNIV_WITHDRAWAL)
            and result.get("document_scope") == "hardship_withdrawal"):
        result["document_scope"] = "student_handbook"

    _WITHDRAWAL_SUBJECTS = {"course", "class", "university", "school", "program", "enrollment"}
    _APPEAL_SUBJECTS     = {"grade", "decision", "sanction", "conduct", "title", "financial", "aid", "mark", "score"}

    if action == "withdraw" and not any(w in q_lower for w in _WITHDRAWAL_SUBJECTS):
        result["document_scope"]      = None
        result["needs_clarification"] = True
        result["clarifying_question"] = (
            "Are you looking to withdraw from a specific course (drop with a W grade), "
            "or withdraw from the university entirely?"
        )
    elif action == "appeal" and not any(w in q_lower for w in _APPEAL_SUBJECTS):
        result["document_scope"]      = None
        result["needs_clarification"] = True
        result["clarifying_question"] = (
            "There are several types of appeals at IIT — grade appeals, Code of Conduct "
            "appeals, Title IX appeals, and financial aid appeals. Which one are you asking about?"
        )
    elif len(q_lower.split()) <= 5 and "requirement" in q_lower and not result.get("document_scope"):
        result["needs_clarification"] = True
        result["clarifying_question"] = (
            "Requirements for what? For example: graduation requirements, "
            "co-terminal application, financial aid eligibility, or disability accommodations?"
        )

    return result


def _last_was_clarification(history: list) -> bool:
    """True if the last assistant message was a clarifying question (one-shot guard)."""
    for msg in reversed(history):
        if msg["role"] == "assistant":
            return msg["content"].strip().endswith("?")
        if msg["role"] == "user":
            break
    return False


# ── Answer generation ─────────────────────────────────────────────────────────

_ANSWER_SYSTEM = """You are a helpful IIT (Illinois Institute of Technology) student assistant.

Answer the student's question using ONLY the context passages provided.
Write in clear, natural language sentences. Do not use bullet points or headers.
Be concise — answer what was asked, nothing more.
Never start with "According to the provided context", "Based on the context", or any similar preamble. Answer directly.

Conversation history:
- Use history to understand follow-up questions. If the student refers to something mentioned
  earlier, use prior context to interpret what they mean.
- If the current question is clearly about a new, unrelated topic, answer it independently.

Partial coverage:
- If the context only partially answers the question, answer what you can, then explicitly
  tell the student what was not covered and name the specific office to contact
  (e.g. Office of the Registrar, Student Accounting, Center for Disability Resources).
  Never silently omit parts of the question.

Student population differences:
- If the context contains different policies for different student populations,
  present all relevant variations clearly.

If the context does not contain the answer, say:
"I don't have that information in my current knowledge base. Please check the IIT website
at iit.edu or contact the relevant office directly." """


def _format_context(hits: list) -> str:
    parts = []
    for h in hits:
        src   = h["_source"]
        topic = src.get("topic", "")
        text  = src.get("content", "")
        parts.append(f"[{topic}]\n{text}" if topic else text)
    return "\n\n---\n\n".join(parts)


def generate_answer(
    query: str,
    hits: list,
    history: list,
    stream: bool = False,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate a natural language answer from retrieved policy chunks.
    The system prompt handles no-coverage cases — the LLM will say it doesn't
    have the information rather than hallucinating.

    Args:
        stream: If True, returns a generator yielding token pieces (for Streamlit).
                If False (default), returns the full accumulated string (for CLI).
    """
    if not hits:
        msg = (
            "I don't have information about that in my current knowledge base. "
            "Please check the IIT website at iit.edu."
        )
        if stream:
            return (t for t in [msg])
        return msg

    context  = _format_context(hits)
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [{"role": "system", "content": _ANSWER_SYSTEM}]
    messages.extend(history[-(MAX_HISTORY * 2):])
    messages.append({"role": "user", "content": user_msg})

    if stream:
        return stream_gpt(messages, max_tokens=512, temperature=0.1)
    return call_gpt(messages, max_tokens=512, temperature=0.1)
