"""
router/router.py
LLM-based intent router — maps a user query to one or more domains using GPT-4o.

Replaces the previous prototype cosine-similarity approach (e5-large-v2 embeddings)
which was fragile when query vocabulary did not match prototype vocabulary.

Algorithm:
  1. Send query to GPT-4o with a zero-shot domain classification prompt.
  2. Parse the returned JSON to get the list of domains.
  3. Return at most 2 domains; multi-domain only when both are genuinely needed.
"""

import json
import logging
from typing import Dict, List

from app.common.llm_client import call_gpt

logger = logging.getLogger(__name__)

DOMAIN_CALENDAR  = "CALENDAR"
DOMAIN_CONTACTS  = "CONTACTS"
DOMAIN_DOCUMENTS = "DOCUMENTS"
DOMAIN_TUITION   = "TUITION"

_VALID_DOMAINS = {DOMAIN_CALENDAR, DOMAIN_CONTACTS, DOMAIN_DOCUMENTS, DOMAIN_TUITION}

_ROUTER_SYSTEM = """You are a query router for an IIT (Illinois Institute of Technology) student assistant chatbot.

Classify the student's question into one or more of these domains:

DOCUMENTS — Academic policies, rules, and procedures:
  • Academic policies: GPA requirements, academic probation, dismissal, grading legend
  • Transcripts: requesting official/unofficial transcripts, Parchment, transcript holds
  • Grades: grade appeals, incomplete grades, NA grade, grade changes, grade types
  • Registration procedures: registration holds, add/drop policies, enrollment verification
  • Course actions: withdrawing from a course (W grade), course repeats, auditing
  • Hardship withdrawal: medical or financial leave, withdrawing from university
  • Co-terminal program: dual degree, accelerated master's policies
  • Student conduct: academic integrity, plagiarism, FERPA, Title IX, code of conduct
  • Housing: residence requirements, on-campus housing policy
  • Health insurance: SHIP, student health insurance waiver, international student insurance
  • Graduation: degree conferral requirements, commencement policies
  • Hawkcard: student ID card rules, lost card replacement
  • Financial aid policies: satisfactory academic progress, co-terminal aid
  • Tuition refund rules: what happens to tuition if you drop or withdraw
  • Billing policy: does tuition change if enrollment status changes, drop below full-time
  • Refund schedules: percentage refunded by week of withdrawal

TUITION — Fee amounts and billing rates (numbers, dollar amounts):
  • Tuition rates by school, program, or enrollment status
  • Mandatory fees: activity fee, student service fee, health fee, U-Pass, CTA pass
  • Per-credit, per-semester, or per-course fee amounts
  • Program-specific fees: LLM, MBA, MDES, MDM, MPA tuition
  • Billing periods, payment plans, tuition by academic year
  • School-specific costs: Chicago-Kent, Institute of Design, Stuart, Mies, IEP
  NOT tuition: refund rules, billing impact of dropping a course, what happens
  to tuition when you withdraw — those are DOCUMENTS (policy) questions.

CALENDAR — Academic calendar dates and deadlines:
  • Semester start and end dates: fall, spring, summer
  • Registration open/close dates, add/drop deadlines
  • Final exam dates and schedules
  • Grade submission deadlines
  • Holidays and breaks: Thanksgiving, spring break, winter break, MLK Day
  • Degree conferral and commencement dates
  • Course schedule publication dates

CONTACTS — People, offices, phone numbers, emails:
  • Department or office contact information
  • Faculty or staff phone numbers / emails
  • Who to contact for a specific issue
  • Office locations and hours
  • Process questions where the answer is "contact this office":
    - Taking a leave of absence / semester off / medical leave → Academic Affairs
    - Hold on account / account balance issues → Student Accounting
    - Advisor approval missing / registration PIN → Registrar or Academic Affairs
    - Transcript request issue → Registrar
    - Financial aid problem → Financial Aid office
    - Disability accommodations → Center for Disability Resources

Output ONLY a JSON object with one key "domains" whose value is a list of 0, 1, or 2 domain strings.
Rules:
  - Include a domain only if the question clearly requires information from that domain.
  - Include 2 domains only when the question genuinely spans both (e.g. "what is the tuition deadline for fall registration?" spans TUITION + CALENDAR).
  - Return {"domains": []} if the question is completely unrelated to IIT academic information.
  - Never return more than 2 domains.
  - Use exact domain names: DOCUMENTS, TUITION, CALENDAR, CONTACTS.

Output ONLY valid JSON. No explanation."""


def get_routing_intent(query: str) -> Dict[str, List[str]]:
    """
    Return the domain(s) that best match the query using GPT-4o classification.

    Returns:
        {"domains": [...], "needs_clarification": False, "sub_queries": {...}}
        domains is [] when the question is not about IIT information.
    """
    if not query or not query.strip():
        return {"domains": []}

    messages = [
        {"role": "system", "content": _ROUTER_SYSTEM},
        {"role": "user",   "content": query},
    ]

    try:
        raw = call_gpt(messages, max_tokens=60, temperature=0.0)

        # Parse JSON — strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw)
        domains = [d for d in result.get("domains", []) if d in _VALID_DOMAINS]
        domains = domains[:2]  # hard cap at 2

        return {
            "domains":             domains,
            "needs_clarification": False,
            "sub_queries":         {d: query for d in domains},
        }

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(f"Router JSON parse failed ({exc}) — raw: {raw!r}")
        return {"domains": [], "needs_clarification": False, "sub_queries": {}}

    except Exception as exc:
        logger.error(f"LLM routing failed: {exc}")
        return {"domains": [], "needs_clarification": False, "sub_queries": {}}
