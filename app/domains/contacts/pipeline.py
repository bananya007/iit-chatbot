"""
domains/contacts/pipeline.py
Slot extraction for the CONTACTS domain.

Mirrors the tuition pattern: one GPT call extracts structured slots,
which drive filtered ES queries instead of relying on score tuning.

Slots
-----
entity_type : "office" | "person" | null
    "office"  — user wants a department, unit, or administrative office
    "person"  — user wants a specific individual (faculty or staff)
    null      — cannot determine intent; fall back to fuzzy search

name : str | null
    The office name OR person name extracted from the query.
    Used as the ES search text instead of the raw user query,
    which may contain filler like "how do I contact…" or "what is the email for…".
"""

import json
import logging

from app.common.llm_client import call_gpt

logger = logging.getLogger(__name__)

_SLOT_SYSTEM = """\
You are a slot extractor for an IIT (Illinois Institute of Technology) contacts chatbot.

Extract two fields from the user query and return ONLY valid JSON:

{
  "entity_type": "office" | "person" | null,
  "name": "<extracted name or null>"
}

Rules:
- entity_type = "office"  when the query asks for a department, office, unit, or service
  (e.g. "registrar", "financial aid", "student accounting", "graduate admissions",
   "disability resources", "dean of students", "bursar", "IT helpdesk")
- entity_type = "person"  when the query asks for a specific individual by name or title
  (e.g. "Professor Smith", "who is Aaron Ruffin", "Dr. Jones contact")
- entity_type = null      when intent is ambiguous or unclear
- name = the office or person name stripped of filler words
  ("how do I contact financial aid" → name = "Financial Aid")
  ("what is the email for the registrar" → name = "Office of The Registrar")
  ("who is Professor Chen" → name = "Professor Chen")
  (query is just "contact information" with no specific target → name = null)
- Return null for name if no specific target is mentioned.
- Return ONLY the JSON object, no explanation.

Intent-to-office mapping — use these when the query describes a problem or process
rather than naming an office directly:
- "hold on account", "account hold", "balance due", "payment issue" → name = "Student Accounting"
- "advisor approval", "registration PIN", "missing PIN", "advisor signature" → name = "Academic Affairs"
- "transcript issue", "transcript problem", "transcript request" → name = "Office of The Registrar"
- "take a semester off", "leave of absence", "medical leave", "stop out" → name = "Academic Affairs"
- "financial aid problem", "aid disbursement", "scholarship issue" → name = "Financial Aid"
- "disability accommodation", "accessibility" → name = "Center For Disability Resources"
- "registration problem", "can't register", "registration error" → name = "Office of The Registrar"
- "housing issue", "residence hall" → name = "Office of Residence Life"
"""


def extract_slots(query: str) -> dict:
    """
    Call GPT to extract entity_type and name from the contacts query.

    Returns:
        dict with keys "entity_type" (str|None) and "name" (str|None).
        Falls back to empty slots on any error.
    """
    try:
        raw = call_gpt(
            messages=[
                {"role": "system", "content": _SLOT_SYSTEM},
                {"role": "user",   "content": query},
            ],
            max_tokens=60,
            temperature=0.0,
        )
        slots = json.loads(raw)
        return {
            "entity_type": slots.get("entity_type"),
            "name":        slots.get("name"),
        }
    except Exception as exc:
        logger.warning(f"Contacts slot extraction failed: {exc}")
        return {"entity_type": None, "name": None}
