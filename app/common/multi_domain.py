"""
common/multi_domain.py
Cross-domain reranking and unified answer generation.

When multiple domains are triggered:
  1. Each domain retrieves its top hits independently (preserving domain-specific
     filters, slot filling, and exact-match logic).
  2. All hits are pooled and re-ranked together by the cross-encoder against
     the user query — placing the most relevant evidence first regardless of domain.
  3. A single LLM call generates one coherent answer from the top-7 ranked passages.

Serialisation per domain (passage fed to cross-encoder and LLM):
  - DOCUMENTS : h["_source"]["content"]
  - TUITION   : h["_source"]["chunk_text"]  (pre-serialised at index time)
  - CALENDAR  : formatted from event_name / term / start_date / end_date
  - CONTACTS  : formatted from name / phone / email fields
"""

from typing import Generator, Union

from app.common.reranker import rerank_chunks
from app.common.llm_client import call_gpt, stream_gpt

TOP_K_PER_DOMAIN = 5   # hits fetched from each domain before pooling
TOP_K_FINAL      = 7   # passages kept after cross-domain reranking

_SYSTEM = (
    "You are an IIT student assistant. Using the retrieved information below, "
    "answer the student's question directly and concisely. "
    "Synthesise information from all sources into one coherent answer. "
    "Do not reference domain names, labels, or source types. "
    "If the question involves a specific date or deadline, reason about it explicitly."
)


def _hit_to_text(hit: dict) -> str:
    """Serialise a hit from any domain into a plain-text passage."""
    src = hit["_source"]

    # Calendar: use semantic_text (synonym-expanded, reads like a passage)
    # Falls back to structured fields if semantic_text is absent
    if "event_name" in src:
        if src.get("semantic_text"):
            return src["semantic_text"]
        name      = src.get("event_name", "")
        term      = src.get("term", "")
        start     = src.get("start_date", "")
        end       = src.get("end_date", "")
        date_part = f"{start} to {end}" if (end and end != start) else start
        return f"{name} ({term}): {date_part}"

    # Tuition: pre-serialised readable sentence stored at index time
    if "chunk_text" in src:
        return src["chunk_text"]

    # Contacts: structured directory record
    if "phone" in src or "email" in src:
        lines = [src.get("name") or src.get("department", "")]
        if src.get("phone"):
            lines.append(f"Phone: {src['phone']}")
        if src.get("email"):
            lines.append(f"Email: {src['email']}")
        return "\n".join(filter(None, lines))

    # Documents (and fallback)
    return src.get("content") or src.get("semantic_text") or ""


def _inject_text(hits: list) -> list:
    """
    Attach a '_passage' key to each hit so the reranker and context builder
    both use the same serialised text without re-computing it twice.
    """
    result = []
    for h in hits:
        h = dict(h)                          # shallow copy — don't mutate originals
        h["_passage"] = _hit_to_text(h)
        if h["_passage"]:
            result.append(h)
    return result


def _pool_and_rerank(query: str, hits_by_domain: dict) -> list:
    """
    Pool hits from all domains, inject passage text, cross-encoder rerank,
    return top TOP_K_FINAL hits.
    """
    pool: list = []
    for hits in hits_by_domain.values():
        pool.extend(_inject_text(hits[:TOP_K_PER_DOMAIN]))

    if not pool:
        return []

    # Temporarily store passage in _source["_x_passage"] so rerank_chunks can find it
    for h in pool:
        h["_source"]["_x_passage"] = h["_passage"]

    # rerank_chunks looks for "content" then "semantic_text" — override with our passage
    for h in pool:
        if not h["_source"].get("content") and not h["_source"].get("semantic_text"):
            h["_source"]["content"] = h["_passage"]

    reranked = rerank_chunks(query, pool, top_k=TOP_K_FINAL)
    return reranked


def cross_domain_answer(
    query: str,
    hits_by_domain: dict,
    history: list,
    stream: bool = False,
) -> Union[str, Generator]:
    """
    Pool hits from multiple domains, cross-encoder rerank, generate one answer.

    Args:
        query:          original user query
        hits_by_domain: {domain_name: [hit, ...]} from each triggered domain
        history:        conversation history for LLM context
        stream:         if True, return a token generator for Streamlit

    Returns:
        str (buffered) or Generator (streaming)
    """
    ranked = _pool_and_rerank(query, hits_by_domain)

    if not ranked:
        return (
            "I couldn't find relevant information for that question. "
            "Please check iit.edu for more details."
        )

    context_lines = []
    for h in ranked:
        passage = h.get("_passage") or _hit_to_text(h)
        if passage:
            context_lines.append(f"- {passage}")

    context  = "\n".join(context_lines)
    hist_ctx = history[-6:] if history else []

    messages = [
        {"role": "system", "content": _SYSTEM},
        *hist_ctx,
        {"role": "user", "content": f"Retrieved information:\n{context}\n\nQuestion: {query}"},
    ]

    if stream:
        return stream_gpt(messages, max_tokens=512, temperature=0.1)
    result = call_gpt(messages, max_tokens=512, temperature=0.1)
    if result.startswith("[GPT"):
        # GPT failed — fall back to concatenating passages
        return "\n".join(context_lines[:3])
    return result
