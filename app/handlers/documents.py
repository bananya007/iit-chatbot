"""
handlers/documents.py
DOCUMENTS domain handler.

Pipeline:
  1. prepare_query() — one LLM call: slot extraction + query rewrite (combined)
  2. Clarification guard
  3. Hybrid search (BM25 + kNN + RRF + cross-encoder rerank)
  4. generate_answer() — one LLM call
     (no-coverage cases handled by the answer LLM's system prompt directly)
  5. Source URL footnotes appended deterministically (no LLM)
"""

from typing import Generator, Union

from app.common.embedding_model import model_large
from app.domains.documents.pipeline import (
    prepare_query,
    generate_answer,
    _last_was_clarification,
    TOP_K,
)
from app.domains.documents.search import search


def _sources_footnote(hits: list) -> str:
    """Build a footnote string from the top-3 unique source URLs in the hit list."""
    seen: list = []
    for h in hits:
        url = h["_source"].get("source_url", "")
        if url and url not in seen:
            seen.append(url)
        if len(seen) == 3:
            break
    if not seen:
        return ""
    return "\n\nSources:\n" + "\n".join(f"• {url}" for url in seen)


def _stream_with_sources(gen: Generator, footnote: str) -> Generator[str, None, None]:
    """Wrap an answer generator to append source URL footnotes after the last token."""
    yield from gen
    if footnote:
        yield footnote


def retrieve(query: str, history: list) -> tuple:
    """
    Run retrieval only — no answer generation.
    Used by the multi-domain cross-reranking path in chat.py.

    Returns:
        hits (list), debug (dict), clarify_reply (str | None)
        clarify_reply is None when retrieval succeeded, or a clarification string.
    """
    debug: dict = {}

    prep           = prepare_query(query, history)
    document_scope = prep.get("document_scope")
    rewritten      = prep.get("rewritten_query") or query

    debug["scope"]   = document_scope
    debug["action"]  = prep.get("action_type")
    debug["rewrite"] = rewritten

    if prep.get("needs_clarification") and not _last_was_clarification(history):
        clarify = prep.get("clarifying_question") or "Could you clarify what you mean?"
        debug["clarification"] = True
        return [], debug, clarify

    vec  = model_large.encode(f"query: {rewritten}", normalize_embeddings=True).tolist()
    hits = search(rewritten, vec, top_k=TOP_K, document_scope=document_scope)
    debug["chunks"] = len(hits)
    return hits, debug, None


def handle(query: str, history: list, stream: bool = False) -> tuple:
    """
    Run the DOCUMENTS pipeline for one turn.

    Args:
        stream: If True, the reply is a generator (for Streamlit st.write_stream).
                If False (default), the reply is a buffered string (for CLI).

    Returns:
        reply (str | Generator), debug (dict), is_clarification (bool)
    """
    debug: dict = {}

    # Step 1: Combined slot extraction + query rewrite (1 LLM call)
    prep           = prepare_query(query, history)
    document_scope = prep.get("document_scope")
    action         = prep.get("action_type")
    rewritten      = prep.get("rewritten_query") or query

    debug["scope"]   = document_scope
    debug["action"]  = action
    debug["rewrite"] = rewritten

    # Step 2: Clarification guard — always returns a string, no streaming needed
    if prep.get("needs_clarification") and not _last_was_clarification(history):
        reply = prep.get("clarifying_question") or "Could you clarify what you mean?"
        debug["clarification"] = True
        return reply, debug, True

    # Step 3: Encode query + hybrid search (BM25 + kNN + RRF + rerank)
    vec  = model_large.encode(f"query: {rewritten}", normalize_embeddings=True).tolist()
    hits = search(rewritten, vec, top_k=TOP_K, document_scope=document_scope)

    debug["chunks"] = len(hits)

    # Step 4: Answer generation (1 LLM call) + source URL footnotes
    footnote = _sources_footnote(hits)

    if stream:
        answer_gen = generate_answer(query, hits, history, stream=True)
        reply = _stream_with_sources(answer_gen, footnote)
    else:
        reply = generate_answer(query, hits, history, stream=False)
        reply += footnote

    return reply, debug, False
