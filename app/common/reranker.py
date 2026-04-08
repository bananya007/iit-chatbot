"""
common/reranker.py
Cross-encoder reranker shared across calendar and documents pipelines.
Streamlit-aware: uses @st.cache_resource when running under Streamlit.
"""

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

try:
    import streamlit as st
except ImportError:
    st = None

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if st is not None:
    @st.cache_resource
    def _load():
        return CrossEncoder(RERANKER_NAME)
else:
    def _load():
        return CrossEncoder(RERANKER_NAME)


reranker = _load()


def rerank_chunks(query: str, hits: list, top_k: int = 3) -> list:
    """Re-rank retrieved chunks using the cross-encoder and return top_k."""
    if not hits:
        return hits

    hits = hits[:20]

    pairs = []
    valid_hits = []
    for h in hits:
        content = h["_source"].get("content") or h["_source"].get("semantic_text")
        if not content:
            continue
        pairs.append((query, content))
        valid_hits.append(h)

    if not valid_hits:
        return []

    if len(valid_hits) <= top_k:
        return valid_hits

    scores = reranker.predict(pairs)
    for hit, score in zip(valid_hits, scores):
        hit["_rerank_score"] = float(score)

    return sorted(valid_hits, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)[:top_k]
