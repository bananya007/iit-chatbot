"""
common/search_utils.py
Shared retrieval utilities: query cleaning and Reciprocal Rank Fusion.
"""

import re


def clean_query(query: str) -> str:
    """Strip punctuation and normalise whitespace for lexical search."""
    text = query.strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rrf_fuse(lexical_hits: list, semantic_hits: list, rrf_k: int = 60) -> list:
    """
    Reciprocal Rank Fusion of two hit lists.
    score(d) = Σ 1 / (rank + 1 + rrf_k)
    Returns hits sorted by descending RRF score, deduped by _id.
    """
    scores: dict = {}
    docs:   dict = {}

    for rank, hit in enumerate(lexical_hits):
        _id = hit["_id"]
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rank + 1 + rrf_k)
        docs[_id]   = hit

    for rank, hit in enumerate(semantic_hits):
        _id = hit["_id"]
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rank + 1 + rrf_k)
        docs[_id]   = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[_id] for _id, _ in ranked]
