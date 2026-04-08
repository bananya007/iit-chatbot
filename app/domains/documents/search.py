"""
domains/documents/search.py
Hybrid BM25 + kNN + RRF + cross-encoder reranking for the iit_policies index.
"""

from app.common.es_client import es
from app.common.reranker import rerank_chunks

INDEX  = "iit_policies"
SOURCE = ["chunk_id", "doc_name", "source_url", "topic", "content"]

# document_scope → source_url substring used to boost BM25 retrieval
SCOPE_URL_MAP = {
    "holds":               "hold-information",
    "co_terminal":         "co-terminal",
    "grades":              "grade",
    "fees":                "mandatory-and-other-fees",
    "transcripts":         "transcripts",
    "commencement":        "commencement",
    "hardship_withdrawal": "hardship-withdrawal",
    "course_withdrawal":   "withdrawing",
    "course_repeats":      "course-repeat",
    "coursera":            "coursera",
    "registration":        "registration",
    "student_handbook":    "Student%20Handbook",
}

# Note: embedding-based out-of-scope detection is not used.
# e5-large-v2 scores are compressed (off-topic queries still score 0.67+ against
# a narrow policy corpus), so no fixed threshold reliably separates relevant from
# irrelevant. The answer LLM's system prompt handles no-coverage cases directly.



def _bm25(query: str, k: int, document_scope: str = None) -> list:
    text_query = {
        "multi_match": {
            "query":  query,
            "fields": ["topic^2", "content"],
            "type":   "best_fields",
        }
    }

    if document_scope and document_scope in SCOPE_URL_MAP:
        url_pattern = SCOPE_URL_MAP[document_scope]
        query_body = {
            "bool": {
                "must":   text_query,
                "should": [{"wildcard": {"source_url": {"value": f"*{url_pattern}*", "boost": 3.0}}}],
            }
        }
    else:
        query_body = text_query

    resp = es.search(
        index=INDEX,
        body={"size": k, "query": query_body, "_source": SOURCE},
    )
    return resp["hits"]["hits"]


def _knn(vec: list, k: int) -> list:
    resp = es.search(
        index=INDEX,
        body={
            "size": k,
            "knn": {
                "field":          "content_vector",
                "query_vector":   vec,
                "num_candidates": k * 5,
                "k":              k,
            },
            "_source": SOURCE,
        },
    )
    return resp["hits"]["hits"]


def _rrf(bm25_hits: list, knn_hits: list, top_k: int, rrf_k: int = 60) -> list:
    scores: dict = {}
    docs:   dict = {}

    for rank, hit in enumerate(bm25_hits):
        _id = hit["_id"]
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rank + 1 + rrf_k)
        docs[_id]   = hit

    for rank, hit in enumerate(knn_hits):
        _id = hit["_id"]
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rank + 1 + rrf_k)
        docs[_id]   = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [docs[_id] for _id, _ in ranked]


def search(
    query: str,
    vec: list,
    top_k: int = 7,
    document_scope: str = None,
) -> list:
    """
    Hybrid BM25 + kNN + RRF, followed by cross-encoder reranking.

    Over-fetches (top_k * 2) from each retriever to give the reranker a wider
    candidate pool, then reranks down to the final top_k.

    Args:
        query:          rewritten search query
        vec:            pre-computed e5-large query embedding (normalised)
        top_k:          number of results to return after reranking
        document_scope: optional scope key for BM25 URL boost

    Returns:
        list of ES hit dicts, reranked by cross-encoder relevance.
    """
    bm25_hits = _bm25(query, k=top_k * 2, document_scope=document_scope)
    knn_hits  = _knn(vec,   k=top_k * 2)
    rrf_hits  = _rrf(bm25_hits, knn_hits, top_k=top_k * 2)
    return rerank_chunks(query, rrf_hits, top_k=top_k)
