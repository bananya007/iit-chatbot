#!/usr/bin/env python3
"""
scripts/reingest_policies.py
Replace iit_policies index with Unstructured chunks k.json (599 chunks).

Steps:
  1. Delete existing iit_policies index
  2. Recreate with same mapping + analyzer
  3. Embed all 599 chunks with intfloat/e5-large-v2
  4. Bulk ingest

Run from repo root:
    python scripts/reingest_policies.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

ES_URL  = os.getenv("ES_URL",  "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.getenv("ES_PASS", "")

es = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_PASS) if ES_USER else None,
    verify_certs=False,
    ssl_show_warn=False,
)

IN_FILE = Path("data/processed/Unstructured data/Unstructured chunks k.json")
INDEX   = "iit_policies"
MODEL   = "intfloat/e5-large-v2"
BATCH   = 32

INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "simple_text": {
                    "type":      "custom",
                    "tokenizer": "standard",
                    "filter":    ["lowercase", "asciifolding"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "chunk_id":       {"type": "keyword"},
            "doc_name":       {"type": "text", "analyzer": "simple_text"},
            "source_url":     {"type": "keyword"},
            "topic":          {"type": "text", "analyzer": "simple_text",
                               "fields": {"keyword": {"type": "keyword"}}},
            "content":        {"type": "text", "analyzer": "simple_text"},
            "num_tokens":     {"type": "integer"},
            "content_vector": {"type": "dense_vector", "dims": 1024,
                               "index": True, "similarity": "cosine"},
        }
    },
}


def load_chunks(path: Path) -> list:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def embed(chunks: list, model) -> list:
    texts = [f"passage: {c['content']}" for c in chunks]
    vecs  = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        embs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vecs.extend(embs.astype(float).tolist())
        print(f"[EMBED] {min(i + BATCH, len(texts))}/{len(texts)}")
    for chunk, vec in zip(chunks, vecs):
        chunk["content_vector"] = vec
    return chunks


def bulk_ingest(chunks: list) -> None:
    ops = []
    for c in chunks:
        doc = {
            "chunk_id":       c.get("chunk_id"),
            "doc_name":       c.get("doc_name"),
            "source_url":     c.get("source_url"),
            "topic":          c.get("Topic") or c.get("topic"),
            "content":        c.get("content"),
            "num_tokens":     c.get("num_tokens"),
            "content_vector": c["content_vector"],
        }
        ops.append({"index": {"_index": INDEX, "_id": c["chunk_id"]}})
        ops.append(doc)

    resp = es.bulk(operations=ops, request_timeout=120)
    if resp.get("errors"):
        failed = [i for i in resp["items"] if i.get("index", {}).get("status", 0) >= 300]
        print(f"[WARN] {len(failed)} errors")
        for item in failed[:3]:
            print(f"       {item}")

    ok = sum(1 for i in resp["items"] if i.get("index", {}).get("status") in (200, 201))
    print(f"[INGEST] Indexed {ok}/{len(chunks)} chunks into '{INDEX}'")


def main():
    if not IN_FILE.exists():
        sys.exit(f"[ERROR] File not found: {IN_FILE}")

    chunks = load_chunks(IN_FILE)
    print(f"[LOAD]  {len(chunks)} chunks from {IN_FILE}")

    # Step 1: Delete existing index
    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
        print(f"[INDEX] Deleted existing '{INDEX}'")

    # Step 2: Recreate index
    es.indices.create(index=INDEX, body=INDEX_SETTINGS)
    print(f"[INDEX] Created '{INDEX}' with mapping")

    # Step 3: Embed
    print(f"[EMBED] Loading model: {MODEL}")
    model = SentenceTransformer(MODEL)
    chunks = embed(chunks, model)

    # Step 4: Ingest
    bulk_ingest(chunks)

    # Verify
    es.indices.refresh(index=INDEX)
    count = es.count(index=INDEX)["count"]
    print(f"[CHECK] iit_policies total docs: {count}")


if __name__ == "__main__":
    main()
