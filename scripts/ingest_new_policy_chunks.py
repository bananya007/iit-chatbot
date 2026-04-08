#!/usr/bin/env python3
"""
scripts/ingest_new_policy_chunks.py
Embed and ingest new_policy_chunks.json into the existing iit_policies index.

Uses the same model as the existing policies index: intfloat/e5-large-v2 (1024 dims).
e5 models expect a "passage: " prefix on documents at index time.

Safe to re-run — uses chunk_id as _id, so existing docs are overwritten idempotently.

Run from the repo root:
    python scripts/ingest_new_policy_chunks.py
"""

import json
import sys
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer
from es_client import ES_URL, ES_USER, ES_PASS, VERIFY

IN_FILE = Path("data/processed/Unstructured data/new_policy_chunks.json")
INDEX   = "iit_policies"
MODEL   = "intfloat/e5-large-v2"
BATCH   = 32


def _auth():
    return (ES_USER, ES_PASS) if ES_USER else None


def load_chunks(path: Path) -> list:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def embed(chunks: list, model) -> list:
    """Add content_vector to each chunk. e5 needs 'passage: ' prefix."""
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
    lines = []
    for c in chunks:
        # Normalise field names to match iit_policies mapping
        doc = {
            "chunk_id":      c.get("chunk_id"),
            "doc_name":      c.get("doc_name"),
            "source_url":    c.get("source_url"),
            "topic":         c.get("Topic") or c.get("topic"),
            "content":       c.get("content"),
            "num_tokens":    c.get("num_tokens"),
            "content_vector": c["content_vector"],
        }
        lines.append(json.dumps({"index": {"_index": INDEX, "_id": c["chunk_id"]}}))
        lines.append(json.dumps(doc, ensure_ascii=False))
    body = "\n".join(lines) + "\n"

    r = requests.post(
        f"{ES_URL}/_bulk",
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
        auth=_auth(), verify=VERIFY, timeout=120,
    )
    r.raise_for_status()
    resp = r.json()

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

    print(f"[EMBED] Loading model: {MODEL}")
    model = SentenceTransformer(MODEL)

    chunks = embed(chunks, model)
    bulk_ingest(chunks)

    # Verify
    requests.post(f"{ES_URL}/{INDEX}/_refresh", auth=_auth(), verify=VERIFY)
    r = requests.get(f"{ES_URL}/{INDEX}/_count", auth=_auth(), verify=VERIFY)
    print(f"[CHECK] iit_policies total docs: {r.json().get('count')}")


if __name__ == "__main__":
    main()
