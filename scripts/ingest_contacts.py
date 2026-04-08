#!/usr/bin/env python3
"""
scripts/ingest_contacts.py
Create (or replace) the iit_contacts index and bulk-ingest the contacts CSV.

Source: data/raw/Contacts data.csv
Target: ES index 'iit_contacts'

Safe to re-run — deletes and recreates the index each time (idempotent).

Run from the repo root:
    python scripts/ingest_contacts.py
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import requests
from es_client import ES_URL, ES_USER, ES_PASS, VERIFY

CSV_FILE = Path("data/raw/Contacts data.csv")
INDEX    = "iit_contacts"

MAPPING = {
    "mappings": {
        "properties": {
            "name":        {"type": "text",    "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
            "department":  {"type": "text",    "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
            "category":    {"type": "keyword"},
            "description": {"type": "text",    "analyzer": "standard"},
            "phone":       {"type": "keyword"},
            "fax":         {"type": "keyword"},
            "email":       {"type": "keyword"},
            "building":    {"type": "text"},
            "address":     {"type": "text"},
            "city":        {"type": "keyword"},
            "state":       {"type": "keyword"},
            "zip":         {"type": "keyword"},
            "source_url":  {"type": "keyword"},
        }
    },
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
    },
}

# CSV header → index field (lowercase)
_FIELD_MAP = {
    "Name":       "name",
    "Department": "department",
    "Category":   "category",
    "Description":"description",
    "Phone":      "phone",
    "Fax":        "fax",
    "Email":      "email",
    "Building":   "building",
    "Address":    "address",
    "City":       "city",
    "State":      "state",
    "Zip":        "zip",
    "Source_url": "source_url",
}


def _auth():
    return (ES_USER, ES_PASS) if ES_USER else None


def recreate_index():
    """Drop (if exists) and create the index with the correct mapping."""
    # Delete if present
    r = requests.delete(
        f"{ES_URL}/{INDEX}",
        auth=_auth(), verify=VERIFY, timeout=30,
    )
    if r.status_code not in (200, 404):
        print(f"[WARN] DELETE returned {r.status_code}: {r.text[:200]}")

    # Create
    r = requests.put(
        f"{ES_URL}/{INDEX}",
        json=MAPPING,
        auth=_auth(), verify=VERIFY, timeout=30,
    )
    r.raise_for_status()
    print(f"[INDEX] Created '{INDEX}'")


def load_csv(path: Path) -> list:
    """Read CSV and normalise field names. Skip rows with empty Name."""
    docs = []
    with path.open(encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            doc = {}
            for csv_col, es_field in _FIELD_MAP.items():
                val = row.get(csv_col, "").strip()
                if val:
                    doc[es_field] = val
            if not doc.get("name"):
                continue
            # Stable doc_id from name so re-runs are idempotent
            doc_id = hashlib.md5(doc["name"].encode()).hexdigest()
            docs.append((doc_id, doc))
    return docs


def bulk_ingest(docs: list) -> None:
    """Bulk-index in batches of 200."""
    BATCH = 200
    total_ok = 0

    for start in range(0, len(docs), BATCH):
        batch = docs[start : start + BATCH]
        lines = []
        for doc_id, doc in batch:
            lines.append(json.dumps({"index": {"_index": INDEX, "_id": doc_id}}))
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
            print(f"[WARN] Batch {start}-{start+BATCH}: {len(failed)} errors")
            for item in failed[:3]:
                print(f"       {item}")

        ok = sum(1 for i in resp["items"] if i.get("index", {}).get("status") in (200, 201))
        total_ok += ok

    print(f"[INGEST] Indexed {total_ok}/{len(docs)} contacts into '{INDEX}'")


def main():
    if not CSV_FILE.exists():
        sys.exit(f"[ERROR] File not found: {CSV_FILE}")

    docs = load_csv(CSV_FILE)
    print(f"[CSV]   Loaded {len(docs)} records from {CSV_FILE}")

    recreate_index()
    bulk_ingest(docs)

    # Quick sanity check
    r = requests.get(
        f"{ES_URL}/{INDEX}/_count",
        auth=_auth(), verify=VERIFY, timeout=10,
    )
    if r.ok:
        print(f"[CHECK] Index count: {r.json().get('count')}")


if __name__ == "__main__":
    main()
