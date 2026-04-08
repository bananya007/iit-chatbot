"""
common/es_client.py
Single Elasticsearch client shared across all domain pipelines.
"""

import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

ES_URL  = os.getenv("ES_URL",  "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.getenv("ES_PASS", "")

try:
    es = Elasticsearch(
        ES_URL,
        basic_auth=(ES_USER, ES_PASS) if ES_USER else None,
        verify_certs=False,
        ssl_show_warn=False,
    )
except Exception as exc:
    raise RuntimeError(f"Failed to connect to Elasticsearch: {exc}") from exc
