"""
common/embedding_model.py
e5-large-v2 sentence transformer, lazy-loaded on first use.

Lazy loading prevents the 1.3 GB model from being downloaded/loaded at
import time — which would OOM the 512 MB Render free tier on startup.
The model loads on the first call to get_model() and is cached thereafter.
"""

from sentence_transformers import SentenceTransformer

MODEL_LARGE_NAME = "intfloat/e5-large-v2"

_model = None


def get_model() -> SentenceTransformer:
    """Return the cached model, loading it on the first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_LARGE_NAME)
    return _model


# Backwards-compatible alias — callers that do `model_large.encode(...)` still work,
# but this now triggers a lazy load on first attribute access rather than at import.
class _LazyModel:
    """Proxy that loads the model on first method call."""
    def encode(self, *args, **kwargs):
        return get_model().encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_model(), name)


model_large = _LazyModel()
