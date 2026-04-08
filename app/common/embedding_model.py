"""
common/embedding_model.py
e5-large-v2 sentence transformer, fully lazy-loaded on first use.

Both the torch import AND the model weights are deferred until get_model()
is first called. This keeps startup memory usage near zero so the FastAPI
server can start on Render's 512 MB free tier.
"""

_model = None


def get_model():
    """Return the cached model, importing torch and loading weights on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/e5-large-v2")
    return _model


class _LazyModel:
    """Proxy that defers torch import + model load until first method call."""
    def encode(self, *args, **kwargs):
        return get_model().encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_model(), name)


model_large = _LazyModel()
