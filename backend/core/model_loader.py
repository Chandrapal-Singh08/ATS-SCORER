from sentence_transformers import SentenceTransformer

from backend.core.config import SENTENCE_TRANSFORMER_MODEL
from sentence_transformers import SentenceTransformer
from backend.core.config import SENTENCE_TRANSFORMER_MODEL

_embedder = None


def get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device="cpu",
        )

    return _embedder

_embedder = None


def get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    return _embedder
