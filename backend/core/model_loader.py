import torch
from sentence_transformers import SentenceTransformer
from backend.core.config import SENTENCE_TRANSFORMER_MODEL

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

_embedder = None

def get_embedder():
    global _embedder

    if _embedder is None:
        _embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device="cpu"
        )
        _embedder.max_seq_length = 256

    return _embedder