import logging

import torch
from sentence_transformers import SentenceTransformer

from backend.core.config import SENTENCE_TRANSFORMER_MODEL

logger = logging.getLogger("ats_resume_scorer")

# Reduce CPU + memory usage on Render
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

_embedder = None


def get_embedder():
    """
    Lazily loads SentenceTransformer only when needed.
    This prevents Render from downloading/loading the model during startup.
    """
    global _embedder

    if _embedder is None:
        logger.info("Loading SentenceTransformer model...")

        _embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device="cpu",
        )

        # Reduce RAM usage
        _embedder.max_seq_length = 256

        logger.info("SentenceTransformer model loaded successfully.")

    return _embedder