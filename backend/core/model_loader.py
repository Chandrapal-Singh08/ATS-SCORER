import logging

import torch
from sentence_transformers import SentenceTransformer

from backend.core.config import SENTENCE_TRANSFORMER_MODEL

logger = logging.getLogger("ats_resume_scorer")

# Limit CPU threads (helps Render Free memory/CPU usage)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Global singleton
_embedder = None


def get_embedder():
    """
    Lazily load the SentenceTransformer model.
    The model loads only on the first /analyze-resume request,
    not during FastAPI startup.
    """
    global _embedder

    if _embedder is None:
        logger.info("Loading SentenceTransformer model...")

        _embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device="cpu",
        )

        # Reduce memory usage
        _embedder.max_seq_length = 256

        logger.info("SentenceTransformer model loaded successfully.")

    return _embedder