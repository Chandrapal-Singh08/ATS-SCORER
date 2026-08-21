import logging
import os

import torch
from sentence_transformers import SentenceTransformer

from backend.core.config import SENTENCE_TRANSFORMER_MODEL

logger = logging.getLogger("ats_resume_scorer")

# Reduce CPU usage on Render Free tier
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_embedder = None


def get_embedder():
    global _embedder

    if _embedder is not None:
        return _embedder

    logger.info("Loading SentenceTransformer model...")

    try:
        _embedder = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device="cpu",
        )

        _embedder.max_seq_length = 256

        logger.info("SentenceTransformer loaded successfully.")
        return _embedder

    except Exception as exc:
        logger.exception(f"Failed to load SentenceTransformer: {exc}")
        raise