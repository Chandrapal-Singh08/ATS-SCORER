import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.config import (
    ALLOWED_ORIGINS,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    SPACY_MODEL_PRIMARY,
    SPACY_MODEL_SECONDARY,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ats_resume_scorer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import spacy

    logger.info("========== ATS BACKEND STARTUP ==========")
    logger.info(f"Loading spaCy model: {SPACY_MODEL_PRIMARY}")

    try:
        app.state.nlp = spacy.load(
            SPACY_MODEL_PRIMARY,
            exclude=[
                "parser",
                "lemmatizer",
                "attribute_ruler",
                "tok2vec",
            ],
        )

    except OSError:
        logger.warning("Primary spaCy model not found. Loading fallback.")

        app.state.nlp = spacy.load(
            SPACY_MODEL_SECONDARY,
            exclude=[
                "parser",
                "lemmatizer",
                "attribute_ruler",
                "tok2vec",
            ],
        )

    # IMPORTANT: Do NOT load SentenceTransformer here.
    app.state.embedder = None

    logger.info("spaCy model loaded successfully.")
    logger.info("Startup completed successfully.")

    yield

    logger.info("========== ATS BACKEND SHUTDOWN ==========")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )