import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import (
    ALLOWED_ORIGINS,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    SPACY_MODEL_PRIMARY,
    SPACY_MODEL_SECONDARY,
)
from backend.api.routes import router

logger = logging.getLogger("ats_resume_scorer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import spacy

    logger.info("=== Startup begins ===")

    try:
        logger.info(f"Loading spaCy model: {SPACY_MODEL_PRIMARY}")

        app.state.nlp = spacy.load(
            SPACY_MODEL_PRIMARY,
            exclude=[
                "parser",
                "lemmatizer",
                "attribute_ruler",
                "tok2vec",
            ],
        )

        logger.info("spaCy loaded successfully.")

    except Exception as e:
        logger.exception(f"Failed to load spaCy: {e}")

        logger.info(f"Trying fallback model: {SPACY_MODEL_SECONDARY}")

        app.state.nlp = spacy.load(
            SPACY_MODEL_SECONDARY,
            exclude=[
                "parser",
                "lemmatizer",
                "attribute_ruler",
                "tok2vec",
            ],
        )

        logger.info("Fallback spaCy loaded successfully.")

    # Lazy loading placeholder
    app.state.embedder = None

    logger.info("Embedder placeholder created.")
    logger.info("=== Startup complete ===")

    yield

    logger.info("Shutting down ATS Resume Analyzer API...")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
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

    # Local development only
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )