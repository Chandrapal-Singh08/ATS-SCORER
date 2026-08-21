import logging
import os
import psutil
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

    process = psutil.Process(os.getpid())

    def log_memory(stage):
        ram = process.memory_info().rss / 1024 / 1024
        logger.info(f"[MEMORY] {stage}: {ram:.1f} MB")

    logger.info("Starting ATS Resume Analyzer API...")
    log_memory("Startup")

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

        log_memory("After spaCy")

    except OSError:
        logger.warning(
            f"{SPACY_MODEL_PRIMARY} not found — falling back to {SPACY_MODEL_SECONDARY}"
        )

        app.state.nlp = spacy.load(
            SPACY_MODEL_SECONDARY,
            exclude=[
                "parser",
                "lemmatizer",
                "attribute_ruler",
                "tok2vec",
            ],
        )

        log_memory("After fallback spaCy")

    app.state.embedder = None
    log_memory("After embedder placeholder")

    logger.info("API started successfully.")

    yield

    logger.info("Shutting down ATS Resume Analyzer API...")


# ---------------- FastAPI App ---------------- #

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
        "name": "ATS Resume Analyzer API",
        "version": APP_VERSION,
        "status": "running",
        "endpoints": {
            "POST /api/v1/analyze-resume": "Analyze Resume",
            "GET /api/v1/history": "Resume History",
            "DELETE /api/v1/history/{id}": "Delete History Entry",
            "GET /api/v1/health": "Health Check",
            "POST /api/v1/generate-pdf": "Generate PDF Report",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )