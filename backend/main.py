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

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ats_resume_scorer")


# -------------------------------------------------------
# Startup / Shutdown
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    import spacy

    logger.info("========== ATS BACKEND STARTUP ==========")

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

        logger.info("spaCy model loaded successfully.")

    except OSError:
        logger.warning(
            f"{SPACY_MODEL_PRIMARY} not found. Falling back to {SPACY_MODEL_SECONDARY}"
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

        logger.info("Fallback spaCy model loaded successfully.")

    # IMPORTANT:
    # SentenceTransformer is NOT loaded here.
    # It will be loaded only when /analyze-resume is called.
    app.state.embedder = None

    logger.info("Startup completed successfully.")

    yield

    logger.info("========== ATS BACKEND SHUTDOWN ==========")


# -------------------------------------------------------
# FastAPI App
# -------------------------------------------------------
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)


# -------------------------------------------------------
# Root Endpoint
# -------------------------------------------------------
@app.get("/")
async def root():
    return {
        "name": "ATS Resume Analyzer API",
        "status": "running",
        "version": APP_VERSION,
    }


# -------------------------------------------------------
# Local Run
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )