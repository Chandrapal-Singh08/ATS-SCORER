import os
from pathlib import Path

# ---------------------------------------------------
# Load .env
# ---------------------------------------------------
try:
    from dotenv import load_dotenv

    ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(ENV_PATH)

except ImportError:
    pass

# ---------------------------------------------------
# API Metadata
# ---------------------------------------------------

APP_TITLE = "ATS RESUME ANALYZER API"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Analyze resumes against job descriptions using NLP + ML."

# ---------------------------------------------------
# CORS
# ---------------------------------------------------

LOCAL_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

DEPLOYED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

ALLOWED_ORIGINS = list(dict.fromkeys([*LOCAL_ORIGINS, *DEPLOYED_ORIGINS]))

# ---------------------------------------------------
# Resume File Validation
# ---------------------------------------------------

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}

# ---------------------------------------------------
# NLP Models (Render Optimized)
# ---------------------------------------------------

SPACY_MODEL_PRIMARY = "en_core_web_sm"
SPACY_MODEL_SECONDARY = "en_core_web_sm"

# Smallest sentence-transformer model (384 dimensions)
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "paraphrase-MiniLM-L3-v2",
)

# ---------------------------------------------------
# ATS Score Weights
# ---------------------------------------------------

SCORE_WEIGHTS = {
    "formatting": 20,
    "keywords": 25,
    "content": 25,
    "skill_validation": 15,
    "ats_compatibility": 15,
}

JD_KEYWORD_WEIGHT = 0.6
JD_SEMANTIC_WEIGHT = 0.4

# ---------------------------------------------------
# Supabase
# ---------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# ---------------------------------------------------
# Groq
# ---------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")