import logging
import traceback

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.models.schemas import (
    AnalysisResponse,
    ComponentScores,
    JDComparison,
    SkillValidationDetails,
)

logger = logging.getLogger("ats_resume_scorer")

# ------------------------------------------------------
# API Router
# ------------------------------------------------------
router = APIRouter(
    prefix="/api/v1",
    tags=["Analysis"],
)


# ------------------------------------------------------
# Health Endpoint
# ------------------------------------------------------
@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "ATS Resume Scorer Backend is running",
    }


# ------------------------------------------------------
# Resume Analysis Endpoint
# ------------------------------------------------------
@router.post("/analyze-resume", response_model=AnalysisResponse)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(""),
):
    logger.info("========== NEW ANALYSIS REQUEST ==========")

    # ✅ Lazy imports (IMPORTANT FOR RENDER)
    from backend.core.model_loader import get_embedder
    from backend.services.resume_parser import parse_resume_file
    from backend.services.resume_analyzer import analyze_full_resume

    user_id = "debug-user"

    try:
        # ---------------- STEP 1 ----------------
        logger.info("Loading spaCy model...")
        nlp = request.app.state.nlp

        # ---------------- STEP 2 ----------------
        logger.info("Loading SentenceTransformer model...")
        embedder = get_embedder()

        # ---------------- STEP 3 ----------------
        file_bytes = await resume.read()
        filename = resume.filename or "resume.pdf"

        logger.info(
            "Received file: %s (%d bytes)",
            filename,
            len(file_bytes),
        )

        resume_text, metadata = parse_resume_file(file_bytes, filename)

        logger.info(
            "Resume parsed successfully (%d characters)",
            len(resume_text),
        )

        # ---------------- STEP 4 ----------------
        result = analyze_full_resume(
            resume_text=resume_text,
            nlp=nlp,
            embedder=embedder,
            job_description=job_description,
        )

        logger.info("Resume analysis completed.")

        # ---------------- STEP 5 ----------------
        jd_result = None

        if result.get("jd_comparison"):
            jd = result["jd_comparison"]

            jd_result = JDComparison(
                match_percentage=round(
                    float(jd.get("match_percentage", 0.0)),
                    1,
                ),
                semantic_similarity=round(
                    float(jd.get("semantic_similarity", 0.0)),
                    3,
                ),
                matched_keywords=jd.get("matched_keywords", [])[:20],
                missing_keywords=jd.get("missing_keywords", [])[:15],
                skills_gap=jd.get("skills_gap", [])[:10],
            )

        # ---------------- STEP 6 ----------------
        response = AnalysisResponse(
            ATS_score=result["ats_score"],

            component_scores=ComponentScores(
                **result["component_scores"]
            ),

            issues_summary=result["issues_summary"],
            detailed_feedback=result["detailed_feedback"],

            jd_match_analysis=jd_result,

            skill_validation_details=SkillValidationDetails(
                **result["skill_validation_details"]
            ),

            # Backward compatibility
            ats_score=result["ats_score"],
            keyword_match=jd_result.match_percentage if jd_result else 0,
            matched_keywords=result["matched_keywords"],
            missing_keywords=result["missing_keywords"],
            skills=result["skills"][:20],
            jd_comparison=jd_result,
            interpretation=result.get("interpretation", ""),
        )

        # ---------------- STEP 7 ----------------
        try:
            from backend.database.supabase_db import save_analysis

            await save_analysis(
                user_id=user_id,
                filename=filename,
                analysis_result=result,
            )

            logger.info("Analysis history saved successfully.")

        except Exception as history_error:
            logger.warning(
                "History save skipped: %s",
                history_error,
            )

        logger.info("========== ANALYSIS REQUEST COMPLETED ==========")

        return response

    except Exception as exc:
        logger.error("========== ANALYSIS FAILED ==========")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )