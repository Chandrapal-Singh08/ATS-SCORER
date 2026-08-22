import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.models.schemas import (
    AnalysisResponse,
    ComponentScores,
    JDComparison,
    SkillValidationDetails,
)
from backend.core.model_loader import get_embedder
from backend.services.resume_analyzer import analyze_full_resume
from backend.services.resume_parser import parse_resume_file

logger = logging.getLogger("ats_resume_scorer")

# ✅ FIX: Add /api/v1 prefix
router = APIRouter(
    prefix="/api/v1",
    tags=["Analysis"],
)


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "ATS Resume Scorer Backend is running",
    }


@router.post("/analyze-resume", response_model=AnalysisResponse)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(""),
):
    logger.info("========== NEW ANALYSIS REQUEST ==========")

    user_id = "debug-user"

    try:
        # STEP 1 — Load spaCy model
        logger.info("Step 1: Loading spaCy model")
        nlp = request.app.state.nlp

        # STEP 2 — Load SentenceTransformer lazily
        logger.info("Step 2: Loading embedder")
        embedder = get_embedder()

        # STEP 3 — Read uploaded resume
        logger.info("Step 3: Reading uploaded resume")
        file_bytes = await resume.read()
        filename = resume.filename or "resume.pdf"

        resume_text, metadata = parse_resume_file(file_bytes, filename)

        logger.info(
            "Resume parsed successfully | filename=%s | chars=%d",
            filename,
            len(resume_text),
        )

        # STEP 4 — Analyze resume
        logger.info("Step 4: Starting resume analysis")

        result = analyze_full_resume(
            resume_text=resume_text,
            nlp=nlp,
            embedder=embedder,
            job_description=job_description,
        )

        logger.info("Resume analysis completed successfully.")

        # STEP 5 — JD comparison
        jd_comparison_result = None

        if result.get("jd_comparison"):
            jd = result["jd_comparison"]

            jd_comparison_result = JDComparison(
                match_percentage=round(float(jd.get("match_percentage", 0.0)), 1),
                semantic_similarity=round(float(jd.get("semantic_similarity", 0.0)), 3),
                matched_keywords=jd.get("matched_keywords", [])[:20],
                missing_keywords=jd.get("missing_keywords", [])[:15],
                skills_gap=jd.get("skills_gap", [])[:10],
            )

        # STEP 6 — Skill validation
        svd = result.get("skill_validation_details", {})

        response = AnalysisResponse(
            ATS_score=result.get("ats_score", 0),
            component_scores=ComponentScores(**result.get("component_scores", {})),
            issues_summary=result.get("issues_summary", []),
            detailed_feedback=result.get("detailed_feedback", []),
            jd_match_analysis=jd_comparison_result,
            skill_validation_details=SkillValidationDetails(**svd),

            # Backward compatibility
            ats_score=result.get("ats_score", 0),
            keyword_match=(
                jd_comparison_result.match_percentage if jd_comparison_result else 0
            ),
            matched_keywords=result.get("matched_keywords", []),
            missing_keywords=result.get("missing_keywords", []),
            skills=result.get("skills", [])[:20],
            jd_comparison=jd_comparison_result,
            interpretation=result.get("interpretation", ""),
        )

        logger.info("Step 5: Saving analysis history")

        try:
            from backend.database.supabase_db import save_analysis

            await save_analysis(
                user_id=user_id,
                filename=filename,
                analysis_result=result,
            )
            logger.info("Analysis history saved successfully.")

        except Exception as history_error:
            logger.warning("History save skipped: %s", history_error)

        logger.info("========== ANALYSIS REQUEST COMPLETED ==========")

        return response

    except Exception as exc:
        logger.exception("========== ANALYSIS FAILED ==========")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {str(exc)}",
        )