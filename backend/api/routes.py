import logging
import traceback

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from backend.core.model_loader import get_embedder
from backend.models.schemas import (
    AnalysisResponse,
    ComponentScores,
    JDComparison,
    SkillValidationDetails,
)
from backend.services.resume_analyzer import analyze_full_resume
from backend.services.resume_parser import parse_resume_file

logger = logging.getLogger("ats_resume_scorer")

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


@router.post(
    "/analyze-resume",
    response_model=AnalysisResponse,
)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(""),
):
    logger.info("========== NEW ANALYSIS REQUEST ==========")

    user_id = "debug-user"

    try:
        # Step 1
        logger.info("Step 1: Loading spaCy model")
        nlp = request.app.state.nlp

        # Step 2
        logger.info("Step 2: Loading SentenceTransformer model")
        embedder = get_embedder()

        # Step 3
        logger.info("Step 3: Reading uploaded resume")

        file_bytes = await resume.read()
        filename = resume.filename or "resume.pdf"

        logger.info(
            "Uploaded resume: %s (%d bytes)",
            filename,
            len(file_bytes),
        )

        resume_text, metadata = parse_resume_file(
            file_bytes,
            filename,
        )

        logger.info(
            "Resume parsed successfully. Characters=%d",
            len(resume_text),
        )

        # Step 4
        logger.info("Step 4: Starting ATS analysis")

        result = analyze_full_resume(
            resume_text=resume_text,
            nlp=nlp,
            embedder=embedder,
            job_description=job_description,
        )

        logger.info("Resume analysis finished.")

        jd_result = None

        if result.get("jd_comparison"):
            jd = result["jd_comparison"]

            jd_result = JDComparison(
                match_percentage=round(
                    float(jd.get("match_percentage", 0)),
                    1,
                ),
                semantic_similarity=round(
                    float(jd.get("semantic_similarity", 0)),
                    3,
                ),
                matched_keywords=jd.get("matched_keywords", [])[:20],
                missing_keywords=jd.get("missing_keywords", [])[:15],
                skills_gap=jd.get("skills_gap", [])[:10],
            )

        svd = result.get(
            "skill_validation_details",
            {},
        )

        response = AnalysisResponse(
            ATS_score=result.get("ats_score", 0),

            component_scores=ComponentScores(
                **result.get("component_scores", {})
            ),

            issues_summary=result.get(
                "issues_summary",
                [],
            ),

            detailed_feedback=result.get(
                "detailed_feedback",
                [],
            ),

            jd_match_analysis=jd_result,

            skill_validation_details=SkillValidationDetails(
                **svd
            ),

            ats_score=result.get("ats_score", 0),

            keyword_match=(
                jd_result.match_percentage
                if jd_result
                else 0
            ),

            matched_keywords=result.get(
                "matched_keywords",
                [],
            ),

            missing_keywords=result.get(
                "missing_keywords",
                [],
            ),

            skills=result.get("skills", [])[:20],

            jd_comparison=jd_result,

            interpretation=result.get(
                "interpretation",
                "",
            ),
        )

        logger.info("Step 5: Saving analysis history")

        try:
            from backend.database.supabase_db import save_analysis

            await save_analysis(
                user_id=user_id,
                filename=filename,
                analysis_result=result,
            )

            logger.info("Analysis history saved.")

        except Exception as history_error:
            logger.warning(
                "History save skipped: %s",
                history_error,
            )

        logger.info("========== ANALYSIS COMPLETED ==========")

        return response

    except Exception as exc:
        logger.error("========== ANALYSIS FAILED ==========")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "type": type(exc).__name__,
            },
        )