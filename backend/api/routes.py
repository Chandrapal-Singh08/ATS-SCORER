import logging
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from backend.api.auth import get_current_user
from backend.core.model_loader import get_embedder
from backend.models.schemas import (
    AnalysisResponse,
    ComponentScores,
    JDComparison,
    SkillValidationDetails,
)

logger = logging.getLogger("ats_resume_scorer")

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post("/analyze-resume", response_model=AnalysisResponse)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(""),
):
    logger.info("=" * 60)
    logger.info("ANALYZE RESUME REQUEST RECEIVED")
    logger.info(f"Resume File: {resume.filename}")
    logger.info(f"Job Description Length: {len(job_description)}")

    user_id = "debug-user"

    # ---------------- STEP 1 ---------------- #
    logger.info("STEP 1: Loading spaCy model")
    nlp = request.app.state.nlp

    # ---------------- STEP 2 ---------------- #
    logger.info("STEP 2: Loading SentenceTransformer")
    try:
        embedder = get_embedder()
        logger.info("SentenceTransformer loaded.")
    except Exception as exc:
        logger.exception("SentenceTransformer failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Embedder failed: {exc}",
        )

    # ---------------- STEP 3 ---------------- #
    logger.info("STEP 3: Reading uploaded resume")
    try:
        file_bytes = await resume.read()
        filename = resume.filename or "resume"
        logger.info(f"Resume size: {len(file_bytes)} bytes")
    except Exception as exc:
        logger.exception("Failed reading uploaded resume.")
        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded file: {exc}",
        )

    # ---------------- STEP 4 ---------------- #
    logger.info("STEP 4: Parsing resume")
    try:
        from backend.services.resume_parser import parse_resume_file

        resume_text, metadata = parse_resume_file(file_bytes, filename)

        logger.info(f"Resume parsed. Characters extracted: {len(resume_text)}")
    except Exception as exc:
        logger.exception("Resume parsing failed.")
        raise HTTPException(
            status_code=422,
            detail=f"Resume parsing failed: {exc}",
        )

    # ---------------- STEP 5 ---------------- #
    logger.info("STEP 5: Running complete ATS analysis")
    try:
        from backend.services.resume_analyzer import analyze_full_resume

        result = analyze_full_resume(
            resume_text=resume_text,
            nlp=nlp,
            embedder=embedder,
            job_description=job_description,
        )

        logger.info("ATS analysis completed.")
    except Exception as exc:
        logger.exception("ATS analysis pipeline failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        )

    # ---------------- STEP 6 ---------------- #
    logger.info("STEP 6: Preparing JD comparison")

    jd_comparison_result = None

    if result.get("jd_comparison"):
        jd = result["jd_comparison"]

        jd_comparison_result = JDComparison(
            match_percentage=round(float(jd.get("match_percentage", 0)), 1),
            semantic_similarity=round(float(jd.get("semantic_similarity", 0)), 3),
            matched_keywords=jd.get("matched_keywords", [])[:20],
            missing_keywords=jd.get("missing_keywords", [])[:15],
            skills_gap=jd.get("skills_gap", [])[:10],
        )

    detailed_feedback = result.get("detailed_feedback", [])

    svd = result.get("skill_validation_details", {})

    skill_validation_details = SkillValidationDetails(
        validated=svd.get("validated", []),
        unvalidated=svd.get("unvalidated", []),
        total=svd.get("total", 0),
        validated_count=svd.get("validated_count", 0),
        validation_pct=svd.get("validation_pct", 0.0),
    )

    # ---------------- STEP 7 ---------------- #
    logger.info("STEP 7: Creating response object")

    response = AnalysisResponse(
        ATS_score=result["ats_score"],
        component_scores=ComponentScores(**result["component_scores"]),
        issues_summary=result["issues_summary"],
        detailed_feedback=detailed_feedback,
        jd_match_analysis=jd_comparison_result,
        skill_validation_details=skill_validation_details,

        # Backward compatibility
        ats_score=result["ats_score"],
        keyword_match=jd_comparison_result.match_percentage if jd_comparison_result else 0.0,
        missing_keywords=result.get("missing_keywords", []),
        matched_keywords=result.get("matched_keywords", []),
        skills=list(result.get("skills", [])[:20]),
        jd_comparison=jd_comparison_result,
        interpretation=result.get("interpretation", ""),
    )

    logger.info("Response object created.")

    # ---------------- STEP 8 ---------------- #
    logger.info("STEP 8: Saving analysis history")

    try:
        from backend.database.supabase_db import save_analysis

        await save_analysis(user_id, filename, result)
        logger.info("History saved successfully.")
    except Exception as exc:
        logger.warning(f"History save skipped: {exc}")

    logger.info("ANALYZE RESUME FINISHED SUCCESSFULLY")
    logger.info("=" * 60)

    return response


# ----------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------
@router.get("/health")
async def health_check(request: Request):
    return {
        "status": "healthy",
        "nlp_loaded": request.app.state.nlp is not None,
        "embedder_loaded": request.app.state.embedder is not None,
        "embedder_mode": "lazy_loading",
    }


# ----------------------------------------------------
# HISTORY
# ----------------------------------------------------
@router.get("/history")
async def get_history(user_id: str = Depends(get_current_user)):
    from backend.database.supabase_db import get_user_history

    try:
        return await get_user_history(user_id)
    except Exception as exc:
        logger.exception("History fetch failed.")
        raise HTTPException(
            status_code=500,
            detail=f"History fetch failed: {exc}",
        )


@router.delete("/history/{analysis_id}")
async def delete_history_entry(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    from backend.database.supabase_db import delete_analysis

    try:
        success = await delete_analysis(analysis_id, user_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail="Analysis not found.",
            )

        return {"status": "deleted", "id": analysis_id}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("History delete failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed: {exc}",
        )


# ----------------------------------------------------
# PDF GENERATION
# ----------------------------------------------------
@router.post("/generate-pdf")
async def generate_pdf(
    data: AnalysisResponse,
    user_id: str = Depends(get_current_user),
):
    from backend.services.report_generator import generate_html_reports
    from backend.services.pdf_export import generate_combined_pdf

    try:
        html_docs = generate_html_reports(data.model_dump())
        pdf_bytes = generate_combined_pdf(html_docs)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=ats_report.pdf"
            },
        )

    except Exception as exc:
        logger.exception("PDF generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {exc}",
        )


@router.get("/history/{analysis_id}/pdf")
async def generate_history_pdf(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
):
    from backend.database.supabase_db import get_user_history
    from backend.services.report_generator import generate_html_reports
    from backend.services.pdf_export import generate_combined_pdf

    history = await get_user_history(user_id)

    analysis_data = next(
        (item["analysis_result"] for item in history if item["id"] == analysis_id),
        None,
    )

    if analysis_data is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    try:
        html_docs = generate_html_reports(analysis_data)
        pdf_bytes = generate_combined_pdf(html_docs)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=ats_report_{analysis_id}.pdf"
            },
        )

    except Exception as exc:
        logger.exception("History PDF generation failed.")
        raise HTTPException(
            status_code=500,
            detail=f"History PDF generation failed: {exc}",
        )