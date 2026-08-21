@router.post("/analyze-resume", response_model=AnalysisResponse)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(...),
    job_description: str = Form(""),
):
    logger.info("========== NEW ANALYSIS REQUEST ==========")

    user_id = "debug-user"

    try:
        logger.info("Step 1: Loading spaCy")
        nlp = request.app.state.nlp

        logger.info("Step 2: Loading Embedder")
        embedder = get_embedder()

        logger.info("Step 3: Reading Resume")
        file_bytes = await resume.read()
        filename = resume.filename or "resume"

        from backend.services.resume_parser import parse_resume_file

        resume_text, _metadata = parse_resume_file(file_bytes, filename)

        logger.info(f"Resume parsed successfully ({len(resume_text)} chars).")

        logger.info("Step 4: Starting Resume Analysis")

        from backend.services.resume_analyzer import analyze_full_resume

        result = analyze_full_resume(
            resume_text=resume_text,
            nlp=nlp,
            embedder=embedder,
            job_description=job_description,
        )

        logger.info("Resume analysis completed.")

        jd_comparison_result = None

        if result.get("jd_comparison"):
            jd_comparison_result = JDComparison(
                match_percentage=round(
                    float(result["jd_comparison"].get("match_percentage", 0.0)),
                    1,
                ),
                semantic_similarity=round(
                    float(result["jd_comparison"].get("semantic_similarity", 0.0)),
                    3,
                ),
                matched_keywords=result["jd_comparison"].get(
                    "matched_keywords",
                    [],
                )[:20],
                missing_keywords=result["jd_comparison"].get(
                    "missing_keywords",
                    [],
                )[:15],
                skills_gap=result["jd_comparison"].get("skills_gap", [])[:10],
            )

        svd = result.get("skill_validation_details", {})

        response = AnalysisResponse(
            ATS_score=result["ats_score"],
            component_scores=ComponentScores(**result["component_scores"]),
            issues_summary=result["issues_summary"],
            detailed_feedback=result["detailed_feedback"],
            jd_match_analysis=jd_comparison_result,
            skill_validation_details=SkillValidationDetails(**svd),

            ats_score=result["ats_score"],
            keyword_match=(
                jd_comparison_result.match_percentage
                if jd_comparison_result
                else 0
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

            await save_analysis(user_id, filename, result)

        except Exception as history_error:
            logger.warning(f"History save skipped: {history_error}")

        logger.info("Analysis request completed successfully.")

        return response

    except Exception as exc:
        logger.exception("ANALYSIS FAILED")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {str(exc)}",
        )