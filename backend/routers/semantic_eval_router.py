"""
semantic_eval_router.py — FastAPI routes for the Semantic Evaluation Engine.

New endpoints:
  GET  /api/semantic-eval/{interview_id}/summary
       → Real-time skill score aggregate (updates after each question)

  GET  /api/semantic-eval/{interview_id}/questions
       → Per-question semantic scores for all evaluated questions

  GET  /api/semantic-eval/{interview_id}/question/{question_id}
       → Full detail for a single question evaluation

  POST /api/semantic-eval/evaluate-now
       → Trigger evaluation synchronously (for testing / manual re-score)

  GET  /api/semantic-eval/{interview_id}/final-report
       → Merged final report: LLM evaluation + semantic scores + skill breakdown
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from services.semantic_eval_db_service import SemanticEvalDBService
from services.semantic_eval_background import run_semantic_eval_background

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/semantic-eval", tags=["semantic-eval"])
_db = SemanticEvalDBService()


# ─── Request / Response schemas ──────────────────────────────────────────────

class EvaluateNowRequest(BaseModel):
    interview_id:    str
    question_id:     int
    question:        str
    transcript_raw:  str
    job_description: str
    resume_context:  Optional[str] = ""
    stage:           Optional[str] = "technical"


class EvaluateNowResponse(BaseModel):
    success:         bool
    interview_id:    str
    question_id:     int
    question_score:  int
    skill_score:     int
    confidence_score:float
    strengths:       list
    weaknesses:      list
    stage_scores:    dict
    elapsed_ms:      int


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get("/{interview_id}/summary")
async def get_skill_summary(interview_id: str):
    """
    Returns the running-average semantic skill scores for an interview.
    Updates in near-real-time after each question evaluation completes.
    Useful for a live dashboard or post-interview drill-down.
    """
    summary = await _db.get_interview_skill_scores(interview_id)
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"No semantic evaluations found for interview {interview_id}. "
                   "Evaluation runs in the background — check back after a question is answered."
        )
    return {"success": True, "summary": summary}


@router.get("/{interview_id}/questions")
async def get_question_evaluations(interview_id: str):
    """
    Returns all per-question semantic evaluation results for an interview,
    ordered by question index.
    """
    questions = await _db.get_question_evaluations(interview_id)
    if not questions:
        raise HTTPException(
            status_code=404,
            detail=f"No question evaluations found for interview {interview_id}."
        )
    return {
        "success":    True,
        "interview_id": interview_id,
        "count":      len(questions),
        "questions":  questions,
    }


@router.get("/{interview_id}/question/{question_id}")
async def get_single_question_evaluation(interview_id: str, question_id: int):
    """Returns full semantic eval detail for one specific question."""
    questions = await _db.get_question_evaluations(interview_id)
    match = next((q for q in questions if q["question_index"] == question_id), None)
    if not match:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for Q{question_id} in interview {interview_id}."
        )
    return {"success": True, "evaluation": match}


@router.get("/{interview_id}/final-report")
async def get_semantic_final_report(interview_id: str):
    """
    Merged final report: semantic skill scores + per-question breakdown.
    Complement to /api/final-report/{interview_id} (which returns the LLM eval).
    Combine both on the frontend for the richest report view.
    """
    summary, questions = await asyncio.gather(
        _db.get_interview_skill_scores(interview_id),
        _db.get_question_evaluations(interview_id),
        return_exceptions=True,
    )

    if isinstance(summary, Exception) or not summary:
        raise HTTPException(
            status_code=404,
            detail=f"No semantic data for interview {interview_id}."
        )

    questions = questions if not isinstance(questions, Exception) else []

    # Build skill radar data for frontend charts
    radar = {
        "concept_coverage":  summary.get("avg_concept_coverage",  0),
        "technical_accuracy":summary.get("avg_technical_accuracy", 0),
        "completeness":      summary.get("avg_completeness",        0),
        "communication":     summary.get("avg_communication",       0),
        "problem_solving":   summary.get("avg_problem_solving",     0),
    }
    if summary.get("avg_star_score") is not None:
        radar["star_compliance"] = summary["avg_star_score"]

    return {
        "success":      True,
        "interview_id": interview_id,
        "skill_summary": {
            **summary,
            "radar_data": radar,
        },
        "questions": questions,
    }


@router.post("/evaluate-now", response_model=EvaluateNowResponse)
async def evaluate_now(body: EvaluateNowRequest):
    """
    Trigger a synchronous semantic evaluation for a single Q&A pair.
    Returns the full result immediately — useful for testing or manual re-scoring.

    NOTE: This runs synchronously (awaited) — use BackgroundTask path in production.
    """
    from services.semantic_evaluator import SemanticEvaluationEngine
    from services.llm_service import OllamaService

    t0  = time.perf_counter()
    llm = OllamaService()

    eval_engine = SemanticEvaluationEngine(llm_fn=llm.generate_short)

    try:
        result = await eval_engine.evaluate_question(
            question_id     = body.question_id,
            interview_id    = body.interview_id,
            question        = body.question,
            transcript_raw  = body.transcript_raw,
            job_description = body.job_description,
            resume_context  = body.resume_context or "",
            stage           = body.stage or "technical",
        )
    except Exception as e:
        logger.exception(f"[evaluate-now] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Persist
    db = SemanticEvalDBService()
    await asyncio.gather(
        db.save_question_evaluation(result),
        return_exceptions=True,
    )
    await db.update_skill_scores(body.interview_id)

    return EvaluateNowResponse(
        success          = True,
        interview_id     = result.interview_id,
        question_id      = result.question_id,
        question_score   = result.question_score,
        skill_score      = result.skill_score,
        confidence_score = result.confidence_score,
        strengths        = result.strengths,
        weaknesses       = result.weaknesses,
        stage_scores     = {
            "concept_coverage":   result.concept_coverage.score,
            "technical_accuracy": result.technical_accuracy.score,
            "completeness":       result.completeness.score,
            "communication":      result.communication.score,
            "problem_solving":    result.problem_solving.score,
            "star_evaluation":    result.star_evaluation.score if result.star_evaluation else None,
        },
        elapsed_ms = int((time.perf_counter() - t0) * 1000),
    )
