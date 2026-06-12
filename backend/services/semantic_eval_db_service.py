"""
semantic_eval_db_service.py — Database service for the Semantic Evaluation Engine.

Provides async-friendly CRUD via run_in_executor so the event loop is never blocked.
All public methods are async wrappers around sync SQLAlchemy calls.

Pattern: each method opens a fresh session scoped to that call (thread-safe).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from config.database import SessionLocal
from config.semantic_eval_models import QuestionEvaluation, InterviewSkillScore
from services.semantic_evaluator import QuestionEvalResult

logger = logging.getLogger(__name__)


def _new_session() -> Session:
    return SessionLocal()


# ─── Sync helpers (called via run_in_executor) ──────────────────────────────

def _save_question_eval_sync(result: QuestionEvalResult) -> Dict:
    """Upsert a QuestionEvaluation row for a single Q&A pair."""
    session = _new_session()
    try:
        row = (
            session.query(QuestionEvaluation)
            .filter_by(interview_id=result.interview_id, question_index=result.question_id)
            .first()
        )
        if not row:
            row = QuestionEvaluation(
                interview_id   = result.interview_id,
                question_index = result.question_id,
            )
            session.add(row)

        row.stage                     = result.concept_coverage.details.get("stage", "unknown")
        row.question_text             = result.question
        row.transcript_raw            = result.transcript_raw
        row.transcript_clean          = result.transcript_clean

        row.concept_coverage_score    = result.concept_coverage.score
        row.technical_accuracy_score  = result.technical_accuracy.score
        row.completeness_score        = result.completeness.score
        row.communication_score       = result.communication.score
        row.problem_solving_score     = result.problem_solving.score
        row.star_score                = result.star_evaluation.score if result.star_evaluation else None

        row.question_score            = result.question_score
        row.skill_score               = result.skill_score
        row.confidence_score          = result.confidence_score
        row.strengths                 = result.strengths
        row.weaknesses                = result.weaknesses

        row.concept_coverage_details  = result.concept_coverage.details
        row.technical_accuracy_details= result.technical_accuracy.details
        row.completeness_details      = result.completeness.details
        row.communication_details     = result.communication.details
        row.problem_solving_details   = result.problem_solving.details
        row.star_details              = result.star_evaluation.details if result.star_evaluation else None

        row.total_eval_ms             = result.total_elapsed_ms
        row.evaluated_at              = datetime.now(timezone.utc)

        session.commit()
        session.refresh(row)
        return {"id": row.id, "question_index": row.question_index}

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _update_skill_scores_sync(interview_id: str) -> Dict:
    """
    Re-compute running averages for InterviewSkillScore from all evaluated
    QuestionEvaluation rows for this interview. Upsert pattern.
    """
    session = _new_session()
    try:
        rows: List[QuestionEvaluation] = (
            session.query(QuestionEvaluation)
            .filter_by(interview_id=interview_id)
            .all()
        )
        if not rows:
            return {}

        def avg(values):
            vals = [v for v in values if v is not None]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        all_strengths  = []
        all_weaknesses = []
        for r in rows:
            all_strengths.extend(r.strengths or [])
            all_weaknesses.extend(r.weaknesses or [])

        # Deduplicate while preserving order
        seen_s, uniq_s = set(), []
        for s in all_strengths:
            if s not in seen_s:
                seen_s.add(s); uniq_s.append(s)
        seen_w, uniq_w = set(), []
        for w in all_weaknesses:
            if w not in seen_w:
                seen_w.add(w); uniq_w.append(w)

        skill_row = session.query(InterviewSkillScore).filter_by(interview_id=interview_id).first()
        if not skill_row:
            skill_row = InterviewSkillScore(interview_id=interview_id)
            session.add(skill_row)

        skill_row.avg_question_score     = avg([r.question_score      for r in rows])
        skill_row.avg_skill_score        = avg([r.skill_score         for r in rows])
        skill_row.avg_confidence         = avg([r.confidence_score    for r in rows])
        skill_row.avg_concept_coverage   = avg([r.concept_coverage_score    for r in rows])
        skill_row.avg_technical_accuracy = avg([r.technical_accuracy_score  for r in rows])
        skill_row.avg_completeness       = avg([r.completeness_score        for r in rows])
        skill_row.avg_communication      = avg([r.communication_score       for r in rows])
        skill_row.avg_problem_solving    = avg([r.problem_solving_score     for r in rows])
        star_scores = [r.star_score for r in rows if r.star_score is not None]
        skill_row.avg_star_score         = avg(star_scores) if star_scores else None
        skill_row.all_strengths          = uniq_s[:10]
        skill_row.all_weaknesses         = uniq_w[:10]
        skill_row.questions_evaluated    = len(rows)
        skill_row.last_updated           = datetime.now(timezone.utc)

        session.commit()

        return {
            "interview_id":            interview_id,
            "questions_evaluated":     skill_row.questions_evaluated,
            "avg_question_score":      skill_row.avg_question_score,
            "avg_skill_score":         skill_row.avg_skill_score,
            "avg_confidence":          skill_row.avg_confidence,
            "avg_concept_coverage":    skill_row.avg_concept_coverage,
            "avg_technical_accuracy":  skill_row.avg_technical_accuracy,
            "avg_completeness":        skill_row.avg_completeness,
            "avg_communication":       skill_row.avg_communication,
            "avg_problem_solving":     skill_row.avg_problem_solving,
            "avg_star_score":          skill_row.avg_star_score,
            "strengths":               skill_row.all_strengths,
            "weaknesses":              skill_row.all_weaknesses,
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _get_interview_skill_scores_sync(interview_id: str) -> Optional[Dict]:
    session = _new_session()
    try:
        row = session.query(InterviewSkillScore).filter_by(interview_id=interview_id).first()
        if not row:
            return None
        return {
            "interview_id":            row.interview_id,
            "questions_evaluated":     row.questions_evaluated,
            "avg_question_score":      row.avg_question_score,
            "avg_skill_score":         row.avg_skill_score,
            "avg_confidence":          row.avg_confidence,
            "avg_concept_coverage":    row.avg_concept_coverage,
            "avg_technical_accuracy":  row.avg_technical_accuracy,
            "avg_completeness":        row.avg_completeness,
            "avg_communication":       row.avg_communication,
            "avg_problem_solving":     row.avg_problem_solving,
            "avg_star_score":          row.avg_star_score,
            "strengths":               row.all_strengths or [],
            "weaknesses":              row.all_weaknesses or [],
            "last_updated":            row.last_updated.isoformat() if row.last_updated else None,
        }
    finally:
        session.close()


def _get_question_evals_sync(interview_id: str) -> List[Dict]:
    session = _new_session()
    try:
        rows = (
            session.query(QuestionEvaluation)
            .filter_by(interview_id=interview_id)
            .order_by(QuestionEvaluation.question_index)
            .all()
        )
        return [
            {
                "question_index":           r.question_index,
                "stage":                    r.stage,
                "question_text":            r.question_text,
                "question_score":           r.question_score,
                "skill_score":              r.skill_score,
                "confidence_score":         r.confidence_score,
                "strengths":                r.strengths or [],
                "weaknesses":               r.weaknesses or [],
                "concept_coverage_score":   r.concept_coverage_score,
                "technical_accuracy_score": r.technical_accuracy_score,
                "completeness_score":       r.completeness_score,
                "communication_score":      r.communication_score,
                "problem_solving_score":    r.problem_solving_score,
                "star_score":               r.star_score,
                "total_eval_ms":            r.total_eval_ms,
                "evaluated_at":             r.evaluated_at.isoformat() if r.evaluated_at else None,
            }
            for r in rows
        ]
    finally:
        session.close()


# ─── Async public API ────────────────────────────────────────────────────────

class SemanticEvalDBService:
    """
    Async database service for the Semantic Evaluation Engine.
    All methods run sync SQLAlchemy in a thread pool.
    """

    async def save_question_evaluation(self, result: QuestionEvalResult) -> Dict:
        loop = asyncio.get_event_loop()
        try:
            saved = await loop.run_in_executor(None, _save_question_eval_sync, result)
            logger.info(
                f"[SEM-DB] Saved Q{result.question_id} for {result.interview_id} "
                f"(q_score={result.question_score}, skill={result.skill_score})"
            )
            return saved
        except Exception as e:
            logger.error(f"[SEM-DB] save_question_evaluation failed: {e}")
            return {}

    async def update_skill_scores(self, interview_id: str) -> Dict:
        loop = asyncio.get_event_loop()
        try:
            updated = await loop.run_in_executor(None, _update_skill_scores_sync, interview_id)
            logger.info(f"[SEM-DB] Updated skill scores for {interview_id}: {updated.get('avg_skill_score')}")
            return updated
        except Exception as e:
            logger.error(f"[SEM-DB] update_skill_scores failed: {e}")
            return {}

    async def get_interview_skill_scores(self, interview_id: str) -> Optional[Dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_interview_skill_scores_sync, interview_id)

    async def get_question_evaluations(self, interview_id: str) -> List[Dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_question_evals_sync, interview_id)
