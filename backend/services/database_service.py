"""
MySQL/SQLAlchemy database service for the AI Interview System.
"""
from datetime import datetime, timezone
import uuid
import logging

from sqlalchemy.orm import Session

from config.database import SessionLocal
from config.models import (
    Candidate,
    InterviewSession,
    Interview,
    Evaluation,
    InterviewLiveSession,
)

logger = logging.getLogger(__name__)


def _row_to_dict(row) -> dict | None:
    """Convert a SQLAlchemy model instance to a plain dict."""
    if row is None:
        return None
    d = {c.name: getattr(row, c.name) for c in row.__table__.columns}
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


class DatabaseService:
    """SQLAlchemy-backed service for all database operations."""

    def __init__(self):
        self._session: Session = SessionLocal()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _commit(self):
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

    def close(self):
        self._session.close()

    # ── Candidates ─────────────────────────────────────────────────────────────

    def create_candidate(self, name: str, resume_path: str, job_description: str) -> dict:
        candidate = Candidate(
            candidate_id=str(uuid.uuid4()),
            name=name,
            resume_path=resume_path,
            job_description=job_description,
            created_at=datetime.now(timezone.utc),
            status="pending",
        )
        self._session.add(candidate)
        self._commit()
        logger.info(f"[DB] Created candidate: {candidate.candidate_id}")
        return _row_to_dict(candidate)

    def get_candidate_by_id(self, candidate_id: str) -> dict | None:
        if not candidate_id:
            return None
        row = self._session.query(Candidate).filter_by(candidate_id=candidate_id).first()
        return _row_to_dict(row)

    # ── Interview sessions ─────────────────────────────────────────────────────

    def create_interview_session(self, candidate_id: str, interview_id: str) -> dict:
        session = InterviewSession(
            interview_id=interview_id,
            candidate_id=candidate_id,
            status="in_progress",
            started_at=datetime.now(timezone.utc),
            questions_asked=[],
            current_question_index=0,
        )
        self._session.add(session)
        self._commit()
        logger.info(f"[DB] Created interview session: {interview_id}")
        return _row_to_dict(session)

    def get_interview_session(self, interview_id: str) -> dict | None:
        row = self._session.query(InterviewSession).filter_by(interview_id=interview_id).first()
        return _row_to_dict(row)

    def mark_interview_cancelled(self, interview_id: str) -> None:
        row = self._session.query(InterviewSession).filter_by(interview_id=interview_id).first()
        if row:
            row.status = "cancelled"
            row.cancelled_at = datetime.now(timezone.utc)
            self._commit()
        logger.info(f"[DB] Interview cancelled: {interview_id}")

    # ── Interactions ───────────────────────────────────────────────────────────

    def save_interview_interaction(
        self,
        interview_id: str,
        question: str,
        answer: str,
        transcript: str,
        audio_path: str,
        scores: dict,
    ) -> dict:
        interaction = Interview(
            interview_id=interview_id,
            question=question,
            answer_transcript=transcript,
            audio_path=audio_path,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
        )
        self._session.add(interaction)

        session_row = (
            self._session.query(InterviewSession)
            .filter_by(interview_id=interview_id)
            .first()
        )
        if session_row:
            questions = list(session_row.questions_asked or [])
            questions.append(question)
            session_row.questions_asked = questions
            session_row.current_question_index = (session_row.current_question_index or 0) + 1

        self._commit()
        return _row_to_dict(interaction)

    def get_all_interactions(self, interview_id: str) -> list[dict]:
        rows = (
            self._session.query(Interview)
            .filter_by(interview_id=interview_id)
            .order_by(Interview.timestamp.asc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]

    def update_live_session(self, interview_id: str, data: dict) -> None:
        row = (
            self._session.query(InterviewLiveSession)
            .filter_by(interview_id=interview_id)
            .first()
        )
        if row:
            existing = dict(row.data or {})
            existing.update(data)
            row.data = existing
            row.updated_at = datetime.now(timezone.utc)
        else:
            row = InterviewLiveSession(
                interview_id=interview_id,
                data=data,
                updated_at=datetime.now(timezone.utc),
            )
            self._session.add(row)
        self._commit()

    # ── Evaluations ────────────────────────────────────────────────────────────

    def save_final_evaluation(self, interview_id: str, evaluation: dict) -> dict:
        """Idempotent upsert for the final evaluation record."""
        row = (
            self._session.query(Evaluation)
            .filter_by(interview_id=interview_id)
            .first()
        )
        doc = {
            "interview_id":              interview_id,
            "candidate_name":            evaluation.get("candidate_name", ""),
            "interview_status":          evaluation.get("interview_status", "completed"),
            "overall_score":             evaluation.get("overall_score", 0),
            "confidence_score":          evaluation.get("confidence_score", 0),
            "communication_score":       evaluation.get("communication_score", 0),
            "problem_solving_score":     evaluation.get("problem_solving_score", 0),
            "technical_knowledge_score": evaluation.get("technical_knowledge_score", 0),
            "role_fitment_score":        evaluation.get("role_fitment_score", 0),
            "clarity_score":             evaluation.get("clarity_score", 0),
            "recommendation":            evaluation.get("recommendation", "Hold"),
            "summary":                   evaluation.get("summary", ""),
            "strengths":                 evaluation.get("strengths", []),
            "improvement_areas":         evaluation.get("improvement_areas", []),
            "created_at":                datetime.now(timezone.utc),
        }

        if row:
            for k, v in doc.items():
                setattr(row, k, v)
        else:
            row = Evaluation(**doc)
            self._session.add(row)

        session_row = (
            self._session.query(InterviewSession)
            .filter_by(interview_id=interview_id)
            .first()
        )
        if session_row:
            session_row.status = (
                "completed" if doc["interview_status"] != "cancelled" else "cancelled"
            )
            session_row.completed_at = datetime.now(timezone.utc)

        self._commit()
        logger.info(
            f"[DB] Saved evaluation: {interview_id} | "
            f"overall={doc['overall_score']} | rec={doc['recommendation']}"
        )
        return {**doc, "created_at": doc["created_at"].isoformat()}

    def get_final_evaluation(self, interview_id: str) -> dict | None:
        row = (
            self._session.query(Evaluation)
            .filter_by(interview_id=interview_id)
            .first()
        )
        return _row_to_dict(row)