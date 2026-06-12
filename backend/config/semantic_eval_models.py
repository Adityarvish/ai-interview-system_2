"""
semantic_eval_models.py — SQLAlchemy ORM additions for the Semantic Evaluation Engine.

Adds two new tables:
  question_evaluations  — per-question semantic scores (all 9 stage outputs)
  interview_skill_scores — aggregated skill scores per interview for final report

These are ADD-ONLY migrations — existing tables are untouched.
Run add_semantic_eval_tables() once at app startup (idempotent via checkfirst=True).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, JSON, ForeignKey, Index
)
from sqlalchemy.orm import relationship

from config.database import Base, engine


def _now():
    return datetime.now(timezone.utc)


class QuestionEvaluation(Base):
    """
    Stores the full semantic evaluation result for a single Q&A pair.
    One row per question answered per interview.
    """
    __tablename__ = "question_evaluations"

    id                   = Column(Integer,     primary_key=True, autoincrement=True)
    interview_id         = Column(String(36),  ForeignKey("interview_sessions.interview_id"), nullable=False, index=True)
    question_index       = Column(Integer,     nullable=False)          # 1-based
    stage                = Column(String(50),  nullable=False)          # "technical", "behavioral", etc.
    question_text        = Column(Text,        nullable=True)
    transcript_raw       = Column(Text,        nullable=True)
    transcript_clean     = Column(Text,        nullable=True)

    # ── Per-stage scores ───────────────────────────────────────────────────
    concept_coverage_score    = Column(Float, default=0.0)
    technical_accuracy_score  = Column(Float, default=0.0)
    completeness_score        = Column(Float, default=0.0)
    communication_score       = Column(Float, default=0.0)
    problem_solving_score     = Column(Float, default=0.0)
    star_score                = Column(Float, nullable=True)    # NULL for non-behavioral

    # ── Aggregated outputs ─────────────────────────────────────────────────
    question_score       = Column(Float, default=0.0)
    skill_score          = Column(Float, default=0.0)
    confidence_score     = Column(Float, default=0.0)

    # ── Qualitative feedback ────────────────────────────────────────────────
    strengths            = Column(JSON,  default=list)
    weaknesses           = Column(JSON,  default=list)

    # ── Stage detail blobs (full JSON from each stage) ─────────────────────
    concept_coverage_details   = Column(JSON, nullable=True)
    technical_accuracy_details = Column(JSON, nullable=True)
    completeness_details       = Column(JSON, nullable=True)
    communication_details      = Column(JSON, nullable=True)
    problem_solving_details    = Column(JSON, nullable=True)
    star_details               = Column(JSON, nullable=True)

    # ── Timing ─────────────────────────────────────────────────────────────
    total_eval_ms        = Column(Integer, default=0)
    evaluated_at         = Column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_qeval_interview_qidx", "interview_id", "question_index"),
    )


class InterviewSkillScore(Base):
    """
    Running aggregate of semantic skill scores per interview.
    Updated after each question evaluation (upsert pattern).
    Used by the final report endpoint for skill breakdown.
    """
    __tablename__ = "interview_skill_scores"

    interview_id          = Column(String(36), ForeignKey("interview_sessions.interview_id"), primary_key=True)

    # Running averages across all evaluated questions
    avg_question_score    = Column(Float, default=0.0)
    avg_skill_score       = Column(Float, default=0.0)
    avg_confidence        = Column(Float, default=0.0)

    # Stage-level averages
    avg_concept_coverage  = Column(Float, default=0.0)
    avg_technical_accuracy= Column(Float, default=0.0)
    avg_completeness      = Column(Float, default=0.0)
    avg_communication     = Column(Float, default=0.0)
    avg_problem_solving   = Column(Float, default=0.0)
    avg_star_score        = Column(Float, nullable=True)

    # All strengths/weaknesses collected across questions
    all_strengths         = Column(JSON,  default=list)
    all_weaknesses        = Column(JSON,  default=list)

    questions_evaluated   = Column(Integer, default=0)
    last_updated          = Column(DateTime(timezone=True), default=_now, onupdate=_now)


def add_semantic_eval_tables():
    """
    Idempotent DDL — only creates tables if they don't exist.
    Call once at FastAPI startup (lifespan or startup event).
    """
    Base.metadata.create_all(
        bind=engine,
        tables=[
            QuestionEvaluation.__table__,
            InterviewSkillScore.__table__,
        ],
        checkfirst=True,
    )
