"""SQLAlchemy ORM models – mirrors the previous MongoDB collections."""
from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, JSON, ForeignKey
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from config.database import Base


def _now():
    return datetime.now(timezone.utc)


class Candidate(Base):
    __tablename__ = "candidates"

    candidate_id   = Column(String(36),  primary_key=True)
    name           = Column(String(255), nullable=False)
    resume_path    = Column(Text,        nullable=True)
    job_description = Column(Text,       nullable=True)
    created_at     = Column(DateTime(timezone=True), default=_now)
    status         = Column(String(50),  default="pending")

    sessions = relationship("InterviewSession", back_populates="candidate", cascade="all, delete-orphan")


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    interview_id           = Column(String(36),  primary_key=True)
    candidate_id           = Column(String(36),  ForeignKey("candidates.candidate_id"), nullable=False)
    status                 = Column(String(50),  default="in_progress")
    started_at             = Column(DateTime(timezone=True), default=_now)
    completed_at           = Column(DateTime(timezone=True), nullable=True)
    cancelled_at           = Column(DateTime(timezone=True), nullable=True)
    questions_asked        = Column(JSON,        default=list)
    current_question_index = Column(Integer,     default=0)

    candidate    = relationship("Candidate",    back_populates="sessions")
    interactions = relationship("Interview",    back_populates="session", cascade="all, delete-orphan")
    evaluation   = relationship("Evaluation",   back_populates="session", uselist=False, cascade="all, delete-orphan")
    live_session = relationship("InterviewLiveSession", back_populates="session", uselist=False, cascade="all, delete-orphan")


class Interview(Base):
    """Stores individual Q&A interactions (previously the 'interviews' collection)."""
    __tablename__ = "interviews"

    id                 = Column(Integer,     primary_key=True, autoincrement=True)
    interview_id       = Column(String(36),  ForeignKey("interview_sessions.interview_id"), nullable=False)
    question           = Column(Text,        nullable=True)
    answer_transcript  = Column(Text,        nullable=True)
    audio_path         = Column(Text,        nullable=True)
    timestamp          = Column(DateTime(timezone=True), default=_now)
    scores             = Column(JSON,        nullable=True)

    session = relationship("InterviewSession", back_populates="interactions")


class Evaluation(Base):
    __tablename__ = "evaluations"

    interview_id              = Column(String(36), ForeignKey("interview_sessions.interview_id"), primary_key=True)
    candidate_name            = Column(String(255), default="")
    interview_status          = Column(String(50),  default="completed")
    overall_score             = Column(Float,       default=0)
    confidence_score          = Column(Float,       default=0)
    communication_score       = Column(Float,       default=0)
    problem_solving_score     = Column(Float,       default=0)
    technical_knowledge_score = Column(Float,       default=0)
    role_fitment_score        = Column(Float,       default=0)
    clarity_score             = Column(Float,       default=0)
    recommendation            = Column(String(50),  default="Hold")
    summary                   = Column(Text,        default="")
    strengths                 = Column(JSON,        default=list)
    improvement_areas         = Column(JSON,        default=list)
    created_at                = Column(DateTime(timezone=True), default=_now)

    session = relationship("InterviewSession", back_populates="evaluation")


class InterviewLiveSession(Base):
    __tablename__ = "interview_live_sessions"

    interview_id = Column(String(36), ForeignKey("interview_sessions.interview_id"), primary_key=True)
    data         = Column(JSON,       nullable=True)
    updated_at   = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    session = relationship("InterviewSession", back_populates="live_session")
