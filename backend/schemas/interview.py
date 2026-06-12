"""
Pydantic request/response models for the AI Interview System.
These replace Flask's request.json / jsonify() patterns.
"""
from typing import Optional, Any
from pydantic import BaseModel


# ── Request models ────────────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    interview_id: Optional[str] = "ondemand"


# ── Response models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    success: bool
    status: str
    service: str


class StartInterviewResponse(BaseModel):
    success: bool
    interview_id: Optional[str] = None
    candidate_id: Optional[str] = None
    first_question: Optional[str] = None
    timing: Optional[dict] = None
    error: Optional[str] = None


class FinalReportResponse(BaseModel):
    success: bool
    evaluation: Optional[dict] = None
    error: Optional[str] = None


class TTSResponse(BaseModel):
    success: bool
    audio: Optional[str] = None
    audio_format: Optional[str] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
