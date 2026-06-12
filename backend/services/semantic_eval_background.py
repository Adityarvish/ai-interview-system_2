"""
semantic_eval_background.py — Background evaluation task runner.

This module is the entry point called by the socket handler immediately after
an answer is received. It:

  1. Fires the semantic evaluation as a non-blocking BackgroundTask.
  2. The HTTP/WebSocket response (next question) returns immediately.
  3. Evaluation completes asynchronously and persists to DB.

Usage in sockets/handlers.py (answer_received handler):
    from services.semantic_eval_background import run_semantic_eval_background

    # After recording the answer, before generating the next question:
    asyncio.create_task(
        run_semantic_eval_background(
            interview_id  = interview_id,
            question_id   = question_count,
            question      = current_question,
            transcript_raw= transcript,
            job_description = engine.candidate_info['job_description'],
            resume_context  = engine.candidate_info.get('resume_text', ''),
            stage           = engine.state.stage,
            llm_fn          = engine.llm.generate_short,
        )
    )

Design notes
------------
- asyncio.create_task() returns immediately so the socket handler continues.
- All CPU/IO in the evaluation pipeline runs via run_in_executor — no blocking
  of the event loop.
- Errors are caught and logged; they never propagate to the socket handler.
- A per-interview semaphore (max_concurrent=2) prevents memory spikes on fast
  back-to-back answers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, Optional

from services.semantic_evaluator import SemanticEvaluationEngine
from services.semantic_eval_db_service import SemanticEvalDBService

logger = logging.getLogger(__name__)

# Per-interview semaphore pool — limits concurrent evaluations per interview
_SEMAPHORES: Dict[str, asyncio.Semaphore] = {}
_MAX_CONCURRENT_PER_INTERVIEW = 2
_SEM_LOCK = asyncio.Lock()


async def _get_semaphore(interview_id: str) -> asyncio.Semaphore:
    async with _SEM_LOCK:
        if interview_id not in _SEMAPHORES:
            _SEMAPHORES[interview_id] = asyncio.Semaphore(_MAX_CONCURRENT_PER_INTERVIEW)
        return _SEMAPHORES[interview_id]


async def run_semantic_eval_background(
    interview_id:    str,
    question_id:     int,
    question:        str,
    transcript_raw:  str,
    job_description: str,
    resume_context:  str,
    stage:           str,
    llm_fn:          Callable[[str], str],
) -> None:
    """
    Run the full 9-stage semantic evaluation in the background.
    Persists results to DB. Never raises — all exceptions are logged.
    """
    t_start = time.perf_counter()
    sem     = await _get_semaphore(interview_id)

    try:
        async with sem:
            logger.info(
                f"[BG-EVAL] Starting evaluation: interview={interview_id} "
                f"Q{question_id} stage={stage}"
            )

            if not transcript_raw or len(transcript_raw.strip()) < 10:
                logger.warning(
                    f"[BG-EVAL] Skipping Q{question_id} — transcript too short "
                    f"({len(transcript_raw)} chars)"
                )
                return

            # ── Run the evaluation engine ──────────────────────────────────
            eval_engine = SemanticEvaluationEngine(llm_fn=llm_fn)
            result = await eval_engine.evaluate_question(
                question_id     = question_id,
                interview_id    = interview_id,
                question        = question,
                transcript_raw  = transcript_raw,
                job_description = job_description,
                resume_context  = resume_context,
                stage           = stage,
            )

            # ── Persist to database ────────────────────────────────────────
            db = SemanticEvalDBService()
            await asyncio.gather(
                db.save_question_evaluation(result),
                return_exceptions=True,
            )
            # Update running skill score aggregate
            await db.update_skill_scores(interview_id)

            elapsed = int((time.perf_counter() - t_start) * 1000)
            logger.info(
                f"[BG-EVAL] COMPLETE: interview={interview_id} Q{question_id} "
                f"q_score={result.question_score} skill={result.skill_score} "
                f"conf={result.confidence_score:.2f} total_ms={elapsed}"
            )

    except asyncio.CancelledError:
        logger.warning(f"[BG-EVAL] Evaluation cancelled for {interview_id} Q{question_id}")
    except Exception as e:
        elapsed = int((time.perf_counter() - t_start) * 1000)
        logger.error(
            f"[BG-EVAL] Evaluation failed for {interview_id} Q{question_id} "
            f"after {elapsed}ms: {e}",
            exc_info=True,
        )


def cleanup_interview_semaphore(interview_id: str) -> None:
    """
    Release semaphore memory when an interview ends.
    Call from the socket handler's interview_end / cleanup path.
    """
    _SEMAPHORES.pop(interview_id, None)
    logger.debug(f"[BG-EVAL] Semaphore released for {interview_id}")
