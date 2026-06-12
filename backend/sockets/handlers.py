"""
Socket.IO async event handlers for the AI Voice Interview System.
"""
import asyncio
import base64
import hashlib
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path

import socketio

from config.settings import Config
from services.database_service import DatabaseService
from services.interview_engine import InterviewEngine
from services.evaluator import EvaluatorService
from services.speech_to_text import stt_service
from services.text_to_speech import tts_service

logger = logging.getLogger(__name__)

# ── Socket.IO server ───────────────────────────────────────────────────────────
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=300,
    ping_interval=25,
    async_mode="asgi",
)

# ── In-memory session stores ───────────────────────────────────────────────────
ENGINES:              dict[str, InterviewEngine] = {}
AUDIO_BUFFERS:        dict[str, list[bytes]]     = {}
CANCELLED_INTERVIEWS: set[str]                   = set()
SID_TO_INTERVIEW:     dict[str, str]             = {}

_PROCESSING_INTERVIEWS: set[str] = set()
_EVALUATING_INTERVIEWS: set[str] = set()

# Module-level database instance
_db = DatabaseService()


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _safe_emit(sid: str, event: str, payload: dict):
    """Emit to the interview room when possible, otherwise direct to sid."""
    interview_id = payload.get('interview_id')
    if interview_id:
        await sio.emit(event, payload, room=interview_id)
    elif sid:
        await sio.emit(event, payload, to=sid)
    else:
        logger.warning(f"[EMIT] No sid or interview_id — broadcasting {event}")
        await sio.emit(event, payload)


def _is_cancelled(interview_id: str) -> bool:
    return interview_id in CANCELLED_INTERVIEWS


def _cleanup_session(interview_id: str):
    """Atomically clear all in-memory session state for an interview."""
    ENGINES.pop(interview_id, None)
    AUDIO_BUFFERS.pop(interview_id, None)
    CANCELLED_INTERVIEWS.discard(interview_id)
    _PROCESSING_INTERVIEWS.discard(interview_id)
    _EVALUATING_INTERVIEWS.discard(interview_id)


def _delete_file_safe(path: str):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception as e:
        logger.debug(f"[CLEANUP] Could not delete {path}: {e}")


def _validate_audio_bytes(data: bytes, min_size: int = 500) -> tuple[bool, str]:
    if not data:
        return False, "No audio data received."
    if len(data) < min_size:
        return False, f"Audio too short ({len(data)} bytes). Please record a longer answer."
    return True, ""


def _convert_to_wav_blocking(input_path: str, output_path: str, timeout: int = 30) -> bool:
    """Run ffmpeg synchronously — always called via run_in_executor."""
    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-ar', '16000',
        '-ac', '1',
        '-f', 'wav',
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error(
                f"[FFMPEG] Conversion failed (rc={result.returncode})\n"
                f"STDERR: {result.stderr[-500:]}"
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"[FFMPEG] Conversion timed out after {timeout}s")
        return False
    except FileNotFoundError:
        logger.error("[FFMPEG] ffmpeg not found — install: apt-get install ffmpeg")
        return False
    except Exception as e:
        logger.error(f"[FFMPEG] Unexpected error: {e}")
        return False


async def _convert_to_wav(input_path: str, output_path: str, timeout: int = 30) -> bool:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _convert_to_wav_blocking, input_path, output_path, timeout
    )


def _tts_b64_blocking(text: str, interview_id: str) -> tuple:
    """Generate TTS synchronously — always called via run_in_executor."""
    from services.text_to_speech import GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT

    t0       = time.perf_counter()
    out_name = f"ai_{interview_id}_{uuid.uuid4().hex}.wav"

    logger.info(
        "[TTS] [%s] provider=GroqOrpheus  model=%s  voice=%s  fmt=%s  chars=%d",
        interview_id, GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT, len(text)
    )

    try:
        path = tts_service.generate_speech(text, out_name)

        if path is None:
            elapsed = int((time.perf_counter() - t0) * 1000)
            logger.error("[TTS_FAILED] [%s] after %dms — generate_speech() returned None.",
                         interview_id, elapsed)
            return None, None

        suffix = Path(path).suffix.lstrip('.')
        with open(path, 'rb') as f:
            audio_bytes = f.read()

        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        timestamp  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        b64        = base64.b64encode(audio_bytes).decode('ascii')
        size_bytes = len(audio_bytes)
        elapsed    = int((time.perf_counter() - t0) * 1000)

        logger.info(
            "[TTS_FILE] [%s]  model=%s  voice=%s  file=%s  size=%d  ts=%s  hash=%s",
            interview_id, GROQ_TTS_MODEL, GROQ_TTS_VOICE,
            out_name, size_bytes, timestamp, audio_hash
        )
        logger.info(
            "[TTS_COMPLETED] [%s] elapsed_ms=%d  size_kb=%d  fmt=%s  voice=%s  hash=%s",
            interview_id, elapsed, size_bytes // 1024, suffix, GROQ_TTS_VOICE, audio_hash
        )
        _delete_file_safe(path)
        return b64, suffix

    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.exception(
            "[TTS_FAILED] [%s] after %dms — %s\n"
            "             Frontend will fall back to browser speechSynthesis.\n"
            "             Run: curl http://localhost:8000/api/tts/health",
            interview_id, elapsed, e
        )
        return None, None


async def _tts_b64(text: str, interview_id: str) -> tuple:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _tts_b64_blocking, text, interview_id)


# ── Core audio processing ──────────────────────────────────────────────────────

async def _handle_audio_bytes(sid: str, interview_id: str, audio_bytes: bytes,
                               extension: str, elapsed: int):
    valid, validation_msg = _validate_audio_bytes(audio_bytes)
    if not valid:
        await _safe_emit(sid, 'error', {'interview_id': interview_id, 'message': validation_msg})
        return

    if extension not in ('webm', 'ogg', 'mp4', 'wav'):
        extension = 'webm'

    raw_path = Config.AUDIO_FOLDER / f"{interview_id}_{uuid.uuid4().hex}.{extension}"
    raw_path.write_bytes(audio_bytes)
    logger.info(f"[AUDIO] [{interview_id}] Wrote {len(audio_bytes)} bytes → {raw_path.name}")

    engine = ENGINES.get(interview_id)

    async def _process():
        t_bg = time.perf_counter()
        _PROCESSING_INTERVIEWS.add(interview_id)

        _keepalive_running = [True]
        _current_status    = ['Transcribing your answer…']
        _keepalive_task    = [None]
        wav_path_str       = [None]

        async def _keepalive_loop():
            while _keepalive_running[0]:
                await asyncio.sleep(5)
                if _keepalive_running[0] and not _is_cancelled(interview_id):
                    await sio.emit('status',
                                   {'interview_id': interview_id,
                                    'message': _current_status[0]},
                                   room=interview_id)
                    await sio.emit('heartbeat',
                                   {'interview_id': interview_id,
                                    'ts': int(time.time() * 1000)},
                                   room=interview_id)

        try:
            if _is_cancelled(interview_id):
                logger.info(f"[PROCESS] [{interview_id}] Cancelled before STT")
                return

            _keepalive_task[0] = asyncio.create_task(_keepalive_loop())

            # ── Convert to WAV ────────────────────────────────────────────────
            _current_status[0] = 'Transcribing your answer…'
            await _safe_emit(sid, 'status',
                             {'interview_id': interview_id,
                              'message': 'Transcribing your answer…'})

            wav_path = Config.AUDIO_FOLDER / f"{interview_id}_{uuid.uuid4().hex}.wav"
            wav_path_str[0] = str(wav_path)
            t_conv    = time.perf_counter()
            converted = await _convert_to_wav(str(raw_path), str(wav_path))
            conv_ms   = int((time.perf_counter() - t_conv) * 1000)

            if converted:
                transcribe_path = str(wav_path)
                logger.info(f"[FFMPEG] [{interview_id}] WAV conversion: {conv_ms}ms")
            else:
                logger.warning(f"[FFMPEG] [{interview_id}] WAV conversion failed — using original")
                transcribe_path = str(raw_path)
                wav_path_str[0] = None

            if _is_cancelled(interview_id):
                return

            # ── STT ───────────────────────────────────────────────────────────
            t_stt = time.perf_counter()
            loop  = asyncio.get_event_loop()
            try:
                transcript = await loop.run_in_executor(
                    None, stt_service.transcribe, transcribe_path
                )
            except Exception as e:
                logger.exception(f"[STT] [{interview_id}] Transcription failed: {e}")
                await _safe_emit(sid, 'error', {
                    'interview_id': interview_id,
                    'message': (
                        'Unable to process audio. Please try recording again. '
                        'If this persists, check that your microphone is working correctly.'
                    ),
                })
                return

            stt_ms = int((time.perf_counter() - t_stt) * 1000)
            logger.info(f"[STT] [{interview_id}] {stt_ms}ms → '{transcript[:80]}'")

            _delete_file_safe(str(raw_path))
            if wav_path_str[0]:
                _delete_file_safe(wav_path_str[0])

            if not transcript or len(transcript.strip()) < 2:
                await _safe_emit(sid, 'error', {
                    'interview_id': interview_id,
                    'message': 'Could not detect speech in your recording. '
                               'Please speak clearly and try again.',
                })
                return

            await _safe_emit(sid, 'transcript',
                             {'interview_id': interview_id, 'transcript': transcript})

            if _is_cancelled(interview_id):
                return

            # ── Persist interaction ───────────────────────────────────────────
            current_q = (engine.state.questions_asked[-1]
                         if engine.state.questions_asked else '')
            await loop.run_in_executor(None, engine.process_answer, transcript)

            async def _save_interaction():
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: _db.save_interview_interaction(
                            interview_id=interview_id,
                            question=current_q,
                            answer=transcript,
                            transcript=transcript,
                            audio_path='[deleted]',
                            scores={},
                        )
                    )
                except Exception as e:
                    logger.error(f"[DB] save_interaction failed: {e}")

            asyncio.create_task(_save_interaction())

            if _is_cancelled(interview_id):
                return

            # ── Generate next question or closing ─────────────────────────────
            count    = len(engine.state.questions_asked)
            is_final = not engine.should_continue_interview(elapsed, count)

            _current_status[0] = 'Generating next question…'
            await _safe_emit(sid, 'status',
                             {'interview_id': interview_id,
                              'message': 'Generating next question…'})

            t_llm = time.perf_counter()
            if is_final:
                ai_text = await loop.run_in_executor(None, engine.generate_closing_statement)
            else:
                ai_text = await loop.run_in_executor(
                    None, engine.generate_follow_up_question, transcript, count + 1
                )
            llm_ms = int((time.perf_counter() - t_llm) * 1000)
            logger.info(f"[LLM] [{interview_id}] Q{count+1} in {llm_ms}ms: '{ai_text[:80]}'")

            _keepalive_running[0] = False
            if _keepalive_task[0]:
                _keepalive_task[0].cancel()
                _keepalive_task[0] = None

            if _is_cancelled(interview_id):
                return

            # ── Inline TTS for closing statement ──────────────────────────────
            inline_audio_b64 = None
            inline_audio_fmt = 'wav'
            inline_tts_ms    = None

            if is_final and not _is_cancelled(interview_id):
                await _safe_emit(sid, 'status',
                                 {'interview_id': interview_id,
                                  'message': 'Generating closing audio…'})
                t_inline_tts = time.perf_counter()
                inline_audio_b64, inline_audio_fmt = await _tts_b64(ai_text, interview_id)
                inline_tts_ms    = int((time.perf_counter() - t_inline_tts) * 1000)
                inline_audio_fmt = inline_audio_fmt or 'wav'
                logger.info(
                    f"[TTS] [{interview_id}] Closing TTS inline: "
                    f"{inline_tts_ms}ms audio={'YES' if inline_audio_b64 else 'NONE'}"
                )

            await _safe_emit(sid, 'next_question', {
                'interview_id':    interview_id,
                'question':        ai_text,
                'audio':           inline_audio_b64,
                'audio_format':    inline_audio_fmt,
                'is_final':        is_final,
                'question_count':  count,
                'interview_stage': engine.state.stage,
                'timing': {
                    'stt_ms':   stt_ms,
                    'llm_ms':   llm_ms,
                    'tts_ms':   inline_tts_ms,
                    'total_ms': int((time.perf_counter() - t_bg) * 1000),
                },
            })

            # ── Backup TTS task ───────────────────────────────────────────────
            async def _deliver_audio():
                if is_final and inline_audio_b64 is not None:
                    logger.debug(
                        f"[TTS] [{interview_id}] Closing audio already sent inline — "
                        f"skipping backup task"
                    )
                    return
                if _is_cancelled(interview_id):
                    return
                t_tts = time.perf_counter()
                await _safe_emit(sid, 'status',
                                 {'interview_id': interview_id,
                                  'message': 'Generating audio…'})
                audio_b64, audio_fmt = await _tts_b64(ai_text, interview_id)
                tts_ms = int((time.perf_counter() - t_tts) * 1000)
                if not _is_cancelled(interview_id):
                    await _safe_emit(sid, 'next_question_audio', {
                        'interview_id': interview_id,
                        'audio':        audio_b64,
                        'audio_format': audio_fmt or 'wav',
                        'is_final':     is_final,
                    })
                logger.info(f"[TTS] [{interview_id}] backup task {tts_ms}ms "
                            f"audio={'YES' if audio_b64 else 'NONE'}")

            asyncio.create_task(_deliver_audio())

        except Exception as e:
            logger.exception(f"[PROCESS] [{interview_id}] Unexpected error: {e}")
            await _safe_emit(sid, 'error', {
                'interview_id': interview_id,
                'message': 'An unexpected error occurred. Please try again.',
            })
        finally:
            _keepalive_running[0] = False
            if _keepalive_task[0]:
                try:
                    _keepalive_task[0].cancel()
                except Exception:
                    pass
            _PROCESSING_INTERVIEWS.discard(interview_id)
            _delete_file_safe(str(raw_path))
            if wav_path_str[0]:
                _delete_file_safe(wav_path_str[0])
            total_ms = int((time.perf_counter() - t_bg) * 1000)
            logger.info(f"[PROCESS] [{interview_id}] Task done in {total_ms}ms")

    asyncio.create_task(_process())


# ── Socket.IO event handlers ───────────────────────────────────────────────────

@sio.event
async def connect(sid, environ, auth=None):
    transport = environ.get('HTTP_UPGRADE', 'polling')
    logger.info(f"[WS] Connected  sid={sid}  transport={transport}")


@sio.event
async def disconnect(sid):
    logger.info(f"[WS] Disconnected  sid={sid}")
    SID_TO_INTERVIEW.pop(sid, None)


@sio.event
async def join_interview(sid, data):
    interview_id = (data.get('interview_id') or '').strip()
    if not interview_id or not sid:
        return
    await sio.enter_room(sid, interview_id)
    SID_TO_INTERVIEW[sid] = interview_id
    logger.info(f"[WS] sid={sid} joined room interview_id={interview_id}")
    await sio.emit('joined_interview', {'interview_id': interview_id, 'sid': sid}, to=sid)


@sio.event
async def audio_chunk(sid, data):
    """Buffer audio chunks streamed from the client."""
    interview_id = (data.get('interview_id') or '').strip()
    chunk_b64    = data.get('data', '')
    if not interview_id or not chunk_b64:
        return
    try:
        chunk_bytes = base64.b64decode(chunk_b64)
    except Exception:
        return
    AUDIO_BUFFERS.setdefault(interview_id, []).append(chunk_bytes)


@sio.event
async def audio_end(sid, data):
    """Flush buffered chunks and begin processing."""
    interview_id = (data.get('interview_id') or '').strip()
    if not interview_id:
        await sio.emit('error', {'interview_id': '', 'message': 'interview_id required'}, to=sid)
        return
    chunks = AUDIO_BUFFERS.pop(interview_id, [])
    if not chunks:
        await sio.emit('error', {'interview_id': interview_id,
                                  'message': 'No audio received. Please use a supported browser.'},
                       to=sid)
        return
    audio_bytes = b''.join(chunks)
    await _handle_audio_bytes(
        sid, interview_id, audio_bytes, extension='webm',
        elapsed=int(data.get('elapsed_time') or 0)
    )


@sio.event
async def audio_upload(sid, data):
    """Primary audio upload — full base64 blob."""
    interview_id = (data.get('interview_id') or '').strip()
    audio_b64    = data.get('audio_data', '')
    extension    = (data.get('extension') or 'webm').strip().lower()
    elapsed      = int(data.get('elapsed_time') or 0)

    if not interview_id:
        await sio.emit('error', {'interview_id': '', 'message': 'interview_id required'}, to=sid)
        return

    engine = ENGINES.get(interview_id)
    if not engine:
        await sio.emit('error', {'interview_id': interview_id,
                                  'message': 'Interview session not found. Please refresh and try again.'},
                       to=sid)
        return

    if not audio_b64:
        await sio.emit('error', {'interview_id': interview_id,
                                  'message': 'No audio data received.'}, to=sid)
        return

    if interview_id in _PROCESSING_INTERVIEWS:
        logger.warning(f"[UPLOAD] [{interview_id}] Already processing — dropping duplicate.")
        await sio.emit('status', {'interview_id': interview_id,
                                   'message': 'Still processing your previous answer…'}, to=sid)
        return

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        logger.warning(f"[UPLOAD] [{interview_id}] Base64 decode failed: {e}")
        await sio.emit('error', {'interview_id': interview_id,
                                  'message': 'Audio upload was corrupted. Please try again.'}, to=sid)
        return

    await _handle_audio_bytes(sid, interview_id, audio_bytes, extension, elapsed)


@sio.event
async def end_interview(sid, data):
    """End the interview and generate a final evaluation."""
    interview_id = (data.get('interview_id') or '').strip()
    if not interview_id:
        await sio.emit('error', {'interview_id': '', 'message': 'interview_id required'}, to=sid)
        return

    CANCELLED_INTERVIEWS.add(interview_id)
    AUDIO_BUFFERS.pop(interview_id, None)
    logger.info(f"[END] [{interview_id}] Cancellation flag set")

    async def _finalize():
        if interview_id in _EVALUATING_INTERVIEWS:
            logger.warning(f"[END] [{interview_id}] Already evaluating — dropping duplicate")
            return
        _EVALUATING_INTERVIEWS.add(interview_id)

        async def _emit_to_room(event, payload):
            await sio.emit(event, payload, room=interview_id)

        try:
            await _emit_to_room('status', {
                'interview_id': interview_id,
                'message': 'Generating final evaluation…',
            })

            # Wait for any in-flight processing to finish
            _wait_start = time.perf_counter()
            while interview_id in _PROCESSING_INTERVIEWS:
                await asyncio.sleep(0.2)
                if time.perf_counter() - _wait_start > 30:
                    logger.warning(f"[END] [{interview_id}] Timed out waiting for processing — proceeding")
                    break

            loop   = asyncio.get_event_loop()
            engine = ENGINES.get(interview_id)

            interactions = await loop.run_in_executor(None, _db.get_all_interactions, interview_id)
            logger.info(f"[END] [{interview_id}] {len(interactions)} interactions")

            if engine:
                name = engine.candidate_info.get('name', 'Candidate')
                jd   = engine.candidate_info.get('job_description', '')
            else:
                session = await loop.run_in_executor(None, _db.get_interview_session, interview_id)
                if not session:
                    await _emit_to_room('error', {
                        'interview_id': interview_id,
                        'message': 'Interview session not found.',
                    })
                    return
                cand = await loop.run_in_executor(
                    None, _db.get_candidate_by_id, session.get('candidate_id'))
                name = (cand or {}).get('name', 'Candidate')
                jd   = (cand or {}).get('job_description', '')

            t_eval = time.perf_counter()

            def _run_eval():
                evaluator  = EvaluatorService()
                evaluation = evaluator.generate_final_evaluation(interactions, name, jd)
                evaluation['interview_status'] = 'completed'
                return _db.save_final_evaluation(interview_id, evaluation)

            saved_doc = await loop.run_in_executor(None, _run_eval)
            eval_ms   = int((time.perf_counter() - t_eval) * 1000)

            logger.info(f"[END] [{interview_id}] eval in {eval_ms}ms "
                        f"overall={saved_doc.get('overall_score')} "
                        f"rec={saved_doc.get('recommendation')}")

            await _emit_to_room('evaluation_ready', {
                'interview_id': interview_id,
                'evaluation':   saved_doc,
            })

        except Exception as e:
            logger.exception(f"[END] [{interview_id}] Failed: {e}")
            await sio.emit('error', {
                'interview_id': interview_id,
                'message': 'Failed to generate evaluation. Please try again.',
            }, room=interview_id)
        finally:
            _cleanup_session(interview_id)

    asyncio.create_task(_finalize())


@sio.event
async def cancel_interview(sid, data):
    """Cancel the interview and generate a partial evaluation."""
    interview_id = (data.get('interview_id') or '').strip()
    if not interview_id:
        await sio.emit('error', {'interview_id': '', 'message': 'interview_id required'}, to=sid)
        return

    CANCELLED_INTERVIEWS.add(interview_id)
    AUDIO_BUFFERS.pop(interview_id, None)
    logger.info(f"[CANCEL] [{interview_id}] Cancellation flag set")

    async def _cancel():
        if interview_id in _EVALUATING_INTERVIEWS:
            logger.warning(f"[CANCEL] [{interview_id}] Already evaluating — dropping duplicate")
            return
        _EVALUATING_INTERVIEWS.add(interview_id)

        async def _emit_to_room(event, payload):
            await sio.emit(event, payload, room=interview_id)

        try:
            _wait_start = time.perf_counter()
            while interview_id in _PROCESSING_INTERVIEWS:
                await asyncio.sleep(0.2)
                if time.perf_counter() - _wait_start > 30:
                    logger.warning(f"[CANCEL] [{interview_id}] Timed out waiting for processing — proceeding")
                    break

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _db.mark_interview_cancelled, interview_id)

            engine       = ENGINES.get(interview_id)
            interactions = await loop.run_in_executor(None, _db.get_all_interactions, interview_id)

            if engine:
                name = engine.candidate_info.get('name', 'Candidate')
                jd   = engine.candidate_info.get('job_description', '')
            else:
                session = await loop.run_in_executor(None, _db.get_interview_session, interview_id)
                cand    = None
                if session:
                    cand = await loop.run_in_executor(
                        None, _db.get_candidate_by_id, session.get('candidate_id'))
                name = (cand or {}).get('name', 'Candidate')
                jd   = (cand or {}).get('job_description', '')

            def _run_eval():
                evaluator  = EvaluatorService()
                evaluation = evaluator.generate_final_evaluation(interactions, name, jd)
                evaluation['interview_status'] = 'cancelled'
                return _db.save_final_evaluation(interview_id, evaluation)

            saved_doc = await loop.run_in_executor(None, _run_eval)

            logger.info(f"[CANCEL] [{interview_id}] Done — interactions={len(interactions)}")
            await _emit_to_room('evaluation_ready', {
                'interview_id': interview_id,
                'evaluation':   saved_doc,
            })

        except Exception as e:
            logger.exception(f"[CANCEL] [{interview_id}] Failed: {e}")
            await sio.emit('error', {
                'interview_id': interview_id,
                'message': 'Failed to generate evaluation.',
            }, room=interview_id)
        finally:
            _cleanup_session(interview_id)

    asyncio.create_task(_cancel())