"""
Speech-to-Text service — v12 (Groq Whisper API)

MIGRATION FROM LOCAL WHISPER → GROQ API
========================================
What was removed (local Whisper code):
  - import whisper                          ← deleted
  - whisper.load_model(model_size)          ← deleted
  - model.transcribe(audio_path, ...)       ← deleted
  - get_whisper() / get_whisper_lock()      ← no longer imported
  - gevent threadpool.apply() wrapper       ← not needed; Groq is a fast HTTP call
  - STT_HARD_TIMEOUT_S / gevent.Timeout     ← replaced with requests-level timeout
  - _whisper_transcribe_lock                ← removed; Groq is stateless/concurrent-safe
  - ffmpeg dependency for WAV conversion    ← Groq Whisper accepts webm/ogg/mp3/wav natively

What was added:
  - groq Python SDK (pip install groq)
  - Groq client initialised once at module load (connection-pooled internally)
  - transcribe_audio(audio_path) module-level function (matches existing call-sites)
  - SpeechToTextService.transcribe() now delegates to transcribe_audio()
  - Structured error handling: missing file, empty file, bad API key, network, API errors
  - Returns "" on any failure (never raises, so callers need no changes)

Install:
  pip install groq>=0.9.0
"""

import logging
import os
import time
from pathlib import Path

# ── Groq SDK ──────────────────────────────────────────────────────────────────
# Official Groq Python SDK.  Install with: pip install groq>=0.9.0
from groq import Groq, AuthenticationError, APIConnectionError, APIStatusError

logger = logging.getLogger(__name__)

# ── Groq Whisper model to use ─────────────────────────────────────────────────
# whisper-large-v3-turbo: fastest Whisper model on Groq, optimised for low latency.
# Other options: "whisper-large-v3" (highest accuracy, slightly slower)
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# ── Groq client — module-level singleton ──────────────────────────────────────
# Initialised once at import time.  The SDK manages its own connection pool,
# so constructing a single client and reusing it is the correct pattern for
# production (avoids TCP handshake overhead on every request).
#
# Reads GROQ_API_KEY from the environment.  If the key is missing the client
# will still construct, but the first API call will raise AuthenticationError —
# caught below in transcribe_audio().
_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """
    Lazily construct and cache the Groq client.

    Using a module-level singleton means:
      - Only one HTTPS connection pool is created per process.
      - No overhead re-constructing the client on every transcription call.
      - Thread-safe: CPython's GIL protects the simple None check / assignment.
    """
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            # Fail early with a clear message rather than a cryptic 401 later.
            logger.error(
                "[STT] GROQ_API_KEY environment variable is not set. "
                "Set it in your .env file or shell before starting the server."
            )
            # Still construct the client so callers get an AuthenticationError
            # (not an AttributeError) — handled gracefully in transcribe_audio().
        _groq_client = Groq(api_key=api_key or None)
        logger.info("[STT] Groq client initialised (model=%s)", GROQ_WHISPER_MODEL)
    return _groq_client


# ── Public transcription function ─────────────────────────────────────────────

def transcribe_audio(audio_path: str, language: str = "en") -> str:
    """
    Transcribe an audio file using the Groq Whisper API.

    Takes an audio file path and returns the transcribed text.
    Returns an empty string on any error so callers never need to handle
    exceptions from this function.

    Supported formats (accepted natively by Groq — no ffmpeg pre-conversion needed):
      flac, mp3, mp4, mpeg, mpga, m4a, ogg, opus, wav, webm

    Args:
        audio_path: Absolute or relative path to the audio file.
        language:   BCP-47 language code (default "en").  Providing an explicit
                    language skips Groq's auto-detection step and reduces latency.

    Returns:
        Transcribed text string, or "" if transcription fails.
    """
    t0 = time.perf_counter()
    path = Path(audio_path)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    # Validate the file exists before opening — gives a clear log message
    # instead of a cryptic FileNotFoundError from inside the SDK.
    if not path.exists():
        logger.error("[STT] Audio file not found: %s", audio_path)
        return ""

    file_size = path.stat().st_size
    if file_size == 0:
        logger.error("[STT] Audio file is empty (0 bytes): %s", audio_path)
        return ""

    logger.info("[STT] Transcribing %s (%d bytes) via Groq …", path.name, file_size)

    # ── Call Groq Whisper API ─────────────────────────────────────────────────
    # The `with open(...)` block guarantees the file handle is closed even if
    # the API call raises an exception — prevents file descriptor leaks.
    try:
        client = _get_groq_client()

        with open(audio_path, "rb") as audio_file:
            # file= accepts a (filename, bytes) tuple so Groq can infer the
            # MIME type from the extension — important for correct decoding.
            transcription = client.audio.transcriptions.create(
                file=(path.name, audio_file.read()),   # read() + close() in one block
                model=GROQ_WHISPER_MODEL,
                response_format="json",                # returns .text attribute
                language=language,                     # skip auto-detect → lower latency
            )
        # File handle is closed here by the `with` block, regardless of outcome.

        transcript = (transcription.text or "").strip()
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "[STT] Done in %dms → %d chars: '%s'",
            elapsed_ms, len(transcript), transcript[:80],
        )
        return transcript

    # ── Structured error handling ─────────────────────────────────────────────

    except AuthenticationError as e:
        # 401 — API key is missing, revoked, or malformed.
        logger.error(
            "[STT] Groq authentication failed — check GROQ_API_KEY. "
            "Details: %s", e
        )
        return ""

    except APIConnectionError as e:
        # Network-level failure: DNS, TCP timeout, TLS error, etc.
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "[STT] Network error reaching Groq after %dms — "
            "check internet connectivity. Details: %s", elapsed_ms, e
        )
        return ""

    except APIStatusError as e:
        # 4xx / 5xx from Groq (rate limit, server error, bad request, …).
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.error(
            "[STT] Groq API error %d after %dms — %s",
            e.status_code, elapsed_ms, e.message
        )
        return ""

    except FileNotFoundError:
        # Shouldn't reach here (caught above), but guard against TOCTOU races.
        logger.error("[STT] Audio file disappeared before upload: %s", audio_path)
        return ""

    except Exception as e:
        # Catch-all: unexpected SDK changes, OS errors, etc.
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception(
            "[STT] Unexpected error after %dms transcribing %s: %s",
            elapsed_ms, path.name, e
        )
        return ""


# ── SpeechToTextService — drop-in replacement ─────────────────────────────────
# Keeps the same class + method interface used throughout socket_routes.py
# so no other files need modification:
#
#   from services.speech_to_text import stt_service
#   transcript = stt_service.transcribe(audio_path)   ← unchanged call-site

class SpeechToTextService:
    """
    Thin wrapper around transcribe_audio() that preserves the class-based
    interface expected by socket_routes.py.

    The heavy lifting (Groq client, error handling, logging) lives in
    transcribe_audio() so it can also be called directly if needed.
    """

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe audio file to text via Groq Whisper API.

        Interface is identical to the old local-Whisper version, so no
        call-sites in socket_routes.py need to change.

        Returns:
            Transcribed text, or "" on failure.

        Note:
            Unlike the old implementation this method does NOT raise
            exceptions — all errors are caught inside transcribe_audio()
            and logged.  Callers that previously wrapped this in try/except
            will continue to work correctly (the except block simply won't
            trigger on API failures).
        """
        return transcribe_audio(audio_path, language=language)


# ── Module-level singleton (matches existing import in socket_routes.py) ───────
stt_service = SpeechToTextService()
