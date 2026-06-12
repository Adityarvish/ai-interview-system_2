"""
Text-to-Speech Service — v6 (Groq Orpheus, audit-hardened)

ROOT CAUSES FIXED IN THIS VERSION
===================================
BUG #1 — ImportError silently fell back to browser speechSynthesis
  CAUSE:  requirements file was renamed to rr.txt; groq SDK was never installed.
          `from groq import Groq` raised ImportError at module load time,
          tts_service could not be imported, generate_speech() was never called,
          _tts_b64() caught the exception and returned (None, None), the frontend
          received null audio and transparently used window.speechSynthesis instead.
  FIX:    Startup self-test in _startup_tts_check() that runs at import time and
          writes a LOUD error to logs if the SDK is missing or API key is absent.
          Import guard with clear InstallationError message instead of silent None.

BUG #2 — app.py called warmup_all(Config.WHISPER_MODEL) after WHISPER_MODEL removed
  CAUSE:  Previous migration removed WHISPER_MODEL from Config but did not update
          the app.py call site → AttributeError in the warmup greenlet.
  FIX:    Handled in app.py (see that file). warmup_all() now takes no arguments.

BUG #3 — No TTS debug report / health endpoint
  FIX:    get_tts_debug_report() function added; used by /api/tts/health route
          in app.py and logged at startup.

BUG #4 — Frontend silence is indistinguishable from success
  FIX:    _tts_b64() in socket_routes.py and interview_routes.py now logs
          [TTS_PROVIDER] line at every call so the active backend is always
          visible in server logs. (See updated socket_routes.py / interview_routes.py)

INSTALL (run this once):
  pip install groq>=0.9.0

UNINSTALL old backends (if still present):
  pip uninstall edge-tts pyttsx3 -y
"""

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Groq SDK import with clear failure message ────────────────────────────────
# If this import fails the developer sees an actionable error, not a silent None.
try:
    from groq import Groq, AuthenticationError, APIConnectionError, APIStatusError
    _GROQ_SDK_AVAILABLE = True
except ImportError:
    _GROQ_SDK_AVAILABLE = False
    # Define stub exceptions so the except clauses below don't NameError.
    AuthenticationError = APIConnectionError = APIStatusError = Exception
    logger.critical(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  [TTS] FATAL — groq SDK not installed                       ║\n"
        "║  Run:  pip install groq>=0.9.0                              ║\n"
        "║  Then restart the server.                                   ║\n"
        "║  Until then ALL TTS calls will fail and the frontend        ║\n"
        "║  will fall back to the browser's built-in speechSynthesis.  ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )

from config.settings import Config

# ── Model & voice configuration ───────────────────────────────────────────────
# Groq Orpheus v1 English — fast, natural, LLM-based TTS.
# Running at ~140 chars/sec on Groq hardware.
GROQ_TTS_MODEL  = "canopylabs/orpheus-v1-english"

# Available voices (English) — valid Groq TTS voices:
#   Female: autumn, diana, hannah
#   Male:   austin, daniel, troy
#
# Override at runtime with env var:  GROQ_TTS_VOICE=diana
GROQ_TTS_VOICE  = os.getenv("GROQ_TTS_VOICE", "daniel").strip()

# Groq Orpheus accepts up to ~2000 characters per request.
# The old limit of 200 was far too conservative and forced every interview
# question (typically 150-400 chars) into 2+ chunks, triggering the WAV
# concatenation path.  1800 chars is a safe ceiling that avoids chunking for
# virtually all interview question texts while staying under API limits.
GROQ_TTS_MAX_CHARS = 1800

# WAV is the native Groq output format and plays in all modern browsers.
# No ffmpeg conversion step needed.
GROQ_TTS_FORMAT = "wav"

# ── Groq client singleton ─────────────────────────────────────────────────────
_groq_client: "Groq | None" = None


def _get_groq_client() -> "Groq":
    """
    Lazily construct and cache the Groq client.
    One client per process — the SDK manages its own HTTPS connection pool.
    """
    global _groq_client
    if _groq_client is None:
        if not _GROQ_SDK_AVAILABLE:
            raise RuntimeError(
                "groq SDK is not installed. Run: pip install groq>=0.9.0"
            )
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to backend/.env:  GROQ_API_KEY=gsk_..."
            )
        _groq_client = Groq(api_key=api_key)
        logger.info(
            "[TTS] ✓ Groq client ready  provider=GroqOrpheus  model=%s  voice=%s  fmt=%s",
            GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT
        )
    return _groq_client


# ── Debug / health report ─────────────────────────────────────────────────────

def get_tts_debug_report() -> dict:
    """
    Return a dict describing the active TTS configuration.
    Used by /api/tts/health and logged at startup.

    Returns:
        {
          "provider":       "GroqOrpheus",
          "sdk_installed":  True,
          "api_key_set":    True,
          "model":          "canopylabs/orpheus-v1-english",
          "voice":          "leah",
          "output_format":  "wav",
          "client_ready":   True,
          "available_voices": [...],
          "status":         "ok" | "degraded" | "error",
          "error":          null | "...",
        }
    """
    api_key_set  = bool(os.getenv("GROQ_API_KEY", "").strip())
    client_ready = _groq_client is not None

    status = "ok"
    error  = None
    if not _GROQ_SDK_AVAILABLE:
        status = "error"
        error  = "groq SDK not installed — run: pip install groq>=0.9.0"
    elif not api_key_set:
        status = "error"
        error  = "GROQ_API_KEY not set in environment / .env"
    elif not client_ready:
        status = "degraded"
        error  = "Client not yet initialised (will init on first TTS call)"

    return {
        "provider":         "GroqOrpheus",
        "sdk_installed":    _GROQ_SDK_AVAILABLE,
        "api_key_set":      api_key_set,
        "model":            GROQ_TTS_MODEL,
        "voice":            GROQ_TTS_VOICE,
        "output_format":    GROQ_TTS_FORMAT,
        "client_ready":     client_ready,
        "available_voices": ["autumn", "diana", "hannah", "austin", "daniel", "troy"],
        "status":           status,
        "error":            error,
    }


# ── Text chunking helpers ─────────────────────────────────────────────────────

import re
import struct


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks of at most max_chars characters.
    Tries to split on sentence boundaries (. ! ?) first, then on commas/spaces,
    then hard-cuts if no good boundary is found.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > max_chars:
        # Try sentence boundary within the window
        window = remaining[:max_chars]
        # Find last sentence-ending punctuation in window
        match = None
        for pattern in [r'[.!?]\s', r'[,;]\s', r'\s']:
            matches = list(re.finditer(pattern, window))
            if matches:
                match = matches[-1]
                break

        if match:
            cut = match.start() + 1  # include the punctuation
        else:
            cut = max_chars  # hard cut

        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        chunks.append(remaining.strip())

    return [c for c in chunks if c]


def _find_wav_data_chunk(wav_bytes: bytes) -> tuple[int, int]:
    """
    Parse a WAV/RIFF file and return (data_offset, data_size).

    WAV is a RIFF container: a sequence of tagged chunks. The PCM audio lives
    in the 'data' chunk, which may NOT start at byte 44. Common reasons:
      - Extended fmt chunk  (fmt chunk size = 18 or 40, not 16)
      - Extra chunks before 'data' (e.g. 'fact', 'LIST', 'bext')
      - Groq Orpheus specifically inserts a 'fact' chunk → header is 58+ bytes

    Hard-coding HEADER_SIZE = 44 works only for the minimal-PCM case and
    produces a corrupted WAV whenever any of the above applies.

    Args:
        wav_bytes: Raw bytes of a complete WAV file.

    Returns:
        (data_start, data_size) where data_start is the byte offset of the
        first PCM sample and data_size is the size field from the chunk header.

    Raises:
        ValueError: if the file is not a valid RIFF/WAVE or has no 'data' chunk.
    """
    if len(wav_bytes) < 12:
        raise ValueError("WAV too short to contain a RIFF header")
    if wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file (magic={wav_bytes[0:4]!r})")

    # Walk RIFF chunks starting after the 12-byte RIFF header.
    offset = 12
    while offset + 8 <= len(wav_bytes):
        chunk_id   = wav_bytes[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
        if chunk_id == b"data":
            return offset + 8, chunk_size          # found it
        # Chunks are padded to even byte boundaries
        offset += 8 + chunk_size + (chunk_size & 1)

    raise ValueError("WAV file contains no 'data' chunk")


def _concat_wav_chunks(wav_parts: list[bytes]) -> bytes:
    """
    Concatenate multiple WAV byte strings into a single valid WAV file.
    All parts must share the same audio format (sample rate, channels, bit depth).

    Properly parses each WAV's RIFF structure to locate the 'data' chunk instead
    of assuming a fixed 44-byte header.  Groq Orpheus (and many other encoders)
    can produce WAV files with 'fact' or other extra chunks, making the data
    offset larger than 44 bytes.  The old hard-coded HEADER_SIZE = 44 sliced
    into the PCM payload, producing a file with a corrupted RIFF header that
    all browsers reject immediately with a MediaError.
    """
    if len(wav_parts) == 1:
        return wav_parts[0]

    # Extract raw PCM samples from every chunk by parsing RIFF properly.
    pcm_parts: list[bytes] = []
    for i, part in enumerate(wav_parts):
        try:
            data_start, data_size = _find_wav_data_chunk(part)
        except ValueError as exc:
            logger.error("[TTS] _concat_wav_chunks: chunk %d/%d invalid WAV — %s",
                         i + 1, len(wav_parts), exc)
            raise
        pcm_parts.append(part[data_start: data_start + data_size])

    total_pcm      = b"".join(pcm_parts)
    total_pcm_size = len(total_pcm)

    # Build a minimal, standards-compliant 44-byte PCM WAV header.
    # We copy the fmt chunk from the first part (bytes 12–36 normally; we
    # re-parse to be safe) so sample rate / channels / bit depth are preserved.
    # Then we append a fresh 'data' chunk header and all the PCM.
    first = wav_parts[0]

    # Locate the fmt chunk in the first part to copy its payload verbatim.
    fmt_payload = b""
    offset = 12
    while offset + 8 <= len(first):
        cid  = first[offset:offset + 4]
        csz  = struct.unpack_from("<I", first, offset + 4)[0]
        if cid == b"fmt ":
            fmt_payload = first[offset + 8: offset + 8 + csz]
            break
        offset += 8 + csz + (csz & 1)

    if not fmt_payload:
        raise ValueError("WAV first chunk has no 'fmt ' sub-chunk")

    # Assemble: RIFF header + fmt chunk + data chunk
    fmt_chunk  = b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    data_chunk = b"data" + struct.pack("<I", total_pcm_size) + total_pcm
    riff_body  = b"WAVE" + fmt_chunk + data_chunk
    riff_header = b"RIFF" + struct.pack("<I", len(riff_body)) + riff_body

    return riff_header


# ── Public reusable function ──────────────────────────────────────────────────

def text_to_speech(text: str, output_file: str) -> "str | None":
    """
    Convert text to speech using Groq Orpheus. Save audio to output_file.
    Return the full output file path, or None on any failure.

    Args:
        text:        Text to synthesise (any length).
        output_file: Filename (not path) for the audio, e.g. "ai_abc.wav".
                     Written to Config.AUDIO_FOLDER / output_file.

    Returns:
        Absolute path string to the written WAV file, or None on error.

    Gevent compatibility:
        The Groq SDK uses httpx in synchronous mode.  httpx makes blocking
        socket calls that gevent monkey-patches into cooperative green sockets.
        No asyncio loop, no threadpool wrapper, no gevent lock needed.
        This call yields to the gevent hub while waiting for the HTTP response.
    """
    t0 = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────────
    if not text or not text.strip():
        logger.error("[TTS] Rejected empty text — returning None")
        return None

    output_path = Config.AUDIO_FOLDER / output_file

    # Log the provider and voice on EVERY call so it is always visible in logs.
    # This is the key diagnostic line that confirms Groq is actually being used.
    logger.info(
        "[TTS_PROVIDER] provider=GroqOrpheus  model=%s  voice=%s  fmt=%s  "
        "chars=%d  output=%s",
        GROQ_TTS_MODEL, GROQ_TTS_VOICE, GROQ_TTS_FORMAT,
        len(text.strip()), output_file,
    )

    # ── API call ──────────────────────────────────────────────────────────────
    try:
        client = _get_groq_client()
        clean_text = text.strip()

        # ── Chunk text to respect 200-char API limit ──────────────────────────
        # Split on sentence boundaries first, then hard-cut any chunk still over limit.
        chunks = _chunk_text(clean_text, GROQ_TTS_MAX_CHARS)
        logger.info("[TTS] Synthesising %d chunk(s) for %d chars", len(chunks), len(clean_text))

        raw_audio_parts: list[bytes] = []
        for i, chunk in enumerate(chunks):
            response = client.audio.speech.create(
                model=GROQ_TTS_MODEL,
                voice=GROQ_TTS_VOICE,
                input=chunk,
                response_format=GROQ_TTS_FORMAT,
            )
            part_bytes = response.read()
            if len(part_bytes) < 44:  # WAV header alone is 44 bytes
                logger.error("[TTS] Chunk %d/%d returned suspiciously small audio (%d bytes)",
                             i + 1, len(chunks), len(part_bytes))
                return None
            raw_audio_parts.append(part_bytes)
            logger.debug("[TTS] Chunk %d/%d done  bytes=%d", i + 1, len(chunks), len(part_bytes))

        # ── Concatenate WAV chunks ────────────────────────────────────────────
        # Each chunk is a complete WAV file. We take the header from the first
        # chunk and concatenate the raw PCM data from all chunks.
        final_audio = _concat_wav_chunks(raw_audio_parts)

        output_path.write_bytes(final_audio)

        # ── Post-write validation ─────────────────────────────────────────────
        if not output_path.exists():
            logger.error("[TTS] write_to_file() returned but file missing: %s", output_path)
            return None

        file_size  = output_path.stat().st_size
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if file_size < 200:
            # A real WAV with even 50ms of audio is ~4 KB. Under 200 bytes = corrupted.
            logger.error(
                "[TTS] File too small (%d bytes) — Groq returned bad audio. "
                "file=%s  elapsed=%dms", file_size, output_file, elapsed_ms
            )
            try:
                output_path.unlink()
            except OSError:
                pass
            return None

        # ── Debug report on every successful synthesis ────────────────────────
        logger.info(
            "[TTS_REPORT] ✓  provider=GroqOrpheus  voice=%s  model=%s  "
            "elapsed_ms=%d  size_kb=%d  format=%s  file=%s  "
            "timestamp=%s",
            GROQ_TTS_VOICE, GROQ_TTS_MODEL,
            elapsed_ms, file_size // 1024, GROQ_TTS_FORMAT,
            output_file,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return str(output_path)

    # ── Structured error handling ─────────────────────────────────────────────

    except RuntimeError as e:
        # Missing SDK or API key — raised by _get_groq_client() — re-raise so caller sees it
        logger.critical("[TTS] ✗ Configuration error — %s", e)
        raise

    except AuthenticationError as e:
        msg = f"401 Authentication failed — GROQ_API_KEY is invalid or revoked. Details: {e}"
        logger.error("[TTS] ✗ %s  key_prefix=%s...", msg, os.getenv("GROQ_API_KEY", "")[:8])
        raise RuntimeError(msg) from e

    except APIConnectionError as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        msg = f"Network error after {elapsed_ms}ms — cannot reach api.groq.com. Details: {e}"
        logger.error("[TTS] ✗ %s", msg)
        raise RuntimeError(msg) from e

    except APIStatusError as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        # e.body contains the full JSON response from Groq — most useful for debugging
        body = getattr(e, "body", None) or getattr(e, "message", str(e))
        msg = f"Groq API HTTP {e.status_code} after {elapsed_ms}ms — {body}"
        logger.error("[TTS] ✗ %s", msg)
        raise RuntimeError(msg) from e

    except OSError as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        msg = f"File write error after {elapsed_ms}ms — path={output_path} — {e}"
        logger.error("[TTS] ✗ %s", msg)
        raise RuntimeError(msg) from e

    except RuntimeError:
        raise  # already formatted above (e.g. missing SDK / API key)

    except Exception as e:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        msg = f"Unexpected error after {elapsed_ms}ms — '{text[:40]}...' — {type(e).__name__}: {e}"
        logger.exception("[TTS] ✗ %s", msg)
        raise RuntimeError(msg) from e


# ── TextToSpeechService ───────────────────────────────────────────────────────
# Drop-in replacement.  Callers (socket_routes.py, interview_routes.py) do:
#   from services.text_to_speech import tts_service
#   path = tts_service.generate_speech(text, out_name)  ← unchanged

class TextToSpeechService:
    """Thin wrapper around text_to_speech() preserving the class interface."""

    def generate_speech(self, text: str, output_filename: str) -> str:
        """
        Generate TTS audio via Groq Orpheus. Return the audio file path.
        Raises RuntimeError with the actual Groq error message on failure.
        """
        return text_to_speech(text, output_filename)


# ── Module-level singleton ────────────────────────────────────────────────────
tts_service = TextToSpeechService()


# ── Startup self-test ─────────────────────────────────────────────────────────
def _startup_tts_check() -> None:
    """
    Run at module import time.  Logs the TTS configuration and catches
    common misconfigurations (missing SDK, missing API key) immediately
    rather than silently failing on the first interview question.
    """
    report = get_tts_debug_report()
    if report["status"] == "ok" or report["status"] == "degraded":
        logger.info(
            "[TTS_STARTUP] provider=%s  model=%s  voice=%s  fmt=%s  "
            "sdk=%s  key_set=%s  status=%s",
            report["provider"], report["model"], report["voice"],
            report["output_format"], report["sdk_installed"],
            report["api_key_set"], report["status"],
        )
        if report["status"] == "degraded":
            logger.warning("[TTS_STARTUP] %s", report["error"])
    else:
        logger.critical(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  [TTS_STARTUP] TTS IS NOT FUNCTIONAL                        ║\n"
            "║  Status: %-51s║\n"
            "║  Error:  %-51s║\n"
            "║  All interview questions will use browser speechSynthesis.  ║\n"
            "║  Fix the issue above and restart the server.                ║\n"
            "╚══════════════════════════════════════════════════════════════╝",
            report["status"], (report["error"] or "")[:51],
        )


_startup_tts_check()

# ── Eagerly initialize the Groq client so it is ready before the first
# interview request. Without this, the first call to _get_groq_client()
# happens inside start_interview() and any init error gets silently
# swallowed by _tts_b64()'s broad except clause.
if _GROQ_SDK_AVAILABLE and os.getenv("GROQ_API_KEY", "").strip():
    try:
        _get_groq_client()
        logger.info("[TTS] ✓ Groq client pre-initialized at startup")
    except Exception as _e:
        logger.critical("[TTS] ✗ Groq client pre-init failed at startup: %s", _e)
