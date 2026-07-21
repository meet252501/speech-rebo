
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import re
import numpy as _np
import threading
from faster_whisper import WhisperModel

# ===========================================================================
# Streaming speech-to-text draft — Hindi + English
#
# Architecture:
#   - whisper-tiny  → fast language detection (~100ms on M1 Pro)
#   - whisper-small → accurate transcription (~1-1.5s on M1 Pro)
#   - beam_size=3 for finals, beam_size=1 for partials
#   - No sample-specific prompts or target-term lists
# ===========================================================================

_model_small = None
_model_tiny = None

_small_lock = threading.Lock()
_tiny_lock = threading.Lock()

def get_small():
    """Main transcription model — whisper-small."""
    global _model_small
    if _model_small is None:
        with _small_lock:
            if _model_small is None:
                _model_small = WhisperModel(
                    "Systran/faster-whisper-small",
                    device="auto", compute_type="int8",
                    cpu_threads=4, local_files_only=False
                )
    return _model_small, _small_lock

def get_tiny():
    """Ultra-fast language detector — whisper-tiny."""
    global _model_tiny
    if _model_tiny is None:
        with _tiny_lock:
            if _model_tiny is None:
                _model_tiny = WhisperModel(
                    "Systran/faster-whisper-tiny",
                    device="auto", compute_type="int8",
                    cpu_threads=4, local_files_only=False
                )
    return _model_tiny, _tiny_lock


# ===========================================================================
# Postprocessing — clean up whitespace only, no domain-specific rewrites
# ===========================================================================

def _postprocess(text: str) -> str:
    """Clean up the final text output."""
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ===========================================================================
# Streaming state
# ===========================================================================

_lang_detected = False
_is_hinglish = False

def draft_reset():
    global _lang_detected, _is_hinglish
    _lang_detected = False
    _is_hinglish = False

def draft(chunk_bytes: bytes, is_final: bool) -> tuple[str, float]:
    global _lang_detected, _is_hinglish

    audio = _np.frombuffer(chunk_bytes, _np.int16).flatten().astype(_np.float32) / 32768.0

    # ---- Language detection at ~2s using tiny model ----
    if not _lang_detected and (len(audio) > 16000 * 2 or is_final):
        _lang_detected = True
        m, lk = get_tiny()
        with lk:
            _, info = m.transcribe(audio, beam_size=1, condition_on_previous_text=False, vad_filter=True)
            _is_hinglish = (info.language == "hi" or info.language == "ur")

    if not _lang_detected:
        return "", 0

    try:
        m, lk = get_small()
        with lk:
            if _is_hinglish:
                segs, _ = m.transcribe(
                    audio,
                    beam_size=3 if is_final else 1,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    language='hi'
                )
            else:
                segs, _ = m.transcribe(
                    audio,
                    beam_size=3 if is_final else 1,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                    vad_filter=True
                )
            text = _postprocess(" ".join(s.text for s in segs).strip())

        if text:
            if is_final:
                return text, len(text)
            else:
                # Commit all but the last ~20 chars to minimize churn
                return text, max(0, len(text) - 20)

        return "", 0
    except Exception:
        return "", 0

# ===========================================================================
# Warmup — ensure models download/load on import before network is blocked.
# MUST be synchronous so stream_server waits for them before printing READY.
# ===========================================================================
get_tiny()
get_small()

