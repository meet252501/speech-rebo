
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import re
import numpy as _np
import threading
from faster_whisper import WhisperModel

# ===========================================================================
# Streaming speech-to-text draft — Hindi + English
#
# Architecture (mirrors RambleFix: fast drafter + accurate finalizer):
#   - whisper_tiny_ct2    → fast partials (~100ms on M1 Pro)
#   - shunyalabs_zero_stt → accurate finals (Hindi-English specialist)
#   - Background pre-transcription: shunyalabs runs in a bg thread during
#     streaming so the final is often pre-cached → near-zero E2F latency
#   - No language detection delay — start transcribing immediately
#   - beam_size=1 everywhere for maximum speed
#   - No sample-specific prompts or glossaries
# ===========================================================================

_model_tiny = None
_model_shunyalabs = None

_tiny_lock = threading.Lock()
_shunyalabs_lock = threading.Lock()


def get_tiny():
    """Ultra-fast partial drafter — whisper-tiny (bundled, 75MB)."""
    global _model_tiny
    if _model_tiny is None:
        with _tiny_lock:
            if _model_tiny is None:
                _model_tiny = WhisperModel(
                    "whisper_tiny_ct2",
                    device="auto", compute_type="int8",
                    cpu_threads=4, local_files_only=True
                )
    return _model_tiny, _tiny_lock


def get_shunyalabs():
    """Accurate Hindi-English finalizer — shunyalabs (bundled, 769MB)."""
    global _model_shunyalabs
    if _model_shunyalabs is None:
        with _shunyalabs_lock:
            if _model_shunyalabs is None:
                _model_shunyalabs = WhisperModel(
                    "shunyalabs_zero_stt_ct2",
                    device="auto", compute_type="int8",
                    cpu_threads=4, local_files_only=True
                )
    return _model_shunyalabs, _shunyalabs_lock


# ===========================================================================
# Postprocessing — clean whitespace only, no domain-specific rewrites
# ===========================================================================

def _postprocess(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ===========================================================================
# Background pre-transcription state
# ===========================================================================

_bg_result = None          # latest shunyalabs transcription (cached)
_bg_result_lock = threading.Lock()
_bg_thread = None          # current background transcription thread
_bg_audio_len = 0          # audio length that bg_result corresponds to


def _bg_transcribe(audio_float: _np.ndarray, audio_len: int):
    """Run shunyalabs in background thread and cache the result."""
    global _bg_result, _bg_audio_len
    try:
        m, lk = get_shunyalabs()
        with lk:
            segs, _ = m.transcribe(
                audio_float,
                beam_size=1,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True
            )
            text = _postprocess(" ".join(s.text for s in segs).strip())
        with _bg_result_lock:
            # Only update if this is for more audio than the current cache
            if audio_len >= _bg_audio_len:
                _bg_result = text
                _bg_audio_len = audio_len
    except Exception:
        pass  # never crash the background thread


# ===========================================================================
# Streaming state
# ===========================================================================

_last_bg_kick = 0  # audio length when we last kicked off a bg thread

def draft_reset():
    global _bg_result, _bg_audio_len, _bg_thread, _last_bg_kick
    with _bg_result_lock:
        _bg_result = None
        _bg_audio_len = 0
    _bg_thread = None
    _last_bg_kick = 0


def draft(chunk_bytes: bytes, is_final: bool) -> tuple[str, int]:
    global _bg_thread, _last_bg_kick

    audio = _np.frombuffer(chunk_bytes, _np.int16).flatten().astype(_np.float32) / 32768.0
    audio_len = len(audio)

    # ---- FINAL: use shunyalabs for maximum accuracy ----
    if is_final:
        # Strategy: check if background thread has a recent result
        # If the bg thread is still running, wait for it (up to 8s)
        if _bg_thread is not None and _bg_thread.is_alive():
            _bg_thread.join(timeout=8.0)

        # Check cached result
        with _bg_result_lock:
            cached = _bg_result

        if cached:
            return (cached, len(cached))

        # Fallback: run shunyalabs synchronously
        try:
            m, lk = get_shunyalabs()
            with lk:
                segs, _ = m.transcribe(
                    audio,
                    beam_size=1,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                    vad_filter=True
                )
                text = _postprocess(" ".join(s.text for s in segs).strip())
            if text:
                return (text, len(text))
        except Exception:
            pass

        # Last resort: use whatever whisper-tiny can produce
        try:
            m, lk = get_tiny()
            with lk:
                segs, _ = m.transcribe(
                    audio, beam_size=1,
                    without_timestamps=True,
                    condition_on_previous_text=False,
                    vad_filter=True
                )
                text = _postprocess(" ".join(s.text for s in segs).strip())
            return (text, len(text)) if text else ("", 0)
        except Exception:
            return ("", 0)

    # ---- PARTIAL: use whisper-tiny for speed ----
    # Need at least ~0.5s of audio before trying
    if audio_len < 16000 * 0.5:
        return ("", 0)

    try:
        m, lk = get_tiny()
        with lk:
            segs, _ = m.transcribe(
                audio,
                beam_size=1,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True
            )
            text = _postprocess(" ".join(s.text for s in segs).strip())

        # Kick off background shunyalabs transcription periodically
        # Only re-kick if we have significantly more audio (>1.5s more)
        if audio_len - _last_bg_kick > 16000 * 1.5:
            if _bg_thread is None or not _bg_thread.is_alive():
                _last_bg_kick = audio_len
                audio_copy = audio.copy()
                _bg_thread = threading.Thread(
                    target=_bg_transcribe,
                    args=(audio_copy, audio_len),
                    daemon=True
                )
                _bg_thread.start()

        if text:
            # Commit all but the last ~15 chars to minimize churn
            return (text, max(0, len(text) - 15))

        return ("", 0)
    except Exception:
        return ("", 0)


# ===========================================================================
# Warmup — load both models before network is blocked.
# MUST be synchronous so stream_server waits before printing READY.
# ===========================================================================
get_tiny()
get_shunyalabs()
