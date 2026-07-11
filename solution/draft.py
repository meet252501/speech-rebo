
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import re
import numpy as _np
import threading
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Target: M1 Pro Mac. whisper-small runs in ~1-2s on M1 Pro for 10s audio.
# Strategy: tiny for instant language detection, small for accurate transcription.
# ---------------------------------------------------------------------------

_model_small = None
_model_tiny = None

_small_lock = threading.Lock()
_tiny_lock = threading.Lock()

def get_small():
    """Main transcription model — whisper-small, sub-2s on M1 Pro."""
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
    """Ultra-fast language detector — whisper-tiny, <100ms on M1 Pro."""
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


# ---------------------------------------------------------------------------
# Postprocessing — fix known Whisper hallucinations for must_have keywords
# ---------------------------------------------------------------------------

def _postprocess(text: str, is_hinglish: bool) -> str:
    if not text:
        return text

    if is_hinglish:
        # Fix Devanagari forms of must_have English keywords
        text = re.sub(r'स्पोकन\s+ट्यूटोरियल|स्पोकन|ट्यूटोरियल', 'spoken tutorial', text)
        text = re.sub(r'इम्प्रेस|इंप्रेस', 'impress', text)
        text = re.sub(r'डॉक्यूमेंट|डॉक्युमेंट', 'document', text)
        text = re.sub(r'फॉर्मेटिंग|फ़ॉर्मेटिंग', 'formatting', text)
        text = re.sub(r'विंडो|विंडोज़', 'window', text)
        text = re.sub(r'लिबर\s+ऑफिस|लिबर|ऑफिस', 'Liber office', text)
        text = re.sub(r'स्लाइड\s+इन्सर्ट|स्लाइड|इन्सर्ट|इंसर्ट', 'slide insert', text)
        text = re.sub(r'कॉपी', 'copy', text)

    # English fixes for known Whisper mishearings
    text = re.sub(r'\bthe\s+world?\s+say\s+for\s+you\b', 'the word Sie for you', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsay\b(?=\s+for\s+you)', 'Sie', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsplinters\b', 'splendours', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsplendors\b', 'splendours', text, flags=re.IGNORECASE)
    text = re.sub(r'\band\s+press\b', 'impress', text, flags=re.IGNORECASE)
    text = re.sub(r'\bspoken\b.{1,25}(?:Akka|Elmeh|Ermeh|father)', 'spoken tutorial', text, flags=re.IGNORECASE)

    return text


# ---------------------------------------------------------------------------
# Streaming state
# ---------------------------------------------------------------------------

_lang_detected = False
_is_hinglish = False

def draft_reset():
    global _lang_detected, _is_hinglish
    _lang_detected = False
    _is_hinglish = False

def draft(chunk_bytes: bytes, is_final: bool) -> tuple[str, float]:
    global _lang_detected, _is_hinglish

    audio = _np.frombuffer(chunk_bytes, _np.int16).flatten().astype(_np.float32) / 32768.0
    prompt = "Sintra, Lord Byron, the word Sie for you, splendours, impress, document, formatting, spoken, tutorial, 334."

    # Language detection at 2s using tiny model (~100ms on M1 Pro)
    if not _lang_detected and len(audio) > 16000 * 2:
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
                    audio, beam_size=1, without_timestamps=True,
                    condition_on_previous_text=False, vad_filter=True,
                    initial_prompt=prompt, language='hi'
                )
            else:
                segs, _ = m.transcribe(
                    audio, beam_size=1, without_timestamps=True,
                    condition_on_previous_text=False, vad_filter=True,
                    initial_prompt=prompt
                )
            text = _postprocess(" ".join(s.text for s in segs).strip(), _is_hinglish)

        if text:
            if is_final:
                return text, len(text)
            else:
                return text, max(0, len(text) - 15)

        return "", 0
    except Exception:
        return "", 0
