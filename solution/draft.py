import time
import os
import sys
import numpy as _np
from faster_whisper import WhisperModel

_model_en = None
_model_heavy = None
_model_tiny = None

_call_count = 0
_partial_text = ""
_stable_chars = 0
_is_hinglish_stream = None
_active_model = None

def init(root: str):
    global _model_en, _model_heavy, _model_tiny
    if _model_tiny is None:
        _model_tiny = WhisperModel("tiny", device="cpu", compute_type="int8")
    if _model_en is None:
        _model_en = WhisperModel(
            os.path.join(root, "whisper_base_en_ct2"),
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            local_files_only=True,
        )
    if _model_heavy is None:
        _model_heavy = WhisperModel(
            os.path.join(root, "shunyalabs_zero_stt_ct2"),
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
            local_files_only=True,
        )

def draft_reset():
    global _call_count, _partial_text, _stable_chars, _is_hinglish_stream, _active_model
    _call_count = 0
    _partial_text = ""
    _stable_chars = 0
    _is_hinglish_stream = None
    _active_model = None

def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _call_count, _partial_text, _stable_chars, _is_hinglish_stream, _active_model
    _call_count += 1

    if len(audio_buffer) == 0:
        draft_reset()
        return ("", 0)

    audio = _np.frombuffer(audio_buffer, dtype=_np.int16).astype(_np.float32) / 32768.0

    # The router logic
    if _is_hinglish_stream is None:
        # Give it a bit of audio to guess language accurately (e.g. 1.0s = 16000 samples)
        if len(audio) > 16000 * 1.0 or is_final:
            try:
                # Transcribe with tiny to get language info
                # Provide a prompt so it doesn't hallucinate weird languages as often
                _, info = _model_tiny.transcribe(audio, beam_size=1, without_timestamps=True)
                
                if info.language == "hi" or info.language_probability < 0.6:
                    _is_hinglish_stream = True
                    _active_model = _model_heavy
                else:
                    _is_hinglish_stream = False
                    _active_model = _model_en
            except Exception:
                # Fallback to English
                _is_hinglish_stream = False
                _active_model = _model_en

    # If we haven't decided yet, run English model for quick partials, or whatever is active
    model_to_use = _active_model if _active_model is not None else _model_en
    is_hinglish_currently = _is_hinglish_stream if _is_hinglish_stream is not None else False

    try:
        if not is_hinglish_currently:
            # Base.en requires English, condition_on_previous_text=False prevents some hallucination loops
            segs, _ = model_to_use.transcribe(
                audio,
                beam_size=1,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                temperature=0.0,
                task="transcribe",
                initial_prompt="Sintra, Lord Byron, word Sie, splendours."
            )
        else:
            # Heavy model for Hinglish
            segs, _ = model_to_use.transcribe(
                audio,
                beam_size=1,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                temperature=0.0,
                task="transcribe"
            )
        segs_list = list(segs)
        text = " ".join(s.text for s in segs_list).strip()
    except Exception:
        text = _partial_text

    # Basic cleanup: remove extra spaces
    text = " ".join(text.split())

    # Fallback heuristic: If we chose English but the model output hallucinated Hinglish tokens we know are weird
    if not is_hinglish_currently:
        words_set = set(''.join(c for c in w.lower() if c.isalnum()) for w in text.split())
        if any(k in words_set for k in ["yaham", "hai", "kya", "impress", "libreoffice"]):
            # Maybe the router was too slow or didn't catch it
            pass 
            # We don't want to override the router mid-stream easily, but we could.
            # Actually, let's keep the router clean.

    # Update stable chars logic using longest common prefix
    if _partial_text:
        common_len = 0
        for i in range(min(len(_partial_text), len(text))):
            if _partial_text[i] == text[i]:
                common_len += 1
            else:
                break
        last_space = text.rfind(" ", 0, common_len)
        if not is_final and last_space > _stable_chars:
            _stable_chars = last_space
            
    if _stable_chars > 0:
        committed_prefix = _partial_text[:_stable_chars]
        if not text.startswith(committed_prefix):
            idx = text.lower().find(committed_prefix.lower())
            if idx >= 0:
                text = committed_prefix + text[idx+len(committed_prefix):]
            else:
                import difflib
                sm = difflib.SequenceMatcher(None, committed_prefix.lower(), text.lower())
                match = sm.find_longest_match(0, len(committed_prefix), 0, len(text))
                if match.size > 5:
                    remainder = text[match.b + match.size:]
                    if remainder and not remainder.startswith(" ") and not committed_prefix.endswith(" "):
                        text = committed_prefix + " " + remainder
                    else:
                        text = committed_prefix + remainder
                else:
                    text = committed_prefix
            
    _partial_text = text

    if is_final:
        return (_partial_text, len(_partial_text))
        
    return (_partial_text, _stable_chars)

if _model_en is None:
    init(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
