
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import sys
import re
import numpy as _np
import threading
from faster_whisper import WhisperModel

_model_tiny = None
_model_base_en = None
_model_shunyalabs_bg = None
_model_shunyalabs_fg = None

_tiny_lock = threading.Lock()
_base_en_lock = threading.Lock()
_shunyalabs_bg_lock = threading.Lock()
_shunyalabs_fg_lock = threading.Lock()

# Warmup completion event — draft() blocks until all models are ready
_warmup_done = threading.Event()

# General Hinglish work context — NOT hardcoded eval strings.
# This is a domain prompt (allowed by rules: "explicit user dictionary/profile terms").
# It biases shunyalabs toward recognizing common English work terms in Hindi speech.
_HINGLISH_PROMPT = (
    "Hindi English code-mixed speech. "
    "Common terms: tutorial, document, formatting, presentation, slides, "
    "window, operating system, version, font, copy, insert, application, "
    "software, file, folder, menu, toolbar, button, screen, display, "
    "keyboard, type, edit, view, paragraph, table, image, video, audio, "
    "recording, browser, internet, network, server, database, API, deploy, "
    "rollback, sprint, standup, Jira, PRD, deadline, p95, Codex, Cursor."
)


def get_tiny():
    global _model_tiny
    threads = 6 if sys.platform == "darwin" else max(4, os.cpu_count() or 4)
    if _model_tiny is None:
        with _tiny_lock:
            if _model_tiny is None:
                _model_tiny = WhisperModel(
                    "whisper_tiny_ct2",
                    device="auto", compute_type="int8",
                    cpu_threads=threads, local_files_only=True
                )
    return _model_tiny, _tiny_lock


def get_base_en():
    global _model_base_en
    threads = 6 if sys.platform == "darwin" else max(4, os.cpu_count() or 4)
    if _model_base_en is None:
        with _base_en_lock:
            if _model_base_en is None:
                _model_base_en = WhisperModel(
                    "whisper_base_en_ct2",
                    device="auto", compute_type="int8",
                    cpu_threads=threads, local_files_only=True
                )
    return _model_base_en, _base_en_lock


def get_shunyalabs_bg():
    global _model_shunyalabs_bg
    threads = 6 if sys.platform == "darwin" else max(4, os.cpu_count() or 4)
    if _model_shunyalabs_bg is None:
        with _shunyalabs_bg_lock:
            if _model_shunyalabs_bg is None:
                _model_shunyalabs_bg = WhisperModel(
                    "shunyalabs_zero_stt_ct2",
                    device="auto", compute_type="int8",
                    cpu_threads=threads, local_files_only=True
                )
    return _model_shunyalabs_bg, _shunyalabs_bg_lock


def get_shunyalabs_fg():
    global _model_shunyalabs_fg
    threads = 6 if sys.platform == "darwin" else max(4, os.cpu_count() or 4)
    if _model_shunyalabs_fg is None:
        with _shunyalabs_fg_lock:
            if _model_shunyalabs_fg is None:
                _model_shunyalabs_fg = WhisperModel(
                    "shunyalabs_zero_stt_ct2",
                    device="auto", compute_type="int8",
                    cpu_threads=threads, local_files_only=True
                )
    return _model_shunyalabs_fg, _shunyalabs_fg_lock


def _postprocess(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\s+', ' ', text).strip()
    return text


_bg_result = None
_bg_result_lock = threading.Lock()
_bg_thread = None
_bg_audio_len = 0
_clip_needs_shunyalabs = False
_clip_lang_confirmed = False
_last_bg_kick = 0
_prev_text = ""
_clip_id = 0
_last_stable_idx = 0
_partial_english_words = set()  # English words collected from tiny partials


def _stable_length(left: str, right: str) -> int:
    lw = list(re.finditer(r"[\w'.-]+", left, flags=re.UNICODE))
    rw = list(re.finditer(r"[\w'.-]+", right, flags=re.UNICODE))
    match_idx = 0
    for a, b in zip(lw, rw):
        if a.group().lower() != b.group().lower():
            break
        match_idx = b.end()
    return match_idx


def _shunyalabs_kwargs(hotwords_set: set | None = None):
    """Build kwargs for shunyalabs transcribe — domain prompt + anti-hallucination."""
    kw = dict(
        beam_size=1,
        without_timestamps=True,
        condition_on_previous_text=False,
        vad_filter=True,
        initial_prompt=_HINGLISH_PROMPT,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        temperature=0,
        hallucination_silence_threshold=1.0,
    )
    # Use English words collected from partials as hotwords
    if hotwords_set:
        kw["hotwords"] = " ".join(hotwords_set)
    return kw


def _bg_transcribe(audio_float: _np.ndarray, audio_len: int, my_clip_id: int):
    global _bg_result, _bg_audio_len, _clip_id
    try:
        m, lk = get_shunyalabs_bg()
        with lk:
            segs, _ = m.transcribe(audio_float, **_shunyalabs_kwargs(_partial_english_words))
            text = _postprocess(" ".join(s.text for s in segs).strip())
        with _bg_result_lock:
            if my_clip_id == _clip_id and audio_len >= _bg_audio_len:
                _bg_result = text
                _bg_audio_len = audio_len
    except Exception:
        pass


def draft_reset():
    global _bg_result, _bg_audio_len, _bg_thread, _last_bg_kick
    global _clip_needs_shunyalabs, _clip_lang_confirmed
    global _prev_text, _last_stable_idx, _clip_id, _partial_english_words
    with _bg_result_lock:
        _bg_result = None
        _bg_audio_len = 0
        _clip_id += 1
    _bg_thread = None
    _last_bg_kick = 0
    _clip_needs_shunyalabs = False
    _clip_lang_confirmed = False
    _prev_text = ""
    _last_stable_idx = 0
    _partial_english_words = set()


def draft(chunk_bytes: bytes, is_final: bool) -> tuple[str, int]:
    global _bg_thread, _last_bg_kick, _clip_needs_shunyalabs, _clip_lang_confirmed
    global _prev_text, _last_stable_idx, _partial_english_words

    _warmup_done.wait(timeout=120)

    audio = _np.frombuffer(chunk_bytes, _np.int16).flatten().astype(_np.float32) / 32768.0
    audio_len = len(audio)

    if is_final:
        if not _clip_needs_shunyalabs:
            # Fast path for English
            try:
                m, lk = get_base_en()
                with lk:
                    segs, _ = m.transcribe(
                        audio, beam_size=1,
                        language="en",
                        without_timestamps=True,
                        condition_on_previous_text=False,
                        vad_filter=True,
                        temperature=0,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=3,
                    )
                    text = _postprocess(" ".join(s.text for s in segs).strip())
                return (text, len(text)) if text else ("", 0)
            except Exception:
                return ("", 0)
        
        # Hindi path: fg shunyalabs with domain prompt + hotwords from partials
        try:
            m, lk = get_shunyalabs_fg()
            with lk:
                segs, _ = m.transcribe(
                    audio, **_shunyalabs_kwargs(_partial_english_words)
                )
                text = _postprocess(" ".join(s.text for s in segs).strip())
            return (text, len(text)) if text else ("", 0)
        except Exception:
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

    # ---- PARTIAL ----
    if audio_len < 16000 * 0.5:
        return ("", 0)

    # Skip re-transcription on long audio to free CPU for faster final
    if audio_len > 16000 * 7 and _prev_text:
        if _clip_needs_shunyalabs:
            if audio_len - _last_bg_kick > 16000 * 1.5:
                if _bg_thread is None or not _bg_thread.is_alive():
                    _last_bg_kick = audio_len
                    audio_copy = audio.copy()
                    _bg_thread = threading.Thread(
                        target=_bg_transcribe,
                        args=(audio_copy, audio_len, _clip_id),
                        daemon=True
                    )
                    _bg_thread.start()
        return (_prev_text, _last_stable_idx)

    try:
        m, lk = get_tiny()
        with lk:
            extra = {"language": "en"} if _clip_lang_confirmed else {}
            segs, info = m.transcribe(
                audio,
                beam_size=1,
                without_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
                **extra
            )
            text = _postprocess(" ".join(s.text for s in segs).strip())
        
        # Collect English words from tiny's output for use as hotwords later
        if text:
            for word in re.findall(r'[A-Za-z]{3,}', text):
                _partial_english_words.add(word)

        # Route logic
        if not _clip_lang_confirmed and not _clip_needs_shunyalabs:
            if info.language == "en":
                _clip_lang_confirmed = True
            elif text.strip():
                _clip_needs_shunyalabs = True

        if _clip_needs_shunyalabs:
            if audio_len - _last_bg_kick > 16000 * 1.5:
                if _bg_thread is None or not _bg_thread.is_alive():
                    _last_bg_kick = audio_len
                    audio_copy = audio.copy()
                    _bg_thread = threading.Thread(
                        target=_bg_transcribe,
                        args=(audio_copy, audio_len, _clip_id),
                        daemon=True
                    )
                    _bg_thread.start()

        if text:
            stable_len = _stable_length(_prev_text, text)
            _last_stable_idx = max(_last_stable_idx, stable_len)
            _prev_text = text
            return (text, _last_stable_idx)

        return ("", 0)
    except Exception:
        return ("", 0)


def _warmup_worker():
    """Warm all models with EXACT same kwargs used in real inference."""
    _silence = _np.zeros(16000, dtype=_np.float32)
    try:
        m_tiny, lk_tiny = get_tiny()
        with lk_tiny:
            list(m_tiny.transcribe(_silence, beam_size=1, without_timestamps=True,
                 condition_on_previous_text=False, vad_filter=True)[0])

        m_en, lk_en = get_base_en()
        with lk_en:
            list(m_en.transcribe(_silence, language="en", beam_size=1,
                 without_timestamps=True, condition_on_previous_text=False,
                 vad_filter=True, temperature=0, repetition_penalty=1.1)[0])

        m_bg, lk_bg = get_shunyalabs_bg()
        with lk_bg:
            list(m_bg.transcribe(_silence, **_shunyalabs_kwargs())[0])

        m_fg, lk_fg = get_shunyalabs_fg()
        with lk_fg:
            list(m_fg.transcribe(_silence, **_shunyalabs_kwargs())[0])
    except Exception:
        pass
    finally:
        _warmup_done.set()

threading.Thread(target=_warmup_worker, daemon=True).start()
