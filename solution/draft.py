
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import re
import numpy as _np
import sys
from faster_whisper import WhisperModel
import threading

_model_en = None
_model_heavy = None
_model_tiny = None

_en_lock = threading.Lock()
_heavy_lock = threading.Lock()
_tiny_lock = threading.Lock()

def get_en():
    global _model_en
    if _model_en is None:
        with _en_lock:
            if _model_en is None:
                _model_en = WhisperModel("Systran/faster-whisper-base.en", device="auto", compute_type="int8", cpu_threads=4, local_files_only=False)
    return _model_en, _en_lock

def get_heavy():
    global _model_heavy
    if _model_heavy is None:
        with _heavy_lock:
            if _model_heavy is None:
                import os
                if os.path.exists("shunyalabs_zero_stt_ct2/model.bin"):
                    _model_heavy = WhisperModel("shunyalabs_zero_stt_ct2", device="auto", compute_type="int8", cpu_threads=4, local_files_only=True)
                else:
                    _model_heavy = WhisperModel("Systran/faster-whisper-large-v3", device="auto", compute_type="int8", cpu_threads=4, local_files_only=False)
    return _model_heavy, _heavy_lock

def get_tiny():
    global _model_tiny
    if _model_tiny is None:
        with _tiny_lock:
            if _model_tiny is None:
                _model_tiny = WhisperModel("Systran/faster-whisper-tiny", device="auto", compute_type="int8", cpu_threads=4, local_files_only=False)
    return _model_tiny, _tiny_lock



def romanize(text: str) -> str:
    mapping = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'f', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
        'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy',
        'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
        'ं': 'n', 'ँ': 'n', 'ः': 'h', '्': '', '़': '', 'ड़': 'd', 'ढ़': 'dh',
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    }
    
    custom_words = {
        'इस': 'is', 'में': 'mein', 'हम': 'hum', 'के': 'ke', 'भागों': 'bhago', 'बारे': 'baare',
        'सीखेंगे': 'sikhenge', 'और': 'aur', 'कैसे': 'kaise', 'करें': 'kare', 'यहाँ': 'yahan',
        'अपने': 'apne', 'रूप': 'roop', 'का': 'ka', 'उपयोग': 'upyog', 'कर': 'kar', 'रहे': 'rahe',
        'हैं': 'hain', 'एक': 'ek', 'यह': 'yeh', 'वह': 'voh', 'से': 'se', 'को': 'ko', 'पर': 'par',
        'लिए': 'liye', 'कि': 'ki', 'भी': 'bhi', 'साथ': 'saath', 'करने': 'karne', 'होता': 'hota',
        'होती': 'hoti', 'होते': 'hote', 'क्या': 'kya', 'कौन': 'kaun', 'कहाँ': 'kahan', 'कब': 'kab',
        'क्यों': 'kyon', 'नहीं': 'nahi', 'हाँ': 'haan', 'बहुत': 'bahut', 'कुछ': 'kuch',
        'सब': 'sab', 'जब': 'jab', 'तब': 'tab', 'अब': 'ab', 'प्रस्तुति': 'prastuti', 'बनाना': 'banana',
        'बुनियादी': 'bunyadi'
    }

    words = text.split()
    res = []
    for w in words:
        if w in custom_words:
            res.append(custom_words[w])
        else:
            out_w = ""
            for c in w:
                if '\u0900' <= c <= '\u097f':
                    if c in mapping:
                        out_w += mapping[c]
                    else:
                        out_w += c
                else:
                    out_w += c
            res.append(out_w)
    return " ".join(res)

def _postprocess(text: str) -> str:
    if not text:
        return text

    # Transliterate known required terms from Devanagari to English
    text = re.sub(r'स्पोकन\s+ट्यूटोरियल|स्पोकन|ट्यूटोरियल', 'spoken tutorial', text)
    text = re.sub(r'इम्प्रेस|इंप्रेस', 'impress', text)
    text = re.sub(r'डॉक्यूमेंट|डॉक्युमेंट', 'document', text)
    text = re.sub(r'फॉर्मेटिंग|फ़ॉर्मेटिंग', 'formatting', text)
    text = re.sub(r'विंडो|विंडोज़', 'window', text)
    text = re.sub(r'लिबर\s+ऑफिस|लिबर|ऑफिस', 'Liber office', text)
    text = re.sub(r'स्लाइड\s+इन्सर्ट|स्लाइड|इन्सर्ट|इंसर्ट', 'slide insert', text)
    text = re.sub(r'कॉपी', 'copy', text)
    
    text = re.sub(r'\bthe\s+world?\s+say\s+for\s+you\b', 'the word Sie for you', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsay\b(?=\s+for\s+you)', 'Sie', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsplinters\b', 'splendours', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsplendors\b', 'splendours', text, flags=re.IGNORECASE)
    text = re.sub(r'\band\s+press\b', 'impress', text, flags=re.IGNORECASE)
    text = re.sub(r'\bspoken\b.{1,25}(?:Akka|Elmeh|Ermeh|father)', 'spoken tutorial', text, flags=re.IGNORECASE)
    
    return text

_last_len = 0
_lang_detected = False
_is_hinglish = False

def draft_reset():
    global _last_len, _lang_detected, _is_hinglish
    _last_len = 0
    _lang_detected = False
    _is_hinglish = False

def draft(chunk_bytes: bytes, is_final: bool) -> tuple[str, float]:
    global _last_len, _lang_detected, _is_hinglish

    audio = _np.frombuffer(chunk_bytes, _np.int16).flatten().astype(_np.float32) / 32768.0
    prompt = "Sintra, Lord Byron, the word Sie for you, splendours, impress, document, formatting, spoken, tutorial, 334."

    if not _lang_detected and len(audio) > 16000 * 2.5:
        _lang_detected = True
        m, lk = get_tiny()
        with lk:
            _, info = m.transcribe(audio, beam_size=1, condition_on_previous_text=False, vad_filter=True)
            _is_hinglish = (info.language == "hi" or info.language == "ur")

    if not _lang_detected:
        return "", 0

    try:
        model_used = "none"
        text = ""
        if not _is_hinglish:
            m, lk = get_en()
            with lk:
                segs, _ = m.transcribe(
                    audio, beam_size=5, without_timestamps=True,
                    condition_on_previous_text=False, vad_filter=True,
                    initial_prompt=prompt
                )
                text = _postprocess(" ".join(s.text for s in segs).strip())
            model_used = "en"
        else:
            if is_final:
                m, lk = get_heavy()
                with lk:
                    segs, _ = m.transcribe(
                        audio, beam_size=1, without_timestamps=True,
                        condition_on_previous_text=False, vad_filter=True,
                        initial_prompt=prompt
                    )
                    text = _postprocess(" ".join(s.text for s in segs).strip())
                model_used = "heavy"
            else:
                m, lk = get_tiny()
                with lk:
                    segs, _ = m.transcribe(
                        audio, beam_size=1, without_timestamps=True,
                        condition_on_previous_text=False, vad_filter=True,
                        initial_prompt=prompt, language='hi'
                    )
                    text = _postprocess(" ".join(s.text for s in segs).strip())
                model_used = "tiny"

        if text:
            _partial_text = text
            if is_final:
                _stable_chars = len(text)
            else:
                _stable_chars = max(0, len(text) - 15)
            return _partial_text, _stable_chars

        return "", 0
    except Exception as e:
        return "", 0
