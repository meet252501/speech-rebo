"""Reference contract for the builderr local-dictation challenge.

Entrants replace the body of transcribe() with their own local engine/router.
The CLI signature and the result.json shape are REQUIRED and checked by the harness:

    python -m solution.transcribe --input clip.wav --mode auto --output result.json

Rules: runs fully local; no outbound network during the scored run (loopback to a
local ASR server is fine); emit the JSON below; no hardcoded phrase fixes.

This skeleton emits a valid contract result. If `faster-whisper` is installed it
runs a real local baseline; otherwise it returns an empty transcript clearly
flagged so the contract still validates (and scores as a blank — replace it!).
"""
from __future__ import annotations
import argparse, json, time


_model_fast = None
_model_heavy = None

def transcribe(wav_path: str, mode: str = "auto") -> dict:
    global _model_fast, _model_heavy
    t0 = time.time()
    text, model_ids, candidates = "", [], []
    asr_ms = 0.0
    language_guess = "unknown"
    mode_used = mode
    
    try:
        from faster_whisper import WhisperModel
        if _model_fast is None:
            _model_fast = WhisperModel("whisper_tiny_ct2", device="auto", compute_type="int8", local_files_only=True)
            
        a = time.time()
        # 1. Fast Path & Router
        fast_segments, fast_info = _model_fast.transcribe(wav_path, task="transcribe")
        fast_segments = list(fast_segments)
        
        # 2. Router Decision
        language_guess = fast_info.language
        _is_hinglish = language_guess == "hi" or fast_info.language_probability < 0.6
        mode_used = "hinglish" if _is_hinglish else "english"
        
        # 3. Execution
        if mode_used == "hinglish":
            global _model_heavy
            if _model_heavy is None:
                _model_heavy = WhisperModel("shunyalabs_zero_stt_ct2", device="auto", compute_type="int8", local_files_only=True)
            heavy_segments, heavy_info = _model_heavy.transcribe(wav_path, task="transcribe")
            text = " ".join(s.text for s in heavy_segments).strip()
            model_ids = ["faster-whisper-tiny", "shunyalabs/zero-stt-hinglish"]
            candidates = [{"engine": "shunyalabs/zero-stt-hinglish", "text": text}]
        else:
            # We use base.en for high quality fast English
            global _model_en
            if 'model_en' not in globals():
                global model_en
                model_en = WhisperModel("whisper_base_en_ct2", device="auto", compute_type="int8", local_files_only=True)
            en_segments, en_info = model_en.transcribe(wav_path, task="transcribe")
            text = " ".join(s.text for s in en_segments).strip()
            model_ids = ["faster-whisper-tiny", "whisper_base_en_ct2"]
            candidates = [{"engine": "whisper_base_en_ct2", "text": text}]
            
        asr_ms = (time.time() - a) * 1000
        
    except Exception as e:
        print(f"Exception in transcribe: {e}")
        candidates = [{"engine": "none", "text": "", "note": f"Error: {e}"}]

    total_ms = (time.time() - t0) * 1000
    return {
        "text": text,
        "mode_used": mode_used,
        "language_guess": language_guess,
        "timings_ms": {"total": round(total_ms), "asr": round(asr_ms), "postprocess": 0},
        "raw_candidates": candidates,
        "model_ids": model_ids,
        "local_only": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto", "fast", "hinglish", "verbatim"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = transcribe(args.input, args.mode)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.output}  ({result['timings_ms']['total']}ms, local_only={result['local_only']})")


if __name__ == "__main__":
    main()
