"""Local dev preview: run YOUR engine over the public dev manifest and score it
exactly like official admission — fully offline. Prints the same rubric the
hidden set uses, so you can iterate before submitting.

    python preview.py                 # uses data/dev/manifest.json + solution.transcribe

The official run is identical but on hidden clips with the network hard-blocked.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scorecard import score_run
from offline_guard import block_network
from solution.transcribe import transcribe


def main():
    manifest = json.load(open(os.path.join(HERE, "samples/manifest.json"), encoding="utf-8"))
    block_network()  # mirror official scoring: no cloud during the run
    rows = []
    for clip in manifest:
        # audio_local when the pool audio is on this machine; else samples/<id>.wav
        # (the Action's fetch_audio.py downloads each clip from HF by provenance id)
        wav = clip.get("audio_local") or os.path.join(HERE, "samples", clip["clip_id"] + ".wav")
        r = transcribe(wav, clip.get("mode", "auto"))
        rows.append({
            "clip_id": clip["clip_id"], "gold": clip["gold"], "pred": r.get("text", ""),
            "must_have": clip.get("must_have", []), "timings_ms": r.get("timings_ms"),
            "local_only": r.get("local_only", False),
            "audit": {"model_ids": r.get("model_ids"), "route": r.get("mode_used")},
        })
    res = score_run(rows)
    print(f"\n  overall score   {res['overall_score']}/100")
    print(f"  meaning (proxy) {res['useful_mean']}   WER {res['wer_mean']}")
    print(f"  p50 {res['p50_ms']}ms  p95 {res['p95_ms']}ms  blanks {res['blank_rate']}  hangs {res['hang_rate']}")
    print(f"  clips capped    {res['clips_capped']}/{res['n']}")
    for c in res["clips"]:
        flag = f"  capped@{c['capped_at']}" if c["capped_at"] else ""
        print(f"    {c['clip_id']:8s} score {c['score']:5}  wer {c['wer']}{flag}  {';'.join(c['reasons'][:2])}")
    print("\n  (dev numbers are illustrative; the hidden set + your latency on the Linux box rank you.)")


if __name__ == "__main__":
    main()
