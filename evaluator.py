"""builderr · STREAMING dictation track — real-time feeder + metric capture.

This is the harness builderr runs. It launches the entrant's sealed
`solution/stream_server.py` on a loopback port, then for each clip:

  1. opens one WebSocket per run,
  2. feeds the WAV at 1x real time as 20ms binary PCM frames (absolute-time paced,
     so jitter never accumulates),
  3. captures partial/final events on a monotonic RECEIVE clock,
  4. repeats 5 times with per-run anti-replay jitter (gain + tiny resample),
  5. scores via streaming_scorecard.score_stream_run — latency on the 5-run
     median, quality on the median-latency run's final.

After READY + a warm-up clip, it calls block_network() so the scored run is
offline (loopback to a local ASR server stays allowed — see offline_guard.py).

Adapted from RambleFix's run_streaming_latency_eval.py. No t_ms / seq / pcm_b64:
every scored timing is the evaluator's own monotonic receive time.

Usage (entrant-facing wrapper is preview_stream.py):
    python evaluator.py --manifest samples/manifest.json --runs 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from streaming_scorecard import score_stream_run  # noqa: E402
from offline_guard import block_network  # noqa: E402

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

FRAME_MS = 20
SR = 16000
FRAME_BYTES = int(SR * FRAME_MS / 1000) * 2  # 20ms s16le mono = 640 bytes


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def _read_pcm_16k_mono(wav_path: str) -> bytes:
    with wave.open(wav_path, "rb") as r:
        sr = r.getframerate()
        ch = r.getnchannels()
        sw = r.getsampwidth()
        raw = r.readframes(r.getnframes())
    if np is None:
        return raw  # best effort; assume already 16k mono s16le
    a = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != SR:
        a = _resample(a, sr, SR)
    return a.astype(np.int16).tobytes()


def _resample(a, src_sr: int, dst_sr: int):
    if src_sr == dst_sr:
        return a
    n_out = int(round(len(a) * dst_sr / src_sr))
    if n_out <= 0:
        return a[:0]
    xp = np.linspace(0.0, 1.0, num=len(a), endpoint=False)
    x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x, xp, a.astype(np.float32)).astype(np.int16)


def _jitter(pcm: bytes, seed: int) -> bytes:
    """Per-run anti-replay perturbation: random gain +/-0.5 dB and +/-0.3%
    resample, deterministic per (clip, run). Defeats fingerprint memoization
    across the 5 warm serial runs without changing the words."""
    if np is None:
        return pcm
    rng = np.random.default_rng(seed)
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    gain_db = rng.uniform(-0.5, 0.5)
    a = a * (10.0 ** (gain_db / 20.0))
    speed = 1.0 + rng.uniform(-0.003, 0.003)
    if abs(speed - 1.0) > 1e-6:
        a16 = np.clip(a, -32768, 32767).astype(np.int16)
        a = _resample(a16, SR, int(round(SR / speed))).astype(np.float32)
    return np.clip(a, -32768, 32767).astype(np.int16).tobytes()


def _frames(pcm: bytes):
    for i in range(0, len(pcm), FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]


# --------------------------------------------------------------------------
# one streamed run
# --------------------------------------------------------------------------

async def _run_once(uri: str, clip_id: str, pcm: bytes) -> dict:
    import websockets
    partials: list[tuple[float, str, int]] = []
    final = None
    dropped = False
    t_start = None
    t_end_audio = None
    pace_ok = True
    expected_s = len(pcm) / (SR * 2)

    try:
        async with websockets.connect(uri, max_size=None) as ws:
            await ws.send(json.dumps({
                "type": "start", "sample_rate": SR, "format": "pcm_s16le",
                "channels": 1, "clip_id": clip_id}))

            async def _receiver():
                nonlocal final
                async for raw in ws:
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    now = time.monotonic()
                    if msg.get("type") == "partial":
                        partials.append((now, msg.get("text", ""), int(msg.get("stable_chars", 0))))
                    elif msg.get("type") == "final":
                        final = (now, msg.get("text", ""))
                        return

            recv_task = asyncio.create_task(_receiver())

            # absolute-time paced feeder — no drift accumulation
            t_start = time.monotonic()
            t_send = t_start
            dt = FRAME_MS / 1000.0
            for frame in _frames(pcm):
                await ws.send(frame)
                t_send += dt
                delay = t_send - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(json.dumps({"type": "end"}))
            t_end_audio = time.monotonic()

            # pace sanity: total send time within +/-5% of clip duration
            sent_s = t_end_audio - t_start
            pace_ok = (np is None) or (abs(sent_s - expected_s) <= 0.05 * max(0.001, expected_s))

            try:
                await asyncio.wait_for(recv_task, timeout=20.0)
            except asyncio.TimeoutError:
                dropped = True
                recv_task.cancel()
    except Exception:  # noqa: BLE001 - any transport failure => dropped clip
        dropped = True

    return {
        "clip_id": clip_id,
        "t_start": t_start,
        "t_end_audio": t_end_audio,
        "partials": partials,
        "final": final,
        "dropped": dropped or final is None,
        "pace_ok": pace_ok,
    }


async def _capture_clip(uri: str, clip: dict, runs: int) -> dict:
    pcm = _read_pcm_16k_mono(clip["_wav"])
    captured = []
    attempt_seed = 0
    while len(captured) < runs and attempt_seed < runs * 4:
        seed = hash((clip["clip_id"], attempt_seed)) & 0xFFFFFFFF
        attempt_seed += 1
        run = await _run_once(uri, clip["clip_id"], _jitter(pcm, seed))
        # pace sanity: an out-of-band run is discarded and re-run, never scored
        if not run.get("pace_ok", True) and not run.get("dropped"):
            continue
        captured.append(run)
    return {
        "clip_id": clip["clip_id"],
        "gold": clip.get("gold", ""),
        "must_have": clip.get("must_have", []),
        "runs": captured,
    }


# --------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(module: str, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", module, "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=HERE, text=True)
    # block on the READY line
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError("stream server exited before READY")
            continue
        if line.startswith(f"READY port={port}"):
            return proc
    raise RuntimeError("stream server did not print READY in time")


async def _evaluate(manifest_path: str, server_module: str, runs: int,
                    enforce_offline: bool) -> dict:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = json.load(open(manifest_path, encoding='utf-8'))

    base = os.path.dirname(os.path.abspath(manifest_path))
    for c in manifest:
        wav = c.get("audio_local") or os.path.join(base, os.path.basename(c.get("audio", c["clip_id"] + ".wav")))
        if not os.path.exists(wav):
            wav = os.path.join(base, c["clip_id"] + ".wav")
        c["_wav"] = wav

    port = _free_port()
    proc = _start_server(server_module, port)
    uri = f"ws://127.0.0.1:{port}"
    try:
        # warm-up on the first clip (unscored), then go offline
        warm = _read_pcm_16k_mono(manifest[0]["_wav"])
        await _run_once(uri, "__warmup__", warm)
        if enforce_offline:
            block_network()  # loopback stays open; cloud is blocked from here on

        clips = [await _capture_clip(uri, c, runs) for c in manifest]
        return score_stream_run(clips)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "samples/manifest.json"))
    ap.add_argument("--server-module", default="solution.stream_server")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--no-offline", action="store_true",
                    help="skip block_network() (dev only; official run always enforces)")
    ap.add_argument("--json", action="store_true", help="dump full JSON result")
    args = ap.parse_args()

    res = asyncio.run(_evaluate(args.manifest, args.server_module, args.runs,
                                enforce_offline=not args.no_offline))
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print(f"\n  streaming score   {res['overall_score']}/100")
    print(f"  meaning {res['meaning_mean']}   WER {res['wer_mean']}   churn {res['churn_mean']}")
    print(f"  median end-to-final {res['median_end_to_final_ms']}ms   median TTFS {res['median_ttfs_ms']}ms")
    print(f"  reliability-ok {res['reliability_ok_rate']}   clips capped {res['clips_capped']}/{res['n']}")
    for c in res["clips"]:
        flag = f"  capped@{c['capped_at']}" if c["capped_at"] else ""
        print(f"    {c['clip_id'][:28]:28s} score {c['score']:6}  e2f {c['median_end_to_final_ms']}ms"
              f"  ttfs {c['median_ttfs_ms']}ms{flag}  {';'.join(c['reasons'][:2])}")


if __name__ == "__main__":
    main()
