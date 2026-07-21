"""SEALED HARNESS — DO NOT EDIT. Implement `solution/draft.py` instead.

This is the streaming server builderr provides. You do NOT build a server: you
write ONE function, `draft()`, in solution/draft.py. This file wraps it in the
wire protocol the evaluator speaks, and is replaced byte-for-byte with the
official copy at scoring time — any edits here are ignored (or disqualified).

Wire protocol (loopback WebSocket, one connection per clip):

  evaluator -> server:
    {"type":"start","sample_rate":16000,"format":"pcm_s16le","channels":1,"clip_id":"..."}
    <binary frames>   raw PCM s16le, 20ms each (640 bytes @16kHz), in arrival order
    {"type":"end"}

  server -> evaluator:
    {"type":"partial","text":"...","stable_chars":N}   zero or more, during audio
    {"type":"final","text":"..."}                       exactly one, after end
    {"type":"meta", ...}                                 optional, unscored audit

On warm + accepting it prints exactly `READY port=<PORT>` to stdout; the harness
blocks on that line. There is NO t_ms / seq / pcm_b64 — all timing is the
evaluator's receive clock. `stable_chars` is the length of the leading prefix of
`text` you promise never to rewrite (must be non-decreasing; the committed prefix
may only be extended). Rewriting committed text is counted as churn.

Run:  python -m solution.stream_server --host 127.0.0.1 --port <PORT>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

try:
    import websockets
except Exception as exc:  # pragma: no cover - dependency guard
    sys.stderr.write(
        "stream_server needs `websockets` (pip install -r requirements-streaming.txt): "
        f"{exc!r}\n"
    )
    raise

from solution.draft import draft, draft_reset

# How often (in 20ms frames) to re-run draft() on the rolling buffer while audio
# is still arriving. 25 frames = ~500ms cadence. Sealed: this is a harness knob,
# not a student knob — your draft() must work at whatever cadence we choose.
DRAFT_EVERY_FRAMES = 25


async def _handle(ws):
    buf = bytearray()
    started = False
    frames_since_draft = 0
    last_partial = ("", 0)
    draft_reset()  # clear any per-clip state the student kept

    async for message in ws:
        if isinstance(message, (bytes, bytearray)):
            # binary is the ONLY audio path
            if not started:
                continue
            buf.extend(message)
            frames_since_draft += 1
            if frames_since_draft >= DRAFT_EVERY_FRAMES:
                frames_since_draft = 0
                text, stable = _safe_draft(bytes(buf), False)
                if text != last_partial[0] or stable != last_partial[1]:
                    last_partial = (text, stable)
                    await ws.send(json.dumps(
                        {"type": "partial", "text": text, "stable_chars": int(stable)}))
            continue

        # control frames are JSON text
        try:
            msg = json.loads(message)
        except (ValueError, TypeError):
            continue
        mtype = msg.get("type")
        if mtype == "start":
            started = True
            buf = bytearray()
            frames_since_draft = 0
            last_partial = ("", 0)
            draft_reset()
        elif mtype == "end":
            text, _ = _safe_draft(bytes(buf), True)
            await ws.send(json.dumps({"type": "final", "text": text}))
            # optional audit (unscored)
            await ws.send(json.dumps(
                {"type": "meta", "local_only": True, "bytes": len(buf)}))
            return


def _safe_draft(audio: bytes, is_final: bool):
    """Never let a buggy draft() crash the connection; clamp the contract."""
    try:
        out = draft(audio, is_final)
    except Exception:  # noqa: BLE001 - reliability: degrade, don't drop
        return ("", 0)
    if not isinstance(out, tuple) or len(out) != 2:
        return ("", 0)
    text, stable = out
    text = text if isinstance(text, str) else ""
    try:
        stable = int(stable)
    except (TypeError, ValueError):
        stable = 0
    stable = max(0, min(stable, len(text)))
    return (text, stable)


async def _serve(host: str, port: int):
    async with websockets.serve(_handle, host, port, max_size=None, ping_interval=None, ping_timeout=None):
        sys.stdout.write(f"READY port={port}\n")
        sys.stdout.flush()
        await asyncio.Future()  # run forever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    try:
        asyncio.run(_serve(args.host, args.port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
