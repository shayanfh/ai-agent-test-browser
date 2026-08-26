import asyncio
import io
import json
import logging
import os
import re
import time
import wave
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from moonshine_voice import ModelArch, Transcriber, TranscriptEventListener, get_model_for_language

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("voicebench")

LLAMA_URL = os.getenv("LLAMA_URL", "http://llama:8080")
PIPER_URL = os.getenv("PIPER_URL", "http://piper:5000")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a concise voice assistant. Reply naturally in one to three short sentences. Avoid markdown.")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "160"))
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-1b-it")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Moonshine Tiny Streaming")
    model_path, model_arch = get_model_for_language("en", ModelArch.TINY_STREAMING)
    # Browser-side VAD provides utterance boundaries; Moonshine's moderately
    # permissive VAD remains enabled as a second guard against room noise.
    options = {
        "vad_threshold": os.getenv("MOONSHINE_VAD_THRESHOLD", "0.35"),
        "vad_max_segment_duration": "120",
    }
    debug_audio_path = os.getenv("MOONSHINE_DEBUG_AUDIO_PATH", "").strip()
    if debug_audio_path:
        options["save_input_wav_path"] = debug_audio_path
    app.state.transcriber = Transcriber(model_path=model_path, model_arch=model_arch, options=options)
    log.info("Moonshine ready from %s", model_path)
    yield
    app.state.transcriber.close()


app = FastAPI(title="VoiceBench", version="1.0.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    downstream = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        # Piper's lightweight Flask server exposes its UI at `/` in all
        # supported releases, while the optional `/info` route varies.
        for name, url in (("llm", f"{LLAMA_URL}/health"), ("tts", f"{PIPER_URL}/")):
            try:
                response = await client.get(url)
                # Health here means the HTTP process is reachable. Some Piper
                # releases omit optional metadata routes and answer 404 while
                # the synthesis endpoint is fully operational.
                downstream[name] = response.status_code < 500
            except httpx.HTTPError:
                downstream[name] = False
    ready = all(downstream.values())
    return JSONResponse({"status": "ok" if ready else "warming", "stt": True, **downstream}, status_code=200 if ready else 503)


class Listener(TranscriptEventListener):
    def __init__(self, turn: "VoiceTurn"):
        self.turn = turn

    def _update(self, text: str, complete: bool = False):
        text = (text or "").strip()
        if not text:
            return
        self.turn.latest_text = text
        if self.turn.first_partial_at is None:
            self.turn.first_partial_at = time.perf_counter()
        if complete:
            self.turn.final_text = text

    def on_line_started(self, event):
        self._update(event.line.text)

    def on_line_text_changed(self, event):
        self._update(event.line.text)

    def on_line_completed(self, event):
        self._update(event.line.text, True)


@dataclass
class VoiceTurn:
    transcriber: Transcriber
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    audio_samples: int = 0
    latest_text: str = ""
    final_text: str = ""
    last_sent_text: str = ""
    first_partial_at: float | None = None
    peak: float = 0.0
    energy_sum: float = 0.0

    def __post_init__(self):
        self.stream = self.transcriber.create_stream(update_interval=0.18)
        self.listener = Listener(self)
        self.stream.add_listener(self.listener)
        self.stream.start()

    def add_pcm16(self, raw: bytes):
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        self.audio_samples += len(samples)
        if len(samples):
            self.peak = max(self.peak, float(np.max(np.abs(samples))))
            self.energy_sum += float(np.dot(samples, samples))
        self.stream.add_audio(samples.tolist(), 16000)

    def stop(self):
        self.ended_at = time.perf_counter()
        self.stream.stop()
        self.stream.close()


async def send_json(websocket: WebSocket, lock: asyncio.Lock, payload: dict):
    async with lock:
        await websocket.send_json(payload)


def split_tts_chunk(buffer: str, force: bool = False):
    if not buffer.strip():
        return None, ""
    match = re.search(r"^(.{20,}?[.!?](?:\s|$))", buffer, re.S)
    if match:
        return match.group(1).strip(), buffer[match.end():]
    if len(buffer) >= 120 or force:
        cut = len(buffer) if force else buffer.rfind(" ", 40, 120)
        if cut < 1:
            cut = min(120, len(buffer))
        return buffer[:cut].strip(), buffer[cut:]
    return None, buffer


def wav_duration(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except (wave.Error, EOFError):
        return 0.0


async def tts_worker(websocket: WebSocket, lock: asyncio.Lock, queue: asyncio.Queue, ended_at: float):
    first = True
    sequence = 0
    async with httpx.AsyncClient(timeout=90.0) as client:
        while True:
            text = await queue.get()
            if text is None:
                queue.task_done()
                return
            started = time.perf_counter()
            try:
                # piper-tts 1.4.x serves synthesis as POST `/`. Newer variants
                # may expose `/synthesize`, but the pinned runtime uses root.
                response = await client.post(f"{PIPER_URL}/", json={"text": text})
                response.raise_for_status()
                audio = response.content
                finished = time.perf_counter()
                duration = wav_duration(audio)
                generation = max(0.001, finished - started)
                event = {
                    "type": "tts_audio", "sequence": sequence, "text": text,
                    "ttfa_ms": generation * 1000, "audio_duration_ms": duration * 1000,
                    "realtime_factor": duration / generation,
                    "end_speech_to_audio_ms": (finished - ended_at) * 1000 if first else None,
                }
                async with lock:
                    await websocket.send_json(event)
                    await websocket.send_bytes(audio)
                first = False
                sequence += 1
            except Exception as exc:
                await send_json(websocket, lock, {"type": "error", "stage": "tts", "message": f"Piper failed: {exc}"})
            finally:
                queue.task_done()


async def process_turn(websocket: WebSocket, lock: asyncio.Lock, turn: VoiceTurn, history: list[dict]):
    text = turn.final_text or turn.latest_text
    if not text:
        duration = turn.audio_samples / 16000
        rms = math.sqrt(turn.energy_sum / max(1, turn.audio_samples))
        if duration < 0.5:
            message = f"Only {duration:.1f}s of audio arrived. Hold the button while speaking for at least one second."
        elif turn.peak < 0.01:
            message = f"The microphone signal is almost silent (peak {turn.peak:.3f}). Check the selected input device and browser microphone level."
        else:
            message = f"Audio arrived (peak {turn.peak:.2f}, RMS {rms:.3f}) but Moonshine produced no English transcript. Try a clear English sentence."
        await send_json(websocket, lock, {"type": "error", "stage": "stt", "message": message})
        return
    duration = turn.audio_samples / 16000
    rms = math.sqrt(turn.energy_sum / max(1, turn.audio_samples))
    normalized = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
    common_silence_hallucinations = {
        "thank you", "thanks", "thank you for watching", "thanks for watching",
        "bye", "goodbye", "you", "the end",
    }
    if rms < 0.007 or duration < 0.55 or (normalized in common_silence_hallucinations and rms < 0.022):
        log.info("Ignoring probable silence hallucination text=%r duration=%.2f rms=%.4f peak=%.3f", text, duration, rms, turn.peak)
        await send_json(websocket, lock, {"type": "stt_ignored", "text": text, "reason": "low_energy", "audio_rms": rms, "audio_peak": turn.peak})
        return
    # Keep complete user/assistant pairs. Trimming after appending the new user
    # can leave an assistant message at the front, which Gemma rejects as an
    # invalid role sequence once a call reaches several turns.
    if history and history[-1].get("role") == "user":
        history.pop()
    history[:] = history[-6:]
    while history and history[0].get("role") != "user":
        history.pop(0)
    history.append({"role": "user", "content": text})
    llm_started = time.perf_counter()
    first_token_at = None
    token_count = 0
    server_tokens_per_second = None
    answer = ""
    tts_buffer = ""
    tts_queue: asyncio.Queue = asyncio.Queue()
    worker = asyncio.create_task(tts_worker(websocket, lock, tts_queue, turn.ended_at or llm_started))
    try:
        payload = {"model": MODEL_NAME, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history], "stream": True, "max_tokens": MAX_TOKENS, "temperature": 0.6}
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{LLAMA_URL}/v1/chat/completions", json=payload) as response:
                if not response.is_success:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"llama.cpp returned HTTP {response.status_code}: {body[:800]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    timings = event.get("timings") or {}
                    usage = event.get("usage") or {}
                    if isinstance(timings.get("predicted_per_second"), (int, float)):
                        server_tokens_per_second = float(timings["predicted_per_second"])
                    if usage.get("completion_tokens"):
                        token_count = int(usage["completion_tokens"])
                    choices = event.get("choices") or [{}]
                    delta = choices[0].get("delta", {}).get("content", "")
                    if not delta:
                        continue
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    if not usage.get("completion_tokens"):
                        token_count += 1
                    answer += delta
                    tts_buffer += delta
                    await send_json(websocket, lock, {"type": "llm_token", "delta": delta, "ttft_ms": (first_token_at - llm_started) * 1000})
                    chunk, tts_buffer = split_tts_chunk(tts_buffer)
                    if chunk:
                        await tts_queue.put(chunk)
        chunk, tts_buffer = split_tts_chunk(tts_buffer, force=True)
        if chunk:
            await tts_queue.put(chunk)
        llm_finished = time.perf_counter()
        elapsed = max(0.001, llm_finished - (first_token_at or llm_started))
        await send_json(websocket, lock, {"type": "llm_done", "total_ms": (llm_finished - llm_started) * 1000, "tokens": token_count, "tokens_per_second": server_tokens_per_second or token_count / elapsed})
        if answer:
            history.append({"role": "assistant", "content": answer})
    except Exception as exc:
        if history and history[-1].get("role") == "user" and history[-1].get("content") == text:
            history.pop()
        await send_json(websocket, lock, {"type": "error", "stage": "llm", "message": f"Gemma failed: {exc}"})
    finally:
        await tts_queue.put(None)
        await tts_queue.join()
        await worker
        await send_json(websocket, lock, {"type": "turn_done"})


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    lock = asyncio.Lock()
    history: list[dict] = []
    turn: VoiceTurn | None = None
    await websocket.send_json({"type": "hello", "sample_rate": 16000, "stt": "moonshine-tiny-streaming", "llm": MODEL_NAME, "tts": "en_US-lessac-medium"})
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                if turn:
                    await asyncio.to_thread(turn.add_pcm16, message["bytes"])
                    if turn.latest_text and turn.latest_text != turn.last_sent_text:
                        turn.last_sent_text = turn.latest_text
                        await send_json(websocket, lock, {"type": "stt_partial", "text": turn.latest_text, "first_partial_ms": ((turn.first_partial_at or time.perf_counter()) - turn.started_at) * 1000})
                continue
            raw = message.get("text")
            if not raw:
                continue
            event = json.loads(raw)
            if event.get("type") == "start":
                if turn:
                    await asyncio.to_thread(turn.stop)
                turn = await asyncio.to_thread(VoiceTurn, app.state.transcriber)
                await send_json(websocket, lock, {"type": "recording"})
            elif event.get("type") == "stop" and turn:
                await asyncio.to_thread(turn.stop)
                ended = turn.ended_at or time.perf_counter()
                rms = math.sqrt(turn.energy_sum / max(1, turn.audio_samples))
                await send_json(websocket, lock, {"type": "stt_final", "text": turn.final_text or turn.latest_text, "speech_duration_ms": turn.audio_samples / 16.0, "stt_final_ms": (time.perf_counter() - ended) * 1000, "audio_peak": turn.peak, "audio_rms": rms})
                await process_turn(websocket, lock, turn, history)
                turn = None
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Voice socket failed")
        try:
            await send_json(websocket, lock, {"type": "error", "message": "The voice session stopped unexpectedly. Check the backend logs."})
        except Exception:
            pass
    finally:
        if turn:
            try:
                await asyncio.to_thread(turn.stop)
            except Exception:
                pass
