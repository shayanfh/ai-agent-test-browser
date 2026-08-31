import asyncio
import io
import json
import logging
import math
import os
import re
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from moonshine_voice import ModelArch, Transcriber, TranscriptEventListener, get_model_for_language

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("voicebench")

LANGUAGES = ("en", "ar")
LANGUAGE_NAMES = {"en": "English", "ar": "العربية"}
LLM_MODELS = {
    "gemma": {
        "name": "Gemma 3 1B",
        "api_model": os.getenv("GEMMA_MODEL_NAME", "gemma-3-1b-it"),
        "url": os.getenv("GEMMA_URL", "http://llama-gemma:8080"),
    },
    "qwen": {
        "name": "Qwen3-1.7B",
        "api_model": os.getenv("QWEN_MODEL_NAME", "Qwen3-1.7B"),
        "url": os.getenv("QWEN_URL", "http://llama-qwen:8080"),
    },
}
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemma")
if DEFAULT_LLM_MODEL not in LLM_MODELS:
    DEFAULT_LLM_MODEL = "gemma"
TTS_URLS = {
    "en": os.getenv("PIPER_EN_URL", os.getenv("PIPER_URL", "http://piper:5000")),
    "ar": os.getenv("NABRA_URL", "http://nabra:8000"),
}
TTS_NAMES = {"en": "Piper Amy", "ar": "Nabra-82M"}
SYSTEM_PROMPTS = {
    "en": os.getenv("SYSTEM_PROMPT_EN", os.getenv("SYSTEM_PROMPT", "You are a concise voice assistant. Reply naturally in one to three short sentences. Avoid markdown.")),
    "ar": os.getenv("SYSTEM_PROMPT_AR", "أنت مساعد صوتي موجز. أجب بالعربية بشكل طبيعي في جملة إلى ثلاث جمل قصيرة، وتجنب تنسيق Markdown."),
}
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "160"))
MAX_TOKENS_AR = int(os.getenv("MAX_TOKENS_AR", "48"))
ARABIC_MAX_WORDS = int(os.getenv("ARABIC_MAX_WORDS", "22"))
HISTORY_MAX_MESSAGES = max(8, int(os.getenv("HISTORY_MAX_MESSAGES", "16")))
MEMORY_MAX_ITEMS = max(4, int(os.getenv("MEMORY_MAX_ITEMS", "12")))
ARABIC_BREVITY_RULE = os.getenv(
    "ARABIC_BREVITY_RULE",
    "قاعدة إلزامية: أجب بجملة عربية واحدة قصيرة لا تتجاوز 22 كلمة، واسأل سؤالاً واحداً فقط في كل دور.",
)
SYSTEM_PROMPTS["ar"] = f"{SYSTEM_PROMPTS['ar']}\n{ARABIC_BREVITY_RULE}"
ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
LATIN_RE = re.compile(r"[A-Za-z]")
ROUTE_PHRASES = {
    "ar": (
        "السلام عليكم", "وعليكم السلام", "صباح الخير", "مساء الخير",
        "مرحبا", "أهلا", "لو سمحت", "من فضلك", "شكرا",
    ),
    "en": (
        "hello", "good morning", "good evening", "how are you",
        "please", "thank you", "thanks",
    ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.transcribers = {}
    for language in LANGUAGES:
        wanted_arch = ModelArch.MEDIUM_STREAMING if language == "en" else ModelArch.TINY_STREAMING
        log.info("Preloading Moonshine %s (%s)", "Medium Streaming" if language == "en" else "Tiny Streaming", language)
        model_path, model_arch = get_model_for_language(language, wanted_arch)
        options = {
            "vad_threshold": os.getenv("MOONSHINE_VAD_THRESHOLD", "0.35"),
            "vad_max_segment_duration": "120",
        }
        debug_audio_path = os.getenv("MOONSHINE_DEBUG_AUDIO_PATH", "").strip()
        if debug_audio_path:
            options["save_input_wav_path"] = f"{debug_audio_path}.{language}.wav"
        app.state.transcribers[language] = Transcriber(model_path=model_path, model_arch=model_arch, options=options)
        log.info("Moonshine %s ready from %s", language, model_path)
    yield
    for transcriber in app.state.transcribers.values():
        transcriber.close()


app = FastAPI(title="VoiceBench", version="2.0.0", lifespan=lifespan)


@app.get("/api/health")
async def health():
    downstream = {}
    checks = {
        "llm_gemma": f"{LLM_MODELS['gemma']['url']}/health",
        "llm_qwen": f"{LLM_MODELS['qwen']['url']}/health",
        "tts_en": f"{TTS_URLS['en']}/",
        "tts_ar": f"{TTS_URLS['ar']}/health",
    }
    async with httpx.AsyncClient(timeout=2.0) as client:
        for name, url in checks.items():
            try:
                response = await client.get(url)
                downstream[name] = response.status_code < 500
            except httpx.HTTPError:
                downstream[name] = False
    llm_ready = downstream["llm_gemma"] and downstream["llm_qwen"]
    tts_ready = downstream["tts_en"] and downstream["tts_ar"]
    ready = llm_ready and tts_ready
    payload = {
        "status": "ok" if ready else "warming", "stt": True,
        "llm": llm_ready, "tts": tts_ready, "routes": downstream,
        "languages": list(LANGUAGES),
        "stt_models": {"en": "moonshine-medium-streaming", "ar": "moonshine-tiny-streaming-ar-27m"},
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


@dataclass
class TranscriptCandidate:
    latest_text: str = ""
    final_text: str = ""
    first_partial_at: float | None = None
    updates: int = 0
    stability: float = 0.5
    revision_chars: int = 0


class RouteListener(TranscriptEventListener):
    def __init__(self, turn: "VoiceTurn", language: str):
        self.turn = turn
        self.language = language

    def _update(self, text: str, complete: bool = False):
        text = (text or "").strip()
        if not text:
            return
        candidate = self.turn.candidates[self.language]
        if text != candidate.latest_text:
            previous = candidate.latest_text
            if previous:
                common = 0
                for old_char, new_char in zip(previous, text):
                    if old_char != new_char:
                        break
                    common += 1
                retained = common / max(1, min(len(previous), len(text)))
                candidate.stability = candidate.stability * 0.65 + retained * 0.35
                candidate.revision_chars += max(0, len(previous) - common)
            candidate.latest_text = text
            candidate.updates += 1
        if candidate.first_partial_at is None:
            candidate.first_partial_at = time.perf_counter()
        if complete:
            candidate.final_text = text

    def on_line_started(self, event):
        self._update(event.line.text)

    def on_line_text_changed(self, event):
        self._update(event.line.text)

    def on_line_completed(self, event):
        self._update(event.line.text, True)


def route_score(language: str, candidate: TranscriptCandidate) -> float:
    text = candidate.final_text or candidate.latest_text
    if not text:
        return -100.0
    arabic = len(ARABIC_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    native = arabic if language == "ar" else latin
    foreign = latin if language == "ar" else arabic
    purity = native / max(1, native + foreign)
    normalized = " ".join(text.lower().split())
    phrase_bonus = max((4.0 if phrase in normalized else 0.0) for phrase in ROUTE_PHRASES[language])
    # Character count is deliberately capped very early. Two monolingual
    # recognizers can both hallucinate fluent-looking text for the same audio;
    # a longer hallucination must not beat a shorter correct transcript.
    presence = min(native, 6) / 3.0
    revision_penalty = min(2.0, candidate.revision_chars / 12.0)
    return (
        presence + purity * 4.0 - foreign * 0.35
        + min(candidate.updates, 4) * 0.35
        + candidate.stability * 1.5 - revision_penalty
        + phrase_bonus + (1.0 if candidate.final_text else 0.0)
    )


@dataclass
class VoiceTurn:
    transcribers: dict[str, Transcriber]
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    audio_samples: int = 0
    peak: float = 0.0
    energy_sum: float = 0.0
    candidates: dict[str, TranscriptCandidate] = field(default_factory=lambda: {language: TranscriptCandidate() for language in LANGUAGES})
    last_snapshot_key: tuple = field(default_factory=tuple)

    def __post_init__(self):
        self.streams = {}
        self.listeners = {}
        for language, transcriber in self.transcribers.items():
            stream = transcriber.create_stream(update_interval=0.18)
            listener = RouteListener(self, language)
            stream.add_listener(listener)
            stream.start()
            self.streams[language] = stream
            self.listeners[language] = listener

    def add_pcm16(self, raw: bytes):
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        self.audio_samples += len(samples)
        if len(samples):
            self.peak = max(self.peak, float(np.max(np.abs(samples))))
            self.energy_sum += float(np.dot(samples, samples))
        audio = samples.tolist()
        for stream in self.streams.values():
            stream.add_audio(audio, 16000)

    def stop(self):
        self.ended_at = time.perf_counter()
        for stream in self.streams.values():
            stream.stop()
        for stream in self.streams.values():
            stream.close()

    def routing_snapshot(self, final: bool = False, preferred_language: str | None = None) -> dict:
        scores = {language: route_score(language, candidate) for language, candidate in self.candidates.items()}
        partial_times = [candidate.first_partial_at for candidate in self.candidates.values() if candidate.first_partial_at is not None]
        if partial_times:
            earliest = min(partial_times)
            for language, candidate in self.candidates.items():
                if candidate.first_partial_at is not None:
                    # First-partial time is only a small tie-breaker. A wrong
                    # recognizer can emit a hallucination before the matching
                    # recognizer has accumulated enough speech.
                    scores[language] += max(0.0, 0.5 - (candidate.first_partial_at - earliest))
        if preferred_language in LANGUAGES and self.candidates[preferred_language].latest_text:
            # Telephone conversations normally stay in one language. Preserve
            # that route with a modest bias, while still allowing a clear
            # code-switch to win.
            scores[preferred_language] += 2.0
        language = max(scores, key=scores.get)
        candidate = self.candidates[language]
        text = candidate.final_text or candidate.latest_text
        if not text:
            other = "ar" if language == "en" else "en"
            other_candidate = self.candidates[other]
            if other_candidate.final_text or other_candidate.latest_text:
                language, candidate = other, other_candidate
                text = candidate.final_text or candidate.latest_text
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
        native_count = len((ARABIC_RE if language == "ar" else LATIN_RE).findall(text))
        stable = bool(text) and native_count >= 2 and (candidate.updates >= 2 or final) and (final or margin >= 1.0)
        confidence = max(0.0, min(0.99, 0.5 + margin / 20.0)) if stable else 0.0
        return {
            "language": language, "language_name": LANGUAGE_NAMES[language],
            "text": text, "confidence": confidence, "stable": stable,
            "first_partial_ms": ((candidate.first_partial_at or time.perf_counter()) - self.started_at) * 1000,
            "scores": {key: round(value, 2) for key, value in scores.items()},
            "candidates": {key: value.final_text or value.latest_text for key, value in self.candidates.items()},
            "router_debug": {
                key: {"updates": value.updates, "stability": round(value.stability, 3), "revision_chars": value.revision_chars}
                for key, value in self.candidates.items()
            },
        }


async def send_json(websocket: WebSocket, lock: asyncio.Lock, payload: dict):
    async with lock:
        await websocket.send_json(payload)


def remember_utterance(memory: list[str], text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return
    normalized = text.casefold().strip(" .،؛؟!?")
    if any(item.casefold().strip(" .،؛؟!?") == normalized for item in memory):
        return
    memory.append(text)
    del memory[:-MEMORY_MAX_ITEMS]


def prompt_with_memory(language: str, memory: list[str]) -> str:
    if not memory:
        return SYSTEM_PROMPTS[language]
    facts = "\n".join(f"{index}. {item}" for index, item in enumerate(memory, 1))
    if language == "ar":
        rule = (
            "\n\nذاكرة المكالمة — هذه عبارات قالها العميل بالفعل:\n"
            f"{facts}\n"
            "قواعد إلزامية: راجع الذاكرة قبل الرد. لا تطلب من العميل أي معلومة سبق أن قالها. "
            "اعتبر الرد القصير إجابة عن سؤالك السابق. علّق على آخر كلامه ثم اسأل فقط عن أول معلومة ناقصة. "
            "إذا ذكر نوع الساندويتش أو اختياره فلا تسأله عن الساندويتش مرة أخرى."
        )
    else:
        rule = (
            "\n\nCall memory — the customer already said:\n"
            f"{facts}\n"
            "Mandatory rules: review this memory before replying. Never ask again for information already provided. "
            "Treat a short reply as the answer to your previous question, acknowledge it, and ask only for the next missing detail."
        )
    return SYSTEM_PROMPTS[language] + rule


def split_tts_chunk(buffer: str, force: bool = False):
    if not buffer.strip():
        return None, ""
    match = re.search(r"^(.{20,}?[.!?؟](?:\s|$))", buffer, re.S)
    if match:
        return match.group(1).strip(), buffer[match.end():]
    if len(buffer) >= 120 or force:
        cut = len(buffer) if force else buffer.rfind(" ", 40, 120)
        if cut < 1:
            cut = min(120, len(buffer))
        return buffer[:cut].strip(), buffer[cut:]
    return None, buffer


def limit_streamed_response(answer: str, delta: str, language: str) -> tuple[str, bool]:
    """Hard-stop verbose Arabic output before extra text reaches the UI/TTS."""
    if language != "ar":
        return delta, False
    combined = answer + delta
    cut = len(combined)
    should_stop = False
    sentence_end = re.search(r"[.!؟](?=\s|$)", combined)
    if sentence_end:
        cut = min(cut, sentence_end.end())
        should_stop = True
    words = list(re.finditer(r"\S+", combined))
    if len(words) >= ARABIC_MAX_WORDS:
        cut = min(cut, words[ARABIC_MAX_WORDS - 1].end())
        should_stop = True
    bounded = combined[:cut]
    return bounded[len(answer):], should_stop


def wav_duration(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except (wave.Error, EOFError):
        return 0.0


async def tts_worker(websocket: WebSocket, lock: asyncio.Lock, queue: asyncio.Queue, ended_at: float, language: str):
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
                endpoint = "/synthesize" if language == "ar" else "/"
                response = await client.post(
                    f"{TTS_URLS[language]}{endpoint}",
                    json={"text": text},
                )
                response.raise_for_status()
                audio = response.content
                finished = time.perf_counter()
                duration = wav_duration(audio)
                generation = max(0.001, finished - started)
                event = {
                    "type": "tts_audio", "sequence": sequence, "text": text, "language": language,
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
                await send_json(websocket, lock, {"type": "error", "stage": "tts", "message": f"{TTS_NAMES[language]} failed: {exc}"})
            finally:
                queue.task_done()


def probable_silence_hallucination(text: str, language: str, rms: float) -> bool:
    if language == "ar":
        normalized = re.sub(r"[^\u0600-\u06ff ]", "", text).strip()
        common = {"شكرا", "شكرا لكم", "شكرا على المشاهدة", "إلى اللقاء", "مع السلامة"}
    else:
        normalized = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
        common = {"thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "goodbye", "you", "the end"}
    return normalized in common and rms < 0.022


async def process_turn(websocket: WebSocket, lock: asyncio.Lock, turn: VoiceTurn, history: list[dict], memory: list[str], route: dict, model_key: str):
    language = route["language"]
    model = LLM_MODELS[model_key]
    text = route["text"]
    duration = turn.audio_samples / 16000
    rms = math.sqrt(turn.energy_sum / max(1, turn.audio_samples))
    if not text:
        if duration < 0.5:
            message = f"Only {duration:.1f}s of audio arrived. Speak for at least one second."
        elif turn.peak < 0.01:
            message = f"The microphone is almost silent (peak {turn.peak:.3f}). Check the selected input device."
        else:
            message = "Audio arrived, but neither English nor Arabic Moonshine produced a transcript."
        await send_json(websocket, lock, {"type": "error", "stage": "stt", "message": message})
        return
    if rms < 0.007 or duration < 0.55 or probable_silence_hallucination(text, language, rms):
        log.info("Ignoring probable %s silence hallucination text=%r duration=%.2f rms=%.4f peak=%.3f", language, text, duration, rms, turn.peak)
        await send_json(websocket, lock, {"type": "stt_ignored", "text": text, "language": language, "reason": "low_energy", "audio_rms": rms, "audio_peak": turn.peak})
        return

    remember_utterance(memory, text)
    if history and history[-1].get("role") == "user":
        history.pop()
    history[:] = history[-HISTORY_MAX_MESSAGES:]
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
    worker = asyncio.create_task(tts_worker(websocket, lock, tts_queue, turn.ended_at or llm_started, language))
    try:
        payload = {
            "model": model["api_model"],
            "messages": [{"role": "system", "content": prompt_with_memory(language, memory)}, *history],
            "stream": True,
            "max_tokens": MAX_TOKENS_AR if language == "ar" else MAX_TOKENS,
            "temperature": 0.5 if language == "ar" else 0.6,
        }
        if model_key == "qwen":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{model['url']}/v1/chat/completions", json=payload) as response:
                if not response.is_success:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(f"llama.cpp {model_key} returned HTTP {response.status_code}: {body[:800]}")
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
                    raw_delta = choices[0].get("delta", {}).get("content", "")
                    if not raw_delta:
                        continue
                    delta, should_stop = limit_streamed_response(answer, raw_delta, language)
                    if not delta and should_stop:
                        break
                    if not delta:
                        continue
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    if not usage.get("completion_tokens"):
                        token_count += 1
                    answer += delta
                    tts_buffer += delta
                    await send_json(websocket, lock, {"type": "llm_token", "delta": delta, "language": language, "model": model_key, "ttft_ms": (first_token_at - llm_started) * 1000})
                    chunk, tts_buffer = split_tts_chunk(tts_buffer)
                    if chunk:
                        await tts_queue.put(chunk)
                    if should_stop:
                        break
        chunk, tts_buffer = split_tts_chunk(tts_buffer, force=True)
        if chunk:
            await tts_queue.put(chunk)
        llm_finished = time.perf_counter()
        elapsed = max(0.001, llm_finished - (first_token_at or llm_started))
        await send_json(websocket, lock, {"type": "llm_done", "language": language, "model": model_key, "memory_items": len(memory), "total_ms": (llm_finished - llm_started) * 1000, "tokens": token_count, "tokens_per_second": server_tokens_per_second or token_count / elapsed})
        if answer:
            history.append({"role": "assistant", "content": answer})
    except Exception as exc:
        if history and history[-1].get("role") == "user" and history[-1].get("content") == text:
            history.pop()
        await send_json(websocket, lock, {"type": "error", "stage": "llm", "message": f"{model['name']} failed: {exc}"})
    finally:
        await tts_queue.put(None)
        await tts_queue.join()
        await worker
        await send_json(websocket, lock, {"type": "turn_done", "language": language, "model": model_key})


@app.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    lock = asyncio.Lock()
    # Keep one conversation across route changes, so callers can switch
    # language mid-call without losing the previous order/context.
    history: list[dict] = []
    memory: list[str] = []
    preferred_language: str | None = None
    selected_model = DEFAULT_LLM_MODEL
    turn: VoiceTurn | None = None
    await websocket.send_json({
        "type": "hello", "sample_rate": 16000, "languages": list(LANGUAGES),
        "stt": {"en": "moonshine-medium-streaming", "ar": "moonshine-tiny-streaming-ar-27m"},
        "llm": {
            "selected": selected_model,
            "models": [{"id": key, "name": value["name"]} for key, value in LLM_MODELS.items()],
        },
        "tts": {"en": "en_US-amy-medium", "ar": "Nabra-82M-v0.1 / af_msa"},
    })
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                if turn:
                    await asyncio.to_thread(turn.add_pcm16, message["bytes"])
                    route = turn.routing_snapshot(preferred_language=preferred_language)
                    key = (route["language"], route["text"], route["stable"])
                    if route["text"] and key != turn.last_snapshot_key:
                        turn.last_snapshot_key = key
                        await send_json(websocket, lock, {"type": "stt_partial", **route})
                continue
            raw = message.get("text")
            if not raw:
                continue
            event = json.loads(raw)
            if event.get("type") == "conversation_start":
                history.clear()
                memory.clear()
                preferred_language = None
                requested_model = event.get("model", DEFAULT_LLM_MODEL)
                if requested_model not in LLM_MODELS:
                    await send_json(websocket, lock, {"type": "error", "stage": "llm", "message": f"Unknown LLM model: {requested_model}"})
                    continue
                selected_model = requested_model
                await send_json(websocket, lock, {"type": "conversation_ready", "language": "auto", "model": selected_model})
            elif event.get("type") == "conversation_end":
                history.clear()
                memory.clear()
                preferred_language = None
            elif event.get("type") == "start":
                if turn:
                    await asyncio.to_thread(turn.stop)
                turn = await asyncio.to_thread(VoiceTurn, app.state.transcribers)
                await send_json(websocket, lock, {"type": "recording", "language": "auto"})
            elif event.get("type") == "stop" and turn:
                await asyncio.to_thread(turn.stop)
                ended = turn.ended_at or time.perf_counter()
                route = turn.routing_snapshot(final=True, preferred_language=preferred_language)
                stt_model = "moonshine-medium-streaming" if route["language"] == "en" else "moonshine-tiny-streaming-ar-27m"
                if route["stable"] and route["confidence"] >= 0.58:
                    preferred_language = route["language"]
                log.info(
                    "Language route=%s confidence=%.3f scores=%s candidates=%s debug=%s",
                    route["language"], route["confidence"], route["scores"],
                    route["candidates"], route["router_debug"],
                )
                rms = math.sqrt(turn.energy_sum / max(1, turn.audio_samples))
                await send_json(websocket, lock, {"type": "language_detected", **route})
                await send_json(websocket, lock, {
                    "type": "stt_final", **route,
                    "speech_duration_ms": turn.audio_samples / 16.0,
                    "stt_final_ms": (time.perf_counter() - ended) * 1000,
                    "stt_model": stt_model,
                    "audio_peak": turn.peak, "audio_rms": rms,
                })
                await process_turn(websocket, lock, turn, history, memory, route, selected_model)
                turn = None
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Voice socket failed")
        try:
            await send_json(websocket, lock, {"type": "error", "message": "The voice session stopped unexpectedly. Check backend logs."})
        except Exception:
            pass
    finally:
        if turn:
            try:
                await asyncio.to_thread(turn.stop)
            except Exception:
                pass
