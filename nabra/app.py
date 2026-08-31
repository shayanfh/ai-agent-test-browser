import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from kokoro import KModel, KPipeline
from kokoro import pipeline as kokoro_pipeline
from pydantic import BaseModel, Field

from arabic_frontend import ArabicDiacritizer, EXTRA_SYMBOLS, clean_phonemes

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("nabra")

MODEL_ROOT = Path(os.getenv("NABRA_MODEL_ROOT", "/models/nabra"))
MODEL_PATH = Path(os.getenv("NABRA_MODEL_PATH", MODEL_ROOT / "kokoro_arabic.pth"))
CONFIG_PATH = Path(os.getenv("NABRA_CONFIG_PATH", MODEL_ROOT / "config.json"))
VOICE_PATH = Path(os.getenv("NABRA_VOICE_PATH", MODEL_ROOT / "af_msa.pt"))
SAMPLE_RATE = 24000
DEFAULT_SPEED = float(os.getenv("NABRA_SPEED", "1.0"))


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    speed: float = Field(default=DEFAULT_SPEED, ge=0.5, le=1.5)


def load_runtime(app: FastAPI):
    missing = [str(path) for path in (MODEL_PATH, CONFIG_PATH, VOICE_PATH) if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing Nabra assets: {', '.join(missing)}")

    threads = max(1, int(os.getenv("NABRA_THREADS", "4")))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    log.info("Loading Nabra-82M on CPU with %d threads", threads)

    model = KModel(
        repo_id="oddadmix/Nabra-82M-v0.1",
        config=str(CONFIG_PATH),
        model=str(MODEL_PATH),
        disable_complex=True,
    ).eval()
    model.vocab.update(EXTRA_SYMBOLS)

    kokoro_pipeline.LANG_CODES.setdefault("ar", "ar")
    pipeline = KPipeline(
        lang_code="ar",
        repo_id="oddadmix/Nabra-82M-v0.1",
        model=model,
    )
    original_g2p = pipeline.g2p

    def arabic_g2p(text: str):
        phonemes, metadata = original_g2p(text)
        return clean_phonemes(phonemes), metadata

    pipeline.g2p = arabic_g2p
    app.state.pipeline = pipeline
    app.state.voice = torch.load(VOICE_PATH, map_location="cpu", weights_only=True)
    app.state.diacritize = ArabicDiacritizer()
    app.state.lock = asyncio.Lock()
    log.info("Nabra-82M ready (voice=af_msa, sample_rate=%d)", SAMPLE_RATE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(load_runtime, app)
    yield


app = FastAPI(title="Nabra Arabic TTS", version="0.1", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "Nabra-82M-v0.1", "voice": "af_msa"}


def synthesize_wav(app: FastAPI, text: str, speed: float) -> tuple[bytes, str, float]:
    started = time.perf_counter()
    diacritized = app.state.diacritize(text)
    if not diacritized:
        raise ValueError("Text became empty after Arabic normalization")

    chunks = []
    for _, _, audio in app.state.pipeline(diacritized, voice=app.state.voice, speed=speed):
        if torch.is_tensor(audio):
            chunks.append(audio.detach().cpu().numpy())
        else:
            chunks.append(np.asarray(audio))
    if not chunks:
        raise RuntimeError("Nabra produced no audio")

    waveform = np.concatenate(chunks).astype(np.float32)
    output = io.BytesIO()
    sf.write(output, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return output.getvalue(), diacritized, (time.perf_counter() - started) * 1000


@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    try:
        async with app.state.lock:
            audio, diacritized, generation_ms = await asyncio.to_thread(
                synthesize_wav, app, request.text.strip(), request.speed
            )
    except Exception as exc:
        log.exception("Nabra synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-Generation-Ms": f"{generation_ms:.1f}",
            "X-Diacritized-Text": diacritized.encode("utf-8").hex(),
        },
    )
