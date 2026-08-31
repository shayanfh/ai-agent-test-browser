# VoiceBench

Self-hosted latency lab for a fully local voice pipeline:

`Browser PCM → Moonshine Tiny EN + AR → partial-STT language router → Gemma / RightNow Arabic → Piper EN / Nabra Arabic → Browser audio`

The browser streams mono 16 kHz PCM over WebSocket. The backend reports timestamps for partial/final STT, LLM TTFT and throughput, TTS generation, and the key end-of-speech-to-first-audio metric.

## Server requirements

- Linux x86_64 or ARM64
- Docker Engine 24+ with Compose v2
- 8 CPU cores recommended
- 8 GB RAM minimum; 12 GB recommended
- About 5 GB free disk for images, runtimes, and model weights
- A domain pointing at the server for microphone access in production (browsers require HTTPS outside localhost)

No Python, Node.js, model runtime, or model file needs to be installed on the host.

## Deploy

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f model-init llama llama-ar piper nabra backend
```

The first start downloads both STT models, both GGUF models, Piper Amy, and Nabra-82M into the persistent `models` volume. The Nabra image also installs and preloads the MSA diacritizer required for natural pronunciation. This can take several minutes. Later starts reuse the model files. The backend waits for every LLM/TTS service to be healthy, then preloads both Moonshine models. Open `http://SERVER_IP:8080` for a LAN smoke test.

The backend has write access to this volume because Moonshine creates temporary lock files while opening cached model assets. The llama.cpp, Piper, and Nabra containers mount the same assets read-only.

For a public server, set `DOMAIN` in `.env`, point its DNS record to the server, open TCP ports 80/443 and UDP 443, then run:

```bash
docker compose --profile https up -d --build
```

Caddy obtains and renews TLS automatically. Open `https://YOUR_DOMAIN`; microphone capture will then be available as a secure browser feature.

## Operations

```bash
# Status
docker compose ps
curl http://127.0.0.1:8080/api/health

# Follow the inference path
docker compose logs -f backend llama llama-ar piper nabra

# Rebuild after a code change (models are preserved)
docker compose up -d --build

# Stop without deleting models
docker compose down
```

To remove downloaded model data as well, explicitly run `docker compose down -v`.

## Configuration

Copy `.env.example` to `.env` and adjust:

- `LLAMA_EN_THREADS`: CPU threads assigned to Gemma.
- `LLAMA_AR_THREADS`: CPU threads assigned to RightNow Arabic. Only the routed model generates, but both remain resident.
- `MAX_TOKENS`: maximum response length; shorter responses reduce total turn time.
- `MAX_TOKENS_AR`: Arabic generation ceiling (default `48`).
- `ARABIC_MAX_WORDS`: hard limit applied before Arabic text reaches TTS (default `22`, one sentence maximum).
- `NABRA_THREADS`: CPU threads assigned to Arabic speech synthesis (default `4`).
- `NABRA_SPEED`: Nabra speaking speed (default `1.0`).
- `SYSTEM_PROMPT`: English voice assistant behavior.
- `SYSTEM_PROMPT_AR`: Arabic voice assistant behavior.
- `HTTP_PORT`: direct HTTP benchmark port (default `8080`).
- `DOMAIN`: required only for the HTTPS profile.

Raw latency is best measured with the browser and all containers on the same server/LAN. Network latency is included in the displayed end-to-end number by design.

## Model sources

- Moonshine Tiny Streaming English and Moonshine Streaming Tiny Arabic 27M are downloaded by the official `moonshine-voice` package. Both process each utterance so their partial transcripts can drive the router.
- Gemma is `ggml-org/gemma-3-1b-it-GGUF`, file `gemma-3-1b-it-Q4_K_M.gguf`.
- Arabic uses `RightNowAI/RightNow-Arabic-0.5B-Turbo`, file `RightNow-Arabic-0.5B-Turbo-q4_k_m.gguf`.
- English TTS uses Piper `en_US-amy-medium`.
- Arabic TTS uses `oddadmix/Nabra-82M-v0.1`, the `af_msa` female voice, and Camel Tools MSA diacritization.

Gemma is subject to its model license; RightNow Arabic and Nabra are Apache-2.0; Piper 1 is GPL-3.0-or-later. The installed Moonshine Arabic runtime currently prints a Moonshine Community License warning, so verify those terms before commercial use even though the public checkpoint card lists MIT.
