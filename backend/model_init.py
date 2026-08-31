import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from moonshine_voice import ModelArch, get_model_for_language

root = Path(os.getenv("MODEL_ROOT", "/models"))
(root / "llm").mkdir(parents=True, exist_ok=True)
(root / "llm-qwen").mkdir(parents=True, exist_ok=True)
(root / "piper").mkdir(parents=True, exist_ok=True)
(root / "nabra").mkdir(parents=True, exist_ok=True)

print("Downloading Gemma 3 1B Q4_K_M…", flush=True)
gguf = hf_hub_download(repo_id="ggml-org/gemma-3-1b-it-GGUF", filename="gemma-3-1b-it-Q4_K_M.gguf", local_dir=root / "llm")

print("Downloading Qwen3-1.7B Q4_K_M…", flush=True)
qwen_gguf = hf_hub_download(
    repo_id="ggml-org/Qwen3-1.7B-GGUF",
    filename="Qwen3-1.7B-Q4_K_M.gguf",
    local_dir=root / "llm-qwen",
)

print("Downloading Moonshine Medium Streaming English and Tiny Streaming Arabic…", flush=True)
moonshine = {}
for language, wanted_arch in (("en", ModelArch.MEDIUM_STREAMING), ("ar", ModelArch.TINY_STREAMING)):
    moonshine_path, arch = get_model_for_language(language, wanted_arch)
    moonshine[language] = {"path": str(moonshine_path), "arch": int(arch)}

for legacy_path in (
    root / "stt-distil-en",
    root / "cache" / "moonshine_voice" / "download.moonshine.ai" / "model" / "tiny-streaming-en",
):
    if legacy_path.exists():
        print(f"Removing unused English STT model: {legacy_path}", flush=True)
        shutil.rmtree(legacy_path)

voice = "en_US-amy-medium"
print(f"Downloading Piper voice: {voice}…", flush=True)
subprocess.run(["python", "-m", "piper.download_voices", "--data-dir", str(root / "piper"), voice], check=True)

print("Downloading Nabra-82M Arabic TTS…", flush=True)
nabra = {}
for filename in ("kokoro_arabic.pth", "af_msa.pt", "config.json"):
    nabra[filename] = hf_hub_download(
        repo_id="oddadmix/Nabra-82M-v0.1",
        filename=filename,
        local_dir=root / "nabra",
    )

manifest = {
    "llm": {"gemma": str(gguf), "qwen": str(qwen_gguf)},
    "stt": moonshine,
    "tts": {"en": {"engine": "piper", "voice": voice}, "ar": {"engine": "nabra", "files": nabra}},
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2), flush=True)
