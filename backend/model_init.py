import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from moonshine_voice import ModelArch, get_model_for_language

root = Path(os.getenv("MODEL_ROOT", "/models"))
(root / "llm").mkdir(parents=True, exist_ok=True)
(root / "llm-ar").mkdir(parents=True, exist_ok=True)
(root / "piper").mkdir(parents=True, exist_ok=True)

print("Downloading Gemma 3 1B Q4_K_M…", flush=True)
gguf = hf_hub_download(repo_id="ggml-org/gemma-3-1b-it-GGUF", filename="gemma-3-1b-it-Q4_K_M.gguf", local_dir=root / "llm")

print("Downloading RightNow Arabic 0.5B Turbo Q4_K_M…", flush=True)
arabic_gguf = hf_hub_download(
    repo_id="RightNowAI/RightNow-Arabic-0.5B-Turbo",
    filename="gguf/RightNow-Arabic-0.5B-Turbo-q4_k_m.gguf",
    local_dir=root / "llm-ar",
)

print("Downloading Moonshine Tiny Streaming English and Arabic…", flush=True)
moonshine = {}
for language in ("en", "ar"):
    moonshine_path, arch = get_model_for_language(language, ModelArch.TINY_STREAMING)
    moonshine[language] = {"path": str(moonshine_path), "arch": int(arch)}

voices = ["en_US-amy-medium", "ar_JO-kareem-medium"]
print(f"Downloading Piper voices: {', '.join(voices)}…", flush=True)
subprocess.run(["python", "-m", "piper.download_voices", "--data-dir", str(root / "piper"), *voices], check=True)

manifest = {
    "llm": {"en": str(gguf), "ar": str(arabic_gguf)},
    "moonshine": moonshine,
    "piper": {"en": voices[0], "ar": voices[1]},
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2), flush=True)
