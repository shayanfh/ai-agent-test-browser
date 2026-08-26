import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from moonshine_voice import ModelArch, get_model_for_language

root = Path(os.getenv("MODEL_ROOT", "/models"))
(root / "llm").mkdir(parents=True, exist_ok=True)
(root / "piper").mkdir(parents=True, exist_ok=True)

print("Downloading Gemma 3 1B Q4_K_M…", flush=True)
gguf = hf_hub_download(repo_id="ggml-org/gemma-3-1b-it-GGUF", filename="gemma-3-1b-it-Q4_K_M.gguf", local_dir=root / "llm")

print("Downloading Moonshine Tiny Streaming…", flush=True)
moonshine_path, arch = get_model_for_language("en", ModelArch.TINY_STREAMING)

print("Downloading Piper en_US-lessac-medium…", flush=True)
subprocess.run(["python", "-m", "piper.download_voices", "--data-dir", str(root / "piper"), "en_US-lessac-medium"], check=True)

manifest = {"gemma": str(gguf), "moonshine": str(moonshine_path), "moonshine_arch": int(arch), "piper": "en_US-lessac-medium"}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2), flush=True)
