#!/usr/bin/env python3
"""Download MiniMax H3 ComfyUI weights onto a persistent volume."""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("HF_HOME", "/workspace/.hf")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

DEST = os.environ.get("H3_MODELS_DIR", "/workspace/ComfyUI/models")
PATTERNS = [
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
]


def main() -> int:
    print("Downloading MiniMax H3 weights into", DEST, flush=True)
    try:
        snapshot_download(
            repo_id="Comfy-Org/MiniMax-H3",
            local_dir=DEST,
            allow_patterns=PATTERNS,
        )
    except Exception:
        traceback.print_exc()
        return 1
    missing = []
    for rel in PATTERNS:
        fp = os.path.join(DEST, rel)
        if os.path.isfile(fp):
            print(f"  ok {rel} {os.path.getsize(fp)} bytes", flush=True)
        else:
            print(f"  MISSING {rel}", flush=True)
            missing.append(rel)
    if missing:
        return 1
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
