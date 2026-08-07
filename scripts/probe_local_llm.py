"""Probe Ollama OpenAI-compat for text JSON + vision."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:11434/v1"


def ensure_image() -> Path:
    for pattern in ("**/*.jpg", "**/*.png"):
        for p in (ROOT / "outputs").glob(pattern):
            if p.is_file() and 1_000 < p.stat().st_size < 400_000:
                return p
    path = ROOT / "_probe_frame.jpg"
    Image.new("RGB", (128, 128), (200, 100, 50)).save(path, quality=85)
    return path


def call(
    model: str,
    *,
    with_image: bool = False,
    response_format: dict | None = None,
    image_b64: str | None = None,
) -> tuple[int, str]:
    if with_image and image_b64:
        content = [
            {"type": "text", "text": 'Describe in JSON only: {"summary":"..."}'},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            },
        ]
        messages = [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": content},
        ]
    else:
        messages = [
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": 'Return exactly: {"ok": true, "msg": "hello"}',
            },
        ]
    payload: dict = {
        "model": model,
        "temperature": 0.2,
        "messages": messages,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    r = httpx.post(f"{BASE}/chat/completions", json=payload, timeout=180.0)
    return r.status_code, r.text[:500]


def native_chat_vision(model: str, image_b64: str) -> tuple[int, str]:
    """Ollama native /api/chat with images[] base64 (no data URI)."""
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": 'Describe in JSON: {"summary":"..."}',
                "images": [image_b64],
            }
        ],
    }
    r = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json=payload,
        timeout=180.0,
    )
    return r.status_code, r.text[:500]


def main() -> None:
    path = ensure_image()
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    print("image", path, "bytes", len(raw))

    for model in ("llama3.2:latest", "llava:latest", "llama3.2", "llava"):
        for rf in (None, {"type": "json_object"}):
            code, text = call(model, with_image=False, response_format=rf)
            print(f"\n{model} TEXT rf={bool(rf)} -> {code}")
            print(text[:220])
        code, text = call(model, with_image=True, image_b64=b64)
        print(f"\n{model} VISION openai-data-uri -> {code}")
        print(text[:220])
        code, text = native_chat_vision(model, b64)
        print(f"\n{model} VISION native /api/chat -> {code}")
        print(text[:220])


if __name__ == "__main__":
    main()
