"""OpenAI-compatible local LLM (Ollama, LM Studio, vLLM, etc.)."""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import httpx

from ..config import Settings


class LocalOpenAIBackend:
    """Chat Completions against a local OpenAI-compatible server."""

    name = "local_openai"

    def __init__(self, settings: Settings):
        self.settings = settings

    def available(self) -> bool:
        if not self.settings.local_llm_enabled:
            return False
        try:
            return self.health() is not None
        except Exception:
            return False

    def health(self) -> dict | None:
        base = self.settings.local_llm_base_url.rstrip("/")
        headers = self._headers()
        timeout = min(8.0, float(self.settings.local_llm_timeout_sec))
        with httpx.Client(timeout=timeout, headers=headers) as client:
            # Prefer /models; fall back to a lightweight root probe.
            for path in ("/models", ""):
                try:
                    url = base if path == "" else f"{base}{path}"
                    r = client.get(url)
                    if r.status_code < 500:
                        data = None
                        try:
                            data = r.json()
                        except Exception:
                            data = {"status_code": r.status_code}
                        return {"url": url, "status_code": r.status_code, "body": data}
                except Exception:
                    continue
        return None

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        key = (self.settings.local_llm_api_key or "ollama").strip()
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _image_part(self, path: Path) -> dict:
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    def generate_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        images: list[Path] | None = None,
        model: str | None = None,
    ) -> tuple[str, str]:
        if not self.settings.local_llm_enabled:
            raise RuntimeError("Local LLM disabled (LOCAL_LLM_ENABLED=false)")

        model_name = (model or self.settings.local_llm_model).strip()
        if not model_name:
            raise RuntimeError("LOCAL_LLM_MODEL is empty")

        base = self.settings.local_llm_base_url.rstrip("/")
        url = f"{base}/chat/completions"

        if images:
            content: list[dict] = [{"type": "text", "text": user}]
            for fp in images:
                if fp.exists():
                    content.append(self._image_part(fp))
            user_msg: dict = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": user}

        payload = {
            "model": model_name,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                user_msg,
            ],
            # Many local servers honor this; harmless if ignored.
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(
            timeout=float(self.settings.local_llm_timeout_sec),
            headers=self._headers(),
        ) as client:
            r = client.post(url, json=payload)
            if r.status_code >= 400:
                # Retry without response_format for pickier servers.
                if r.status_code in (400, 422) and "response_format" in payload:
                    payload.pop("response_format", None)
                    r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected local LLM response: {data!r}"[:400]) from exc
        if not (text or "").strip():
            raise RuntimeError("Local LLM returned empty content")
        return text.strip(), f"local_openai:{model_name}"
