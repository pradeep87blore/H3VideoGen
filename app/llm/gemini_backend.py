"""Gemini cloud backend with multi-model cascade."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..config import Settings


def _is_quota_error(exc: BaseException) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _is_capacity_error(exc: BaseException) -> bool:
    """Overload / temporary unavailability (try another model or retry)."""
    text = str(exc)
    low = text.lower()
    return any(
        token in text or token in low
        for token in (
            "503",
            "UNAVAILABLE",
            "high demand",
            "overloaded",
            "temporarily unavailable",
            "Server disconnected",
            "server disconnected",
            "DeadlineExceeded",
            "deadline exceeded",
            "timed out",
            "timeout",
            "502",
            "504",
        )
    )


def _is_retryable_model_error(exc: BaseException) -> bool:
    text = str(exc)
    if _is_quota_error(exc) or _is_capacity_error(exc):
        return True
    return any(
        token in text
        for token in (
            "404",
            "NOT_FOUND",
            "no longer available",
            "not found",
            "not supported",
            "is not available",
        )
    )


def _retry_seconds(exc: BaseException, default: float = 15.0) -> float:
    match = re.search(r"[Rr]etry in ([0-9]+(?:\.[0-9]+)?)", str(exc))
    if match:
        return min(float(match.group(1)) + 1.0, 60.0)
    return default


def model_list(primary: str, settings: Settings) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in [primary, *settings.gemini_model_fallbacks]:
        m = (m or "").strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


class GeminiBackend:
    name = "gemini"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def available(self) -> bool:
        return bool(self.settings.gemini_api_key)

    def _client_or_raise(self):
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def generate_text(
        self,
        *,
        models: list[str],
        system: str,
        user: str,
        temperature: float,
        images: list[Path] | None = None,
    ) -> tuple[str, str]:
        from google.genai import types

        client = self._client_or_raise()
        contents: Any
        if images:
            parts: list[Any] = [user]
            for fp in images:
                if not fp.exists():
                    continue
                mime = "image/jpeg" if fp.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                parts.append(types.Part.from_bytes(data=fp.read_bytes(), mime_type=mime))
            contents = parts
        else:
            contents = user

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
        )

        errors: list[str] = []
        for model in models:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    text = (response.text or "").strip()
                    if not text:
                        raise RuntimeError("empty Gemini response")
                    return text, f"gemini:{model}"
                except Exception as exc:  # noqa: BLE001 — cascade models
                    errors.append(f"{model}: {exc}")
                    # Permanent problems on this request → stop (bad key, etc.)
                    if not _is_retryable_model_error(exc):
                        raise
                    # Quota / high demand: brief pause then retry same model once more
                    if (_is_quota_error(exc) or _is_capacity_error(exc)) and attempt < 2:
                        time.sleep(_retry_seconds(exc, default=8.0 + attempt * 4.0))
                        continue
                    # Then try next model in the cascade
                    break
        raise RuntimeError("All Gemini models failed: " + " | ".join(errors[:6]))
