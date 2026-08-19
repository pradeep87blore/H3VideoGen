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


# Paid-tier USD per 1M tokens (input, output). Used only for log estimates.
_GEMINI_RATES_USD: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.6-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.30, 2.50),
    "gemini-3-flash": (1.50, 9.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-flash": (0.30, 2.50),
}


def rates_for_model(model: str) -> tuple[float, float] | None:
    name = (model or "").lower()
    for key, rates in sorted(_GEMINI_RATES_USD.items(), key=lambda kv: len(kv[0]), reverse=True):
        if name == key or name.startswith(key):
            return rates
    return None


def extract_usage(response: Any, *, model: str, image_count: int = 0) -> dict[str, Any]:
    """Normalize Gemini usage_metadata into a JSON-friendly dict."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None and isinstance(response, dict):
        meta = response.get("usage_metadata")
    prompt = _usage_int(meta, "prompt_token_count")
    output = _usage_int(meta, "candidates_token_count")
    thinking = _usage_int(meta, "thoughts_token_count")
    cached = _usage_int(meta, "cached_content_token_count")
    total = _usage_int(meta, "total_token_count") or (prompt + output)
    in_rate, out_rate = rates_for_model(model) or (None, None)
    est = None
    if in_rate is not None and out_rate is not None:
        est = round((prompt * in_rate + output * out_rate) / 1_000_000.0, 6)
    return {
        "model": model,
        "prompt_tokens": prompt,
        "output_tokens": output,
        "thinking_tokens": thinking,
        "cached_tokens": cached,
        "total_tokens": total,
        "images": image_count,
        "est_usd": est,
    }


def format_usage_line(usage: dict[str, Any] | None) -> str:
    if not usage:
        return ""
    model = usage.get("model") or "gemini"
    pin = int(usage.get("prompt_tokens") or 0)
    pout = int(usage.get("output_tokens") or 0)
    think = int(usage.get("thinking_tokens") or 0)
    total = int(usage.get("total_tokens") or (pin + pout))
    imgs = int(usage.get("images") or 0)
    extra = f", {think} think" if think else ""
    img = f", {imgs} image{'s' if imgs != 1 else ''}" if imgs else ""
    usd = usage.get("est_usd")
    money = f" ≈ ${usd:.4f}" if isinstance(usd, (int, float)) else ""
    return (
        f"Gemini usage ({model}): {pin} in + {pout} out{extra}{img} "
        f"= {total} tokens{money}"
    )


def _usage_int(meta: Any, field: str) -> int:
    if meta is None:
        return 0
    val = getattr(meta, field, None)
    if val is None and isinstance(meta, dict):
        val = meta.get(field)
    try:
        return int(val or 0)
    except (TypeError, ValueError):
        return 0


class GeminiBackend:
    name = "gemini"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        self.last_usage: dict[str, Any] | None = None
        self.session_usage: dict[str, Any] = {
            "calls": 0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "est_usd": 0.0,
        }

    def _record_usage(self, usage: dict[str, Any]) -> None:
        self.last_usage = usage
        s = self.session_usage
        s["calls"] = int(s.get("calls") or 0) + 1
        s["prompt_tokens"] = int(s.get("prompt_tokens") or 0) + int(usage.get("prompt_tokens") or 0)
        s["output_tokens"] = int(s.get("output_tokens") or 0) + int(usage.get("output_tokens") or 0)
        s["total_tokens"] = int(s.get("total_tokens") or 0) + int(usage.get("total_tokens") or 0)
        if usage.get("est_usd") is not None:
            s["est_usd"] = round(float(s.get("est_usd") or 0) + float(usage["est_usd"]), 6)

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
        image_count = 0
        if images:
            parts: list[Any] = [user]
            for fp in images:
                if not fp.exists():
                    continue
                mime = "image/jpeg" if fp.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                parts.append(types.Part.from_bytes(data=fp.read_bytes(), mime_type=mime))
                image_count += 1
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
                    usage = extract_usage(response, model=model, image_count=image_count)
                    self._record_usage(usage)
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
