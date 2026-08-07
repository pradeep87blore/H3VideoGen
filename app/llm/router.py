"""Route director/critic calls: Gemini → local OpenAI → offline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..job_control import CancelledError, job_control
from .gemini_backend import GeminiBackend, model_list
from .json_util import extract_json
from .local_openai import LocalOpenAIBackend
from .offline import critic_review_json, director_plan_json


LogFn = Callable[[str], None]


@dataclass
class LLMResponse:
    data: dict[str, Any]
    provider: str


class LLMRouter:
    def __init__(self, settings: Settings, log: LogFn | None = None):
        self.settings = settings
        self.log = log
        self.gemini = GeminiBackend(settings)
        self.local = LocalOpenAIBackend(settings)
        self.last_provider: str = ""
        self.last_errors: list[str] = []

    def _emit(self, msg: str) -> None:
        if self.log:
            self.log(msg)

    def _order(self) -> list[str]:
        order = [x.strip().lower() for x in self.settings.llm_fallback_order if x.strip()]
        return order or ["gemini", "local_openai", "offline"]

    def status(self) -> dict[str, Any]:
        local_health = None
        local_ok = False
        if self.settings.local_llm_enabled:
            try:
                local_health = self.local.health()
                local_ok = local_health is not None
            except Exception as exc:
                local_health = {"error": str(exc)}
        local_models: list[dict] = []
        vision_model = None
        text_model = None
        if local_ok:
            try:
                local_models = [
                    {"id": e["id"], "vision": bool(e.get("vision"))}
                    for e in self.local.list_models()
                ]
                text_model = self.local.resolve_model(images=False)
                vision_model = self.local.resolve_model(images=True)
            except Exception as exc:
                local_health = {**(local_health or {}), "model_resolve_error": str(exc)}
        return {
            "order": self._order(),
            "gemini_key_set": bool(self.settings.gemini_api_key),
            "gemini_models": model_list(self.settings.gemini_director_model, self.settings),
            "local_llm_enabled": self.settings.local_llm_enabled,
            "local_llm_base_url": self.settings.local_llm_base_url,
            "local_llm_model": self.settings.local_llm_model,
            "local_llm_vision_model": self.settings.local_llm_vision_model,
            "local_llm_resolved_text_model": text_model,
            "local_llm_resolved_vision_model": vision_model,
            "local_llm_models": local_models,
            "local_llm_reachable": local_ok,
            "local_llm_health": local_health,
            "offline_always_available": True,
        }

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.7,
        primary_gemini_model: str | None = None,
        images: list[Path] | None = None,
        offline_factory: Callable[[], dict[str, Any]] | None = None,
        purpose: str = "llm",
    ) -> LLMResponse:
        """Try providers in order until JSON is produced."""
        self.last_errors = []
        models = model_list(
            primary_gemini_model or self.settings.gemini_director_model,
            self.settings,
        )

        for name in self._order():
            job_control.check()

            try:
                if name == "gemini":
                    if not self.gemini.available():
                        raise RuntimeError("Gemini key not set")
                    self._emit(f"{purpose}: trying Gemini ({', '.join(models[:3])}…)")
                    text, provider = self.gemini.generate_text(
                        models=models,
                        system=system,
                        user=user,
                        temperature=temperature,
                        images=images,
                    )
                    job_control.check()
                    data = extract_json(text)
                    self.last_provider = provider
                    self._emit(f"{purpose}: ok via {provider}")
                    return LLMResponse(data=data, provider=provider)

                if name in ("local_openai", "local", "ollama"):
                    if not self.settings.local_llm_enabled:
                        raise RuntimeError("Local LLM disabled")
                    want_imgs = bool(images)
                    try:
                        resolved = self.local.resolve_model(images=want_imgs)
                    except Exception:
                        resolved = (
                            self.settings.local_llm_vision_model
                            if want_imgs
                            else self.settings.local_llm_model
                        )
                    self._emit(
                        f"{purpose}: trying local LLM "
                        f"{self.settings.local_llm_base_url} model={resolved}"
                        + (" (vision)" if want_imgs else " (text)")
                    )
                    text, provider = self.local.generate_text(
                        system=system,
                        user=user,
                        temperature=temperature,
                        images=images,
                    )
                    job_control.check()
                    data = extract_json(text)
                    self.last_provider = provider
                    self._emit(f"{purpose}: ok via {provider}")
                    return LLMResponse(data=data, provider=provider)

                if name == "offline":
                    if offline_factory is None:
                        raise RuntimeError("No offline factory for this call")
                    self._emit(f"{purpose}: using offline fallback")
                    data = offline_factory()
                    self.last_provider = "offline"
                    self._emit(f"{purpose}: ok via offline")
                    return LLMResponse(data=data, provider="offline")

                raise RuntimeError(f"Unknown LLM backend '{name}'")
            except CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — cascade
                err = f"{name}: {exc}"
                self.last_errors.append(err)
                self._emit(f"{purpose}: {name} failed — {exc}")
                continue

        raise RuntimeError(
            f"All LLM backends failed for {purpose}: " + " | ".join(self.last_errors[:5])
        )

    def director_plan_payload(
        self,
        *,
        system: str,
        user: str,
        offline_kwargs: dict[str, Any],
    ) -> LLMResponse:
        return self.generate_json(
            system=system,
            user=user,
            temperature=0.7,
            primary_gemini_model=self.settings.gemini_director_model,
            offline_factory=lambda: director_plan_json(**offline_kwargs),
            purpose="Director",
        )

    def critic_review_payload(
        self,
        *,
        system: str,
        user: str,
        images: list[Path],
        offline_kwargs: dict[str, Any],
    ) -> LLMResponse:
        return self.generate_json(
            system=system,
            user=user,
            temperature=0.2,
            primary_gemini_model=self.settings.gemini_critic_model,
            images=images,
            offline_factory=lambda: critic_review_json(**offline_kwargs),
            purpose="Critic",
        )
