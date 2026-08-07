"""OpenAI-compatible local LLM (Ollama, LM Studio, vLLM, etc.)."""
from __future__ import annotations

import base64
import io
import mimetypes
import re
from pathlib import Path
from typing import Any

import httpx

from ..config import Settings

# Models known to accept images (substring match on id, case-insensitive).
_VISION_NAME_HINTS = (
    "llava",
    "bakllava",
    "vision",
    "minicpm-v",
    "minicpm_v",
    "qwen2-vl",
    "qwen2.5-vl",
    "qwen-vl",
    "moondream",
    "llama3.2-vision",
    "llama-3.2-vision",
    "gemma3",  # recent gemma multimodal tags
    "pixtral",
    "internvl",
)

_MULTIMODAL_ERR = re.compile(
    r"multimodal|does not support (multimodal|vision|image)|"
    r"image.*(not supported|unsupported)|vision.*(not supported|unsupported)",
    re.I,
)


class LocalOpenAIBackend:
    """Chat Completions against a local OpenAI-compatible server."""

    name = "local_openai"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._catalog_cache: list[dict[str, Any]] | None = None

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
            for path in ("/models", ""):
                try:
                    url = base if path == "" else f"{base}{path}"
                    r = client.get(url)
                    if r.status_code < 500:
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

    def _openai_root(self) -> str:
        return self.settings.local_llm_base_url.rstrip("/")

    def _server_root(self) -> str:
        """Host root without /v1 (Ollama native APIs live here)."""
        base = self._openai_root()
        if base.endswith("/v1"):
            return base[:-3]
        return base

    def _looks_like_ollama(self) -> bool:
        base = self.settings.local_llm_base_url.lower()
        if "11434" in base or "ollama" in base:
            return True
        # Probe native tags endpoint once we try catalog
        return False

    def list_models(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Return model catalog entries: {id, vision: bool}."""
        if self._catalog_cache is not None and not force:
            return self._catalog_cache

        entries: list[dict[str, Any]] = []
        timeout = min(8.0, float(self.settings.local_llm_timeout_sec))
        headers = self._headers()

        # Prefer Ollama /api/tags (has capabilities / vision flags).
        try:
            with httpx.Client(timeout=timeout, headers=headers) as client:
                r = client.get(f"{self._server_root()}/api/tags")
                if r.status_code < 400:
                    for m in (r.json() or {}).get("models") or []:
                        name = str(m.get("name") or m.get("model") or "").strip()
                        if not name:
                            continue
                        caps = m.get("capabilities") or []
                        families = ((m.get("details") or {}).get("families")) or []
                        vision = (
                            "vision" in caps
                            or "clip" in [str(f).lower() for f in families]
                            or self._name_suggests_vision(name)
                        )
                        entries.append({"id": name, "vision": bool(vision), "raw": m})
        except Exception:
            pass

        if not entries:
            try:
                with httpx.Client(timeout=timeout, headers=headers) as client:
                    r = client.get(f"{self._openai_root()}/models")
                    if r.status_code < 400:
                        for m in (r.json() or {}).get("data") or []:
                            mid = str(m.get("id") or "").strip()
                            if mid:
                                entries.append(
                                    {
                                        "id": mid,
                                        "vision": self._name_suggests_vision(mid),
                                        "raw": m,
                                    }
                                )
            except Exception:
                pass

        self._catalog_cache = entries
        return entries

    @staticmethod
    def _name_suggests_vision(name: str) -> bool:
        n = name.lower()
        return any(h in n for h in _VISION_NAME_HINTS)

    def _normalize_model_id(self, name: str, catalog: list[dict[str, Any]]) -> str:
        """Match configured short names to pulled tags (llama3.2 → llama3.2:latest)."""
        want = (name or "").strip()
        if not want:
            return want
        ids = [e["id"] for e in catalog]
        if want in ids:
            return want
        # Strip :latest for comparison
        want_base = want.split(":")[0]
        for mid in ids:
            if mid == want or mid.split(":")[0] == want_base:
                return mid
        return want

    def resolve_model(self, *, images: bool, model: str | None = None) -> str:
        """Pick text vs vision model. Prefer explicit config, then catalog."""
        catalog = self.list_models()
        if model:
            return self._normalize_model_id(model, catalog)

        text_model = self._normalize_model_id(self.settings.local_llm_model, catalog)
        vision_cfg = (self.settings.local_llm_vision_model or "").strip()
        vision_model = (
            self._normalize_model_id(vision_cfg, catalog) if vision_cfg else ""
        )

        if not images:
            if not text_model:
                raise RuntimeError("LOCAL_LLM_MODEL is empty")
            return text_model

        catalog_ids = {e["id"] for e in catalog}
        catalog_bases = {e["id"].split(":")[0] for e in catalog}

        def _installed(name: str) -> bool:
            if not catalog:
                return True  # unknown server — trust config
            return name in catalog_ids or name.split(":")[0] in catalog_bases

        # Multimodal path
        if vision_model and _installed(vision_model):
            return vision_model

        # Auto: any installed vision model
        for e in catalog:
            if e.get("vision"):
                return str(e["id"])

        if vision_model:
            # Config set but not found — still try (user may have not listed)
            return vision_model

        # Configured text model might itself be multimodal
        if text_model and (
            self._name_suggests_vision(text_model)
            or any(
                e.get("vision")
                and (
                    e["id"] == text_model
                    or e["id"].split(":")[0] == text_model.split(":")[0]
                )
                for e in catalog
            )
        ):
            return text_model

        if not text_model:
            raise RuntimeError(
                "No local model configured. Set LOCAL_LLM_MODEL and "
                "LOCAL_LLM_VISION_MODEL (e.g. llava) for critic frames."
            )
        return text_model

    def _prepare_images(self, paths: list[Path]) -> list[tuple[Path, bytes, str]]:
        """Load and optionally downscale images for local VLM context windows."""
        max_side = max(256, int(self.settings.local_llm_vision_max_side))
        out: list[tuple[Path, bytes, str]] = []
        for fp in paths:
            if not fp.exists():
                continue
            raw = fp.read_bytes()
            mime, _ = mimetypes.guess_type(str(fp))
            if not mime:
                mime = "image/jpeg" if fp.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            try:
                from PIL import Image

                with Image.open(io.BytesIO(raw)) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    scale = min(1.0, float(max_side) / float(max(w, h)))
                    if scale < 0.999:
                        im = im.resize(
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            Image.Resampling.LANCZOS,
                        )
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=85, optimize=True)
                    raw = buf.getvalue()
                    mime = "image/jpeg"
            except Exception:
                pass
            out.append((fp, raw, mime))
        return out

    def _image_part(self, raw: bytes, mime: str) -> dict:
        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        try:
            data = response.json()
            err = data.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err
                return str(msg)[:800]
            if err:
                return str(err)[:800]
            return str(data)[:800]
        except Exception:
            return (response.text or "")[:800]

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

        img_paths = [p for p in (images or []) if p and Path(p).exists()]
        prepared = self._prepare_images([Path(p) for p in img_paths]) if img_paths else []
        want_vision = bool(prepared)

        model_name = self.resolve_model(images=want_vision, model=model)
        if not model_name:
            raise RuntimeError("LOCAL_LLM_MODEL is empty")

        last_native: Exception | None = None
        # Prefer Ollama native chat for vision — `format: json` + images[] is reliable
        if want_vision and (self._looks_like_ollama() or self._ollama_tags_ok()):
            try:
                return self._ollama_native(
                    model_name=model_name,
                    system=system,
                    user=user,
                    temperature=temperature,
                    prepared=prepared,
                )
            except Exception as native_exc:
                last_native = native_exc

        try:
            return self._openai_chat(
                model_name=model_name,
                system=system,
                user=user,
                temperature=temperature,
                prepared=prepared,
            )
        except Exception as openai_exc:
            err_s = str(openai_exc)
            if want_vision and _MULTIMODAL_ERR.search(err_s):
                vision_cfg = (self.settings.local_llm_vision_model or "").strip()
                alt = ""
                if vision_cfg:
                    alt = self.resolve_model(images=True, model=vision_cfg)
                if not alt or alt == model_name:
                    # Auto-pick any other vision model from catalog
                    for e in self.list_models():
                        if e.get("vision") and e["id"] != model_name:
                            alt = str(e["id"])
                            break
                if alt and alt != model_name:
                    try:
                        if self._looks_like_ollama() or self._ollama_tags_ok():
                            return self._ollama_native(
                                model_name=alt,
                                system=system,
                                user=user,
                                temperature=temperature,
                                prepared=prepared,
                            )
                        return self._openai_chat(
                            model_name=alt,
                            system=system,
                            user=user,
                            temperature=temperature,
                            prepared=prepared,
                        )
                    except Exception as alt_exc:
                        raise RuntimeError(
                            f"Local vision failed on {model_name} and {alt}: {alt_exc}. "
                            "Pull a vision model: `ollama pull llava` and set "
                            "LOCAL_LLM_VISION_MODEL=llava"
                        ) from alt_exc
                raise RuntimeError(
                    f"Local model '{model_name}' cannot accept images. "
                    "Pull a vision model (`ollama pull llava`) and set "
                    "LOCAL_LLM_VISION_MODEL=llava for critic frame QA."
                ) from openai_exc
            if last_native is not None:
                raise RuntimeError(
                    f"Local LLM failed (openai: {openai_exc}; native: {last_native})"
                ) from openai_exc
            raise

    def _ollama_tags_ok(self) -> bool:
        try:
            with httpx.Client(timeout=3.0, headers=self._headers()) as client:
                r = client.get(f"{self._server_root()}/api/tags")
                return r.status_code < 400
        except Exception:
            return False

    def _ollama_native(
        self,
        *,
        model_name: str,
        system: str,
        user: str,
        temperature: float,
        prepared: list[tuple[Path, bytes, str]],
    ) -> tuple[str, str]:
        root = self._server_root()
        url = f"{root}/api/chat"
        b64_images = [base64.b64encode(raw).decode("ascii") for _, raw, _ in prepared]

        sys_prompt = (
            (system or "").strip()
            + "\n\nYou MUST reply with a single valid JSON object only. "
            "No markdown fences, no commentary before or after the JSON."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": user,
                **({"images": b64_images} if b64_images else {}),
            },
        ]
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": float(temperature),
                "num_predict": int(self.settings.local_llm_max_tokens),
            },
        }

        with httpx.Client(
            timeout=float(self.settings.local_llm_timeout_sec),
            headers=self._headers(),
        ) as client:
            r = client.post(url, json=payload)
            if r.status_code >= 400:
                # Retry without format= if server rejects it
                if r.status_code in (400, 422) and "format" in payload:
                    payload.pop("format", None)
                    r = client.post(url, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(
                    f"Ollama /api/chat HTTP {r.status_code}: {self._error_text(r)}"
                )
            data = r.json()

        text = ""
        try:
            text = (data.get("message") or {}).get("content") or ""
        except Exception:
            text = ""
        if not (text or "").strip():
            raise RuntimeError(f"Ollama returned empty content: {data!r}"[:400])
        return text.strip(), f"local_openai:{model_name}"

    def _openai_chat(
        self,
        *,
        model_name: str,
        system: str,
        user: str,
        temperature: float,
        prepared: list[tuple[Path, bytes, str]],
    ) -> tuple[str, str]:
        url = f"{self._openai_root()}/chat/completions"
        sys_prompt = (
            (system or "").strip()
            + "\n\nYou MUST reply with a single valid JSON object only. "
            "No markdown fences, no commentary before or after the JSON."
        )

        if prepared:
            content: list[dict] = [{"type": "text", "text": user}]
            for _, raw, mime in prepared:
                content.append(self._image_part(raw, mime))
            user_msg: dict = {"role": "user", "content": content}
        else:
            user_msg = {"role": "user", "content": user}

        payload: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": sys_prompt},
                user_msg,
            ],
            "max_tokens": int(self.settings.local_llm_max_tokens),
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(
            timeout=float(self.settings.local_llm_timeout_sec),
            headers=self._headers(),
        ) as client:
            r = client.post(url, json=payload)
            if r.status_code >= 400:
                err = self._error_text(r)
                # Drop response_format / max_tokens for pickier servers
                if r.status_code in (400, 422):
                    for key in ("response_format", "max_tokens"):
                        if key in payload:
                            payload.pop(key, None)
                            r = client.post(url, json=payload)
                            if r.status_code < 400:
                                break
                            err = self._error_text(r)
                if r.status_code >= 400:
                    raise RuntimeError(
                        f"Local /chat/completions HTTP {r.status_code}: {err}"
                    )
            data = r.json()

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected local LLM response: {data!r}"[:400]) from exc
        if not (text or "").strip():
            raise RuntimeError("Local LLM returned empty content")
        return text.strip(), f"local_openai:{model_name}"
