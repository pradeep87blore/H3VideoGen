"""ElevenLabs voice / documentary-style narration."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import httpx

from ..config import Settings
from ..models import ProductionPlan, ShotPlan

LogFn = Callable[[str], None]


class VoiceAgent:
    """Synthesize full-track narration via ElevenLabs Text-to-Speech."""

    API_BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, settings: Settings, log: LogFn | None = None):
        self.settings = settings
        self.log = log

    def _emit(self, msg: str) -> None:
        if self.log:
            self.log(msg)

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enable_voice and (self.settings.elevenlabs_api_key or "").strip())

    @property
    def voice_id(self) -> str:
        return (self.settings.elevenlabs_voice_id or "").strip()

    def build_script(self, plan: ProductionPlan) -> str:
        """Join per-shot VO lines, or fall back to a compact plan-based script."""
        if (plan.narration_script or "").strip():
            return re.sub(r"\s+", " ", plan.narration_script.strip())
        lines: list[str] = []
        for s in plan.shots or []:
            line = (s.narration_line or "").strip()
            if line:
                lines.append(line.rstrip(".") + ".")
        if lines:
            return " ".join(lines)
        # Last resort: title + logline only
        title = (plan.title or "This short").strip()
        logline = (plan.logline or "").strip()
        if logline:
            return f"{title}. {logline}"
        return title

    def synthesize(self, text: str, out_path: Path) -> Path | None:
        if not self.enabled:
            self._emit("Narration skipped (ENABLE_VOICE=false or no ELEVENLABS_API_KEY)")
            return None
        text = re.sub(r"\s+", " ", (text or "").strip())
        if not text:
            self._emit("Narration skipped (empty script)")
            return None
        voice = self.voice_id
        if not voice:
            raise RuntimeError(
                "ELEVENLABS_VOICE_ID is empty — set a voice id in .env to use narration"
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model = (self.settings.elevenlabs_model_id or "eleven_multilingual_v2").strip()
        url = f"{self.API_BASE}/text-to-speech/{voice}"
        headers = {
            "xi-api-key": self.settings.elevenlabs_api_key.strip(),
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text[:4500],
            "model_id": model,
            "voice_settings": {
                "stability": float(self.settings.elevenlabs_stability),
                "similarity_boost": float(self.settings.elevenlabs_similarity),
                "style": 0.15,
                "use_speaker_boost": True,
            },
        }
        self._emit(f"ElevenLabs TTS ({model}, voice={voice[:12]}…)…")
        with httpx.Client(timeout=float(self.settings.elevenlabs_timeout_sec)) as client:
            r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                detail = r.text[:400]
                raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {detail}")
            out_path.write_bytes(r.content)
        if not out_path.exists() or out_path.stat().st_size < 500:
            raise RuntimeError("ElevenLabs returned empty audio")
        self._emit(f"Narration audio ready: {out_path.name} ({out_path.stat().st_size} bytes)")
        return out_path

    def narrate_plan(self, plan: ProductionPlan, out_path: Path) -> tuple[Path | None, str]:
        """Build script from plan and synthesize. Returns (path, script)."""
        script = self.build_script(plan)
        path = self.synthesize(script, out_path)
        return path, script
