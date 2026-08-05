"""ElevenLabs voice/narration — stubbed off until enabled."""
from __future__ import annotations

from pathlib import Path

from ..config import Settings


class VoiceAgent:
    """Placeholder for ElevenLabs narration.

    The full pipeline will later:
    1) Ask Gemini for narration script timed to shots
    2) Synthesize per-shot or full-track audio via ElevenLabs
    3) Mux under the assembled video with ducking
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.enable_voice and settings.elevenlabs_api_key)

    def synthesize(self, text: str, out_path: Path) -> Path | None:
        if not self.enabled:
            return None
        raise NotImplementedError(
            "ElevenLabs narration is intentionally disabled for this build. "
            "Set ENABLE_VOICE=true and implement API calls when ready."
        )
