"""Application settings loaded from environment / .env."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_GEMINI_FALLBACKS = (
    "gemini-3.5-flash,gemini-3.6-flash,gemini-3.1-flash-lite,"
    "gemini-flash-latest,gemini-3-flash-preview,gemini-3.5-flash-lite"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="", alias="ELEVENLABS_VOICE_ID")

    comfyui_host: str = Field(default="127.0.0.1", alias="COMFYUI_HOST")
    comfyui_port: int = Field(default=8188, alias="COMFYUI_PORT")
    comfyui_timeout_sec: int = Field(default=3600, alias="COMFYUI_TIMEOUT_SEC")

    output_root: Path = Field(default=ROOT / "outputs", alias="OUTPUT_ROOT")
    comfy_output_root: Path = Field(default=Path(r"E:/AI/ComfyUI/output"), alias="COMFY_OUTPUT_ROOT")
    ai_root: Path = Field(default=Path(r"E:/AI"), alias="AI_ROOT")
    ffmpeg_path: str = Field(default="ffmpeg", alias="FFMPEG_PATH")
    ffprobe_path: str = Field(default="ffprobe", alias="FFPROBE_PATH")

    h3_unet: str = Field(default="minimax_h3_fl2va_pruned_int8_convrot.safetensors", alias="H3_UNET")
    h3_unet_r2v: str = Field(
        default="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        alias="H3_UNET_R2V",
    )
    h3_clip: str = Field(default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", alias="H3_CLIP")
    h3_video_vae: str = Field(default="minimax_h3_video_vae_fp16.safetensors", alias="H3_VIDEO_VAE")
    h3_audio_vae: str = Field(default="minimax_h3_audio_vae_fp32.safetensors", alias="H3_AUDIO_VAE")
    # r2v | t2v  (r2v uses reference images for character consistency)
    h3_mode: str = Field(default="r2v", alias="H3_MODE")
    h3_ref_image_size: str = Field(default="match", alias="H3_REF_IMAGE_SIZE")  # match | max
    h3_use_prev_shot_ref: bool = Field(default=True, alias="H3_USE_PREV_SHOT_REF")
    # auto | gemini | h3 | manual | none — how to create identity stills
    character_board_mode: str = Field(default="auto", alias="CHARACTER_BOARD_MODE")
    # Multi-view sheet: how many poses for lead vs supporting cast
    character_sheet_poses_primary: int = Field(default=5, alias="CHARACTER_SHEET_POSES_PRIMARY")
    character_sheet_poses_secondary: int = Field(default=2, alias="CHARACTER_SHEET_POSES_SECONDARY")
    # Max reference images attached per R2V shot (H3 hard cap is 9)
    character_sheet_max_refs_per_shot: int = Field(default=4, alias="CHARACTER_SHEET_MAX_REFS_PER_SHOT")
    # If Gemini stills fail / incomplete, run a short H3 T2V turnaround
    character_sheet_use_h3: bool = Field(default=True, alias="CHARACTER_SHEET_USE_H3")
    gemini_director_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_DIRECTOR_MODEL")
    gemini_critic_model: str = Field(default="gemini-3.5-flash", alias="GEMINI_CRITIC_MODEL")
    gemini_model_fallbacks: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [m.strip() for m in _DEFAULT_GEMINI_FALLBACKS.split(",") if m.strip()],
        alias="GEMINI_MODEL_FALLBACKS",
    )
    gemini_image_models: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
        ],
        alias="GEMINI_IMAGE_MODELS",
    )

    # 960x544 is safer on ~16GB VRAM for R2V; bump to 1344x768 when you have headroom
    default_width: int = Field(default=960, alias="DEFAULT_WIDTH")
    default_height: int = Field(default=544, alias="DEFAULT_HEIGHT")
    default_fps: int = Field(default=24, alias="DEFAULT_FPS")
    default_steps: int = Field(default=16, alias="DEFAULT_STEPS")
    default_length_frames: int = Field(default=124, alias="DEFAULT_LENGTH_FRAMES")
    max_retakes: int = Field(default=2, alias="MAX_RETAKES")
    critic_pass_threshold: float = Field(default=7.5, alias="CRITIC_PASS_THRESHOLD")
    # How many generation jobs may run at once (1 = serial queue; raise carefully on VRAM)
    max_parallel_jobs: int = Field(default=1, alias="MAX_PARALLEL_JOBS", ge=1, le=8)

    # Local OpenAI-compatible server (Ollama default port, also works with LM Studio :1234)
    local_llm_enabled: bool = Field(default=True, alias="LOCAL_LLM_ENABLED")
    local_llm_base_url: str = Field(default="http://127.0.0.1:11434/v1", alias="LOCAL_LLM_BASE_URL")
    local_llm_api_key: str = Field(default="ollama", alias="LOCAL_LLM_API_KEY")
    # Text planner (director). Keep a fast non-vision model here.
    local_llm_model: str = Field(default="llama3.2", alias="LOCAL_LLM_MODEL")
    # Vision critic for frame stills. Required for cast QA when Gemini is down.
    # Empty → auto-pick any installed vision model (llava, etc.).
    local_llm_vision_model: str = Field(default="llava", alias="LOCAL_LLM_VISION_MODEL")
    local_llm_timeout_sec: int = Field(default=300, alias="LOCAL_LLM_TIMEOUT_SEC")
    local_llm_max_tokens: int = Field(default=2048, alias="LOCAL_LLM_MAX_TOKENS")
    # Downscale review frames before sending to local VLMs (speed + context).
    local_llm_vision_max_side: int = Field(default=512, alias="LOCAL_LLM_VISION_MAX_SIDE")

    # Provider preference: gemini → local_openai → offline
    llm_fallback_order: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["gemini", "local_openai", "offline"],
        alias="LLM_FALLBACK_ORDER",
    )

    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=7860, alias="PORT")

    # Voice / ElevenLabs narration (documentary-style VO for all narrative modes when enabled)
    enable_voice: bool = Field(default=True, alias="ENABLE_VOICE")
    elevenlabs_model_id: str = Field(
        default="eleven_multilingual_v2",
        alias="ELEVENLABS_MODEL_ID",
    )
    elevenlabs_stability: float = Field(default=0.45, alias="ELEVENLABS_STABILITY")
    elevenlabs_similarity: float = Field(default=0.75, alias="ELEVENLABS_SIMILARITY")
    elevenlabs_timeout_sec: int = Field(default=120, alias="ELEVENLABS_TIMEOUT_SEC")
    # Lower ambient under VO (0–1 scale factor on original clip audio)
    narration_ambient_mix: float = Field(default=0.22, alias="NARRATION_AMBIENT_MIX")
    narration_voice_gain: float = Field(default=1.0, alias="NARRATION_VOICE_GAIN")

    # Auto-launch local deps when generate / resume starts
    auto_start_comfy: bool = Field(default=True, alias="AUTO_START_COMFY")
    auto_start_ollama: bool = Field(default=True, alias="AUTO_START_OLLAMA")
    comfyui_root: Path = Field(default=Path(r"E:/AI/ComfyUI"), alias="COMFYUI_ROOT")
    comfyui_python: str = Field(default="", alias="COMFYUI_PYTHON")  # empty → <root>/.venv/Scripts/python.exe
    comfyui_extra_args: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["--lowvram"],
        alias="COMFYUI_EXTRA_ARGS",
    )
    comfy_start_timeout_sec: int = Field(default=300, alias="COMFY_START_TIMEOUT_SEC")
    # If something is listening on COMFYUI_PORT but lacks H3 nodes, free the port and start COMFYUI_ROOT
    comfy_replace_non_h3: bool = Field(default=True, alias="COMFY_REPLACE_NON_H3")
    # Fail generate when H3 nodes are missing (after any replace attempt)
    comfy_require_h3_nodes: bool = Field(default=True, alias="COMFY_REQUIRE_H3_NODES")
    ollama_cmd: str = Field(default="ollama", alias="OLLAMA_CMD")
    ollama_start_timeout_sec: int = Field(default=60, alias="OLLAMA_START_TIMEOUT_SEC")
    # Max time to wait for essential tools (ComfyUI, etc.) before failing with a clear prompt
    essentials_wait_sec: int = Field(default=300, alias="ESSENTIALS_WAIT_SEC")

    @field_validator(
        "gemini_model_fallbacks",
        "llm_fallback_order",
        "gemini_image_models",
        "comfyui_extra_args",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, v: Any) -> list[str]:
        if v is None or v == "":
            return []
        if isinstance(v, str):
            # allow space-separated extra args or comma-separated lists
            if " " in v and "," not in v and any(v.startswith(s) for s in ("-",)):
                return [p for p in v.split() if p]
            return [p.strip() for p in v.split(",") if p.strip()]
        if isinstance(v, list):
            return [str(p).strip() for p in v if str(p).strip()]
        return v

    @property
    def comfy_base_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
