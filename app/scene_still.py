"""Pre-clip scene still generation for critic preflight (before full H3 video)."""
from __future__ import annotations

import base64
import shutil
from pathlib import Path
from typing import Callable

from .comfy_h3 import ComfyError, ComfyH3Client
from .config import Settings
from .job_control import CancelledError
from .media import extract_frames
from .models import ProductionPlan, ShotPlan

LogFn = Callable[[str], None]


def build_scene_still_prompt(
    plan: ProductionPlan,
    shot: ShotPlan,
    render_prompt_body: str = "",
    critic_notes: str = "",
) -> str:
    """Composition-focused still prompt (shared style/cast world with H3)."""
    parts = [
        "Single cinematic keyframe / concept still for a video shot.",
        "Not an animation sequence. One frozen moment only.",
        f"Production title: {plan.title}",
        f"Style: {plan.style_bible}",
        f"Shot {shot.id} — {shot.name}",
        f"Story beat: {shot.beat}",
        f"Camera: {shot.camera}" if shot.camera else "",
        f"Presence: {shot.character_presence}" if shot.character_presence else "",
        (shot.visual_prompt or "").strip(),
        "Match the style bible exactly. No on-screen text, logos, subtitles, or watermarks.",
        "Readable composition, strong subject focus, YouTube-thumbnail clarity.",
    ]
    if critic_notes:
        parts.append("MANDATORY FIXES FROM PREVIOUS PREVIEW CRITIC:\n" + critic_notes.strip())
    body = (render_prompt_body or "").strip()
    if body:
        parts.append("Scene brief (visual):\n" + body[:1800])
    return "\n\n".join(p for p in parts if p)


def generate_scene_still(
    settings: Settings,
    *,
    plan: ProductionPlan,
    shot: ShotPlan,
    out_path: Path,
    render_prompt: str = "",
    critic_notes: str = "",
    seed: int = 42,
    comfy: ComfyH3Client | None = None,
    gen_mode: str = "t2v",
    ref_image_paths: list[Path] | None = None,
    project_tag: str = "preclip",
    log: LogFn | None = None,
) -> Path | None:
    """
    Create a preview still for pre-clip QA.

    Modes (settings.preclip_still_mode):
      - gemini: Gemini image models only
      - h3_probe: short H3 clip → extract one frame
      - auto: gemini, then h3_probe fallback
      - none: returns None
    """
    mode = (settings.preclip_still_mode or "auto").strip().lower()
    if mode in ("none", "off", "disabled"):
        return None

    prompt = build_scene_still_prompt(plan, shot, render_prompt, critic_notes)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tried: list[str] = []
    if mode in ("auto", "gemini"):
        tried.append("gemini")
        path = _gemini_scene_still(settings, prompt, out_path, log=log)
        if path:
            return path
        if mode == "gemini":
            return None

    if mode in ("auto", "h3_probe", "h3"):
        tried.append("h3_probe")
        if comfy is None:
            if log:
                log("Preclip still: H3 probe skipped (no Comfy client)")
            return None
        path = _h3_probe_still(
            settings,
            comfy,
            prompt=render_prompt or prompt,
            out_path=out_path,
            seed=seed,
            gen_mode=gen_mode,
            ref_image_paths=ref_image_paths,
            project_tag=project_tag,
            log=log,
        )
        if path:
            return path

    if log:
        log(f"Preclip still: no image produced (tried {', '.join(tried) or mode})")
    return None


def _gemini_scene_still(
    settings: Settings,
    prompt: str,
    out_path: Path,
    log: LogFn | None = None,
) -> Path | None:
    if not settings.gemini_api_key:
        if log:
            log("Preclip still: no GEMINI_API_KEY — skip Gemini image")
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"Preclip still: google-genai unavailable ({exc})")
        return None

    client = genai.Client(api_key=settings.gemini_api_key)
    models = [m.strip() for m in (settings.gemini_image_models or []) if m.strip()]
    last_err: Exception | None = None
    for model in models:
        try:
            if log:
                log(f"Preclip still: Gemini image via {model}…")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                ),
            )
            path = _save_gemini_image(response, out_path)
            if path:
                if log:
                    log(f"Preclip still: saved {path.name} ({path.stat().st_size} bytes)")
                return path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if log:
                log(f"Preclip still: {model} failed ({exc})")
            continue
    if last_err and log:
        log(f"Preclip still: all Gemini image models failed ({last_err})")
    return None


def _save_gemini_image(response: object, out_path: Path) -> Path | None:
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if not inline:
                continue
            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", "image/png") or "image/png"
            if not data:
                continue
            raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
            stem = out_path.with_suffix("")
            ext = ".png" if "png" in mime else ".jpg"
            dest = Path(f"{stem}{ext}")
            dest.write_bytes(raw)
            if dest.stat().st_size > 500:
                return dest
    return None


def _h3_probe_still(
    settings: Settings,
    comfy: ComfyH3Client,
    *,
    prompt: str,
    out_path: Path,
    seed: int,
    gen_mode: str,
    ref_image_paths: list[Path] | None,
    project_tag: str,
    log: LogFn | None = None,
) -> Path | None:
    length = int(settings.preclip_h3_length_frames)
    mode = (gen_mode or "t2v").lower()
    refs = list(ref_image_paths or [])
    if mode == "r2v" and not refs:
        mode = "t2v"
    prefix = f"video/H3VideoGen/{project_tag}/preclip_{out_path.stem}"
    try:
        if log:
            log(f"Preclip still: H3 probe {mode.upper()} length={length}…")
        src, _pid, used = comfy.generate(
            prompt,
            length=length,
            seed=seed,
            filename_prefix=prefix,
            mode=mode,
            ref_image_paths=refs if mode == "r2v" else None,
            project_tag=project_tag,
        )
        tmp_dir = out_path.parent / f"_probe_{out_path.stem}"
        frames = extract_frames(settings, src, tmp_dir, times=[0.4, 0.8])
        if not frames:
            return None
        dest = out_path.with_suffix(".jpg")
        shutil.copy2(frames[0], dest)
        if log:
            log(f"Preclip still: H3 probe frame from {used} → {dest.name}")
        return dest
    except CancelledError:
        raise
    except ComfyError as exc:
        if log:
            log(f"Preclip still: H3 probe failed ({exc})")
        return None
    except Exception as exc:  # noqa: BLE001
        if log:
            log(f"Preclip still: H3 probe error ({exc})")
        return None
