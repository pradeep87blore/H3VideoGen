"""Film director: concept → production plan + shot list (Gemini with local/offline fallback)."""
from __future__ import annotations

from typing import Callable

from ..character_board import ensure_character_designs
from ..config import Settings
from ..llm import LLMRouter
from ..models import CharacterDesign, ProductionPlan, ShotPlan

LogFn = Callable[[str], None]

DIRECTOR_SYSTEM = """You are an elite film DIRECTOR for short AI-generated YouTube videos.
You design production packages that a video generation model (MiniMax H3) will render as
short clips (~5s each) with optional ambient audio (no voiceover required yet).

Hard constraints for H3 clips:
- Each shot is a SINGLE continuous take (no multi-cut montage inside one clip unless
  the model can hold one continuous camera idea).
- Duration snaps to ~5s (124 frames @ 24fps) unless you specially request longer.
- Frame: 16:9, ~1344x768, cinematic.
- No on-screen text, logos, watermarks, subtitles, or UI.
- Prefer unified stylized 3D / animated / cinematic look as styled by the user — not
  mixed photoreal + cartoon unless the style says so.
- Prefer clear action, readable silhouette, strong scale/emotion for YouTube retention.
- Write visual prompts as rich production prose a video model can follow.
- Include character locks so continuity holds across shots.
- Specify who is ON SCREEN for each shot to avoid “everyone in every frame” leakage.
- Define a short list of main CHARACTERS (id C01..) with precise look bible text and
  board_prompt describing a multi-view design sheet (face + body identity for stills).
- For each shot set ref_character_ids to which character sheet ids appear on screen.

YouTube bar (your job at planning stage):
- Hook in first shot (3 seconds of attention).
- Clear emotional arc even in 60–120s.
- Varied shot sizes: wide / medium / close for rhythm.
- Consistent style bible.

Return ONLY valid JSON matching the schema. No markdown fences."""


def _align_frames(seconds: float, fps: int = 24) -> int:
    """Snap to MiniMax H3 length grid: n % 17 == 5."""
    n = max(5, int(round(seconds * fps)))
    while n % 17 != 5:
        n += 1
    return n


class DirectorAgent:
    def __init__(self, settings: Settings, log: LogFn | None = None):
        self.settings = settings
        self.log = log
        self.llm = LLMRouter(settings, log=log)
        self.last_provider = ""

    def plan(
        self,
        user_prompt: str,
        style: str,
        target_duration_sec: float = 60.0,
        max_shots: int = 12,
    ) -> ProductionPlan:
        shot_count = max(2, min(max_shots, int(round(target_duration_sec / 5.0))))
        per_shot = target_duration_sec / shot_count

        user_msg = f"""Create a full production plan for this YouTube short.

STORY / CONCEPT:
{user_prompt}

VISUAL STYLE (separate from story — enforce strictly):
{style}

TARGET DURATION: ~{target_duration_sec:.0f} seconds
SHOT BUDGET: exactly {shot_count} shots (~{per_shot:.1f}s each)
RESOLUTION: 1344x768 16:9 @ 24fps

JSON schema:
{{
  "title": "string",
  "logline": "string",
  "target_duration_sec": number,
  "aspect_ratio": "16:9",
  "style_bible": "one dense paragraph locking look, lighting, materials, era",
  "character_lock": "explicit character designs for continuity (summary)",
  "characters": [
    {{
      "id": "C01",
      "name": "character name",
      "look": "face, hair, outfit, age, proportions — precise for lock",
      "board_prompt": "identity bible for multi-view design sheet: face, hair, outfit, proportions; plain studio bg; single character"
    }}
  ],
  "color_grade": "string",
  "audio_bed": "diegetic + music notes without spoken dialogue unless necessary SFX",
  "youtube_notes": "hook + retention strategy",
  "raw_director_notes": "anything else",
  "shots": [
    {{
      "id": "S01",
      "name": "short name",
      "beat": "story function",
      "duration_sec": {per_shot:.1f},
      "visual_prompt": "full render prompt body (scene action)",
      "camera": "lens / move",
      "audio_notes": "string",
      "character_presence": "who/what must appear / must NOT appear",
      "ref_character_ids": ["C01"]
    }}
  ]
}}

Characters: max 4 main cast. Every shot that shows a cast member must list their id in ref_character_ids.
Shot IDs must be S01..S{shot_count:02d} in order.
Make visual_prompt detailed enough for a diffusion video model.
"""

        result = self.llm.director_plan_payload(
            system=DIRECTOR_SYSTEM,
            user=user_msg,
            offline_kwargs={
                "user_prompt": user_prompt,
                "style": style,
                "target_duration_sec": target_duration_sec,
                "shot_count": shot_count,
                "per_shot": per_shot,
            },
        )
        self.last_provider = result.provider
        data = result.data
        # tolerate missing characters
        if "characters" not in data or not data.get("characters"):
            data["characters"] = []
        plan = ProductionPlan.model_validate(data)
        ensure_character_designs(plan)

        # Normalize frames / ids
        shots: list[ShotPlan] = []
        known_ids = {c.id for c in plan.characters}
        for i, s in enumerate(plan.shots[:shot_count]):
            sid = f"S{i+1:02d}"
            frames = _align_frames(s.duration_sec or per_shot)
            refs = [r for r in (s.ref_character_ids or []) if r in known_ids]
            if not refs and plan.characters:
                # default: main character if presence empty or mentions them
                refs = [plan.characters[0].id]
            shots.append(
                ShotPlan(
                    id=sid,
                    name=s.name or f"Shot {i+1}",
                    beat=s.beat,
                    duration_sec=frames / 24.0,
                    length_frames=frames,
                    visual_prompt=s.visual_prompt,
                    camera=s.camera,
                    audio_notes=s.audio_notes,
                    character_presence=s.character_presence,
                    ref_character_ids=refs,
                    seed=None,
                )
            )
        while len(shots) < shot_count:
            i = len(shots)
            frames = _align_frames(per_shot)
            shots.append(
                ShotPlan(
                    id=f"S{i+1:02d}",
                    name=f"Shot {i+1}",
                    beat="Continue story momentum",
                    duration_sec=frames / 24.0,
                    length_frames=frames,
                    visual_prompt=f"{user_prompt}. Continuous cinematic take matching style: {style}",
                    camera="medium, steady push-in",
                    audio_notes="Diegetic ambience",
                    character_presence="Story subject only",
                    ref_character_ids=[plan.characters[0].id] if plan.characters else [],
                    seed=None,
                )
            )
        plan.shots = shots
        plan.target_duration_sec = sum(s.duration_sec for s in shots)
        if not plan.style_bible:
            plan.style_bible = style
        if result.provider.startswith("offline"):
            note = "Offline template plan (no LLM)."
            plan.raw_director_notes = f"{plan.raw_director_notes}\n{note}".strip()
        return plan

    def build_render_prompt(
        self,
        plan: ProductionPlan,
        shot: ShotPlan,
        critic_notes: str = "",
        *,
        r2v: bool = False,
        picture_map: dict[str, int] | None = None,
        picture_meta: list[dict] | None = None,
        extra_picture_notes: list[str] | None = None,
    ) -> str:
        parts: list[str] = []
        if r2v and (picture_meta or picture_map):
            id_lines: list[str] = []
            if picture_meta:
                for m in picture_meta:
                    pic = m.get("picture")
                    name = m.get("name") or m.get("character_id") or "Character"
                    pose = m.get("label") or m.get("pose_id") or "view"
                    look = (m.get("look") or "")[:160]
                    id_lines.append(
                        f"- <Picture {pic}>: {name} — {pose} reference. "
                        f"Match face, hair, outfit, proportions exactly; "
                        f"do not copy this reference pose/camera. {look}"
                    )
            elif picture_map:
                for cid, pic in sorted(picture_map.items(), key=lambda kv: kv[1]):
                    char = next((c for c in plan.characters if c.id == cid), None)
                    label = char.name if char else cid
                    look = (char.look if char else "")[:180]
                    id_lines.append(
                        f"- {label} identity is <Picture {pic}> "
                        f"(match face, hair, outfit exactly; do not copy the reference pose). {look}"
                    )
            if id_lines:
                parts.append(
                    "REFERENCE IDENTITY LOCK (MiniMax H3 R2V multi-view sheet — follow precisely):\n"
                    "These images are the same character(s) from different angles for identity only.\n"
                    + "\n".join(id_lines)
                )
            if extra_picture_notes:
                parts.append("ADDITIONAL REFS:\n" + "\n".join(extra_picture_notes))

        parts.extend(
            [
                plan.style_bible.strip(),
                f"CHARACTER LOCK: {plan.character_lock}".strip() if plan.character_lock else "",
                f"COLOR GRADE: {plan.color_grade}".strip() if plan.color_grade else "",
                f"AUDIO: {plan.audio_bed}. Shot audio: {shot.audio_notes}".strip(),
                f"SHOT {shot.id} — {shot.name}. Story beat: {shot.beat}.",
                f"CHARACTER PRESENCE: {shot.character_presence}" if shot.character_presence else "",
                f"CAMERA: {shot.camera}" if shot.camera else "",
                shot.visual_prompt.strip(),
                "Single continuous animated/cinematic shot. No on-screen text, logos, subtitles, watermarks.",
            ]
        )
        if critic_notes:
            parts.append(
                "DIRECTOR RETAKE NOTES (mandatory fixes from previous take):\n" + critic_notes.strip()
            )
        return "\n\n".join(p for p in parts if p)
