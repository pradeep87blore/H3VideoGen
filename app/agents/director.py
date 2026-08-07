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
- CAST EXCLUSIVITY IS CRITICAL: H3 video models invent extra cast if the prompt
  names or implies the full ensemble. Prevent this carefully:
  • ref_character_ids = ONLY character ids physically visible this take (never everyone "for continuity").
  • character_presence must list who APPEARS and who is ABSENT by name ("Goldilocks alone;
    Papa/Mama/Baby Bear must not appear: no silhouettes, faces, or bodies anywhere in frame").
  • visual_prompt for alone/empty-house beats must not stage the absent cast in the background.
  • When the story is "cottage of the three bears" but only Goldilocks is home, describe
    empty furniture/set dressing — do NOT show the bears watching or waiting.
- Define a short list of main CHARACTERS (id C01..) with precise look bible text and
  board_prompt describing a multi-view design sheet (face + body identity for stills).

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

Characters: max 4 main cast. C01 MUST be the story protagonist (e.g. Goldilocks),
not a supporting ensemble member. Supporting family (bears, etc.) are C02+.
Every shot that shows a cast member must list their id in ref_character_ids.
Shots with no living cast use ref_character_ids: [] and character_presence "no living characters on screen; empty set only".
Do NOT put the full cast in every ref_character_ids "for continuity" — that forces everyone on screen.
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
        ensure_character_designs(plan, story_hint=user_prompt)

        # Normalize frames / ids
        shots: list[ShotPlan] = []
        known_ids = {c.id for c in plan.characters}
        for i, s in enumerate(plan.shots[:shot_count]):
            sid = f"S{i+1:02d}"
            frames = _align_frames(s.duration_sec or per_shot)
            refs = [r for r in (s.ref_character_ids or []) if r in known_ids]
            deduped: list[str] = []
            for r in refs:
                if r not in deduped:
                    deduped.append(r)
            refs = deduped
            # Empty refs allowed (empty set / prop-only). Default lead only when cast was omitted.
            if not refs and plan.characters:
                presence_l = (s.character_presence or "").lower()
                empty_intent = any(
                    k in presence_l
                    for k in (
                        "no character",
                        "no one",
                        "empty",
                        "none on",
                        "prop only",
                        "no living",
                        "just the",  # e.g. just the three bowls
                    )
                )
                if not empty_intent:
                    refs = [plan.characters[0].id]
            presence = enforce_character_presence_text(
                plan, refs, existing=s.character_presence or ""
            )
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
                    character_presence=presence,
                    ref_character_ids=refs,
                    seed=None,
                )
            )
        while len(shots) < shot_count:
            i = len(shots)
            frames = _align_frames(per_shot)
            refs = [plan.characters[0].id] if plan.characters else []
            presence = enforce_character_presence_text(plan, refs, existing="Story subject only")
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
                    character_presence=presence,
                    ref_character_ids=refs,
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
        on_ids, off_chars = _cast_split_for_shot(plan, shot, picture_meta, picture_map)
        parts: list[str] = []

        # Hard exclusivity FIRST so the model prioritizes it over story leakage
        exclusivity = _exclusivity_block(plan, on_ids, off_chars)
        if exclusivity:
            parts.append(exclusivity)

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
                    "These images are ONLY identity for characters already allowed ON SCREEN.\n"
                    "Do not invent additional cast because other names appear in the story.\n"
                    + "\n".join(id_lines)
                )
            if extra_picture_notes:
                parts.append("ADDITIONAL REFS:\n" + "\n".join(extra_picture_notes))

        # Shot-scoped look lock — do NOT paste full-cast character_lock (it causes leakage)
        lock_lines = _on_screen_look_lock(plan, on_ids)
        parts.extend(
            [
                plan.style_bible.strip(),
                lock_lines,
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


def _cast_split_for_shot(
    plan: ProductionPlan,
    shot: ShotPlan,
    picture_meta: list[dict] | None,
    picture_map: dict[str, int] | None,
) -> tuple[list[str], list[CharacterDesign]]:
    on: list[str] = []
    if picture_meta:
        for m in picture_meta:
            cid = m.get("character_id")
            if cid and cid not in on:
                on.append(str(cid))
    if not on and picture_map:
        on = [cid for cid, _ in sorted(picture_map.items(), key=lambda kv: kv[1])]
    if not on:
        on = list(shot.ref_character_ids or [])
    known = {c.id: c for c in plan.characters or []}
    on = [cid for cid in on if cid in known]
    off = [c for c in (plan.characters or []) if c.id not in set(on)]
    return on, off


def _on_screen_look_lock(plan: ProductionPlan, on_ids: list[str]) -> str:
    if not on_ids:
        return (
            "CHARACTER LOCK (this take): No living cast members. "
            "Empty set / props only — do not invent story characters."
        )
    lines = [
        "CHARACTER LOCK (this take ONLY — ignore other cast not listed here):",
    ]
    by_id = {c.id: c for c in plan.characters or []}
    for cid in on_ids:
        c = by_id.get(cid)
        if not c:
            continue
        look = (c.look or "").strip() or "consistent with identity sheet"
        lines.append(f"- {c.name} ({cid}): {look}")
    return "\n".join(lines)


def _exclusivity_block(
    plan: ProductionPlan,
    on_ids: list[str],
    off_chars: list[CharacterDesign],
) -> str:
    if not plan.characters:
        return ""
    by_id = {c.id: c for c in plan.characters}
    on_names = [by_id[i].name for i in on_ids if i in by_id]
    lines = [
        "CAST EXCLUSIVITY (HARD RULE — higher priority than story lore):",
    ]
    if on_names:
        lines.append(
            "ON SCREEN living characters (ONLY these may appear as people/animals/figures): "
            + ", ".join(on_names)
            + "."
        )
    else:
        lines.append(
            "ON SCREEN living characters: NONE. This is a prop/environment take only."
        )
    if off_chars:
        ban = ", ".join(f"{c.name} ({c.id})" for c in off_chars)
        lines.append(
            f"BANNED this take — must not appear in ANY form: {ban}. "
            "No faces, bodies, silhouettes, shadows cast by them, "
            "watchers in windows, background cameos, plush stand-ins, or crowd fillers. "
            "If the story setup mentions their home/items, show empty furniture and set dressing only."
        )
    lines.append(
        "Violation of bans is a failed take. Prefer empty space over inventing banned cast."
    )
    return "\n".join(lines)


def enforce_character_presence_text(
    plan: ProductionPlan,
    ref_ids: list[str],
    existing: str = "",
) -> str:
    """Rewrite/augment presence so absent cast is named explicitly."""
    by_id = {c.id: c for c in plan.characters or []}
    on_names = [by_id[i].name for i in ref_ids if i in by_id]
    off = [c for c in (plan.characters or []) if c.id not in set(ref_ids)]
    off_names = [c.name for c in off]

    base = (existing or "").strip()
    # Drop previously auto-appended APPEARS/ABSENT blocks if re-normalizing
    for marker in ("APPEARS:", "ABSENT (do not show):"):
        if marker in base:
            base = base.split(marker)[0].strip(" .;")

    parts: list[str] = []
    if base:
        parts.append(base.rstrip(" ."))
    if on_names:
        parts.append(f"APPEARS: {', '.join(on_names)} only")
    else:
        parts.append("APPEARS: no living cast (empty set / props only)")
    if off_names:
        parts.append(
            f"ABSENT (do not show): {', '.join(off_names)} — "
            "no silhouettes, faces, bodies, or background watchers"
        )
    return ". ".join(parts) + "."


def normalize_plan_cast_presence(plan: ProductionPlan) -> ProductionPlan:
    """Recompute character_presence bans from ref_character_ids (safe on resume)."""
    if not plan.characters:
        return plan
    for shot in plan.shots or []:
        refs = list(shot.ref_character_ids or [])
        shot.character_presence = enforce_character_presence_text(
            plan, refs, existing=shot.character_presence or ""
        )
    return plan
