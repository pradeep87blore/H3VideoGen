"""Film director: concept → production plan + shot list (mode-aware)."""
from __future__ import annotations

from typing import Callable

from ..character_board import ensure_character_designs
from ..config import Settings
from ..llm import LLMRouter
from ..models import CharacterDesign, NarrativeMode, ProductionPlan, ShotPlan, normalize_narrative_mode

LogFn = Callable[[str], None]

DIRECTOR_SYSTEM_CHARACTER = """You are an elite film DIRECTOR for short AI-generated YouTube videos.
You design production packages that a video generation model (MiniMax H3) will render as
short clips (~5s each) with optional ambient audio and a separate narrator track.

Hard constraints for H3 clips:
- Each shot is a SINGLE continuous take (no multi-cut montage inside one clip).
- Duration snaps to ~5s (124 frames @ 24fps) unless you specially request longer.
- Frame: 16:9, cinematic.
- No on-screen text, logos, watermarks, subtitles, or UI.
- Prefer unified stylized look as styled by the user.
- Prefer clear action, readable silhouette for YouTube retention.
- Write visual prompts as rich production prose a video model can follow.
- Include character locks so continuity holds across shots.
- CAST EXCLUSIVITY IS CRITICAL: H3 invents extra cast if the prompt names the full ensemble.
  • ref_character_ids = ONLY ids physically visible this take.
  • character_presence lists APPEARS and ABSENT by name.
  • empty house/prop shots leave cast out of visual_prompt.
- Define main CHARACTERS (id C01..) with precise look bible + board_prompt.
- audio_notes: concrete diegetic SFX and room tone H3 will generate in stereo
  (footsteps, fabric, weather, object impacts). Not narrator VO.
- narration_line: one short sentence of third-person documentary-style narrator VO for that shot
  (not character dialogue). Natural pacing for ~5s. Mixed later; do not put VO inside visual_prompt.

Return ONLY valid JSON matching the schema. No markdown fences."""

DIRECTOR_SYSTEM_DOCUMENTARY = """You are an elite documentary DIRECTOR for short commemorative / history YouTube videos.
MiniMax H3 will render ~5s continuous cinematic clips. A professional narrator (ElevenLabs) will speak later.

Hard constraints:
- NO protagonist character-lock packages. Prefer places, armies, machines, nature, crowds as masses.
- Do NOT invent a named main cast list unless a specific historical figure is essential (at most 1–2).
  Prefer empty characters array.
- ref_character_ids must always be [] when characters is empty.
- Each shot is one continuous take; no on-screen text/logos/charts with readable labels.
- Visuals convey history through environment, scale, era detail, and action.
- audio_notes: concrete diegetic SFX / room / crowd (not the narrator).
- narration_line: calm documentary VO covering that beat (spoken English, ~5s worth of words).
- Style unified hyper-real OR stylized documentary look from the style bible.
- Hook early; clear cause→effect arc; historical tone (not parody unless prompt asks).

Return ONLY valid JSON. No markdown fences."""

DIRECTOR_SYSTEM_EXPLAINER = """You are an elite educational DIRECTOR for short concept explainer YouTube videos
(e.g. inflation, Bitcoin, gravity). MiniMax H3 renders ~5s continuous clips; ElevenLabs narrates.

Hard constraints:
- Teach ONE clear idea via visual metaphors — not a character fairy tale.
- characters array should be EMPTY (or at most one optional "prop mascot" if useful — prefer empty).
- ref_character_ids always [] when no characters.
- No on-screen text, charts with numbers, logos, watermarks, UI.
- Prefer abstract but concrete metaphors (coins, factories, scales, digital ledgers as objects/places).
- Shot structure typically: hook problem → simple definition → mechanism → example → takeaway.
- audio_notes: diegetic SFX for the metaphor world (not teacher VO).
- narration_line: clear plain-language teacher VO for ~5s (no jargon dump; one idea per shot).
- Keep style bible coherent so the metaphor world feels like one channel brand.

Return ONLY valid JSON. No markdown fences."""


def _align_frames(seconds: float, fps: int = 24) -> int:
    """Snap to MiniMax H3 length grid: n % 17 == 5."""
    n = max(5, int(round(seconds * fps)))
    while n % 17 != 5:
        n += 1
    return n


def _mode_system(mode: str) -> str:
    m = normalize_narrative_mode(mode)
    if m == NarrativeMode.documentary.value:
        return DIRECTOR_SYSTEM_DOCUMENTARY
    if m == NarrativeMode.explainer.value:
        return DIRECTOR_SYSTEM_EXPLAINER
    return DIRECTOR_SYSTEM_CHARACTER


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
        narrative_mode: str = "character",
    ) -> ProductionPlan:
        mode = normalize_narrative_mode(narrative_mode)
        shot_count = max(2, min(max_shots, int(round(target_duration_sec / 5.0))))
        per_shot = target_duration_sec / shot_count
        user_msg = self._user_prompt(
            user_prompt, style, target_duration_sec, shot_count, per_shot, mode
        )

        result = self.llm.director_plan_payload(
            system=_mode_system(mode),
            user=user_msg,
            offline_kwargs={
                "user_prompt": user_prompt,
                "style": style,
                "target_duration_sec": target_duration_sec,
                "shot_count": shot_count,
                "per_shot": per_shot,
                "narrative_mode": mode,
            },
        )
        self.last_provider = result.provider
        data = result.data
        if "characters" not in data or data.get("characters") is None:
            data["characters"] = []
        data["narrative_mode"] = mode
        plan = ProductionPlan.model_validate(data)
        plan.narrative_mode = mode

        invent_cast = mode == NarrativeMode.character.value
        if invent_cast:
            ensure_character_designs(plan, story_hint=user_prompt)
        else:
            # Keep only explicit characters with names; drop empty invented noise
            plan.characters = [
                c for c in (plan.characters or []) if (c.name or "").strip()
            ][:4]
            for i, c in enumerate(plan.characters):
                c.id = f"C{i+1:02d}"

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
            if invent_cast and not refs and plan.characters:
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
                        "just the",
                    )
                )
                if not empty_intent:
                    refs = [plan.characters[0].id]
            if not invent_cast:
                refs = [r for r in refs if r in known_ids]
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
                    narration_line=(s.narration_line or "").strip(),
                    seed=None,
                )
            )
        while len(shots) < shot_count:
            i = len(shots)
            frames = _align_frames(per_shot)
            refs = [plan.characters[0].id] if invent_cast and plan.characters else []
            presence = enforce_character_presence_text(
                plan, refs, existing="Story subject only" if invent_cast else "Visual metaphor only"
            )
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
                    narration_line="",
                    seed=None,
                )
            )
        plan.shots = shots
        plan.target_duration_sec = sum(s.duration_sec for s in shots)
        if not plan.style_bible:
            plan.style_bible = style
        # Build full narration script if LLM left it empty
        if not (plan.narration_script or "").strip():
            joined = " ".join(
                (s.narration_line.rstrip(".") + ".")
                for s in plan.shots
                if (s.narration_line or "").strip()
            )
            plan.narration_script = joined.strip()
        if result.provider.startswith("offline"):
            note = "Offline template plan (no LLM)."
            plan.raw_director_notes = f"{plan.raw_director_notes}\n{note}".strip()
        return plan

    def _user_prompt(
        self,
        user_prompt: str,
        style: str,
        target_duration_sec: float,
        shot_count: int,
        per_shot: float,
        mode: str,
    ) -> str:
        common_head = f"""Create a full production plan for this YouTube short.

NARRATIVE MODE: {mode}
STORY / CONCEPT:
{user_prompt}

VISUAL STYLE (separate from story — enforce strictly):
{style}

TARGET DURATION: ~{target_duration_sec:.0f} seconds
SHOT BUDGET: exactly {shot_count} shots (~{per_shot:.1f}s each)
RESOLUTION: 16:9 @ 24fps (single continuous take per shot)

Each shot must include narration_line: ~one spoken sentence of professional narrator VO.
Write narration as continuous spoken English that will be read aloud; do not describe camera in VO.
"""

        if mode == NarrativeMode.character.value:
            return common_head + f"""
JSON schema:
{{
  "title": "string",
  "logline": "string",
  "target_duration_sec": number,
  "aspect_ratio": "16:9",
  "narrative_mode": "{mode}",
  "style_bible": "one dense paragraph locking look, lighting, materials, era",
  "character_lock": "explicit character designs for continuity (summary)",
  "characters": [
    {{
      "id": "C01",
      "name": "character name",
      "look": "face, hair, outfit, age, proportions — precise for lock",
      "board_prompt": "identity for multi-view design sheet; plain studio bg; single character"
    }}
  ],
  "color_grade": "string",
  "audio_bed": "diegetic ambience + optional light score (H3 generates this; narrator is separate)",
  "youtube_notes": "hook + retention",
  "narration_script": "optional full VO string (else built from shot lines)",
  "raw_director_notes": "anything else",
  "shots": [
    {{
      "id": "S01",
      "name": "short name",
      "beat": "story function",
      "duration_sec": {per_shot:.1f},
      "visual_prompt": "full render prompt body (scene action)",
      "camera": "lens / move",
      "audio_notes": "diegetic SFX/room tone for H3 stereo (not VO)",
      "character_presence": "who APPEARS / ABSENT",
      "ref_character_ids": ["C01"],
      "narration_line": "Spoken VO sentence for this shot"
    }}
  ]
}}

Characters: max 4. C01 = protagonist. Do not put full cast on every shot.
Shot IDs S01..S{shot_count:02d}. visual_prompt detailed for a diffusion video model.
"""

        # Documentary & explainer share a lighter character schema
        extra = (
            "Documentary: focus on events, places, machines, crowds as mass. "
            "Prefer characters: []. Historical figure allowed only if essential."
            if mode == NarrativeMode.documentary.value
            else
            "Explainer: teach the concept with metaphors. Prefer characters: []. "
            "Shots: hook → definition → mechanism → example → takeaway as fits the budget."
        )
        return common_head + f"""
{extra}

JSON schema:
{{
  "title": "string",
  "logline": "string",
  "target_duration_sec": number,
  "aspect_ratio": "16:9",
  "narrative_mode": "{mode}",
  "style_bible": "dense paragraph locking look, lighting, materials, era/mood",
  "character_lock": "empty or brief prop identity note",
  "characters": [],
  "color_grade": "string",
  "audio_bed": "ambience/score H3 can generate; narrator is separate",
  "youtube_notes": "hook + clarity strategy",
  "narration_script": "optional full VO",
  "raw_director_notes": "anything else",
  "shots": [
    {{
      "id": "S01",
      "name": "short name",
      "beat": "story / teaching function",
      "duration_sec": {per_shot:.1f},
      "visual_prompt": "full render prompt for the shot (no on-screen text)",
      "camera": "lens / move",
      "audio_notes": "diegetic SFX/room tone for H3 stereo (not narrator VO)",
      "character_presence": "what appears (places/objects/masses) — not a cast list",
      "ref_character_ids": [],
      "narration_line": "Spoken VO sentence for this shot"
    }}
  ]
}}

Shot IDs must be S01..S{shot_count:02d} in order. Keep ref_character_ids empty.
visual_prompt detailed for a diffusion video model. No readable screen text.
"""

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
        video_notes: list[str] | None = None,
        keyframe_mode: str | None = None,
        clip_duration_sec: float | None = None,
    ) -> str:
        mode = normalize_narrative_mode(getattr(plan, "narrative_mode", None) or "character")
        on_ids, off_chars = _cast_split_for_shot(plan, shot, picture_meta, picture_map)
        parts: list[str] = []
        dur = float(clip_duration_sec or shot.duration_sec or 5.0)

        if keyframe_mode == "fl2va":
            parts.append(
                "How the reference pictures align with the target video — "
                "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
                f"Picture 2 (from Shot 1) aligns with the {dur:.2f}-second mark of the target video."
            )
        elif keyframe_mode == "i2va":
            parts.append(
                "For the target video, at 0.00 seconds into the target video, "
                "<Picture 1> (from [Shot 1]) is fully referenced."
            )
        elif keyframe_mode == "l2va":
            parts.append(
                "How the reference pictures align with the target video — "
                f"<Picture 1> (from [Shot 1]) aligns with the {dur:.2f}-second mark of the target video."
            )

        if mode == NarrativeMode.character.value or plan.characters:
            exclusivity = _exclusivity_block(plan, on_ids, off_chars)
            if exclusivity:
                parts.append(exclusivity)

        if r2v and (picture_meta or picture_map):
            id_lines: list[str] = []
            if picture_meta:
                for m in picture_meta:
                    pic = m.get("picture")
                    name = m.get("name") or m.get("character_id") or "Subject"
                    pose = m.get("label") or m.get("pose_id") or "view"
                    look = (m.get("look") or "")[:160]
                    job = m.get("job") or "identity / appearance / outfit"
                    id_lines.append(
                        f"- <Picture {pic}> assigned job: {job} for {name} ({pose}). "
                        f"Match identity exactly; do not copy this reference pose or camera. {look}"
                    )
            elif picture_map:
                for cid, pic in sorted(picture_map.items(), key=lambda kv: kv[1]):
                    char = next((c for c in plan.characters if c.id == cid), None)
                    label = char.name if char else cid
                    look = (char.look if char else "")[:180]
                    id_lines.append(
                        f"- <Picture {pic}> assigned job: identity / appearance for {label} "
                        f"(match exactly; do not copy the reference pose). {look}"
                    )
            if id_lines:
                parts.append(
                    "REFERENCE IDENTITY LOCK (MiniMax H3 R2V — <Picture N>):\n"
                    + "\n".join(id_lines)
                )
            if extra_picture_notes:
                parts.append("ADDITIONAL IMAGE REFS:\n" + "\n".join(extra_picture_notes))

        if r2v and video_notes:
            parts.append(
                "REFERENCE VIDEO LOCK (MiniMax H3 R2V — <Video k> / <Audio j>):\n"
                + "\n".join(video_notes)
            )

        if mode == NarrativeMode.character.value or on_ids:
            lock_lines = _on_screen_look_lock(plan, on_ids)
        else:
            lock_lines = (
                "CONTINUITY: Keep era, materials, palette, and metaphor world consistent with style bible. "
                "No invented named main character unless prompted."
            )

        mode_note = {
            NarrativeMode.documentary.value: "Documentary continuous take — historical / event feel.",
            NarrativeMode.explainer.value: "Educational explainer metaphor shot — concept clarity first.",
            NarrativeMode.character.value: "Narrative fiction continuous take.",
        }.get(mode, "")

        cam = (shot.camera or "").strip()
        style_head = (plan.style_bible or "").strip()
        action = (shot.visual_prompt or "").strip()
        presence = (shot.character_presence or "").strip()
        timeline = (
            f"integrated_multimodal_description: [Shot 1] {style_head} "
            f"{mode_note} "
            + (f"Camera: {cam}. " if cam else "")
            + (f"{presence}. " if presence else "")
            + action
        )
        parts.extend(
            [
                lock_lines,
                f"COLOR GRADE: {plan.color_grade}".strip() if plan.color_grade else "",
                f"SHOT {shot.id} — {shot.name}. Story beat: {shot.beat}.",
                timeline,
                "Single continuous cinematic shot. No on-screen text, logos, subtitles, watermarks.",
            ]
        )
        if self.settings.h3_prompt_native_audio:
            parts.extend(_h3_audio_block(plan, shot, enable_voice=self.settings.enable_voice))
        elif plan.audio_bed or shot.audio_notes:
            parts.append(f"AUDIO: {plan.audio_bed}. Shot audio: {shot.audio_notes}".strip())
        if critic_notes:
            parts.append(
                "DIRECTOR RETAKE NOTES (mandatory fixes from previous take):\n" + critic_notes.strip()
            )
        return "\n\n".join(p for p in parts if p)


def _h3_audio_block(plan: ProductionPlan, shot: ShotPlan, *, enable_voice: bool) -> list[str]:
    ambience = " ".join(
        x.strip() for x in (plan.audio_bed, shot.audio_notes) if (x or "").strip()
    ).strip()
    if not ambience:
        ambience = (
            "Natural diegetic room tone matching the scene, with physical action sounds "
            "timed to visible motion."
        )
    lines = [
        f"overall_soundscape: {ambience} Synchronize footsteps, fabric, weather, and object "
        "impacts to on-screen action. Stereo cinematic mix."
    ]
    bed_l = (plan.audio_bed or "").lower()
    if any(k in bed_l for k in ("score", "music", "orchestra", "piano", "strings", "underscore")):
        lines.append(f"non_diegetic_music: {plan.audio_bed}")
    else:
        lines.append(
            "non_diegetic_music: Sparse cinematic underscore at low volume, or N/A if the "
            "scene is driven by diegetic sound only."
        )
    if enable_voice:
        lines.append(
            "H3 AUDIO RULE: No spoken narration or character dialogue in this clip. "
            "Keep mouths closed or natural non-speech motion. A separate narrator is mixed later."
        )
    elif (shot.narration_line or "").strip():
        vo = shot.narration_line.strip().replace("<", "").replace(">", "")
        lines.append(
            "A calm off-screen narrator (S1) says in an off-screen voiceover: "
            f"<d>[English] {vo}</d> while on-screen lips remain completely closed."
        )
    return lines


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
            "CHARACTER LOCK (this take): No living cast members from a character board. "
            "Focus on environment / props / masses only."
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
            "watchers in windows, background cameos, plush stand-ins, or crowd fillers."
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
    if not plan.characters:
        base = (existing or "").strip()
        return base or "No named character cast — environment, props, or mass figures only."

    by_id = {c.id: c for c in plan.characters or []}
    on_names = [by_id[i].name for i in ref_ids if i in by_id]
    off = [c for c in (plan.characters or []) if c.id not in set(ref_ids)]
    off_names = [c.name for c in off]

    base = (existing or "").strip()
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
    for shot in plan.shots or []:
        refs = list(shot.ref_character_ids or [])
        shot.character_presence = enforce_character_presence_text(
            plan, refs, existing=shot.character_presence or ""
        )
    return plan
