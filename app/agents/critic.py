"""Harsh YouTube critic for generated video frames (Gemini with local/offline fallback)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..llm import LLMRouter
from ..models import (
    CharacterDesign,
    CharacterSheetPose,
    CriticReview,
    CriticVerdict,
    NarrativeMode,
    ProductionPlan,
    ShotPlan,
    normalize_narrative_mode,
)

LogFn = Callable[[str], None]

CRITIC_SYSTEM_CHARACTER = """You are a HARSH film critic and YouTube content QA lead.
You review AI-generated video stills (and metadata) for a short that must be good enough
to upload on YouTube without embarrassment.

Be strict. Average random AI clips should FAIL or RETAKE. Only pass clips that look
intentional, continuous-character, and stylistically coherent.

Score 0–10 on:
- composition
- character_fidelity (matches lock / presence rules)
- style_consistency
- motion_readability (from freeze + description of intended action)
- story_clarity (does this beat read?)
- youtube_polish (artifacts, mush, weird anatomy, text leakage, gen-AI cheese)

Rules:
- overall_score < 7.5 → cannot be PASS (use RETAKE or REJECT)
- text/logos/watermarks → REJECT or RETAKE with fix
- CAST LEAKAGE is a hard fail: if CHARACTER PRESENCE / ABSENT / CAST EXCLUSIVITY bans a character
  score character_fidelity ≤ 3 and verdict RETAKE.
- wrong characters present → RETAKE
- photoreal vs style mismatch when style is animated → RETAKE
- boring empty frame with no story → RETAKE
- If pass, youtube_ready=true only if overall_score >= 8.0 AND no critical issues

Always provide revised_prompt: an improved visual-only prompt body for a retake.
Return ONLY JSON."""

CRITIC_SYSTEM_DOCUMENTARY = """You are a harsh documentary finishing critic for short history/event films.
Score 0–10: composition, character_fidelity (reuse for subject continuity of place/vehicles —
not face lock), style_consistency, motion_readability, story_clarity (does the historical beat read?),
youtube_polish.

Rules:
- overall_score < 7.5 → RETAKE or REJECT
- on-screen text/logos/watermarks → RETAKE
- anachronistic mix that breaks the era look → RETAKE
- empty frame with no documentary information → RETAKE
- Do NOT fail solely because there is no named protagonist face.
Always provide revised_prompt as improved scene action. Return ONLY JSON."""

CRITIC_SYSTEM_EXPLAINER = """You are a harsh educational explainer critic for short concept videos
(inflation, Bitcoin, science). Score composition, character_fidelity (treat as metaphor consistency),
style_consistency, motion_readability, story_clarity (does the teaching beat read without VO?),
youtube_polish.

Rules:
- overall_score < 7.5 → RETAKE
- on-screen text/charts with numbers → RETAKE (models mangle text)
- metaphor too abstract / unreadable → RETAKE
- style drift that breaks the education brand → RETAKE
Always provide revised_prompt. Return ONLY JSON."""


def _critic_system(mode: str) -> str:
    m = normalize_narrative_mode(mode)
    if m == NarrativeMode.documentary.value:
        return CRITIC_SYSTEM_DOCUMENTARY
    if m == NarrativeMode.explainer.value:
        return CRITIC_SYSTEM_EXPLAINER
    return CRITIC_SYSTEM_CHARACTER


CRITIC_SYSTEM_PRECLIP_STILL = """You are reviewing a PRE-CLIP KEYFRAME / STORYBOARD STILL — not a finished video.
This image is a cheap preflight before expensive video generation.

Score 0–10 on:
- composition (framing, subject readability)
- character_fidelity (cast presence / identity rules when applicable)
- style_consistency (matches style bible)
- story_clarity (does this beat read as a frozen moment?)
- youtube_polish (artifacts, mush, weird anatomy, text leakage)

For motion_readability: score neutral-high (7–8) unless the still completely fails to imply the
intended action (pose unreadable). Do NOT fail solely for lack of motion.

Rules:
- overall_score < threshold → RETAKE (do not PASS)
- CAST LEAKAGE / wrong banned characters → hard RETAKE
- style mismatch, empty unreadable frame → RETAKE
- Always provide retake_instructions and revised_prompt for the next still/video prompt
Return ONLY JSON."""

CRITIC_SYSTEM_CHARACTER_SHEET = """You are a harsh character-bible art director for AI video.
You review multi-view CHARACTER SHEET stills used as identity locks (R2V references).

Score 0–10 on:
- composition (clear pose, readable silhouette, usable as a ref)
- character_fidelity (matches the identity description / lock; single correct character)
- style_consistency (matches the production style bible)
- story_clarity (for sheets: pose intent is clear — face/outfit readable)
- youtube_polish (hands, eyes, anatomy, text, mush, unwanted props/extra people)

For motion_readability: score ~7–8 unless the pose is completely unreadable.

Rules:
- overall_score < threshold → RETAKE
- Multiple people, text, logos, watermarks → RETAKE
- Face unreadable when the pose should show the face → RETAKE
- Wrong outfit / hair / species vs identity lock → RETAKE
- Style that fights the style bible (e.g. photoreal when animated) → RETAKE
- Plain studio / clean background preferred; busy scenes that bury identity → RETAKE
Always provide retake_instructions and revised_prompt for the next sheet still.
Return ONLY JSON."""


class CriticAgent:
    def __init__(self, settings: Settings, log: LogFn | None = None):
        self.settings = settings
        self.log = log
        self.llm = LLMRouter(settings, log=log)
        self.last_provider = ""
        self.last_usage: dict[str, Any] | None = None

    def _stamp_review(self, review: CriticReview, result: Any) -> CriticReview:
        self.last_provider = getattr(result, "provider", "") or ""
        self.last_usage = getattr(result, "usage", None)
        review.provider = self.last_provider
        if self.last_usage:
            review.usage = dict(self.last_usage)
        return review

    def review(
        self,
        plan: ProductionPlan,
        shot: ShotPlan,
        frame_paths: list[Path],
        take: int,
        video_meta: dict[str, Any] | None = None,
        render_prompt_used: str = "",
    ) -> CriticReview:
        existing = [fp for fp in frame_paths if fp.exists()]
        if not existing:
            return CriticReview(
                shot_id=shot.id,
                take=take,
                verdict=CriticVerdict.retake,
                overall_score=0,
                youtube_ready=False,
                issues=["No review frames available"],
                retake_instructions="Regenerate; ensure frame extraction works.",
                summary="Cannot review without frames.",
            )

        on_names: list[str] = []
        ban_names: list[str] = []
        by_id = {c.id: c for c in (plan.characters or [])}
        for cid in shot.ref_character_ids or []:
            if cid in by_id:
                on_names.append(by_id[cid].name)
        for c in plan.characters or []:
            if c.id not in set(shot.ref_character_ids or []):
                ban_names.append(c.name)

        allowed = ", ".join(on_names) if on_names else "NONE (prop/environment only — no living cast)"
        banned = ", ".join(ban_names) if ban_names else "(none)"

        meta = video_meta or {}
        is_preclip = str(meta.get("phase") or "") == "preclip_still"
        phase_label = (
            "PRE-CLIP KEYFRAME (storyboard still — not final video)"
            if is_preclip
            else "generated take for YouTube release quality"
        )

        brief = f"""Review this {phase_label}.

TITLE: {plan.title}
STYLE BIBLE: {plan.style_bible}
CHARACTER LOCK (production-wide — identity only; not everyone must appear): {plan.character_lock}

SHOT: {shot.id} — {shot.name}
BEAT: {shot.beat}
CHARACTER PRESENCE RULES: {shot.character_presence}

CAST CHECKLIST FOR THIS SHOT (use the stills; this is mandatory QA):
- ALLOWED ON SCREEN: {allowed}
- BANNED ON SCREEN: {banned}
If any BANNED character appears (foreground, background, silhouette, window, crowd, substitute figure):
verdict MUST be RETAKE, character_fidelity ≤ 3, and retake_instructions + revised_prompt must order their complete removal.

INTENDED CAMERA: {shot.camera}
INTENDED VISUAL PROMPT:
{shot.visual_prompt}

RENDER PROMPT USED (truncated ok):
{render_prompt_used[:2500]}

TECHNICAL META:
{json.dumps(meta, indent=2)}

TAKE NUMBER: {take}
{"PRECLIP: focus on composition, cast, style, story beat. Soft on pure motion." if is_preclip else ""}

JSON schema:
{{
  "shot_id": "{shot.id}",
  "take": {take},
  "verdict": "PASS" | "RETAKE" | "REJECT",
  "overall_score": 0-10 number,
  "youtube_ready": boolean,
  "scores": {{
    "composition": 0-10,
    "character_fidelity": 0-10,
    "style_consistency": 0-10,
    "motion_readability": 0-10,
    "story_clarity": 0-10,
    "youtube_polish": 0-10
  }},
  "strengths": ["..."],
  "issues": ["critical problems"],
  "retake_instructions": "mandatory fixes for next take",
  "revised_prompt": "improved scene action prompt for retake",
  "summary": "2-3 sentences in a harsh critic voice"
}}
"""

        if is_preclip:
            system = CRITIC_SYSTEM_PRECLIP_STILL
        else:
            system = _critic_system(getattr(plan, "narrative_mode", None) or "character")

        result = self.llm.critic_review_payload(
            system=system,
            user=brief,
            images=existing,
            offline_kwargs={
                "shot_id": shot.id,
                "take": take,
                "frame_paths": existing,
                "shot_visual": shot.visual_prompt,
                "allowed_names": on_names,
                "banned_names": ban_names,
            },
        )
        data = result.data

        v = str(data.get("verdict", "RETAKE")).upper()
        if v == "PASS":
            data["verdict"] = CriticVerdict.pass_
        elif v == "REJECT":
            data["verdict"] = CriticVerdict.reject
        else:
            data["verdict"] = CriticVerdict.retake
        data["shot_id"] = shot.id
        data["take"] = take

        review = self._stamp_review(CriticReview.model_validate(data), result)

        # Enforce harsh threshold locally (preclip can use a slightly lower bar)
        threshold = (
            float(self.settings.preclip_critic_threshold)
            if is_preclip
            else float(self.settings.critic_pass_threshold)
        )
        if review.overall_score < threshold:
            if review.verdict == CriticVerdict.pass_:
                review.verdict = CriticVerdict.retake
                review.youtube_ready = False
                review.issues = list(review.issues) + [
                    f"Score {review.overall_score} below threshold {threshold}"
                ]
        if is_preclip:
            # youtube_ready only applies to finished clips
            review.youtube_ready = False
        elif review.verdict == CriticVerdict.pass_ and review.overall_score < 8.0:
            review.youtube_ready = False

        return review

    def review_character_pose(
        self,
        plan: ProductionPlan,
        character: CharacterDesign,
        pose: CharacterSheetPose,
        image_path: Path,
        *,
        take: int = 1,
        render_prompt_used: str = "",
        peer_images: list[Path] | None = None,
    ) -> CriticReview:
        """QA a single character-sheet still (optionally with peer views for identity lock)."""
        paths = [Path(image_path)]
        for p in peer_images or []:
            if p and Path(p).exists() and Path(p) not in paths:
                paths.append(Path(p))
        existing = [p for p in paths if p.exists()]
        side = f"{character.id}_{pose.pose_id}"
        if not existing:
            return CriticReview(
                shot_id=side,
                take=take,
                verdict=CriticVerdict.retake,
                overall_score=0,
                youtube_ready=False,
                issues=["No sheet still to review"],
                retake_instructions="Regenerate this character sheet pose still.",
                summary="Cannot review character sheet without an image.",
            )

        pose_label = pose.label or pose.pose_id
        peer_note = ""
        if len(existing) > 1:
            peer_note = (
                "\nIDENTITY CHECK: Additional images are OTHER views of the SAME character. "
                "They must match face, hair, body, and outfit. If they diverge, RETAKE and "
                "describe how to unify identity."
            )

        brief = f"""Review this CHARACTER SHEET still for identity-lock quality.

PRODUCTION: {plan.title}
STYLE BIBLE: {plan.style_bible}
CHARACTER LOCK: {plan.character_lock}

SHEET SUBJECT:
- id: {character.id}
- name: {character.name}
- look: {character.look or character.board_prompt or "(see lock)"}
- pose: {pose.pose_id} ({pose_label})
- pose intent: {pose.prompt or pose_label}

Image 1 is the pose under review.{peer_note}

PROMPT USED (truncated):
{(render_prompt_used or "")[:2000]}

TAKE: {take}

JSON schema:
{{
  "shot_id": "{side}",
  "take": {take},
  "verdict": "PASS" | "RETAKE" | "REJECT",
  "overall_score": 0-10 number,
  "youtube_ready": false,
  "scores": {{
    "composition": 0-10,
    "character_fidelity": 0-10,
    "style_consistency": 0-10,
    "motion_readability": 0-10,
    "story_clarity": 0-10,
    "youtube_polish": 0-10
  }},
  "strengths": ["..."],
  "issues": ["critical problems"],
  "retake_instructions": "mandatory fixes for next sheet still",
  "revised_prompt": "improved character sheet still prompt for this pose",
  "summary": "2-3 sentences as art director"
}}
"""
        result = self.llm.critic_review_payload(
            system=CRITIC_SYSTEM_CHARACTER_SHEET,
            user=brief,
            images=existing,
            offline_kwargs={
                "shot_id": side,
                "take": take,
                "frame_paths": existing,
                "shot_visual": pose.prompt or character.board_prompt or character.look,
                "allowed_names": [character.name],
                "banned_names": [
                    c.name
                    for c in (plan.characters or [])
                    if c.id != character.id and c.name
                ],
            },
        )
        data = result.data
        v = str(data.get("verdict", "RETAKE")).upper()
        if v == "PASS":
            data["verdict"] = CriticVerdict.pass_
        elif v == "REJECT":
            data["verdict"] = CriticVerdict.reject
        else:
            data["verdict"] = CriticVerdict.retake
        data["shot_id"] = side
        data["take"] = take
        data["youtube_ready"] = False

        review = self._stamp_review(CriticReview.model_validate(data), result)
        threshold = float(self.settings.character_sheet_critic_threshold)
        if review.overall_score < threshold and review.verdict == CriticVerdict.pass_:
            review.verdict = CriticVerdict.retake
            review.issues = list(review.issues) + [
                f"Score {review.overall_score} below sheet threshold {threshold}"
            ]
        review.youtube_ready = False
        return review
