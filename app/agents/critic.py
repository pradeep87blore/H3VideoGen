"""Harsh YouTube critic for generated video frames (Gemini with local/offline fallback)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from ..llm import LLMRouter
from ..models import CriticReview, CriticVerdict, ProductionPlan, ShotPlan

LogFn = Callable[[str], None]

CRITIC_SYSTEM = """You are a HARSH film critic and YouTube content QA lead.
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
- wrong characters present → RETAKE
- photoreal vs style mismatch when style is animated → RETAKE
- boring empty frame with no story → RETAKE
- If pass, youtube_ready=true only if overall_score >= 8.0 AND no critical issues

Always provide revised_prompt: an improved visual-only prompt body for a retake
(not full style bible—just the action/scene improvements).

Return ONLY JSON."""


class CriticAgent:
    def __init__(self, settings: Settings, log: LogFn | None = None):
        self.settings = settings
        self.log = log
        self.llm = LLMRouter(settings, log=log)
        self.last_provider = ""

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

        brief = f"""Review this generated take for YouTube release quality.

TITLE: {plan.title}
STYLE BIBLE: {plan.style_bible}
CHARACTER LOCK: {plan.character_lock}
USER/STYLE CONTEXT already baked into style bible.

SHOT: {shot.id} — {shot.name}
BEAT: {shot.beat}
CHARACTER PRESENCE RULES: {shot.character_presence}
INTENDED CAMERA: {shot.camera}
INTENDED VISUAL PROMPT:
{shot.visual_prompt}

RENDER PROMPT USED (truncated ok):
{render_prompt_used[:2500]}

TECHNICAL META:
{json.dumps(video_meta or {}, indent=2)}

TAKE NUMBER: {take}

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

        result = self.llm.critic_review_payload(
            system=CRITIC_SYSTEM,
            user=brief,
            images=existing,
            offline_kwargs={
                "shot_id": shot.id,
                "take": take,
                "frame_paths": existing,
                "shot_visual": shot.visual_prompt,
            },
        )
        self.last_provider = result.provider
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

        review = CriticReview.model_validate(data)

        # Enforce harsh threshold locally
        if review.overall_score < self.settings.critic_pass_threshold:
            if review.verdict == CriticVerdict.pass_:
                review.verdict = CriticVerdict.retake
                review.youtube_ready = False
                review.issues = list(review.issues) + [
                    f"Score {review.overall_score} below threshold {self.settings.critic_pass_threshold}"
                ]
        if review.verdict == CriticVerdict.pass_ and review.overall_score < 8.0:
            review.youtube_ready = False

        return review
