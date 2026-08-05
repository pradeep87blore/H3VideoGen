"""Deterministic offline director plan + frame heuristic critic (no network)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_CAMERAS = [
    "wide establishing, slow push-in",
    "medium tracking shot, eye level",
    "close-up, shallow depth of field",
    "low-angle hero framing, steady",
    "orbiting medium, cinematic drift",
    "over-the-shoulder medium, subtle handheld",
]

_BEATS = [
    "Hook — grab attention in under three seconds",
    "Setup — introduce world and stakes",
    "Build — deepen conflict or wonder",
    "Turn — unexpected shift or reveal",
    "Peak — emotional / visual climax",
    "Payoff — resolve or land the joke",
    "Button — memorable last beat for retention",
]


def _title_from_prompt(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", prompt.strip())
    if not cleaned:
        return "Untitled Short"
    words = cleaned.split()
    snippet = " ".join(words[:8])
    if len(snippet) > 48:
        snippet = snippet[:45].rstrip() + "…"
    return snippet[:1].upper() + snippet[1:]


def director_plan_json(
    user_prompt: str,
    style: str,
    target_duration_sec: float,
    shot_count: int,
    per_shot: float,
) -> dict[str, Any]:
    """Produce a valid production plan without an LLM."""
    title = _title_from_prompt(user_prompt)
    character_hint = user_prompt.strip()[:220] or "main character from the story prompt"
    shots: list[dict[str, Any]] = []
    for i in range(shot_count):
        beat = _BEATS[i % len(_BEATS)]
        camera = _CAMERAS[i % len(_CAMERAS)]
        sid = f"S{i+1:02d}"
        shots.append(
            {
                "id": sid,
                "name": f"Beat {i+1}",
                "beat": beat,
                "duration_sec": round(per_shot, 1),
                "visual_prompt": (
                    f"{user_prompt.strip()}. Beat: {beat}. "
                    f"Single continuous take, action reads clearly for YouTube. "
                    f"Keep subjects readable; strong silhouette; cinematic motion."
                ),
                "camera": camera,
                "audio_notes": "Diegetic ambience matching scene; light score swell if needed",
                "character_presence": (
                    f"Only characters implied by the story. Focus: {character_hint[:120]}"
                ),
                "ref_character_ids": ["C01"],
            }
        )

    return {
        "title": title,
        "logline": user_prompt.strip()[:280],
        "target_duration_sec": target_duration_sec,
        "aspect_ratio": "16:9",
        "style_bible": (
            style.strip()
            or "Stylized cinematic animation, coherent art direction, no photoreal/cartoon mix"
        ),
        "character_lock": (
            f"Hold consistent character design, wardrobe, and color identity across shots. "
            f"Story subject: {character_hint}"
        ),
        "characters": [
            {
                "id": "C01",
                "name": "Main Character",
                "look": character_hint,
                "board_prompt": (
                    f"Character design portrait of the main character from: {character_hint}. "
                    f"Style: {style}. Full body, clear face, plain studio background, no text."
                ),
            }
        ],
        "color_grade": "Cohesive grade matching the style bible; avoid random palette drift",
        "audio_bed": "Light ambient bed + diegetic SFX; no spoken dialogue unless inherent SFX",
        "youtube_notes": "Hook in S01; vary shot size; keep motion readable; no on-screen text",
        "raw_director_notes": "Offline template plan (no LLM). Refine prompts on retakes if needed.",
        "shots": shots,
    }


def _frame_stats(path: Path) -> dict[str, float]:
    from PIL import Image, ImageStat

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        stat = ImageStat.Stat(im)
        # mean of R,G,B
        mean = sum(stat.mean) / 3.0
        # simple contrast proxy via stdev average
        stdev = sum(stat.stdev) / 3.0 if stat.stdev else 0.0
        return {"w": float(w), "h": float(h), "mean": float(mean), "stdev": float(stdev)}


def critic_review_json(
    shot_id: str,
    take: int,
    frame_paths: list[Path],
    shot_visual: str,
) -> dict[str, Any]:
    """Heuristic review of extracted frames when no vision LLM is available."""
    if not frame_paths:
        return {
            "shot_id": shot_id,
            "take": take,
            "verdict": "RETAKE",
            "overall_score": 0.0,
            "youtube_ready": False,
            "scores": {
                "composition": 0,
                "character_fidelity": 0,
                "style_consistency": 0,
                "motion_readability": 0,
                "story_clarity": 0,
                "youtube_polish": 0,
            },
            "strengths": [],
            "issues": ["No review frames available"],
            "retake_instructions": "Regenerate; ensure frame extraction works.",
            "revised_prompt": shot_visual,
            "summary": "Offline critic: cannot review without frames.",
        }

    issues: list[str] = []
    strengths: list[str] = []
    stats_list: list[dict[str, float]] = []
    for fp in frame_paths:
        try:
            stats_list.append(_frame_stats(fp))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Unreadable frame {fp.name}: {exc}")

    if not stats_list:
        return {
            "shot_id": shot_id,
            "take": take,
            "verdict": "RETAKE",
            "overall_score": 1.0,
            "youtube_ready": False,
            "scores": {
                "composition": 1,
                "character_fidelity": 1,
                "style_consistency": 1,
                "motion_readability": 1,
                "story_clarity": 1,
                "youtube_polish": 1,
            },
            "strengths": [],
            "issues": issues or ["Frames unreadable"],
            "retake_instructions": "Regenerate clip; verify output is a valid video.",
            "revised_prompt": shot_visual,
            "summary": "Offline critic: frames could not be opened.",
        }

    mean_lum = sum(s["mean"] for s in stats_list) / len(stats_list)
    mean_std = sum(s["stdev"] for s in stats_list) / len(stats_list)
    min_side = min(min(s["w"], s["h"]) for s in stats_list)

    # Composition proxies
    composition = 7.0
    polish = 7.0
    if mean_lum < 18:
        issues.append("Frame is extremely dark (possible generation failure)")
        composition -= 2.5
        polish -= 2.0
    elif mean_lum > 245:
        issues.append("Frame is blown-out / near white")
        composition -= 2.0
        polish -= 1.5
    else:
        strengths.append("Luminance is in a usable range")

    if mean_std < 8:
        issues.append("Very low contrast — looks flat or blank")
        composition -= 2.0
        polish -= 2.0
    else:
        strengths.append("Some contrast / texture present")

    if min_side < 256:
        issues.append("Suspiciously small frame resolution")
        polish -= 1.5

    character = 7.0
    style_c = 7.0
    motion = 6.5
    story = 6.5

    overall = (composition + character + style_c + motion + story + polish) / 6.0
    overall = max(0.0, min(10.0, round(overall, 2)))

    if issues:
        verdict = "RETAKE"
        youtube_ready = False
        retake = (
            "Increase subject clarity and lighting; strengthen composition; "
            "avoid empty/blank frames; keep style consistent with bible."
        )
        summary = "Offline heuristic critic found quality concerns: " + "; ".join(issues[:3])
    else:
        # Slightly soft pass path when only crude image stats are good.
        verdict = "PASS" if overall >= 7.0 else "RETAKE"
        youtube_ready = overall >= 8.0
        retake = (
            "Push clearer action and stronger silhouette; "
            "tighten character continuity with the lock."
        )
        summary = (
            "Offline heuristic critic: frames look generically usable "
            "(no vision model — treat scores as approximate)."
        )
        if not strengths:
            strengths.append("Frames present and not obviously failed")

    revised = shot_visual.strip()
    if issues:
        revised = (
            f"{shot_visual.strip()}\n"
            "Fix offline issues: brighter readable subject, better contrast, "
            "clear silhouette, intentional composition."
        )

    return {
        "shot_id": shot_id,
        "take": take,
        "verdict": verdict,
        "overall_score": overall,
        "youtube_ready": youtube_ready,
        "scores": {
            "composition": round(composition, 1),
            "character_fidelity": round(character, 1),
            "style_consistency": round(style_c, 1),
            "motion_readability": round(motion, 1),
            "story_clarity": round(story, 1),
            "youtube_polish": round(polish, 1),
        },
        "strengths": strengths,
        "issues": issues,
        "retake_instructions": retake,
        "revised_prompt": revised,
        "summary": summary,
    }


# Convenience for debugging / router symmetry
def dump(obj: dict[str, Any]) -> str:
    return json.dumps(obj)
