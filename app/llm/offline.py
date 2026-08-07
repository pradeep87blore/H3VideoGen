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
    narrative_mode: str = "character",
) -> dict[str, Any]:
    """Produce a valid production plan without an LLM."""
    mode = (narrative_mode or "character").lower()
    title = _title_from_prompt(user_prompt)
    character_hint = user_prompt.strip()[:220] or "main subject from the prompt"
    shots: list[dict[str, Any]] = []
    for i in range(shot_count):
        beat = _BEATS[i % len(_BEATS)]
        camera = _CAMERAS[i % len(_CAMERAS)]
        sid = f"S{i+1:02d}"
        if mode == "explainer":
            narr = f"In simple terms: {beat.lower()}. {character_hint[:80]}."
            vis = (
                f"Educational visual metaphor for: {user_prompt.strip()}. Beat: {beat}. "
                f"Clear readable subject, one idea, no on-screen text, cinematic motion."
            )
            presence = "Metaphor objects / environments only; no invented main character cast"
            refs: list[str] = []
        elif mode == "documentary":
            narr = f"{beat}. {character_hint[:100]}."
            vis = (
                f"Documentary continuous take: {user_prompt.strip()}. Beat: {beat}. "
                f"Historical scale, era materials, no on-screen text."
            )
            presence = "Places, machines, or mass figures; no fictional star cast"
            refs = []
        else:
            narr = f"{beat}."
            vis = (
                f"{user_prompt.strip()}. Beat: {beat}. "
                f"Single continuous take, action reads clearly for YouTube. "
                f"Keep subjects readable; strong silhouette; cinematic motion."
            )
            presence = f"Only characters implied by the story. Focus: {character_hint[:120]}"
            refs = ["C01"]
        shots.append(
            {
                "id": sid,
                "name": f"Beat {i+1}",
                "beat": beat,
                "duration_sec": round(per_shot, 1),
                "visual_prompt": vis,
                "camera": camera,
                "audio_notes": "Diegetic ambience matching scene",
                "character_presence": presence,
                "ref_character_ids": refs,
                "narration_line": narr[:220],
            }
        )
    chars: list[dict[str, Any]] = []
    if mode == "character":
        char_lock = (
            f"Hold consistent character design. Story subject: {character_hint}"
        )
        chars = [
            {
                "id": "C01",
                "name": "Main Character",
                "look": character_hint,
                "board_prompt": (
                    f"Character design portrait: {character_hint}. Style: {style}. "
                    f"Full body, clear face, plain studio background, no text."
                ),
            }
        ]
    else:
        char_lock = "No locked cast — continuity of era / metaphor / palette."
    joined = " ".join(
        (s.get("narration_line") or "").rstrip(".") + "."
        for s in shots
        if s.get("narration_line")
    )
    return {
        "title": title,
        "logline": user_prompt.strip()[:280],
        "target_duration_sec": target_duration_sec,
        "aspect_ratio": "16:9",
        "narrative_mode": mode,
        "style_bible": style.strip()
        or "Stylized cinematic look, coherent art direction",
        "character_lock": char_lock,
        "characters": chars,
        "color_grade": "Cohesive grade matching the style bible",
        "audio_bed": "Ambient bed; narrator is separate VO",
        "youtube_notes": "Hook in S01; keep idea clear; no on-screen text",
        "narration_script": joined,
        "raw_director_notes": f"Offline template plan (mode={mode}).",
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
        # Pillow exposes stddev (list of channel stdevs); older docs also say stdev
        std_list = getattr(stat, "stddev", None) or getattr(stat, "stdev", None) or []
        stdev = sum(std_list) / 3.0 if std_list else 0.0
        return {"w": float(w), "h": float(h), "mean": float(mean), "stdev": float(stdev)}


def critic_review_json(
    shot_id: str,
    take: int,
    frame_paths: list[Path],
    shot_visual: str,
    allowed_names: list[str] | None = None,
    banned_names: list[str] | None = None,
) -> dict[str, Any]:
    """Heuristic review of extracted frames when no vision LLM is available.

    Cannot verify cast exclusivity from pixels — when banned cast is configured,
    score character_fidelity low so we do not fake-PASS leakages.
    """
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

    # Composition proxies (baseline high enough that usable frames can clear
    # CRITIC_PASS_THRESHOLD when offline — vision models still override this path)
    composition = 7.6
    polish = 7.6
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

    style_c = 7.6
    motion = 7.5
    story = 7.5

    bans = [n for n in (banned_names or []) if str(n).strip()]
    allows = [n for n in (allowed_names or []) if str(n).strip()]
    # Pixel heuristics cannot verify cast exclusivity — fail closed when bans apply.
    if bans:
        character = 4.0
        issues.append(
            "Offline critic cannot verify cast exclusivity for banned: "
            + ", ".join(bans[:8])
            + " — will not fake-PASS (use LOCAL_LLM_VISION_MODEL=llava)"
        )
    else:
        character = 7.4
        if allows:
            strengths.append(
                f"Cast exclusivity not applied offline (allowed: {', '.join(allows[:4])})"
            )

    overall = (composition + character + style_c + motion + story + polish) / 6.0
    overall = max(0.0, min(10.0, round(overall, 2)))

    if issues:
        verdict = "RETAKE"
        youtube_ready = False
        retake = (
            "Increase subject clarity and lighting; strengthen composition; "
            "avoid empty/blank frames; keep style consistent with bible."
        )
        if bans:
            retake = (
                "Cannot offline-verify cast. Ensure render only shows: "
                + (", ".join(allows) if allows else "no living cast")
                + ". Completely remove banned: "
                + ", ".join(bans)
                + ". Prefer enabling LOCAL_LLM_VISION_MODEL=llava for real QA."
            )
        summary = "Offline heuristic critic found quality concerns: " + "; ".join(issues[:3])
    else:
        # Offline cannot truly QA character identity — pass when stats look sane.
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
