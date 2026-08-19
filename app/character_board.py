"""Multi-view character sheets for MiniMax H3 R2V (<Picture n> tags, ≤9 images)."""
from __future__ import annotations

import base64
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .config import Settings
from .job_control import CancelledError, job_control
from .models import CharacterDesign, CharacterSheetPose, ProductionPlan, ShotPlan

if TYPE_CHECKING:
    from .comfy_h3 import ComfyH3Client

LogFn = Callable[[str], None]

# Official H3 R2V max reference images
R2V_MAX_IMAGES = 9

# Default multi-view pack for lead characters
PRIMARY_POSES: list[tuple[str, str, str]] = [
    (
        "front_full",
        "front full-body",
        "Full-body standing portrait, facing camera, arms relaxed, clear face and outfit.",
    ),
    (
        "three_quarter",
        "three-quarter",
        "Three-quarter view standing, face clearly visible, same outfit and proportions.",
    ),
    (
        "side",
        "side profile",
        "Straight side profile full-body, same face silhouette hair and outfit.",
    ),
    (
        "closeup_face",
        "face close-up",
        "Tight head-and-shoulders close-up, face fills frame, same features lighting neutral.",
    ),
    (
        "action",
        "action pose",
        "Dynamic but readable action pose mid-movement, same character design locked.",
    ),
    (
        "back",
        "back three-quarter",
        "Back three-quarter view showing hair and outfit rear details, head slightly turned.",
    ),
]

SECONDARY_POSES = PRIMARY_POSES[:2]  # front + three_quarter for supporting cast


def ensure_character_designs(plan: ProductionPlan, story_hint: str = "") -> list[CharacterDesign]:
    """Normalize plan.characters; invent from character_lock if director omitted them.

    Reorders so the story protagonist is C01 (exclusivity / sheets treat C01 as lead).
    """
    chars = list(plan.characters or [])
    if chars:
        for i, c in enumerate(chars):
            if not c.id:
                c.id = f"C{i+1:02d}"
            if not c.board_prompt:
                c.board_prompt = (
                    f"Character design portrait of {c.name}. {c.look or plan.character_lock}. "
                    f"Style: {plan.style_bible}. Neutral standing pose, full body preferred, "
                    f"clear face, plain studio background, single character only, no text."
                )
        chars = _prioritize_lead_character(plan, chars, story_hint)
        for i, c in enumerate(chars):
            c.id = f"C{i+1:02d}"
            _ensure_sheet_slots(c, primary=(i == 0))
        plan.characters = chars[:6]
        return plan.characters

    lock = plan.character_lock or plan.logline or plan.title
    names = _guess_names(lock) or ["Main Character"]
    chars = []
    for i, name in enumerate(names[:4]):
        c = CharacterDesign(
            id=f"C{i+1:02d}",
            name=name,
            look=lock[:400],
            board_prompt=(
                f"Character design portrait of {name}. {lock[:500]}. "
                f"Style: {plan.style_bible}. Neutral standing pose, full body preferred, "
                f"clear face, plain studio background, single character only, no text."
            ),
        )
        chars.append(c)
    chars = _prioritize_lead_character(plan, chars, story_hint)
    for i, c in enumerate(chars):
        c.id = f"C{i+1:02d}"
        _ensure_sheet_slots(c, primary=(i == 0))
    plan.characters = chars
    return chars


def _prioritize_lead_character(
    plan: ProductionPlan,
    chars: list[CharacterDesign],
    story_hint: str = "",
) -> list[CharacterDesign]:
    """Put the named protagonist first (e.g. Goldilocks before the bears)."""
    if len(chars) < 2:
        return chars
    blob = " ".join(
        [
            story_hint or "",
            plan.title or "",
            plan.logline or "",
            plan.raw_director_notes or "",
        ]
    ).lower()

    def score(c: CharacterDesign) -> float:
        n = (c.name or "").lower().strip()
        if not n:
            return -1.0
        s = 0.0
        if n in blob:
            idx = blob.find(n)
            s += 50 + max(0, 30 - min(idx, 30))
        for hero in (
            "goldilocks",
            "girl",
            "boy",
            "child",
            "kid",
            "hero",
            "heroine",
            "princess",
            "prince",
            "main",
            "protagonist",
        ):
            if hero in n:
                s += 40
        for dem in ("papa", "mama", "baby", "father", "mother", "brother", "sister"):
            if dem in n and ("goldilocks" in blob or "girl" in blob or "child" in blob):
                s -= 25
        if "bear" in n and "goldilocks" in blob:
            s -= 15
        return s

    ranked = sorted(enumerate(chars), key=lambda iv: (-score(iv[1]), iv[0]))
    if score(ranked[0][1]) > score(chars[0]) + 5:
        return [c for _, c in ranked]
    return chars


def _ensure_sheet_slots(char: CharacterDesign, *, primary: bool) -> None:
    if char.sheet:
        return
    poses = PRIMARY_POSES if primary else SECONDARY_POSES
    char.sheet = [
        CharacterSheetPose(pose_id=pid, label=label, prompt=suffix)
        for pid, label, suffix in poses
    ]


def _guess_names(text: str) -> list[str]:
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text or "")
    skip = {"The", "A", "An", "Style", "Character", "Papa", "Mama", "Baby"}
    out: list[str] = []
    for c in candidates:
        if c in skip or len(c) < 3:
            continue
        if c not in out:
            out.append(c)
    return out[:4]


class CharacterBoardBuilder:
    def __init__(
        self,
        settings: Settings,
        board_dir: Path,
        log: LogFn | None = None,
        comfy: "ComfyH3Client | None" = None,
        on_character_done: Callable[[CharacterDesign], None] | None = None,
    ):
        self.settings = settings
        self.board_dir = Path(board_dir)
        self.board_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.comfy = comfy
        self.on_character_done = on_character_done

    def _emit(self, msg: str) -> None:
        if self.log:
            self.log(msg)

    def _notify_char_done(self, c: CharacterDesign) -> None:
        if self.on_character_done:
            try:
                self.on_character_done(c)
            except Exception:
                pass

    def build(self, plan: ProductionPlan) -> list[CharacterDesign]:
        """Build multi-view character sheets (Gemini stills and/or H3 turnaround)."""
        from datetime import datetime, timezone

        def _iso() -> str:
            return datetime.now(timezone.utc).isoformat()

        def _elapsed(start: str | None, end: str | None) -> float | None:
            if not start:
                return None
            try:
                a = datetime.fromisoformat(start.replace("Z", "+00:00"))
                b = (
                    datetime.fromisoformat(end.replace("Z", "+00:00"))
                    if end
                    else datetime.now(timezone.utc)
                )
                return round(max(0.0, (b - a).total_seconds()), 2)
            except Exception:
                return None

        job_control.check()
        chars = ensure_character_designs(plan)
        mode = (self.settings.character_board_mode or "auto").lower()
        max_primary = max(1, min(6, self.settings.character_sheet_poses_primary))
        max_secondary = max(0, min(4, self.settings.character_sheet_poses_secondary))

        # Clamp sheet slots by config
        for i, c in enumerate(chars):
            limit = max_primary if i == 0 else max_secondary
            if not c.sheet:
                _ensure_sheet_slots(c, primary=(i == 0))
            c.sheet = c.sheet[:limit]

        self._adopt_manual_files(chars)

        for c in chars:
            already = self._primary_path(c)
            if already and not self._missing_poses(c):
                # Fully on disk already (resume / manual)
                c.sheet_status = "ready"
                if c.sheet_duration_sec is None and c.sheet_started_at and c.sheet_finished_at:
                    c.sheet_duration_sec = _elapsed(c.sheet_started_at, c.sheet_finished_at)
                if not c.sheet_source:
                    c.sheet_source = "manual" if already else "none"
                if c.sheet_finished_at is None and c.sheet_duration_sec is None:
                    # Resume with existing assets — don't invent a long duration
                    c.sheet_duration_sec = 0.0
                    if not c.sheet_started_at:
                        c.sheet_started_at = _iso()
                    c.sheet_finished_at = c.sheet_started_at
                self._notify_char_done(c)
                continue

            c.sheet_started_at = c.sheet_started_at or _iso()
            c.sheet_status = "building"
            c.sheet_finished_at = None
            used_sources: list[str] = []
            if any(p.image_path for p in c.sheet):
                used_sources.append("manual")

            if mode in ("auto", "gemini"):
                gemini_n = self._fill_gemini_for_character(c, plan)
                if gemini_n:
                    used_sources.append("gemini")

            if (
                mode in ("auto", "h3")
                and self.comfy
                and self.settings.character_sheet_use_h3
                and self._missing_poses(c)
            ):
                try:
                    h3_n = self._fill_h3_turnaround(c, plan)
                    if h3_n:
                        used_sources.append("h3")
                except Exception as exp:  # noqa: BLE001
                    self._emit(f"Character sheet H3 turnaround failed for {c.name}: {exp}")

            # Critic QA + optional Gemini retakes for each pose still
            if (
                self.settings.character_sheet_critic_enabled
                and any(p.image_path for p in c.sheet)
            ):
                fixed = self._critic_review_character_sheet(c, plan)
                if fixed:
                    used_sources.append("critic")

            self._sync_primary_image([c])
            primary = self._primary_path(c)
            c.sheet_finished_at = _iso()
            c.sheet_duration_sec = _elapsed(c.sheet_started_at, c.sheet_finished_at)
            if not used_sources:
                used_sources = ["none"]
            c.sheet_source = (
                "mixed"
                if len(set(used_sources)) > 1
                else used_sources[0]
            )
            poses_ready = sum(
                1 for p in c.sheet if p.image_path and Path(p.image_path).exists()
            )
            if primary:
                c.sheet_status = "ready"
                self._emit(
                    f"Character sheet finalized: {c.name} ({c.id}) in "
                    f"{c.sheet_duration_sec:.1f}s — {poses_ready} view(s), source={c.sheet_source}"
                )
            else:
                c.sheet_status = "failed"
                self._emit(
                    f"Character sheet incomplete: {c.name} ({c.id}) after "
                    f"{c.sheet_duration_sec or 0:.1f}s — no primary still"
                )
            self._notify_char_done(c)

        self._sync_primary_image(chars)
        self._write_manifest(plan, chars)
        plan.characters = chars

        total_poses = sum(
            1
            for c in chars
            for p in c.sheet
            if p.image_path and Path(p.image_path).exists()
        )
        chars_ready = sum(1 for c in chars if self._primary_path(c))
        self._emit(
            f"Character sheet ready: {chars_ready}/{len(chars)} cast, "
            f"{total_poses} view stills "
            f"(missing views bootstrap from scene frames if needed)"
        )
        return chars

    def _missing_poses(self, char: CharacterDesign) -> list[CharacterSheetPose]:
        return [p for p in char.sheet if not (p.image_path and Path(p.image_path).exists())]

    def _primary_path(self, char: CharacterDesign) -> Path | None:
        if char.image_path and Path(char.image_path).exists():
            return Path(char.image_path)
        for prefer in ("front_full", "three_quarter", "closeup_face"):
            for p in char.sheet:
                if p.pose_id == prefer and p.image_path and Path(p.image_path).exists():
                    return Path(p.image_path)
        for p in char.sheet:
            if p.image_path and Path(p.image_path).exists():
                return Path(p.image_path)
        return None

    def _sync_primary_image(self, chars: list[CharacterDesign]) -> None:
        for c in chars:
            primary = self._primary_path(c)
            c.image_path = str(primary) if primary else None
            c.picture_index = None  # selection is dynamic per shot

    def _adopt_manual_files(self, chars: list[CharacterDesign]) -> None:
        for c in chars:
            for pose in c.sheet:
                if pose.image_path and Path(pose.image_path).exists():
                    continue
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    # C01_front_full.png, C01-front.png, name_front_full.png
                    candidates = [
                        self.board_dir / f"{c.id}_{pose.pose_id}{ext}",
                        self.board_dir / f"{c.id}-{pose.pose_id}{ext}",
                        self.board_dir / f"{_safe_name(c.name)}_{pose.pose_id}{ext}",
                    ]
                    for cand in candidates:
                        if cand.exists():
                            pose.image_path = str(cand)
                            break
                    if pose.image_path:
                        break
            # Legacy single file C01.png → front slot
            if not any(p.image_path for p in c.sheet):
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    legacy = self.board_dir / f"{c.id}{ext}"
                    if legacy.exists() and c.sheet:
                        c.sheet[0].image_path = str(legacy)
                        break

    def _fill_gemini_for_character(
        self, c: CharacterDesign, plan: ProductionPlan
    ) -> int:
        """Fill missing poses via Gemini. Returns count of new stills."""
        n = 0
        for pose in self._missing_poses(c):
            job_control.check()
            try:
                path = self._gemini_still(c, plan, pose)
                if path:
                    pose.image_path = str(path)
                    n += 1
                    self._emit(f"Character sheet: {c.name} / {pose.pose_id} → {path.name}")
            except CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._emit(f"Character sheet Gemini failed {c.name}/{pose.pose_id}: {exc}")
        return n

    def _critic_review_character_sheet(
        self, character: CharacterDesign, plan: ProductionPlan
    ) -> int:
        """
        Review each sheet pose; on RETAKE regenerate via Gemini (with notes).
        Returns number of retake regenerations performed.
        """
        from .agents.critic import CriticAgent
        from .models import CriticVerdict

        if not self.settings.character_sheet_critic_enabled:
            return 0

        critic = CriticAgent(self.settings, log=self.log)
        reviews_dir = self.board_dir / "sheet_reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        max_rt = int(self.settings.character_sheet_critic_max_retakes)
        require_pass = bool(self.settings.character_sheet_critic_require_pass)
        regenerations = 0

        self._emit(f"Character sheet critic: reviewing {character.name}…")

        for pose in character.sheet:
            job_control.check()
            if not pose.image_path or not Path(pose.image_path).exists():
                continue

            notes = ""
            best_path = Path(pose.image_path)
            best_score = -1.0
            passed = False

            for attempt in range(1, max_rt + 2):
                job_control.check()
                img = Path(pose.image_path) if pose.image_path else None
                if not img or not img.exists():
                    break

                render_used = self._sheet_still_prompt(character, plan, pose, notes)
                review = critic.review_character_pose(
                    plan,
                    character,
                    pose,
                    img,
                    take=attempt,
                    render_prompt_used=render_used,
                )
                (reviews_dir / f"{character.id}_{pose.pose_id}_t{attempt}.json").write_text(
                    review.model_dump_json(indent=2), encoding="utf-8"
                )
                self._emit(
                    f"Sheet critic ({critic.last_provider or '?'}): "
                    f"{character.name}/{pose.pose_id} attempt {attempt} "
                    f"→ {review.verdict.value} score={review.overall_score} — "
                    f"{(review.summary or '')[:120]}"
                )
                if review.usage:
                    from .llm.gemini_backend import format_usage_line

                    line = format_usage_line(review.usage)
                    if line:
                        self._emit(line)

                if (review.overall_score or 0) >= best_score:
                    best_score = review.overall_score or 0
                    best_path = img

                if review.verdict == CriticVerdict.pass_:
                    if review.revised_prompt:
                        pose.prompt = review.revised_prompt.strip() or pose.prompt
                    passed = True
                    break

                notes = (review.retake_instructions or review.summary or "").strip()
                if review.revised_prompt:
                    pose.prompt = review.revised_prompt.strip()

                if attempt > max_rt:
                    break

                # Retake: regenerate still with critic notes
                self._emit(
                    f"Character sheet RETAKE {character.name}/{pose.pose_id}: "
                    f"{notes[:160]}"
                )
                try:
                    new_path = self._gemini_still(
                        character, plan, pose, critic_notes=notes, attempt=attempt + 1
                    )
                    if new_path:
                        pose.image_path = str(new_path)
                        regenerations += 1
                    else:
                        self._emit(
                            f"Character sheet retake failed to produce image for "
                            f"{character.name}/{pose.pose_id}"
                        )
                        break
                except CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self._emit(
                        f"Character sheet retake error {character.name}/{pose.pose_id}: {exc}"
                    )
                    break

            if not passed:
                if best_path and best_path.exists():
                    pose.image_path = str(best_path)
                    self._emit(
                        f"Sheet critic: {character.name}/{pose.pose_id} kept best "
                        f"(score={best_score})"
                        + ("; require-pass ignored soft" if require_pass else "")
                    )
                if require_pass and best_score < float(
                    self.settings.character_sheet_critic_threshold
                ):
                    # Soft-soft: only drop if truly unusable empty; keep primary refs
                    self._emit(
                        f"Sheet critic: {character.name}/{pose.pose_id} below "
                        f"threshold after retakes (score={best_score})"
                    )

        # Optional multi-view identity check (front + closeup)
        if self.settings.character_sheet_critic_identity_check:
            regenerations += self._critic_identity_lock(character, plan, critic, reviews_dir)

        return regenerations

    def _critic_identity_lock(
        self,
        character: CharacterDesign,
        plan: ProductionPlan,
        critic: object,
        reviews_dir: Path,
    ) -> int:
        """Compare front + face views; if identity diverges, regen weaker pose once."""
        from .agents.critic import CriticAgent
        from .models import CriticVerdict

        if not isinstance(critic, CriticAgent):
            return 0

        by_id = {
            p.pose_id: p
            for p in character.sheet
            if p.image_path and Path(p.image_path).exists()
        }
        front = by_id.get("front_full") or by_id.get("three_quarter")
        face = by_id.get("closeup_face")
        if not front or not face or front is face:
            return 0

        job_control.check()
        self._emit(f"Character sheet identity lock: {character.name} front vs close-up…")
        review = critic.review_character_pose(
            plan,
            character,
            face,
            Path(face.image_path),  # type: ignore[arg-type]
            take=1,
            render_prompt_used=self._sheet_still_prompt(character, plan, face, ""),
            peer_images=[Path(front.image_path)],  # type: ignore[arg-type]
        )
        (reviews_dir / f"{character.id}_identity_t1.json").write_text(
            review.model_dump_json(indent=2), encoding="utf-8"
        )
        self._emit(
            f"Sheet identity ({critic.last_provider or '?'}): "
            f"{review.verdict.value} score={review.overall_score} — "
            f"{(review.summary or '')[:120]}"
        )
        if review.verdict == CriticVerdict.pass_:
            return 0

        notes = (review.retake_instructions or review.summary or "").strip()
        # Prefer fixing the lower-fidelity close-up unless instructions say otherwise
        target = face
        if review.revised_prompt:
            target.prompt = review.revised_prompt.strip()
        try:
            new_path = self._gemini_still(
                character, plan, target, critic_notes=notes, attempt=2
            )
            if new_path:
                target.image_path = str(new_path)
                self._emit(
                    f"Identity lock retake: regenerated {character.name}/{target.pose_id}"
                )
                return 1
        except CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._emit(f"Identity lock retake failed: {exc}")
        return 0

    def _sheet_still_prompt(
        self,
        character: CharacterDesign,
        plan: ProductionPlan,
        pose: CharacterSheetPose,
        critic_notes: str = "",
    ) -> str:
        pose_bit = pose.prompt or pose.label or pose.pose_id
        parts = [
            f"Character design sheet still of {character.name}.",
            f"Identity lock: {character.look or character.board_prompt or plan.character_lock}",
            f"View: {pose_bit}",
            f"Art style: {plan.style_bible}",
            "Single character only. Plain seamless studio background. Consistent face, hair, "
            "outfit, proportions across the character bible. No text, no logos, no watermarks.",
        ]
        if critic_notes:
            parts.append("MANDATORY FIXES FROM SHEET CRITIC:\n" + critic_notes.strip())
        return "\n".join(parts)

    def _gemini_still(
        self,
        character: CharacterDesign,
        plan: ProductionPlan,
        pose: CharacterSheetPose,
        critic_notes: str = "",
        attempt: int = 1,
    ) -> Path | None:
        if not self.settings.gemini_api_key:
            return None
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.settings.gemini_api_key)
        models = [m.strip() for m in self.settings.gemini_image_models if m.strip()]
        prompt = self._sheet_still_prompt(character, plan, pose, critic_notes)
        stem = f"{character.id}_{pose.pose_id}"
        if attempt > 1:
            stem = f"{stem}_r{attempt}"
        last_err: Exception | None = None
        for model in models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                    ),
                )
                path = self._save_gemini_images(response, stem)
                if path:
                    return path
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if last_err:
            raise last_err
        return None

    def _save_gemini_images(self, response: object, stem: str) -> Path | None:
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
                ext = ".png" if "png" in mime else ".jpg"
                out = self.board_dir / f"{stem}{ext}"
                out.write_bytes(raw)
                if out.stat().st_size > 500:
                    return out
        return None

    def _fill_gemini_sheets(self, chars: list[CharacterDesign], plan: ProductionPlan) -> None:
        for c in chars:
            self._fill_gemini_for_character(c, plan)

    def _fill_h3_turnaround(self, char: CharacterDesign, plan: ProductionPlan) -> int:
        """One short studio turnaround clip → extract frames into missing poses.

        Returns number of poses filled from the turnaround.
        """
        if not self.comfy:
            return 0
        job_control.check()
        missing = self._missing_poses(char)
        if not missing:
            return 0
        from .media import extract_frames

        seed = 9000 + sum(ord(ch) for ch in char.id)
        prompt = (
            f"{plan.style_bible}\n"
            f"CHARACTER DESIGN TURNTABLE of {char.name}. {char.look or plan.character_lock}\n"
            f"Neutral seamless studio backdrop, even soft lighting, single character only.\n"
            f"Slow orbit: full body front facing camera, then three-quarter, then side profile, "
            f"then slight turn toward back. Clear face when facing camera. No on-screen text, "
            f"no logos, no props, no crowd."
        )
        out_prefix = f"video/H3VideoGen/{self.board_dir.name}/sheet_{char.id}"
        self._emit(f"Character sheet: H3 turnaround for {char.name}…")
        video, _, _ = self.comfy.generate(
            prompt,
            length=self.settings.default_length_frames,
            seed=seed,
            filename_prefix=out_prefix,
            mode="t2v",
            project_tag=self.board_dir.name,
        )
        frames_dir = self.board_dir / f"_turn_{char.id}"
        times = [0.4, 1.2, 2.0, 2.8, 3.6, 4.4][: len(missing)]
        frames = extract_frames(self.settings, video, frames_dir, times=times)
        filled = 0
        for pose, frame in zip(missing, frames):
            dest = self.board_dir / f"{char.id}_{pose.pose_id}.jpg"
            shutil.copy2(frame, dest)
            pose.image_path = str(dest)
            filled += 1
            self._emit(f"Character sheet: {char.name} / {pose.pose_id} from H3 turn → {dest.name}")
        return filled

    def adopt_frame(
        self,
        character: CharacterDesign,
        frame: Path,
        pose_id: str = "front_full",
    ) -> Path | None:
        if not frame.exists():
            return None
        dest = self.board_dir / f"{character.id}_{pose_id}.jpg"
        shutil.copy2(frame, dest)
        for pose in character.sheet:
            if pose.pose_id == pose_id:
                pose.image_path = str(dest)
                break
        else:
            character.sheet.append(
                CharacterSheetPose(pose_id=pose_id, label=pose_id, image_path=str(dest))
            )
        if pose_id in ("front_full", "three_quarter") or not character.image_path:
            character.image_path = str(dest)
        return dest

    def bootstrap_from_frame(self, plan: ProductionPlan, frame: Path) -> list[CharacterDesign]:
        chars = ensure_character_designs(plan)
        if any(self._primary_path(c) for c in chars):
            self._sync_primary_image(chars)
            plan.characters = chars
            return chars
        if not frame.exists():
            return chars
        primary = chars[0]
        self.adopt_frame(primary, frame, "front_full")
        # also stash as closeup if no face still
        if not any(p.pose_id == "closeup_face" and p.image_path for p in primary.sheet):
            self.adopt_frame(primary, frame, "closeup_face")
        self._sync_primary_image(chars)
        self._write_manifest(plan, chars)
        plan.characters = chars
        self._emit(f"Character sheet bootstrapped from frame → {primary.name}")
        return chars

    def enrich_from_accepted_frame(
        self, plan: ProductionPlan, frame: Path, shot: ShotPlan
    ) -> None:
        """Add an extra continuity-ish view from a passed take if slots remain empty."""
        if not frame.exists():
            return
        wanted = shot.ref_character_ids or ([plan.characters[0].id] if plan.characters else [])
        for cid in wanted:
            char = next((c for c in plan.characters if c.id == cid), None)
            if not char:
                continue
            miss = self._missing_poses(char)
            if not miss:
                continue
            # Prefer filling action or three_quarter from live footage
            prefer = ["action", "three_quarter", "front_full", "closeup_face"]
            target = None
            for pid in prefer:
                target = next((p for p in miss if p.pose_id == pid), None)
                if target:
                    break
            target = target or miss[0]
            self.adopt_frame(char, frame, target.pose_id)
            self._emit(f"Character sheet enriched {char.name}/{target.pose_id} from accepted take")
            break
        self._sync_primary_image(plan.characters)
        self._write_manifest(plan, plan.characters)

    def paths_in_picture_order(self, plan: ProductionPlan) -> list[Path]:
        """Flat list of all ready sheet stills (legacy helper)."""
        paths: list[Path] = []
        for c in plan.characters or []:
            for p in c.sheet:
                if p.image_path and Path(p.image_path).exists():
                    paths.append(Path(p.image_path))
            primary = self._primary_path(c)
            if primary and primary not in paths:
                paths.insert(0, primary)
        return paths

    def select_refs_for_shot(
        self,
        plan: ProductionPlan,
        shot: ShotPlan,
        last_frame: Path | None = None,  # unused; continuity is video / T2V keyframes
        prev_ref_ids: list[str] | None = None,  # unused; exclusivity handled by pipeline
    ) -> tuple[list[Path], list[dict], list[str]]:
        """
        Pick ≤9 reference images for this shot (identity stills only).
        Previous-clip continuity is attached by the pipeline as <Video 1> or T2V first_frame.
        Returns (paths, meta entries for prompt, extra notes).
        meta item: {picture, character_id, name, pose_id, label, look}
        """
        ensure_character_designs(plan)
        budget = min(R2V_MAX_IMAGES, max(1, self.settings.character_sheet_max_refs_per_shot))

        wanted_ids = list(shot.ref_character_ids or [])
        if not wanted_ids and plan.characters:
            # Solo-default to lead only — never auto-expand to full cast (causes leakage)
            if self._primary_path(plan.characters[0]):
                wanted_ids = [plan.characters[0].id]

        # Previous-shot continuity is a video ref (or T2V first_frame), not an extra still.

        # Lead cast first
        ordered_chars: list[CharacterDesign] = []
        for cid in wanted_ids:
            ch = next((c for c in plan.characters if c.id == cid), None)
            if ch:
                ordered_chars.append(ch)
        for c in plan.characters:
            if c not in ordered_chars and self._primary_path(c):
                # only if mentioned somehow — skip extras when budget tight
                pass

        pose_pref = _pose_preference_for_shot(shot)
        selected: list[tuple[Path, dict]] = []
        seen: set[str] = set()

        def try_add(char: CharacterDesign, pose: CharacterSheetPose) -> bool:
            if not pose.image_path:
                return False
            p = Path(pose.image_path)
            if not p.exists():
                return False
            key = str(p.resolve())
            if key in seen:
                return False
            if len(selected) >= budget:
                return False
            seen.add(key)
            selected.append(
                (
                    p,
                    {
                        "character_id": char.id,
                        "name": char.name,
                        "pose_id": pose.pose_id,
                        "label": pose.label or pose.pose_id,
                        "look": (char.look or "")[:160],
                    },
                )
            )
            return True

        # Pass 1: for each cast member on shot, best matching poses in order
        per_char_cap = max(1, budget // max(1, len(ordered_chars)))
        # Lead gets more views
        for i, char in enumerate(ordered_chars):
            cap = per_char_cap + (1 if i == 0 and budget >= 3 else 0)
            added = 0
            # preferred poses first, then remaining sheet
            poses_ranked = sorted(
                [p for p in char.sheet if p.image_path],
                key=lambda p: pose_pref.index(p.pose_id)
                if p.pose_id in pose_pref
                else 50 + PRIMARY_POSES_INDEX.get(p.pose_id, 99),
            )
            for pose in poses_ranked:
                if added >= cap:
                    break
                if try_add(char, pose):
                    added += 1

        # Pass 2: fill remaining budget with more views of lead
        if ordered_chars and len(selected) < budget:
            lead = ordered_chars[0]
            for pose in lead.sheet:
                if len(selected) >= budget:
                    break
                try_add(lead, pose)

        # Fallback: single primary image per char
        if not selected:
            for char in ordered_chars:
                primary = self._primary_path(char)
                if not primary:
                    continue
                key = str(primary.resolve())
                if key in seen:
                    continue
                seen.add(key)
                selected.append(
                    (
                        primary,
                        {
                            "character_id": char.id,
                            "name": char.name,
                            "pose_id": "primary",
                            "label": "identity",
                            "look": (char.look or "")[:160],
                        },
                    )
                )
                if len(selected) >= budget:
                    break

        paths = [p for p, _ in selected]
        meta = []
        for i, (_, m) in enumerate(selected, start=1):
            entry = dict(m)
            entry["picture"] = i
            meta.append(entry)

        extra: list[str] = []
        return paths, meta, extra

    def _write_manifest(self, plan: ProductionPlan, chars: list[CharacterDesign]) -> None:
        data = {
            "title": plan.title,
            "characters": [
                {
                    "id": c.id,
                    "name": c.name,
                    "look": c.look,
                    "primary": c.image_path,
                    "sheet": [
                        {
                            "pose_id": p.pose_id,
                            "label": p.label,
                            "image_path": p.image_path,
                        }
                        for p in c.sheet
                    ],
                }
                for c in chars
            ],
        }
        (self.board_dir / "sheet_manifest.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )


PRIMARY_POSES_INDEX = {pid: i for i, (pid, _, _) in enumerate(PRIMARY_POSES)}


def _pose_preference_for_shot(shot: ShotPlan) -> list[str]:
    cam = f"{shot.camera} {shot.visual_prompt} {shot.name} {shot.beat}".lower()
    if any(k in cam for k in ("close", "closeup", "close-up", "portrait", "face", "eyes")):
        return ["closeup_face", "three_quarter", "front_full", "action", "side", "back"]
    if any(k in cam for k in ("profile", "side view", "silhouette")):
        return ["side", "three_quarter", "front_full", "closeup_face", "back", "action"]
    if any(k in cam for k in ("wide", "establish", "establishing", "long shot", "bird")):
        return ["front_full", "three_quarter", "action", "side", "back", "closeup_face"]
    if any(k in cam for k in ("run", "action", "fight", "chase", "jump", "dance")):
        return ["action", "three_quarter", "front_full", "closeup_face", "side", "back"]
    return ["front_full", "three_quarter", "closeup_face", "side", "action", "back"]


def _safe_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "—", "–"):
            keep.append("_")
    return "".join(keep)[:40] or "char"
