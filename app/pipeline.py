"""End-to-end: Director → H3 generate → Critic loop → Assemble."""
from __future__ import annotations

import json
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .agents.critic import CriticAgent
from .agents.director import DirectorAgent, normalize_plan_cast_presence
from .agents.voice import VoiceAgent
from .character_board import CharacterBoardBuilder, ensure_character_designs
from .comfy_h3 import ComfyError, ComfyH3Client
from .config import Settings, get_settings
from .job_control import CancelledError, JobControl
from .media import assemble_master, extract_frames, probe
from .models import (
    GenerateRequest,
    ProductionPlan,
    ProjectState,
    ResumeRequest,
    ShotPlan,
    ShotRecord,
    ShotStatus,
    CriticVerdict,
)
from .services import ensure_runtime_services, comfy_reachable

LogFn = Callable[[str], None]


class ProductionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        log: LogFn | None = None,
        control: JobControl | None = None,
    ):
        self.settings = settings or get_settings()
        self.control = control
        self.director = DirectorAgent(self.settings, log=log)
        self.critic = CriticAgent(self.settings, log=log)
        self.comfy = ComfyH3Client(self.settings, control=control)
        self.voice = VoiceAgent(self.settings)

    def _check_cancel(self) -> None:
        if self.control:
            self.control.check()

    def _project_dir(self, project_id: str) -> Path:
        d = self.settings.output_root / project_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "shots").mkdir(exist_ok=True)
        (d / "takes").mkdir(exist_ok=True)
        (d / "reviews").mkdir(exist_ok=True)
        return d

    def _save_state(self, state: ProjectState) -> None:
        root = self._project_dir(state.project_id)
        path = root / "state.json"
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        # Plain-text log alongside assets (easy to open / attach / search)
        try:
            (root / "run.log").write_text(
                "\n".join(state.log) + ("\n" if state.log else ""),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _log(self, state: ProjectState, msg: str, log: LogFn | None = None) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        state.log.append(line)
        if log:
            log(line)
        self._save_state(state)

    def plan_only(
        self,
        prompt: str,
        style: str,
        target_duration_sec: float = 60.0,
        max_shots: int = 12,
    ) -> ProductionPlan:
        return self.director.plan(prompt, style, target_duration_sec, max_shots)

    def run(
        self,
        req: GenerateRequest,
        log: LogFn | None = None,
        on_start: Callable[[str], None] | None = None,
    ) -> ProjectState:
        project_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        state = ProjectState(
            project_id=project_id,
            user_prompt=req.prompt,
            style=req.style,
            status="planning",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        root = self._project_dir(project_id)
        self._save_state(state)
        if self.control:
            self.control.set_project_id(project_id)
        if on_start:
            try:
                on_start(project_id)
            except Exception:
                pass

        try:
            self._ensure_services(state, log)
            try:
                self.comfy.health()
            except Exception as e:
                state.status = "error"
                self._log(state, f"ComfyUI not reachable at {self.settings.comfy_base_url}: {e}", log)
                return state

            self._check_cancel()
            # Rebind agent logs for this run so fallback attempts show in job output.
            self.director = DirectorAgent(self.settings, log=lambda m: self._log(state, m, log))
            self.critic = CriticAgent(self.settings, log=lambda m: self._log(state, m, log))

            mode = (req.h3_mode or self.settings.h3_mode or "r2v").lower()
            if mode not in ("r2v", "t2v", "auto"):
                mode = "r2v"
            state.h3_mode = mode

            self._log(state, "Director: building production plan (Gemini → local → offline)…", log)
            plan = self.director.plan(
                req.prompt,
                req.style,
                target_duration_sec=req.target_duration_sec,
                max_shots=req.max_shots,
            )
            plan = normalize_plan_cast_presence(plan)
            self._check_cancel()
            state.plan = plan
            state.shots = [ShotRecord(plan=s) for s in plan.shots]
            (root / "production.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            self._log(
                state,
                f"Plan ready via {self.director.last_provider or 'unknown'}: "
                f"“{plan.title}” — {len(plan.shots)} shots, "
                f"~{plan.target_duration_sec:.0f}s · H3 mode={mode}",
                log,
            )

            board_dir = root / "character_board"
            board_dir.mkdir(exist_ok=True)
            state.character_board_dir = str(board_dir)
            board = CharacterBoardBuilder(
                self.settings,
                board_dir,
                log=lambda m: self._log(state, m, log),
                comfy=self.comfy,
            )
            if mode in ("r2v", "auto"):
                self._log(
                    state,
                    "Building multi-view character sheets for R2V "
                    f"(≤{self.settings.character_sheet_max_refs_per_shot} refs/shot)…",
                    log,
                )
                board.build(plan)
                (root / "production.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                self._save_state(state)

            max_retakes = req.max_retakes if req.max_retakes is not None else self.settings.max_retakes
            self._render_remaining_shots(
                state,
                board=board,
                mode=mode,
                max_retakes=max_retakes,
                seed_base=req.seed_base,
                auto_assemble=req.auto_assemble,
                log=log,
            )

        except CancelledError:
            state.status = "cancelled"
            try:
                self.comfy.interrupt()
            except Exception:
                pass
            self._log(state, "Generation stopped by user.", log)
        except Exception as e:
            state.status = "error"
            self._log(state, f"Pipeline error: {e}", log)
            self._log(state, traceback.format_exc()[-1500:], log)

        self._save_state(state)
        return state

    def resume(
        self,
        project_id: str,
        req: ResumeRequest | None = None,
        log: LogFn | None = None,
        on_start: Callable[[str], None] | None = None,
    ) -> ProjectState:
        """Continue a prior project from the first unfinished shot."""
        req = req or ResumeRequest()
        state = load_state(project_id, self.settings)
        if not state:
            raise FileNotFoundError(f"Project not found: {project_id}")
        if not state.plan or not state.plan.shots:
            raise ValueError("Project has no production plan to resume")

        root = self._project_dir(project_id)
        if self.control:
            self.control.set_project_id(project_id)
        if on_start:
            try:
                on_start(project_id)
            except Exception:
                pass

        # Normalize shots list against plan
        if not state.shots or len(state.shots) != len(state.plan.shots):
            by_id = {r.plan.id: r for r in (state.shots or [])}
            rebuilt: list[ShotRecord] = []
            for s in state.plan.shots:
                rebuilt.append(by_id.get(s.id) or ShotRecord(plan=s))
            state.shots = rebuilt

        try:
            self._ensure_services(state, log)
            try:
                self.comfy.health()
            except Exception as e:
                state.status = "error"
                self._log(state, f"ComfyUI not reachable at {self.settings.comfy_base_url}: {e}", log)
                return state

            self._check_cancel()
            self.director = DirectorAgent(self.settings, log=lambda m: self._log(state, m, log))
            self.critic = CriticAgent(self.settings, log=lambda m: self._log(state, m, log))

            mode = (req.h3_mode or state.h3_mode or self.settings.h3_mode or "r2v").lower()
            if mode not in ("r2v", "t2v", "auto"):
                mode = "r2v"
            state.h3_mode = mode
            if state.plan:
                state.plan = normalize_plan_cast_presence(state.plan)
                # Keep shot records in sync with rewritten presence on the plan
                plan_by_id = {s.id: s for s in state.plan.shots}
                for rec in state.shots or []:
                    if rec.plan.id in plan_by_id:
                        rec.plan = plan_by_id[rec.plan.id]
            state.status = "running"
            self._log(
                state,
                f"Resuming project {project_id} "
                f"({sum(1 for r in state.shots if r.status == ShotStatus.passed)}/"
                f"{len(state.shots)} shots already passed) · H3 mode={mode}",
                log,
            )

            # Recover "passed" status from on-disk clip files when state was interrupted mid-write
            for record in state.shots:
                if record.status == ShotStatus.passed and record.final_video and Path(record.final_video).exists():
                    continue
                if record.final_video and Path(record.final_video).exists():
                    record.status = ShotStatus.passed
                    continue
                shots_dir = root / "shots"
                if shots_dir.is_dir():
                    sid = record.plan.id
                    candidates = sorted(
                        [
                            p
                            for p in shots_dir.glob(f"{sid}_*.mp4")
                            if p.is_file() and p.stat().st_size > 1024
                        ],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if candidates:
                        record.final_video = str(candidates[0].resolve())
                        record.status = ShotStatus.passed
                        record.error = None
                        self._log(
                            state,
                            f"{sid}: recovered passed clip from disk ({candidates[0].name})",
                            log,
                        )

            plan = state.plan
            board_dir = Path(state.character_board_dir) if state.character_board_dir else root / "character_board"
            board_dir.mkdir(parents=True, exist_ok=True)
            state.character_board_dir = str(board_dir)
            board = CharacterBoardBuilder(
                self.settings,
                board_dir,
                log=lambda m: self._log(state, m, log),
                comfy=self.comfy,
            )
            if mode in ("r2v", "auto"):
                self._log(state, "Refreshing character sheets (reuse existing stills)…", log)
                board.build(plan)
                (root / "production.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

            # Reset non-passed shots so they re-queue
            for record in state.shots:
                ok = (
                    record.status == ShotStatus.passed
                    and record.final_video
                    and Path(record.final_video).exists()
                )
                if ok:
                    continue
                if record.status == ShotStatus.failed and not req.redo_failed:
                    continue
                if record.status == ShotStatus.passed:
                    self._log(
                        state,
                        f"{record.plan.id} marked passed but video missing — will re-render",
                        log,
                    )
                record.status = ShotStatus.pending
                record.error = None

            max_retakes = req.max_retakes if req.max_retakes is not None else self.settings.max_retakes
            self._render_remaining_shots(
                state,
                board=board,
                mode=mode,
                max_retakes=max_retakes,
                seed_base=req.seed_base,
                auto_assemble=req.auto_assemble,
                log=log,
            )

        except CancelledError:
            state.status = "cancelled"
            try:
                self.comfy.interrupt()
            except Exception:
                pass
            self._log(state, "Generation stopped by user.", log)
        except Exception as e:
            state.status = "error"
            self._log(state, f"Pipeline error: {e}", log)
            self._log(state, traceback.format_exc()[-1500:], log)

        self._save_state(state)
        return state

    def _ensure_services(self, state: ProjectState, log: LogFn | None) -> None:
        from .services import essentials_report

        report_pre = essentials_report(self.settings)
        if report_pre.get("prompt"):
            self._log(state, report_pre["prompt"].replace("\n", " | "), log)

        deadline_note = min(self.settings.essentials_wait_sec, self.settings.comfy_start_timeout_sec)
        self._log(
            state,
            f"Checking essentials (ComfyUI / Ollama / FFmpeg) — will wait ≤{deadline_note}s then fail if still down…",
            log,
        )
        report = ensure_runtime_services(
            self.settings,
            log=lambda m: self._log(state, m, log),
            need_comfy=True,
            need_ollama=True,
        )
        comfy = report.get("comfy") or {}
        if not comfy.get("ok"):
            msg = (
                comfy.get("error")
                or f"ComfyUI not ready at {self.settings.comfy_base_url}"
            )
            raise RuntimeError(
                f"ESSENTIALS FAILED (within {deadline_note}s): {msg} "
                "Start/replace ComfyUI with MiniMax H3 nodes (COMFYUI_ROOT), then Retry / Resume."
            )
        if self.settings.comfy_require_h3_nodes:
            from .services import comfy_h3_status

            h3 = comfy.get("h3") or comfy_h3_status(self.settings)
            if not h3.get("ok"):
                raise RuntimeError(
                    "ESSENTIALS FAILED: MiniMax H3 nodes not found "
                    f"({', '.join(h3.get('missing_nodes') or [])}). "
                    f"COMFYUI_ROOT={self.settings.comfyui_root}; "
                    "set COMFY_REPLACE_NON_H3=true if another app owns the port."
                )
        from .services import ffmpeg_available

        if not ffmpeg_available(self.settings):
            raise RuntimeError(
                f"ESSENTIALS FAILED: FFmpeg not found ({self.settings.ffmpeg_path}). "
                "Install FFmpeg and set FFMPEG_PATH in .env."
            )
        oll = report.get("ollama") or {}
        if oll and not oll.get("ok") and oll.get("status") not in (
            "disabled",
            "skipped",
            "already_running",
        ):
            self._log(
                state,
                f"Local LLM: {oll.get('status')} — {oll.get('error') or 'not available'}; "
                "continuing (Gemini/offline fallback).",
                log,
            )
        post = essentials_report(self.settings)
        if post.get("warnings"):
            for w in post["warnings"]:
                self._log(state, f"Warning: {w}", log)
        if not post.get("ok"):
            self._log(
                state,
                "Essentials still have issues: " + " | ".join(post.get("blocking") or []),
                log,
            )

    def _render_remaining_shots(
        self,
        state: ProjectState,
        *,
        board: CharacterBoardBuilder,
        mode: str,
        max_retakes: int,
        seed_base: int,
        auto_assemble: bool,
        log: LogFn | None,
    ) -> None:
        plan = state.plan
        assert plan is not None
        last_frame: Path | None = None
        last_ref_ids: list[str] | None = None

        for idx, record in enumerate(state.shots):
            self._check_cancel()
            if (
                record.status == ShotStatus.passed
                and record.final_video
                and Path(record.final_video).exists()
            ):
                if record.final_frame and Path(record.final_frame).exists():
                    last_frame = Path(record.final_frame)
                last_ref_ids = list(record.plan.ref_character_ids or [])
                self._log(state, f"{record.plan.id} already passed — skip", log)
                continue
            if record.status == ShotStatus.failed:
                # left failed intentionally (redo_failed=false)
                self._log(state, f"{record.plan.id} left as failed — skip", log)
                continue

            last_frame = self._produce_shot(
                state,
                record,
                idx,
                seed_base,
                max_retakes,
                log,
                board=board,
                last_frame=last_frame,
                prev_ref_ids=last_ref_ids,
                mode=mode,
            ) or last_frame
            last_ref_ids = list(record.plan.ref_character_ids or [])

        self._check_cancel()
        passed = [r for r in state.shots if r.status == ShotStatus.passed and r.final_video]
        if auto_assemble and passed:
            state.status = "assembling"
            self._log(state, f"Assembling master from {len(passed)} passed shots…", log)
            clips = [Path(r.final_video) for r in passed if r.final_video]
            master = self._project_dir(state.project_id) / "master" / f"{_safe(plan.title)}.mp4"
            assemble_master(
                self.settings,
                clips,
                master,
                title=plan.title,
                subtitle=plan.logline[:80] if plan.logline else state.style[:60],
                add_cards=True,
            )
            state.master_path = str(master)
            self._log(state, f"Master ready: {master}", log)
            state.status = "completed"
        elif not passed:
            state.status = "failed"
            self._log(state, "No shots passed critic — no master assembled.", log)
        else:
            state.status = "completed_no_assemble"

        if self.voice.enabled:
            self._log(state, "Voice enabled but not implemented in this build.", log)
        else:
            self._log(state, "Voice/narration skipped (ElevenLabs disabled).", log)

    def _produce_shot(
        self,
        state: ProjectState,
        record: ShotRecord,
        idx: int,
        seed_base: int,
        max_retakes: int,
        log: LogFn | None,
        *,
        board: CharacterBoardBuilder | None = None,
        last_frame: Path | None = None,
        prev_ref_ids: list[str] | None = None,
        mode: str = "r2v",
    ) -> Path | None:
        plan = state.plan
        assert plan is not None
        shot = record.plan
        root = self._project_dir(state.project_id)
        take_dir = root / "takes" / shot.id
        take_dir.mkdir(parents=True, exist_ok=True)

        critic_notes = ""
        visual_override = shot.visual_prompt
        accepted_frame: Path | None = None

        for take in range(1, max_retakes + 2):  # initial + retakes
            if self.control:
                self.control.check()
            record.status = ShotStatus.generating
            seed = seed_base + idx * 17 + take * 3

            shot_for_prompt = shot.model_copy(update={"visual_prompt": visual_override})
            gen_mode, ref_paths, picture_meta, picture_map, extra_notes = self._resolve_refs(
                plan, shot, mode, board, last_frame, prev_ref_ids=prev_ref_ids
            )
            render_prompt = self.director.build_render_prompt(
                plan,
                shot_for_prompt,
                critic_notes,
                r2v=(gen_mode == "r2v"),
                picture_map=picture_map,
                picture_meta=picture_meta,
                extra_picture_notes=extra_notes,
            )
            # Keep exclusivity in retake notes when critic already complained about cast
            if take == 1 and (shot.ref_character_ids is not None):
                off = [
                    c.name
                    for c in (plan.characters or [])
                    if c.id not in set(shot.ref_character_ids or [])
                ]
                if off:
                    self._log(
                        state,
                        f"{shot.id}: exclusive cast={shot.ref_character_ids or []} "
                        f"(banned: {', '.join(off)})",
                        log,
                    )

            ref_labels = ", ".join(
                f"P{m.get('picture')}:{m.get('name','?')}/{m.get('pose_id','?')}"
                for m in (picture_meta or [])
            ) or "none"
            self._log(
                state,
                f"{shot.id} take {take}: H3 {gen_mode.upper()} "
                f"(seed={seed}, refs={len(ref_paths)} [{ref_labels}])…",
                log,
            )

            prefix = f"video/H3VideoGen/{state.project_id}/{shot.id}_t{take}"
            try:
                src, prompt_id, used_mode = self.comfy.generate(
                    render_prompt,
                    length=shot.length_frames,
                    seed=seed,
                    filename_prefix=prefix,
                    mode=gen_mode,
                    ref_image_paths=ref_paths,
                    project_tag=state.project_id,
                )
            except CancelledError:
                raise
            except ComfyError as e:
                # If R2V fails (model/node), fall back once to T2V
                if gen_mode == "r2v":
                    self._log(
                        state,
                        f"{shot.id} R2V failed ({e}); falling back to T2V for this take…",
                        log,
                    )
                    try:
                        if self.control:
                            self.control.check()
                        t2v_prompt = self.director.build_render_prompt(
                            plan, shot_for_prompt, critic_notes, r2v=False
                        )
                        src, prompt_id, used_mode = self.comfy.generate(
                            t2v_prompt,
                            length=shot.length_frames,
                            seed=seed,
                            filename_prefix=prefix + "_t2v",
                            mode="t2v",
                            project_tag=state.project_id,
                        )
                        render_prompt = t2v_prompt
                    except CancelledError:
                        raise
                    except ComfyError as e2:
                        record.status = ShotStatus.failed
                        record.error = str(e2)
                        self._log(state, f"{shot.id} generation failed: {e2}", log)
                        return accepted_frame
                else:
                    record.status = ShotStatus.failed
                    record.error = str(e)
                    self._log(state, f"{shot.id} generation failed: {e}", log)
                    return accepted_frame

            dest = take_dir / f"{shot.id}_take{take}.mp4"
            try:
                shutil.copy2(src, dest)
            except Exception as copy_err:
                # Last chance: re-resolve/download if path went stale
                self._log(state, f"{shot.id}: copy failed ({copy_err}); re-fetching from Comfy…", log)
                raise ComfyError(f"Could not copy generated clip: {copy_err}") from copy_err
            if not dest.exists() or dest.stat().st_size < 1024:
                raise ComfyError(f"Copied clip is empty/missing: {dest}")
            try:
                frames = extract_frames(self.settings, dest, take_dir / f"frames_t{take}")
            except Exception as fe:
                self._log(state, f"{shot.id} take {take}: frame extract failed ({fe})", log)
                frames = []
            mid = frames[min(1, len(frames) - 1)] if frames else None
            try:
                meta = probe(self.settings, dest)
            except Exception as pe:
                self._log(state, f"{shot.id} take {take}: probe failed ({pe})", log)
                meta = {}
            if not frames:
                self._log(
                    state,
                    f"{shot.id} take {take}: no critic frames from {dest.name} "
                    f"({dest.stat().st_size} bytes) — will still record take",
                    log,
                )

            # Bootstrap multi-view sheet from first good mid-frame if still empty
            if (
                board
                and mid
                and mode in ("r2v", "auto")
                and not board.paths_in_picture_order(plan)
            ):
                board.bootstrap_from_frame(plan, mid)
                (root / "production.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

            record.status = ShotStatus.reviewing
            self._log(state, f"{shot.id} take {take}: critic review ({used_mode})…", log)
            review = self.critic.review(
                plan,
                shot,
                frames,
                take=take,
                video_meta={
                    "duration": (meta.get("format") or {}).get("duration"),
                    "size": (meta.get("format") or {}).get("size"),
                    "streams": meta.get("streams"),
                    "prompt_id": prompt_id,
                    "h3_mode": used_mode,
                    "ref_count": len(ref_paths),
                },
                render_prompt_used=render_prompt,
            )
            record.reviews.append(review)
            (root / "reviews" / f"{shot.id}_t{take}.json").write_text(
                review.model_dump_json(indent=2), encoding="utf-8"
            )
            self._log(
                state,
                f"{shot.id} critic ({self.critic.last_provider or '?'}): "
                f"{review.verdict.value} score={review.overall_score} "
                f"youtube_ready={review.youtube_ready} — {review.summary[:160]}",
                log,
            )

            take_info = {
                "take": take,
                "seed": seed,
                "video": str(dest),
                "frame": str(mid) if mid else None,
                "prompt_id": prompt_id,
                "score": review.overall_score,
                "verdict": review.verdict.value,
                "h3_mode": used_mode,
                "ref_count": len(ref_paths),
            }
            record.takes.append(take_info)

            if review.verdict == CriticVerdict.pass_:
                final = root / "shots" / f"{shot.id}_{_safe(shot.name)}.mp4"
                shutil.copy2(dest, final)
                record.final_video = str(final)
                record.final_frame = str(mid) if mid else None
                record.status = ShotStatus.passed
                accepted_frame = Path(mid) if mid else None
                if board and mid and mode in ("r2v", "auto"):
                    try:
                        board.enrich_from_accepted_frame(plan, Path(mid), shot)
                        (root / "production.json").write_text(
                            plan.model_dump_json(indent=2), encoding="utf-8"
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._log(state, f"Sheet enrich skipped: {exc}", log)
                self._log(state, f"{shot.id} PASSED — keep take {take} ({used_mode})", log)
                return accepted_frame

            if take > max_retakes or (review.verdict == CriticVerdict.reject and take > max_retakes):
                best = max(record.takes, key=lambda t: t.get("score") or 0, default=None)
                if best and (best.get("score") or 0) >= self.settings.critic_pass_threshold - 1.0:
                    final = root / "shots" / f"{shot.id}_{_safe(shot.name)}_best.mp4"
                    shutil.copy2(best["video"], final)
                    record.final_video = str(final)
                    record.final_frame = best.get("frame")
                    record.status = ShotStatus.passed
                    if best.get("frame"):
                        accepted_frame = Path(best["frame"])
                    self._log(
                        state,
                        f"{shot.id} accepted best effort take {best['take']} "
                        f"(score={best.get('score')}) after max retakes",
                        log,
                    )
                    return accepted_frame
                record.status = ShotStatus.failed
                record.error = review.summary or "Failed critic after retakes"
                self._log(state, f"{shot.id} failed after retakes", log)
                return accepted_frame

            record.status = ShotStatus.retake
            critic_notes = review.retake_instructions or review.summary
            if review.revised_prompt:
                visual_override = review.revised_prompt
            self._log(state, f"{shot.id} RETAKE ordered: {critic_notes[:200]}", log)

        return accepted_frame

    def _resolve_refs(
        self,
        plan: ProductionPlan,
        shot: ShotPlan,
        mode: str,
        board: CharacterBoardBuilder | None,
        last_frame: Path | None,
        prev_ref_ids: list[str] | None = None,
    ) -> tuple[str, list[Path], list[dict], dict[str, int], list[str]]:
        """Return (mode, ref_paths, picture_meta, picture_map, extra_notes)."""
        if mode == "t2v":
            return "t2v", [], [], {}, []

        ensure_character_designs(plan)
        if board:
            paths, meta, extra = board.select_refs_for_shot(
                plan, shot, last_frame, prev_ref_ids=prev_ref_ids
            )
            picture_map: dict[str, int] = {}
            for m in meta:
                cid = m.get("character_id")
                pic = m.get("picture")
                if cid and pic and cid not in picture_map:
                    picture_map[cid] = int(pic)
            if paths:
                return "r2v", paths, meta, picture_map, extra
            return "t2v", [], [], {}, []

        # No board builder — single primary stills only
        picture_map = {}
        paths: list[Path] = []
        meta: list[dict] = []
        extra: list[str] = []
        wanted_ids = list(shot.ref_character_ids or [])
        if not wanted_ids and plan.characters:
            wanted_ids = [plan.characters[0].id] if plan.characters[0].image_path else []
        for cid in wanted_ids:
            char = next((c for c in plan.characters if c.id == cid), None)
            if not char or not char.image_path:
                continue
            p = Path(char.image_path)
            if not p.exists() or p in paths:
                continue
            paths.append(p)
            pic = len(paths)
            picture_map[cid] = pic
            meta.append(
                {
                    "picture": pic,
                    "character_id": cid,
                    "name": char.name,
                    "pose_id": "primary",
                    "label": "identity",
                    "look": (char.look or "")[:160],
                }
            )
        use_prev = (
            self.settings.h3_use_prev_shot_ref
            and last_frame
            and last_frame.exists()
            and len(paths) < 9
        )
        if use_prev and prev_ref_ids is not None:
            if set(prev_ref_ids) - set(wanted_ids):
                use_prev = False
        if use_prev:
            paths.append(last_frame)  # type: ignore[arg-type]
            extra.append(
                f"- Continuity / lighting from previous shot is <Picture {len(paths)}> "
                "(match costume continuity of ON-SCREEN cast only; "
                "do not reintroduce banned cast; new camera and action are OK)."
            )
        if paths:
            return "r2v", paths, meta, picture_map, extra
        return "t2v", [], [], {}, []


def _safe(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "—", "–"):
            keep.append("_")
    return "".join(keep)[:60] or "shot"


def load_state(project_id: str, settings: Settings | None = None) -> ProjectState | None:
    settings = settings or get_settings()
    path = settings.output_root / project_id / "state.json"
    if not path.exists():
        return None
    return ProjectState.model_validate_json(path.read_text(encoding="utf-8"))


def project_is_resumable(data: dict) -> bool:
    """True when a plan exists and work remains (not a finished master)."""
    plan = data.get("plan") or {}
    shots = plan.get("shots") or data.get("shots") or []
    if not shots and not plan.get("shots"):
        # ProjectState has shots separately
        shot_records = data.get("shots") or []
        if not shot_records and not plan:
            return False
    status = (data.get("status") or "").lower()
    if status in ("completed",) and data.get("master_path"):
        mp = data.get("master_path")
        if mp and Path(mp).exists():
            return False
    # Need a plan
    if not plan and not data.get("shots"):
        return False
    if not plan.get("shots") and not data.get("shots"):
        return False
    # Any incomplete shot record or sticky / error status
    records = data.get("shots") or []
    if records:
        for r in records:
            st = (r.get("status") or "pending").lower()
            fv = r.get("final_video")
            if st != "passed" or not fv or not Path(str(fv)).exists():
                return True
        # all passed but missing master
        if not data.get("master_path") or not Path(str(data.get("master_path"))).exists():
            return True
        return False
    return status not in ("completed",)


def list_projects(settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    root = settings.output_root
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for d in sorted(root.iterdir(), reverse=True):
        st = d / "state.json"
        if st.exists():
            try:
                data = json.loads(st.read_text(encoding="utf-8"))
                items.append(
                    {
                        "project_id": data.get("project_id", d.name),
                        "status": data.get("status"),
                        "title": (data.get("plan") or {}).get("title"),
                        "prompt": data.get("user_prompt", "")[:120],
                        "master_path": data.get("master_path"),
                        "resumable": project_is_resumable(data),
                        "shots_total": len((data.get("plan") or {}).get("shots") or data.get("shots") or []),
                        "shots_passed": sum(
                            1
                            for r in (data.get("shots") or [])
                            if (r.get("status") or "").lower() == "passed"
                            and r.get("final_video")
                        ),
                    }
                )
            except Exception:
                continue
    return items
