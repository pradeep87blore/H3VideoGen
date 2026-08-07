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
from .media import assemble_master, extract_frames, probe, mux_narration
from .models import (
    GenerateRequest,
    NarrativeMode,
    ProductionPlan,
    ProjectState,
    ResumeRequest,
    ShotPlan,
    ShotRecord,
    ShotStatus,
    StageTiming,
    CriticVerdict,
    normalize_narrative_mode,
)
from .scene_still import generate_scene_still
from .services import ensure_runtime_services, comfy_reachable

LogFn = Callable[[str], None]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _elapsed_sec(started: str | None, ended: str | None = None) -> float | None:
    a = _parse_iso(started)
    if not a:
        return None
    b = _parse_iso(ended) if ended else _utc_now()
    if not b:
        return None
    return round(max(0.0, (b - a).total_seconds()), 2)


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

    def _sync_character_stage_rows(self, state: ProjectState) -> None:
        """Mirror finalized character sheet timings into the stage table."""
        plan = state.plan
        if not plan or not plan.characters:
            return
        for c in plan.characters:
            key = f"char:{c.id}"
            label = f"Sheet · {c.name}"
            row = self._stage_get(state, key)
            if row is None:
                row = StageTiming(key=key, label=label)
                # Insert after character_sheets block if present
                cs_i = next(
                    (i for i, r in enumerate(state.stage_timings) if r.key == "character_sheets"),
                    None,
                )
                if cs_i is not None:
                    # keep chars grouped after character_sheets, before shots
                    insert_at = cs_i + 1
                    while (
                        insert_at < len(state.stage_timings)
                        and state.stage_timings[insert_at].key.startswith("char:")
                    ):
                        insert_at += 1
                    state.stage_timings.insert(insert_at, row)
                else:
                    state.stage_timings.append(row)
            row.label = label
            row.started_at = c.sheet_started_at or row.started_at or ""
            row.ended_at = c.sheet_finished_at
            row.duration_sec = c.sheet_duration_sec
            st = (c.sheet_status or "pending").lower()
            if st == "ready":
                row.status = "done"
            elif st == "building":
                row.status = "running"
                if row.started_at:
                    row.duration_sec = _elapsed_sec(row.started_at, None)
            elif st == "failed":
                row.status = "error"
            else:
                row.status = st if st in ("pending", "skipped") else "pending"
            poses = sum(1 for p in (c.sheet or []) if p.image_path)
            row.detail = (
                f"{poses} view(s)"
                + (f" · {c.sheet_source}" if c.sheet_source else "")
            )
        self._save_state(state)

    def _stage_get(self, state: ProjectState, key: str) -> StageTiming | None:
        for row in state.stage_timings:
            if row.key == key:
                return row
        return None

    def _stage_start(
        self,
        state: ProjectState,
        key: str,
        label: str,
        *,
        detail: str = "",
        log: LogFn | None = None,
        silent: bool = False,
    ) -> StageTiming:
        now = _iso_now()
        row = self._stage_get(state, key)
        if row is None:
            row = StageTiming(key=key, label=label)
            state.stage_timings.append(row)
        row.label = label
        row.started_at = now
        row.ended_at = None
        row.duration_sec = None
        row.status = "running"
        if detail:
            row.detail = detail
        if not silent:
            self._save_state(state)
        return row

    def _stage_end(
        self,
        state: ProjectState,
        key: str,
        *,
        status: str = "done",
        detail: str | None = None,
        log: LogFn | None = None,
        silent: bool = False,
    ) -> StageTiming | None:
        row = self._stage_get(state, key)
        if row is None:
            return None
        if row.status == "running" or row.ended_at is None:
            row.ended_at = _iso_now()
            row.duration_sec = _elapsed_sec(row.started_at, row.ended_at)
        row.status = status
        if detail is not None:
            row.detail = detail
        if not silent:
            self._save_state(state)
        return row

    def _stage_skip(
        self,
        state: ProjectState,
        key: str,
        label: str,
        *,
        detail: str = "",
    ) -> None:
        row = self._stage_get(state, key)
        if row is None:
            row = StageTiming(key=key, label=label)
            state.stage_timings.append(row)
        row.label = label
        row.status = "skipped"
        row.detail = detail or row.detail
        if not row.started_at:
            row.started_at = _iso_now()
        if not row.ended_at:
            row.ended_at = row.started_at
            row.duration_sec = 0.0
        self._save_state(state)

    def _stage_bump_running(self, state: ProjectState) -> None:
        """Refresh elapsed seconds on open stages (for UI polling)."""
        for row in state.stage_timings:
            if row.status == "running" and row.started_at:
                row.duration_sec = _elapsed_sec(row.started_at, None)
        total = self._stage_get(state, "total")
        if total and total.status == "running" and state.job_started_at:
            total.duration_sec = _elapsed_sec(state.job_started_at, None)

    def _finalize_job_clock(self, state: ProjectState, *, status: str | None = None) -> None:
        state.job_finished_at = _iso_now()
        for row in state.stage_timings:
            if row.status == "running":
                row.ended_at = state.job_finished_at
                row.duration_sec = _elapsed_sec(row.started_at, row.ended_at)
                if status in ("error", "cancelled", "failed"):
                    row.status = status if status != "failed" else "error"
                else:
                    row.status = "done"
        total = self._stage_get(state, "total")
        if total:
            total.ended_at = state.job_finished_at
            total.duration_sec = _elapsed_sec(
                state.job_started_at or total.started_at or state.created_at,
                total.ended_at,
            )
            total.status = "done" if (status or state.status or "").startswith("completed") else (
                status or total.status or "done"
            )
            if total.status == "failed":
                total.status = "error"
        self._save_state(state)

    def plan_only(
        self,
        prompt: str,
        style: str,
        target_duration_sec: float = 60.0,
        max_shots: int = 12,
        narrative_mode: str = "character",
    ) -> ProductionPlan:
        return self.director.plan(
            prompt,
            style,
            target_duration_sec,
            max_shots,
            narrative_mode=narrative_mode,
        )

    def run(
        self,
        req: GenerateRequest,
        log: LogFn | None = None,
        on_start: Callable[[str], None] | None = None,
    ) -> ProjectState:
        project_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        started = _iso_now()
        state = ProjectState(
            project_id=project_id,
            user_prompt=req.prompt,
            style=req.style,
            status="planning",
            created_at=started,
            job_started_at=started,
            stage_timings=[
                StageTiming(
                    key="total",
                    label="Total job",
                    started_at=started,
                    status="running",
                )
            ],
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
            self._stage_start(state, "essentials", "Essentials (Comfy / Ollama / FFmpeg)", silent=True)
            self._ensure_services(state, log)
            self._stage_end(state, "essentials", detail="ready")
            try:
                self.comfy.health()
            except Exception as e:
                state.status = "error"
                self._stage_end(state, "essentials", status="error", detail=str(e)[:120])
                self._log(state, f"ComfyUI not reachable at {self.settings.comfy_base_url}: {e}", log)
                self._finalize_job_clock(state, status="error")
                return state

            self._check_cancel()
            # Rebind agent logs for this run so fallback attempts show in job output.
            self.director = DirectorAgent(self.settings, log=lambda m: self._log(state, m, log))
            self.critic = CriticAgent(self.settings, log=lambda m: self._log(state, m, log))
            self.voice = VoiceAgent(self.settings, log=lambda m: self._log(state, m, log))

            narr_mode = normalize_narrative_mode(req.narrative_mode)
            state.narrative_mode = narr_mode

            mode = (req.h3_mode or self.settings.h3_mode or "r2v").lower()
            if mode not in ("r2v", "t2v", "auto"):
                mode = "r2v"
            # Documentary / explainer default to text-to-video (no cast sheets)
            if narr_mode in (
                NarrativeMode.documentary.value,
                NarrativeMode.explainer.value,
            ) and req.h3_mode is None:
                mode = "t2v"
            state.h3_mode = mode

            self._stage_start(state, "director", f"Director plan ({narr_mode})")
            self._log(
                state,
                f"Director: building production plan · narrative_mode={narr_mode} "
                f"(Gemini → local → offline)…",
                log,
            )
            plan = self.director.plan(
                req.prompt,
                req.style,
                target_duration_sec=req.target_duration_sec,
                max_shots=req.max_shots,
                narrative_mode=narr_mode,
            )
            plan = normalize_plan_cast_presence(plan)
            self._check_cancel()
            state.plan = plan
            state.shots = [ShotRecord(plan=s) for s in plan.shots]
            (root / "production.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            self._stage_end(
                state,
                "director",
                detail=(
                    f"{plan.title} — {len(plan.shots)} shots · {narr_mode} via "
                    f"{self.director.last_provider or 'unknown'}"
                ),
            )
            self._log(
                state,
                f"Plan ready via {self.director.last_provider or 'unknown'}: "
                f"“{plan.title}” — {len(plan.shots)} shots, "
                f"~{plan.target_duration_sec:.0f}s · narrative={narr_mode} · H3 mode={mode}",
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
                on_character_done=lambda _c: self._sync_character_stage_rows(state),
            )
            need_sheets = (
                narr_mode == NarrativeMode.character.value
                and mode in ("r2v", "auto")
                and bool(plan.characters)
            )
            if need_sheets:
                self._stage_start(state, "character_sheets", "Character sheets")
                self._log(
                    state,
                    "Building multi-view character sheets for R2V "
                    f"(≤{self.settings.character_sheet_max_refs_per_shot} refs/shot)…",
                    log,
                )
                board.build(plan)
                self._sync_character_stage_rows(state)
                (root / "production.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                ready = sum(
                    1
                    for c in plan.characters or []
                    if (c.sheet_status or "") == "ready" or c.image_path
                )
                sheet_times = [
                    f"{c.name}={c.sheet_duration_sec:.1f}s"
                    for c in (plan.characters or [])
                    if c.sheet_duration_sec is not None
                ]
                detail = f"{ready}/{len(plan.characters or [])} cast ready"
                if sheet_times:
                    detail += " · " + ", ".join(sheet_times)
                self._stage_end(state, "character_sheets", detail=detail)
                self._save_state(state)
            else:
                reason = (
                    f"skipped ({narr_mode} mode / H3={mode})"
                    if narr_mode != NarrativeMode.character.value
                    else "skipped (t2v or no cast)"
                )
                self._stage_skip(
                    state,
                    "character_sheets",
                    "Character sheets",
                    detail=reason,
                )

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
            self._finalize_job_clock(state, status=state.status)

        except CancelledError:
            state.status = "cancelled"
            try:
                self.comfy.interrupt()
            except Exception:
                pass
            self._log(state, "Generation stopped by user.", log)
            self._finalize_job_clock(state, status="cancelled")
        except Exception as e:
            state.status = "error"
            self._log(state, f"Pipeline error: {e}", log)
            self._log(state, traceback.format_exc()[-1500:], log)
            self._finalize_job_clock(state, status="error")

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

        if not state.job_started_at:
            state.job_started_at = state.created_at or _iso_now()
        state.job_finished_at = None
        if not self._stage_get(state, "total"):
            state.stage_timings.insert(
                0,
                StageTiming(
                    key="total",
                    label="Total job",
                    started_at=state.job_started_at,
                    status="running",
                    detail="resumed",
                ),
            )
        else:
            total = self._stage_get(state, "total")
            if total:
                total.status = "running"
                total.ended_at = None
                total.detail = (total.detail + " · resumed").strip(" ·")

        try:
            self._stage_start(state, "essentials", "Essentials (Comfy / Ollama / FFmpeg)", silent=True)
            self._ensure_services(state, log)
            self._stage_end(state, "essentials", detail="ready")
            try:
                self.comfy.health()
            except Exception as e:
                state.status = "error"
                self._stage_end(state, "essentials", status="error", detail=str(e)[:120])
                self._log(state, f"ComfyUI not reachable at {self.settings.comfy_base_url}: {e}", log)
                self._finalize_job_clock(state, status="error")
                return state

            self._check_cancel()
            self.director = DirectorAgent(self.settings, log=lambda m: self._log(state, m, log))
            self.critic = CriticAgent(self.settings, log=lambda m: self._log(state, m, log))
            self.voice = VoiceAgent(self.settings, log=lambda m: self._log(state, m, log))

            if req.narrative_mode:
                state.narrative_mode = normalize_narrative_mode(req.narrative_mode)
            elif state.plan and getattr(state.plan, "narrative_mode", None):
                state.narrative_mode = normalize_narrative_mode(state.plan.narrative_mode)
            narr_mode = normalize_narrative_mode(state.narrative_mode)

            mode = (req.h3_mode or state.h3_mode or self.settings.h3_mode or "r2v").lower()
            if mode not in ("r2v", "t2v", "auto"):
                mode = "r2v"
            if (
                narr_mode
                in (NarrativeMode.documentary.value, NarrativeMode.explainer.value)
                and req.h3_mode is None
                and not state.h3_mode
            ):
                mode = "t2v"
            state.h3_mode = mode
            if state.plan:
                state.plan.narrative_mode = narr_mode
                state.plan = normalize_plan_cast_presence(state.plan)
            if state.plan:
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
                on_character_done=lambda _c: self._sync_character_stage_rows(state),
            )
            need_sheets = (
                narr_mode == NarrativeMode.character.value
                and mode in ("r2v", "auto")
                and bool(plan and plan.characters)
            )
            if need_sheets:
                self._stage_start(state, "character_sheets", "Character sheets (refresh)")
                self._log(state, "Refreshing character sheets (reuse existing stills)…", log)
                board.build(plan)
                self._sync_character_stage_rows(state)
                (root / "production.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                ready = sum(
                    1
                    for c in plan.characters or []
                    if (c.sheet_status or "") == "ready" or c.image_path
                )
                self._stage_end(
                    state,
                    "character_sheets",
                    detail=f"{ready}/{len(plan.characters or [])} cast",
                )
            else:
                self._stage_skip(
                    state,
                    "character_sheets",
                    "Character sheets",
                    detail=f"skipped ({narr_mode} / H3={mode})",
                )

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
            self._finalize_job_clock(state, status=state.status)

        except CancelledError:
            state.status = "cancelled"
            try:
                self.comfy.interrupt()
            except Exception:
                pass
            self._log(state, "Generation stopped by user.", log)
            self._finalize_job_clock(state, status="cancelled")
        except Exception as e:
            state.status = "error"
            self._log(state, f"Pipeline error: {e}", log)
            self._log(state, traceback.format_exc()[-1500:], log)
            self._finalize_job_clock(state, status="error")

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

        remaining = [
            r
            for r in state.shots
            if not (
                r.status == ShotStatus.passed
                and r.final_video
                and Path(r.final_video).exists()
            )
            and r.status != ShotStatus.failed
        ]
        if remaining:
            self._stage_start(
                state,
                "shots",
                "Shot generation + critic",
                detail=f"{len(remaining)} shot(s) remaining",
            )
        else:
            self._stage_skip(
                state,
                "shots",
                "Shot generation + critic",
                detail="all shots already passed",
            )

        aborted_on_shot_fail = False
        abort_shot_id = ""
        for idx, record in enumerate(state.shots):
            self._check_cancel()
            self._stage_bump_running(state)
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

            # Exhausting retakes (or hard gen failure) ends the whole job —
            # later shots cannot recover continuity / product quality.
            if record.status == ShotStatus.failed:
                aborted_on_shot_fail = True
                abort_shot_id = record.plan.id if record.plan else f"shot{idx}"
                err = (record.error or "shot failed").strip()
                self._log(
                    state,
                    f"Aborting job: {abort_shot_id} failed after all retakes "
                    f"({max_retakes + 1} take budget) — {err[:200]}",
                    log,
                )
                skipped = 0
                for later in state.shots[idx + 1 :]:
                    if later.status in (ShotStatus.passed, ShotStatus.failed):
                        continue
                    if (
                        later.status == ShotStatus.passed
                        and later.final_video
                        and Path(later.final_video).exists()
                    ):
                        continue
                    later.status = ShotStatus.skipped
                    later.error = f"Aborted: {abort_shot_id} failed"
                    skipped += 1
                if skipped:
                    self._log(
                        state,
                        f"Skipped {skipped} remaining shot(s) after {abort_shot_id} failed.",
                        log,
                    )
                break

        if remaining:
            passed_n = sum(
                1
                for r in state.shots
                if r.status == ShotStatus.passed and r.final_video
            )
            self._stage_end(
                state,
                "shots",
                status="error" if aborted_on_shot_fail else "done",
                detail=(
                    f"aborted on {abort_shot_id} · {passed_n}/{len(state.shots)} passed"
                    if aborted_on_shot_fail
                    else f"{passed_n}/{len(state.shots)} passed"
                ),
            )

        if aborted_on_shot_fail:
            state.status = "failed"
            self._stage_skip(
                state,
                "assemble",
                "Assemble master",
                detail=f"aborted: {abort_shot_id} failed",
            )
            self._stage_skip(
                state,
                "narration",
                "ElevenLabs narration",
                detail="aborted after shot failure",
            )
            self._log(
                state,
                f"Job failed — {abort_shot_id} exhausted retakes; no further shots or master.",
                log,
            )
            if not self.voice.enabled:
                self._log(state, "Voice/narration skipped (ENABLE_VOICE=false).", log)
            return

        self._check_cancel()
        passed = [r for r in state.shots if r.status == ShotStatus.passed and r.final_video]
        narr_mode = normalize_narrative_mode(
            state.narrative_mode or getattr(plan, "narrative_mode", None) or "character"
        )
        add_cards = (
            narr_mode == NarrativeMode.character.value
            and float(plan.target_duration_sec or 0) >= 40
        )
        if auto_assemble and passed:
            state.status = "assembling"
            self._stage_start(state, "assemble", "Assemble master")
            self._log(state, f"Assembling master from {len(passed)} passed shots…", log)
            clips = [Path(r.final_video) for r in passed if r.final_video]
            master = self._project_dir(state.project_id) / "master" / f"{_safe(plan.title)}.mp4"
            assemble_master(
                self.settings,
                clips,
                master,
                title=plan.title,
                subtitle=plan.logline[:80] if plan.logline else state.style[:60],
                add_cards=add_cards,
            )
            state.master_path = str(master)
            self._log(state, f"Master ready: {master}", log)
            self._stage_end(state, "assemble", detail=Path(master).name)

            # Documentary-style narration under the master (all modes when voice enabled)
            if self.voice.enabled and plan:
                try:
                    self._stage_start(state, "narration", "ElevenLabs narration")
                    self.voice = VoiceAgent(
                        self.settings, log=lambda m: self._log(state, m, log)
                    )
                    narr_path = (
                        self._project_dir(state.project_id) / "master" / "narration.mp3"
                    )
                    audio, script = self.voice.narrate_plan(plan, narr_path)
                    if audio and audio.exists():
                        state.narration_path = str(audio)
                        vo_master = (
                            self._project_dir(state.project_id)
                            / "master"
                            / f"{_safe(plan.title)}_narrated.mp4"
                        )
                        mux_narration(self.settings, master, audio, vo_master)
                        state.master_path = str(vo_master)
                        self._log(
                            state,
                            f"Narration mixed: {vo_master.name} "
                            f"(script ~{len(script)} chars)",
                            log,
                        )
                        self._stage_end(
                            state,
                            "narration",
                            detail=vo_master.name,
                        )
                    else:
                        self._stage_end(
                            state,
                            "narration",
                            status="skipped",
                            detail="no audio synthesized",
                        )
                except Exception as exc:  # noqa: BLE001
                    self._log(state, f"Narration failed (master kept without VO): {exc}", log)
                    self._stage_end(
                        state,
                        "narration",
                        status="error",
                        detail=str(exc)[:120],
                    )
            else:
                self._stage_skip(
                    state,
                    "narration",
                    "ElevenLabs narration",
                    detail="voice disabled or no plan",
                )

            state.status = "completed"
        elif not passed:
            state.status = "failed"
            self._stage_skip(state, "assemble", "Assemble master", detail="no passed shots")
            self._log(state, "No shots passed critic — no master assembled.", log)
        else:
            state.status = "completed_no_assemble"
            self._stage_skip(state, "assemble", "Assemble master", detail="auto_assemble=false")

        if not self.voice.enabled:
            self._log(state, "Voice/narration skipped (ENABLE_VOICE=false).", log)

    def _finish_shot_timing(
        self,
        state: ProjectState,
        record: ShotRecord,
        shot_key: str,
        *,
        status: str = "done",
        detail: str = "",
    ) -> None:
        record.finished_at = _iso_now()
        record.duration_sec = _elapsed_sec(record.started_at, record.finished_at)
        det = detail
        if record.duration_sec is not None:
            det = (det + f" · {record.duration_sec:.1f}s").strip(" ·")
        self._stage_end(state, shot_key, status=status, detail=det)

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

        shot_key = f"shot:{shot.id}"
        record.started_at = _iso_now()
        record.finished_at = None
        record.duration_sec = None
        self._stage_start(
            state,
            shot_key,
            f"Shot {shot.id} — {shot.name}",
            detail=f"up to {max_retakes + 1} take(s)",
            silent=True,
        )
        state.status = "running"

        for take in range(1, max_retakes + 2):  # initial + retakes
            if self.control:
                self.control.check()
            record.status = ShotStatus.generating
            self._stage_bump_running(state)
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

            # --- Pre-clip still gate (cheap) → full H3 only after still PASS or soft skip ---
            first_frame: Path | None = None
            if self.settings.preclip_still_enabled and (
                (self.settings.preclip_still_mode or "auto").lower()
                not in ("none", "off", "disabled")
            ):
                still_ok, critic_notes, visual_override, first_frame = self._preclip_still_gate(
                    state=state,
                    record=record,
                    plan=plan,
                    shot=shot,
                    take=take,
                    seed=seed,
                    take_dir=take_dir,
                    gen_mode=gen_mode,
                    ref_paths=ref_paths,
                    picture_map=picture_map,
                    picture_meta=picture_meta,
                    extra_notes=extra_notes,
                    critic_notes=critic_notes,
                    visual_override=visual_override,
                    log=log,
                )
                shot_for_prompt = shot.model_copy(update={"visual_prompt": visual_override})
                render_prompt = self.director.build_render_prompt(
                    plan,
                    shot_for_prompt,
                    critic_notes,
                    r2v=(gen_mode == "r2v"),
                    picture_map=picture_map,
                    picture_meta=picture_meta,
                    extra_picture_notes=extra_notes,
                )
                if not still_ok and self.settings.preclip_require_pass:
                    # Skip expensive H3; treat as video retake note carry-forward
                    self._log(
                        state,
                        f"{shot.id} take {take}: preclip still never PASSED — "
                        f"skipping full H3 (PRECLIP_REQUIRE_PASS=true)",
                        log,
                    )
                    if take > max_retakes:
                        record.status = ShotStatus.failed
                        record.error = critic_notes or "Preclip still failed all attempts"
                        self._finish_shot_timing(
                            state,
                            record,
                            shot_key,
                            status="error",
                            detail="preclip still failed",
                        )
                        return accepted_frame
                    record.status = ShotStatus.retake
                    continue

            self._log(
                state,
                f"{shot.id} take {take}: H3 {gen_mode.upper()} "
                f"(seed={seed}, refs={len(ref_paths)} [{ref_labels}])…",
                log,
            )

            prefix = f"video/H3VideoGen/{state.project_id}/{shot.id}_t{take}"
            use_ff = (
                first_frame
                if (
                    first_frame
                    and self.settings.preclip_use_as_first_frame
                    and gen_mode == "t2v"
                )
                else None
            )
            try:
                src, prompt_id, used_mode = self.comfy.generate(
                    render_prompt,
                    length=shot.length_frames,
                    seed=seed,
                    filename_prefix=prefix,
                    mode=gen_mode,
                    ref_image_paths=ref_paths,
                    first_frame_path=use_ff,
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
                        t2v_ff = (
                            first_frame
                            if (
                                first_frame
                                and self.settings.preclip_use_as_first_frame
                            )
                            else None
                        )
                        src, prompt_id, used_mode = self.comfy.generate(
                            t2v_prompt,
                            length=shot.length_frames,
                            seed=seed,
                            filename_prefix=prefix + "_t2v",
                            mode="t2v",
                            first_frame_path=t2v_ff,
                            project_tag=state.project_id,
                        )
                        render_prompt = t2v_prompt
                    except CancelledError:
                        raise
                    except ComfyError as e2:
                        record.status = ShotStatus.failed
                        record.error = str(e2)
                        self._log(state, f"{shot.id} generation failed: {e2}", log)
                        self._finish_shot_timing(
                            state, record, shot_key, status="error", detail=str(e2)[:80]
                        )
                        return accepted_frame
                else:
                    record.status = ShotStatus.failed
                    record.error = str(e)
                    self._log(state, f"{shot.id} generation failed: {e}", log)
                    self._finish_shot_timing(
                        state, record, shot_key, status="error", detail=str(e)[:80]
                    )
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
                self._finish_shot_timing(
                    state,
                    record,
                    shot_key,
                    detail=f"PASSED take {take}",
                )
                return accepted_frame

            if take > max_retakes or (review.verdict == CriticVerdict.reject and take > max_retakes):
                best = max(record.takes, key=lambda t: t.get("score") or 0, default=None)
                if best and (best.get("score") or 0) >= self.settings.critic_pass_threshold - 1.0:
                    final = root / "shots" / f"{shot.id}_{_safe(shot.name)}_best.mp4"
                    shutil.copy2(best["video"], final)
                    record.final_video = str(final)
                    record.final_frame = best.get("frame")
                    record.status = ShotStatus.passed
                    accepted_frame = Path(best["frame"]) if best.get("frame") else None
                    self._log(
                        state,
                        f"{shot.id} kept best take (score={best.get('score')}) after retakes",
                        log,
                    )
                    self._finish_shot_timing(
                        state,
                        record,
                        shot_key,
                        detail=f"best-of after {take} takes",
                    )
                    return accepted_frame
                record.status = ShotStatus.failed
                record.error = review.retake_instructions or review.summary
                self._log(state, f"{shot.id} FAILED after {take} take(s)", log)
                self._finish_shot_timing(
                    state,
                    record,
                    shot_key,
                    status="error",
                    detail=f"failed after {take} take(s)",
                )
                return accepted_frame

            critic_notes = review.retake_instructions or review.summary
            visual_override = review.revised_prompt or visual_override
            record.status = ShotStatus.retake
            self._log(
                state,
                f"{shot.id} RETAKE ordered: {critic_notes[:200]}",
                log,
            )

        self._finish_shot_timing(
            state, record, shot_key, status="error", detail="exhausted takes"
        )
        return accepted_frame

    def _preclip_still_gate(
        self,
        *,
        state: ProjectState,
        record: ShotRecord,
        plan: ProductionPlan,
        shot: ShotPlan,
        take: int,
        seed: int,
        take_dir: Path,
        gen_mode: str,
        ref_paths: list[Path],
        picture_map: dict,
        picture_meta: list,
        extra_notes: list[str],
        critic_notes: str,
        visual_override: str,
        log: LogFn | None,
    ) -> tuple[bool, str, str, Path | None]:
        """
        Generate + critic preview stills before full H3.

        Returns (passed_or_soft_ok, critic_notes, visual_override, approved_still_path).
        Soft-ok: stills failed/unavailable but PRECLIP_REQUIRE_PASS is false → proceed to video.
        """
        max_still = int(self.settings.preclip_max_retakes)
        notes = critic_notes
        visual = visual_override
        best_still: Path | None = None
        best_score = -1.0
        root = self._project_dir(state.project_id)
        reviews_dir = root / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        any_still = False

        for s_try in range(1, max_still + 2):
            if self.control:
                self.control.check()
            shot_for_prompt = shot.model_copy(update={"visual_prompt": visual})
            render_prompt = self.director.build_render_prompt(
                plan,
                shot_for_prompt,
                notes,
                r2v=(gen_mode == "r2v"),
                picture_map=picture_map,
                picture_meta=picture_meta,
                extra_picture_notes=extra_notes,
            )
            out_stem = take_dir / f"preview_t{take}_s{s_try}"
            self._log(
                state,
                f"{shot.id} take {take}: preclip still {s_try}/{max_still + 1}…",
                log,
            )
            still = generate_scene_still(
                self.settings,
                plan=plan,
                shot=shot_for_prompt,
                out_path=out_stem,
                render_prompt=render_prompt,
                critic_notes=notes,
                seed=seed + s_try * 11,
                comfy=self.comfy,
                gen_mode=gen_mode,
                ref_image_paths=ref_paths,
                project_tag=state.project_id,
                log=lambda m: self._log(state, m, log),
            )
            if not still or not still.exists():
                self._log(
                    state,
                    f"{shot.id} take {take}: preclip still {s_try} unavailable — "
                    f"{'continue soft' if not self.settings.preclip_require_pass else 'no image'}",
                    log,
                )
                # No image: stop trying more stills in this cycle
                break

            any_still = True
            pre_review = self.critic.review(
                plan,
                shot_for_prompt,
                [still],
                take=take,
                video_meta={
                    "phase": "preclip_still",
                    "still_try": s_try,
                    "still_path": str(still),
                },
                render_prompt_used=render_prompt,
            )
            record.reviews.append(pre_review)
            (reviews_dir / f"{shot.id}_t{take}_preclip_s{s_try}.json").write_text(
                pre_review.model_dump_json(indent=2), encoding="utf-8"
            )
            self._log(
                state,
                f"{shot.id} preclip critic ({self.critic.last_provider or '?'}): "
                f"{pre_review.verdict.value} score={pre_review.overall_score} — "
                f"{pre_review.summary[:140]}",
                log,
            )
            if (pre_review.overall_score or 0) >= best_score:
                best_score = pre_review.overall_score or 0
                best_still = still

            if pre_review.verdict == CriticVerdict.pass_:
                if pre_review.revised_prompt:
                    visual = pre_review.revised_prompt
                if (pre_review.retake_instructions or "").strip():
                    notes = pre_review.retake_instructions.strip()
                self._log(
                    state,
                    f"{shot.id} take {take}: preclip still PASSED — proceeding to full H3",
                    log,
                )
                return True, notes, visual, best_still

            notes = pre_review.retake_instructions or pre_review.summary or notes
            if pre_review.revised_prompt:
                visual = pre_review.revised_prompt
            if s_try <= max_still:
                self._log(
                    state,
                    f"{shot.id} preclip RETAKE: {notes[:180]}",
                    log,
                )

        # Exhausted or no image
        if any_still and best_still:
            self._log(
                state,
                f"{shot.id} take {take}: preclip stills exhausted "
                f"(best_score={best_score}); "
                f"{'blocking H3' if self.settings.preclip_require_pass else 'proceeding to H3 soft'}",
                log,
            )
            return (not self.settings.preclip_require_pass), notes, visual, best_still

        # Generator unavailable — never block video when we couldn't even make a still
        self._log(
            state,
            f"{shot.id} take {take}: preclip skipped (no still generated) — full H3 as usual",
            log,
        )
        return True, notes, visual, None

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
