"""FastAPI app: UI + API for H3VideoGen."""
from __future__ import annotations

import asyncio
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .comfy_h3 import ComfyH3Client
from .config import Settings, get_settings
from .job_control import (
    CancelledError,
    JobControl,
    bind_job_control,
    reset_job_control,
)
from .models import DirectorOnlyRequest, GenerateRequest, ResumeRequest
from .pipeline import ProductionPipeline, list_projects, load_state, _elapsed_sec
from .queue_store import (
    durable_item,
    load_queue_document,
    save_queue_document,
)
from .services import (
    comfy_reachable,
    essentials_report,
    heal_runtime_services,
    ollama_reachable,
)
from .style_library import get_style, styles_for_api, build_style_prompt

ROOT = Path(__file__).resolve().parent.parent
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Install missing AI_ROOT prereqs, restore queue; flush queue on shutdown."""
    try:
        from .prereq_install import start_bootstrap_background

        start_bootstrap_background(settings)
    except Exception:
        traceback.print_exc()
    try:
        await asyncio.to_thread(_restore_queue_from_disk)
    except Exception:
        traceback.print_exc()
    yield
    try:
        await asyncio.to_thread(_persist_queue_safe)
    except Exception:
        traceback.print_exc()


app = FastAPI(title="H3 Video Gen", version="0.1.0", lifespan=lifespan)
templates = Jinja2Templates(directory=str(ROOT / "web" / "templates"))

# project jobs: job_key / project_id -> status info
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
# FIFO of pending work items (not yet assigned a worker)
_queue: list[dict[str, Any]] = []
# job_key -> live worker thread
_workers: dict[str, threading.Thread] = {}
# job_key -> JobControl for that worker
_controls: dict[str, JobControl] = {}
# Runtime override for parallel slots (None → settings.max_parallel_jobs)
_parallel_override: int | None = None
# While restoring, avoid thrashing disk writes
_restoring_queue = False
# One-time bootstrap log lines surfaced into jobs UI
_restore_notes: list[str] = []

static_dir = ROOT / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class ParallelJobsBody(BaseModel):
    max_parallel_jobs: int = Field(ge=1, le=8)


class StopBody(BaseModel):
    """Stop a specific job or all live workers; optionally clear the queue."""

    job_ref: str | None = None
    project_id: str | None = None
    stop_all: bool = False
    clear_queue: bool = False


def _effective_parallel() -> int:
    with _lock:
        n = _parallel_override if _parallel_override is not None else settings.max_parallel_jobs
    return max(1, min(8, int(n or 1)))


def _any_worker_alive() -> bool:
    with _lock:
        for t in _workers.values():
            if t is not None and t.is_alive():
                return True
        return False


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "enable_voice": settings.enable_voice,
            "comfy_url": settings.comfy_base_url,
            "critic_threshold": settings.critic_pass_threshold,
            "preclip_still_enabled": settings.preclip_still_enabled,
            "max_parallel_jobs": _effective_parallel(),
        },
    )


@app.get("/api/health")
async def health():
    """
    Status + essentials. Runs off the event loop so long Comfy probes
    cannot freeze queue/jobs UI while a generation is using Comfy.
    """
    return await asyncio.to_thread(_health_snapshot)


def _health_snapshot() -> dict[str, Any]:
    comfy_ok = comfy_reachable(settings, timeout=2.0)
    comfy_info: Any = None
    h3_info: Any = None
    err = None
    if comfy_ok:
        try:
            # Short probe; h3 capability is cached to avoid stampeding object_info
            from .services import comfy_h3_status

            client = ComfyH3Client(settings)
            try:
                comfy_info = client.health()
            except Exception as e:
                comfy_info = {"error": str(e)}
            h3_info = comfy_h3_status(settings, timeout=8.0, use_cache=True)
            if settings.comfy_require_h3_nodes and not h3_info.get("ok"):
                miss = ", ".join(h3_info.get("missing_nodes") or [])
                err = f"H3 nodes missing ({miss})"
                comfy_ok = False
        except Exception as e:
            err = str(e)
            comfy_ok = False
    else:
        err = f"Unreachable at {settings.comfy_base_url}"

    from .llm import LLMRouter

    try:
        llm_status = LLMRouter(settings).status()
    except Exception as e:
        llm_status = {"error": str(e)}
    try:
        essentials = essentials_report(settings)
    except Exception as e:
        essentials = {
            "ready_for_generate": False,
            "blocking": [f"Health check error: {e}"],
            "services": [],
            "wait_limit_sec": settings.essentials_wait_sec,
        }
    with _lock:
        running = any(_is_running_status(v.get("status")) for v in _jobs.values())
        cancel_any = any(c.is_cancelled() for c in _controls.values())
        workers_alive = sum(1 for t in _workers.values() if t and t.is_alive())
        qdepth = len(_queue)
    from datetime import datetime, timezone

    checked_at = datetime.now(timezone.utc).isoformat()
    return {
        "gemini_key_set": bool(settings.gemini_api_key),
        "voice_enabled": settings.enable_voice,
        "comfy_ok": comfy_ok,
        "comfy_url": settings.comfy_base_url,
        "comfy_error": err,
        "comfy_info": comfy_info,
        "comfy_h3": h3_info,
        "auto_start_comfy": settings.auto_start_comfy,
        "comfy_replace_non_h3": settings.comfy_replace_non_h3,
        "auto_start_ollama": settings.auto_start_ollama,
        "ollama_ok": ollama_reachable(settings),
        "llm": llm_status,
        "job_running": running,
        "cancel_requested": cancel_any,
        "max_parallel_jobs": _effective_parallel(),
        "workers_alive": workers_alive,
        "queue_depth": qdepth,
        "essentials": essentials,
        "bootstrap": _bootstrap_snapshot(),
        "heartbeat": {
            "interval_sec": 10,
            "checked_at": checked_at,
            "ready_for_generate": (essentials or {}).get("ready_for_generate", False),
            "tools": (essentials or {}).get("services") or [],
        },
    }


def _bootstrap_snapshot() -> dict[str, Any]:
    try:
        from .prereq_install import bootstrap_status, scan_prereqs

        st = bootstrap_status()
        if not st.get("done") and not st.get("running"):
            # Idle — attach a lightweight scan for UI
            try:
                scan = scan_prereqs(settings)
                st = {**st, "scan": scan.to_dict()}
            except Exception as e:
                st = {**st, "scan_error": str(e)}
        return st
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/bootstrap")
async def bootstrap_get():
    """Prerequisite scan / install progress under AI_ROOT."""
    return await asyncio.to_thread(_bootstrap_snapshot)


@app.post("/api/bootstrap")
async def bootstrap_run(force: bool = False):
    """Run (or re-run) AI_ROOT prerequisite install on a background thread."""
    from .prereq_install import ensure_all_prereqs, start_bootstrap_background, bootstrap_status

    if force:
        def _run() -> dict[str, Any]:
            return ensure_all_prereqs(settings, force=True).to_dict()

        # Force is blocking in a worker thread so CLI-like install can complete
        report = await asyncio.to_thread(_run)
        return {"ok": report.get("ok"), "report": report, "status": bootstrap_status()}
    started = start_bootstrap_background(settings)
    return {"ok": True, "started": started, "status": bootstrap_status()}


@app.post("/api/services/ensure")
async def services_ensure():
    """Start ComfyUI / Ollama if configured and not already running (self-heal)."""
    lines: list[str] = []

    def log(m: str) -> None:
        lines.append(m)

    def _run() -> dict[str, Any]:
        return heal_runtime_services(
            settings,
            log=log,
            need_comfy=True,
            need_ollama=True,
            reason="api_ensure",
        )

    report = await asyncio.to_thread(_run)
    essentials = await asyncio.to_thread(essentials_report, settings)
    comfy_rep = report.get("comfy") or {}
    return {
        "ok": bool(comfy_rep.get("ok")),
        "report": report,
        "log": lines,
        "comfy_ok": bool(comfy_rep.get("ok")),
        "ollama_ok": await asyncio.to_thread(ollama_reachable, settings),
        "essentials": essentials,
        "self_healed": True,
    }


@app.post("/api/plan")
async def plan_only(req: DirectorOnlyRequest):
    try:
        plan = ProductionPipeline(settings).plan_only(
            req.prompt,
            req.style,
            req.target_duration_sec,
            req.max_shots,
            narrative_mode=req.narrative_mode,
        )
        return plan.model_dump()
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/api/styles")
async def list_style_library():
    """Style reference library with composed prompts + thumbnail URLs."""
    try:
        return styles_for_api()
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.get("/api/styles/{slug}")
async def style_detail(slug: str):
    st = get_style(slug)
    if not st:
        raise HTTPException(404, f"Style not found: {slug}")
    from .style_library import _slugify as style_slugify

    sslug = style_slugify(str(st.get("slug") or st.get("name") or slug))
    return {
        "slug": sslug,
        "name": st.get("name"),
        "category": st.get("category"),
        "description": st.get("description"),
        "style_prompt": build_style_prompt(st),
        "sample_prompt": st.get("sample_prompt"),
        "thumb_url": f"/static/style_thumbs/{sslug}.jpg",
        "raw": st,
    }


def _thumb_rel_path(project_id: str, image_path: str | None, settings: Settings) -> str | None:
    """Return project-relative path under character_board/ for UI thumbs."""
    if not image_path:
        return None
    p = Path(image_path)
    root = (settings.output_root / project_id).resolve()
    try:
        resolved = p.resolve()
        if str(resolved).startswith(str(root)):
            rel = resolved.relative_to(root).as_posix()
            return rel
    except Exception:
        pass
    name = p.name
    if name:
        return f"character_board/{name}"
    return None


def _job_snapshot(state_or_dict: Any, *, status: str | None = None) -> dict[str, Any]:
    """Normalize a job dict for the UI (logs from memory or disk)."""
    if hasattr(state_or_dict, "model_dump"):
        s = state_or_dict
        for row in s.stage_timings or []:
            if (row.status or "").lower() == "running" and row.started_at:
                row.duration_sec = _elapsed_sec(row.started_at, None)
        if (s.status or "").lower() not in (
            "completed",
            "completed_no_assemble",
            "failed",
            "error",
            "cancelled",
        ):
            start = s.job_started_at or s.created_at
            for row in s.stage_timings or []:
                if row.key == "total" and (row.status or "").lower() == "running" and start:
                    row.duration_sec = _elapsed_sec(start, None)
        chars = []
        if s.plan and s.plan.characters:
            for c in s.plan.characters:
                primary = c.image_path
                if not primary:
                    for pose in c.sheet or []:
                        if pose.image_path:
                            primary = pose.image_path
                            break
                thumb = _thumb_rel_path(s.project_id, primary, settings)
                sheet_thumbs = []
                for pose in c.sheet or []:
                    if not pose.image_path:
                        continue
                    rel = _thumb_rel_path(s.project_id, pose.image_path, settings)
                    if rel:
                        sheet_thumbs.append(
                            {
                                "pose_id": pose.pose_id,
                                "label": pose.label or pose.pose_id,
                                "path": rel,
                            }
                        )
                chars.append(
                    {
                        "id": c.id,
                        "name": c.name,
                        "look": (c.look or "")[:220],
                        "image_path": primary,
                        "thumb_path": thumb,
                        "sheet_count": sum(1 for p in (c.sheet or []) if p.image_path),
                        "sheet_status": c.sheet_status,
                        "sheet_duration_sec": c.sheet_duration_sec,
                        "sheet_source": c.sheet_source,
                        "sheet_started_at": c.sheet_started_at,
                        "sheet_finished_at": c.sheet_finished_at,
                        "sheet_thumbs": sheet_thumbs,
                    }
                )
        stages = []
        for row in s.stage_timings or []:
            stages.append(
                {
                    "key": row.key,
                    "label": row.label,
                    "started_at": row.started_at,
                    "ended_at": row.ended_at,
                    "duration_sec": row.duration_sec,
                    "status": row.status,
                    "detail": row.detail,
                }
            )
        shot_rows = []
        for rec in s.shots or []:
            shot_rows.append(
                {
                    "id": rec.plan.id if rec.plan else "?",
                    "name": rec.plan.name if rec.plan else "",
                    "status": rec.status.value if hasattr(rec.status, "value") else rec.status,
                    "started_at": rec.started_at,
                    "finished_at": rec.finished_at,
                    "duration_sec": rec.duration_sec,
                    "takes": len(rec.takes or []),
                }
            )
        return {
            "status": status or s.status,
            "project_id": s.project_id,
            "log": list(s.log),
            "last_message": s.log[-1] if s.log else "",
            "master_path": s.master_path,
            "title": s.plan.title if s.plan else None,
            "characters": chars,
            "stage_timings": stages,
            "shot_timings": shot_rows,
            "job_started_at": s.job_started_at or s.created_at,
            "job_finished_at": s.job_finished_at,
            "created_at": s.created_at,
            "temp": False,
        }
    return dict(state_or_dict)


def _merge_state_into_job(
    j: dict[str, Any], project_id: str, *, mem_logs: list[str] | None = None
) -> None:
    """Refresh job dict fields from on-disk state (cast, timings, stage table)."""
    try:
        st = load_state(project_id, settings)
        if not st:
            return
        snap = _job_snapshot(st, status=j.get("status") or st.status)
        for k in (
            "characters",
            "stage_timings",
            "shot_timings",
            "title",
            "job_started_at",
            "job_finished_at",
            "created_at",
            "master_path",
        ):
            if snap.get(k) is not None:
                j[k] = snap[k]
        disk_log = snap.get("log") or []
        if mem_logs is not None and len(mem_logs) >= len(disk_log):
            j["log"] = mem_logs[-500:]
        elif disk_log:
            j["log"] = list(disk_log)[-500:]
    except Exception:
        pass


def _set_job(key: str, data: dict[str, Any], *aliases: str) -> None:
    with _lock:
        data.setdefault("job_key", key)
        # Preserve durable fields across snapshot overwrites
        prev = _jobs.get(key) or {}
        for field in (
            "kind",
            "generate_payload",
            "resume_payload",
            "enqueued_at",
            "label",
            "prompt_preview",
            "restored",
        ):
            if data.get(field) is None and prev.get(field) is not None:
                data[field] = prev.get(field)
        _jobs[key] = data
        for a in aliases:
            if a and a != key:
                _jobs[a] = data
    _persist_queue_safe()


def _is_running_status(status: str | None) -> bool:
    return (status or "").lower() in {
        "running",
        "planning",
        "assembling",
        "generating",
        "reviewing",
    }


def _is_active_status(status: str | None) -> bool:
    s = (status or "").lower()
    return s in {
        "queued",
        "running",
        "planning",
        "assembling",
        "generating",
        "reviewing",
        "cancelling",
        "interrupted",
    }


def _is_terminal_job_status(status: str | None) -> bool:
    return (status or "").lower() in {
        "completed",
        "completed_no_assemble",
        "cancelled",
        "error",
        "failed",
    }


def _is_incomplete_project_status(status: str | None) -> bool:
    s = (status or "").lower()
    return s in {
        "running",
        "planning",
        "assembling",
        "generating",
        "reviewing",
        "retake",
        "cancelling",
        "interrupted",
        "created",
    }


def _resume_payload_from_generate(gen: dict[str, Any] | None) -> dict[str, Any]:
    gen = gen or {}
    return {
        "max_retakes": gen.get("max_retakes"),
        "auto_assemble": gen.get("auto_assemble", True),
        "seed_base": gen.get("seed_base", 42),
        "h3_mode": gen.get("h3_mode"),
        "narrative_mode": gen.get("narrative_mode"),
        "redo_failed": True,
    }


def _job_to_durable(
    j: dict[str, Any],
    job_key: str,
    *,
    status: str,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    fb = fallback or {}
    kind = j.get("kind") or fb.get("kind") or "generate"
    gen = (
        j.get("generate_payload")
        if j.get("generate_payload") is not None
        else fb.get("generate_payload")
    )
    res = (
        j.get("resume_payload")
        if j.get("resume_payload") is not None
        else fb.get("resume_payload")
    )
    pid = j.get("project_id") or fb.get("project_id")
    if kind == "resume" and res is None:
        res = {}
    if kind == "generate" and not gen and not (
        pid and not str(pid).startswith(("pending_", "resume_"))
    ):
        return None
    if kind == "resume" and not pid:
        return None
    return durable_item(
        job_key=job_key,
        kind=str(kind),
        label=j.get("label") or fb.get("label") or "",
        project_id=str(pid) if pid else None,
        prompt_preview=j.get("prompt_preview") or fb.get("prompt_preview") or "",
        title=j.get("title") or fb.get("title"),
        status=status,
        generate_payload=gen,
        resume_payload=res,
        enqueued_at=j.get("enqueued_at") or fb.get("enqueued_at"),
    )


def _build_persist_document_unlocked() -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for jk, t in list(_workers.items()):
        if not t or not t.is_alive():
            continue
        j = _jobs.get(jk) or {}
        d = _job_to_durable(j, jk, status="running")
        if d:
            items.append(d)
            seen.add(jk)
    for q in _queue:
        jk = q.get("job_key")
        if not jk or jk in seen:
            continue
        j = _jobs.get(jk) or {}
        d = _job_to_durable(j, jk, status="queued", fallback=q)
        if d:
            items.append(d)
            seen.add(jk)
    return {
        "version": 1,
        "items": items,
        "parallel_override": _parallel_override,
    }


def _persist_queue_safe() -> None:
    if not settings.queue_persist or _restoring_queue:
        return
    try:
        with _lock:
            doc = _build_persist_document_unlocked()
        save_queue_document(settings, doc)
    except Exception:
        traceback.print_exc()


def _make_task_from_durable(
    item: dict[str, Any],
) -> tuple[Callable[[JobControl], None], str, dict[str, Any]]:
    job_key = str(item.get("job_key") or f"restored_{uuid.uuid4().hex[:12]}")
    kind = (item.get("kind") or "generate").lower()
    pid = item.get("project_id")
    gen_raw = item.get("generate_payload")
    res_raw = item.get("resume_payload") or {}
    label = item.get("label") or "Restored job…"
    prompt_preview = item.get("prompt_preview") or ""
    title = item.get("title")

    if pid and not str(pid).startswith(("pending_", "resume_")):
        st = load_state(str(pid), settings)
        if st and st.plan and st.plan.shots:
            kind = "resume"
            if not res_raw and isinstance(gen_raw, dict):
                res_raw = _resume_payload_from_generate(gen_raw)
            label = f"Resuming {pid} (restored)…"
            title = title or (st.plan.title if st.plan else None)
            prompt_preview = prompt_preview or (st.user_prompt or "")[:120]

    if kind == "resume":
        if not pid:
            raise ValueError("resume durable item missing project_id")
        req = ResumeRequest(**(res_raw or {}))

        def task(control: JobControl, _pid=str(pid), _req=req, _jk=job_key) -> None:
            _run_resume_job(_pid, _req, _jk, control)

        fields = {
            "kind": "resume",
            "generate_payload": gen_raw if isinstance(gen_raw, dict) else None,
            "resume_payload": req.model_dump(),
            "project_id": str(pid),
            "label": label,
            "prompt_preview": prompt_preview,
            "title": title,
            "restored": True,
        }
        return task, job_key, fields

    if not isinstance(gen_raw, dict):
        raise ValueError("generate durable item missing generate_payload")
    gen_req = GenerateRequest(**gen_raw)

    def task(control: JobControl, _req=gen_req, _jk=job_key) -> None:
        _run_job(_req, _jk, control)

    fields = {
        "kind": "generate",
        "generate_payload": gen_req.model_dump(),
        "resume_payload": None,
        "project_id": str(pid) if pid and not str(pid).startswith("pending_") else None,
        "label": label or "Starting pipeline (restored)…",
        "prompt_preview": prompt_preview or gen_req.prompt[:120],
        "title": title,
        "restored": True,
    }
    return task, job_key, fields


def _skip_restored_item(item: dict[str, Any]) -> str | None:
    pid = item.get("project_id")
    if not pid or str(pid).startswith(("pending_", "resume_")):
        if item.get("kind") == "generate" and item.get("generate_payload"):
            return None
        if item.get("kind") == "resume":
            return "resume without project_id"
        return "no recoverable payload" if not item.get("generate_payload") else None

    st = load_state(str(pid), settings)
    if not st:
        if item.get("kind") == "generate" and item.get("generate_payload"):
            return None
        return f"project {pid} missing"
    status = (st.status or "").lower()
    if status in ("completed", "completed_no_assemble") and st.master_path:
        from pathlib import Path as _P

        if _P(str(st.master_path)).exists():
            return f"project {pid} already completed"
    if status == "cancelled":
        return f"project {pid} was cancelled"
    return None


def _restore_queue_from_disk() -> None:
    """Re-load job_queue.json and incomplete projects into the in-memory queue."""
    global _parallel_override, _restoring_queue, _restore_notes
    if not settings.queue_persist:
        return

    _restoring_queue = True
    notes: list[str] = []
    recovered = 0
    recovered_pids: set[str] = set()

    try:
        doc = load_queue_document(settings)
        if doc.get("parallel_override") is not None:
            try:
                _parallel_override = max(1, min(8, int(doc["parallel_override"])))
            except Exception:
                pass

        for item in list(doc.get("items") or []):
            if not isinstance(item, dict):
                continue
            reason = _skip_restored_item(item)
            if reason:
                notes.append(f"Skipped restored job: {reason}")
                continue
            try:
                task, job_key, fields = _make_task_from_durable(item)
            except Exception as exc:
                notes.append(f"Could not restore job {item.get('job_key')}: {exc}")
                continue
            pid = fields.get("project_id")
            try:
                _enqueue_job(
                    task,
                    job_key,
                    fields.get("label") or "Restored job…",
                    project_id=pid if pid else None,
                    prompt_preview=fields.get("prompt_preview") or "",
                    kind=str(fields.get("kind") or "generate"),
                    generate_payload=fields.get("generate_payload"),
                    resume_payload=fields.get("resume_payload"),
                    title=fields.get("title"),
                    persist=False,
                    restored=True,
                )
                recovered += 1
                if pid:
                    recovered_pids.add(str(pid))
            except HTTPException as he:
                notes.append(f"Skip restore {job_key}: {he.detail}")
            except Exception as exc:
                notes.append(f"Skip restore {job_key}: {exc}")

        if settings.queue_auto_resume_interrupted:
            for p in list_projects(settings):
                pid = p.get("project_id")
                if not pid or str(pid) in recovered_pids:
                    continue
                if not p.get("resumable"):
                    continue
                st_name = (p.get("status") or "").lower()
                if st_name == "cancelled":
                    continue
                if not _is_incomplete_project_status(st_name):
                    continue

                state = load_state(str(pid), settings)
                if not state or not state.plan or not state.plan.shots:
                    continue
                try:
                    if (state.status or "").lower() != "interrupted":
                        state.status = "interrupted"
                        state.log = list(state.log or []) + [
                            "Interrupted by process exit — re-queued on server restart."
                        ]
                        root = settings.output_root / str(pid)
                        (root / "state.json").write_text(
                            state.model_dump_json(indent=2), encoding="utf-8"
                        )
                except Exception:
                    pass

                temp_id = f"resume_{pid}_{uuid.uuid4().hex[:8]}"
                req = ResumeRequest(redo_failed=True)
                title = state.plan.title if state.plan else str(pid)

                def task(
                    control: JobControl, _pid=str(pid), _req=req, _jk=temp_id
                ) -> None:
                    _run_resume_job(_pid, _req, _jk, control)

                try:
                    _enqueue_job(
                        task,
                        temp_id,
                        f"Resuming {pid} (interrupted)…",
                        project_id=str(pid),
                        prompt_preview=title,
                        kind="resume",
                        resume_payload=req.model_dump(),
                        title=title,
                        persist=False,
                        restored=True,
                    )
                    recovered += 1
                    recovered_pids.add(str(pid))
                    notes.append(f"Re-queued interrupted project {pid}")
                except HTTPException:
                    pass
                except Exception as exc:
                    notes.append(f"Failed to re-queue {pid}: {exc}")

        if recovered:
            notes.insert(
                0, f"Restored {recovered} job(s) from previous session — resuming queue."
            )
        _restore_notes = notes[-40:]
    finally:
        _restoring_queue = False
        _persist_queue_safe()


def _prune_dead_workers() -> None:
    dead = [k for k, t in _workers.items() if not t or not t.is_alive()]
    for k in dead:
        _workers.pop(k, None)
        _controls.pop(k, None)


def _queue_positions() -> None:
    """Stamp queue_position on queued job dicts (1-based)."""
    for i, item in enumerate(_queue):
        jk = item.get("job_key")
        if not jk:
            continue
        j = _jobs.get(jk)
        if j and (j.get("status") or "").lower() == "queued":
            j["queue_position"] = i + 1
            j["last_message"] = f"Queued (#{i + 1})"


def _max_parallel_locked() -> int:
    n = _parallel_override if _parallel_override is not None else settings.max_parallel_jobs
    return max(1, min(8, int(n or 1)))


def _pump_queue() -> None:
    """Start workers from the queue until parallel slots are full."""
    with _lock:
        _prune_dead_workers()
        max_p = _max_parallel_locked()
        while len(_workers) < max_p and _queue:
            item = _queue.pop(0)
            job_key = item["job_key"]
            j = _jobs.get(job_key)
            if not j:
                continue
            if (j.get("status") or "").lower() in ("cancelled", "error", "failed"):
                continue
            control = JobControl()
            control.job_key = job_key
            control.reset()
            pid = j.get("project_id")
            if pid and not str(pid).startswith(("pending_", "resume_")):
                control.set_project_id(str(pid))
            _controls[job_key] = control
            j["status"] = "running"
            j["queue_position"] = None
            j["last_message"] = item.get("label") or "Starting…"
            logs = list(j.get("log") or [])
            logs.append(item.get("label") or "Starting…")
            j["log"] = logs[-500:]
            for k in ("kind", "generate_payload", "resume_payload", "enqueued_at"):
                if item.get(k) is not None and j.get(k) is None:
                    j[k] = item.get(k)
            _jobs[job_key] = j

            task_fn: Callable[[JobControl], None] = item["task"]

            def _runner(
                task=task_fn,
                control=control,
                key=job_key,
            ) -> None:
                token = bind_job_control(control)
                control.bind_thread(threading.current_thread())
                try:
                    task(control)
                except Exception as exc:  # noqa: BLE001
                    with _lock:
                        j = _jobs.get(key) or {}
                        st = (j.get("status") or "").lower()
                        if st in (
                            "",
                            "queued",
                            "running",
                            "planning",
                            "generating",
                            "reviewing",
                            "assembling",
                        ):
                            j["status"] = "error"
                            j["last_message"] = str(exc)
                            logs = list(j.get("log") or [])
                            logs.append(f"Worker crashed: {exc}")
                            logs.append(traceback.format_exc()[-1500:])
                            j["log"] = logs[-500:]
                            j["job_key"] = key
                            _jobs[key] = j
                            pid = j.get("project_id")
                            if pid:
                                _jobs[str(pid)] = j
                finally:
                    try:
                        control.set_prompt_id(None)
                        control.bind_thread(None)
                    except Exception:
                        pass
                    try:
                        reset_job_control(token)
                    except Exception:
                        pass
                    with _lock:
                        _workers.pop(key, None)
                        _controls.pop(key, None)
                    _persist_queue_safe()
                    try:
                        _pump_queue()
                    except Exception:
                        traceback.print_exc()

            t = threading.Thread(target=_runner, name=f"h3-job-{job_key[:24]}", daemon=True)
            _workers[job_key] = t
            t.start()
        _queue_positions()
    _persist_queue_safe()


def _enqueue_job(
    task: Callable[[JobControl], None],
    temp_id: str,
    label: str,
    *,
    project_id: str | None = None,
    prompt_preview: str = "",
    kind: str = "generate",
    generate_payload: dict[str, Any] | None = None,
    resume_payload: dict[str, Any] | None = None,
    title: str | None = None,
    persist: bool = True,
    restored: bool = False,
) -> dict[str, Any]:
    """Enqueue a generate/resume job; starts immediately if a parallel slot is free."""
    import time as _time

    with _lock:
        if project_id:
            existing = _jobs.get(project_id)
            if existing and _is_active_status(existing.get("status")):
                ex_key = existing.get("job_key")
                if not (restored and ex_key == temp_id):
                    raise HTTPException(
                        409,
                        f"Project {project_id} is already queued or running",
                    )

        _queue[:] = [i for i in _queue if i.get("job_key") != temp_id]

        job = {
            "status": "queued",
            "project_id": project_id or temp_id,
            "job_key": temp_id,
            "log": [
                f"{label} — restored from previous session"
                if restored
                else f"{label} — queued"
            ],
            "last_message": "Queued (restored)" if restored else "Queued",
            "temp": not bool(project_id),
            "queue_position": len(_queue) + 1,
            "label": label,
            "prompt_preview": (prompt_preview or "")[:120],
            "title": title,
            "kind": kind,
            "generate_payload": generate_payload,
            "resume_payload": resume_payload,
            "enqueued_at": _time.time(),
            "restored": restored,
        }
        _jobs[temp_id] = job
        if project_id:
            _jobs[project_id] = job
        _queue.append(
            {
                "job_key": temp_id,
                "label": label,
                "task": task,
                "project_id": project_id,
                "kind": kind,
                "generate_payload": generate_payload,
                "resume_payload": resume_payload,
                "prompt_preview": (prompt_preview or "")[:120],
                "title": title,
                "enqueued_at": job["enqueued_at"],
            }
        )
        depth = len(_queue)
        live_n = sum(1 for t in _workers.values() if t and t.is_alive())
        max_p = _max_parallel_locked()
        will_start_soon = live_n < max_p

    _pump_queue()
    if persist:
        _persist_queue_safe()

    with _lock:
        j2 = _jobs.get(temp_id) or job
        started = _is_running_status(j2.get("status"))
        pos = j2.get("queue_position")

    if started:
        msg = label
    elif will_start_soon:
        msg = f"{label} — starting"
    else:
        msg = f"{label} — queued (#{pos or depth})"

    return {
        "ok": True,
        "message": msg,
        "job_ref": temp_id,
        "project_id": project_id,
        "status": "running" if started else "queued",
        "queue_position": None if started else (pos or depth),
        "max_parallel_jobs": _effective_parallel(),
        "restored": restored,
    }


def _find_job_keys_for_target(job_ref: str | None, project_id: str | None) -> list[str]:
    """Resolve job_key(s) to stop from UI target."""
    keys: list[str] = []
    with _lock:
        targets = [t for t in (job_ref, project_id) if t]
        if not targets:
            return list(_controls.keys())
        for t in targets:
            if t in _controls:
                keys.append(t)
                continue
            j = _jobs.get(t)
            if j:
                jk = j.get("job_key") or t
                if jk not in keys:
                    keys.append(jk)
            for k, v in _jobs.items():
                if v.get("project_id") == t:
                    jk = v.get("job_key") or k
                    if jk not in keys:
                        keys.append(jk)
    return keys


def _run_job(req: GenerateRequest, temp_id: str, control: JobControl) -> None:
    logs: list[str] = []
    project_id_holder: dict[str, str] = {}

    def on_start(project_id: str) -> None:
        project_id_holder["id"] = project_id
        control.set_project_id(project_id)
        with _lock:
            j = _jobs.get(temp_id) or {
                "status": "running",
                "log": [],
                "last_message": "",
            }
            j["project_id"] = project_id
            j["temp"] = False
            j["status"] = "running"
            j["job_key"] = temp_id
            j["kind"] = j.get("kind") or "generate"
            _jobs[temp_id] = j
            _jobs[project_id] = j
        _persist_queue_safe()

    def log(msg: str) -> None:
        logs.append(msg)
        with _lock:
            j = _jobs.get(temp_id) or {}
            j["log"] = logs[-500:]
            j["last_message"] = msg
            cur = (j.get("status") or "").lower()
            if cur not in ("cancelled", "error", "completed", "completed_no_assemble", "failed"):
                if control.is_cancelled():
                    j["status"] = "cancelling"
                else:
                    j["status"] = "running"
            pid = project_id_holder.get("id")
            if pid:
                j["project_id"] = pid
                j["temp"] = False
                _merge_state_into_job(j, pid, mem_logs=logs)
                _jobs[pid] = j
            j["job_key"] = temp_id
            _jobs[temp_id] = j

    try:
        pipe = ProductionPipeline(settings, control=control)
        state = pipe.run(req, log=log, on_start=on_start)
        snap = _job_snapshot(state)
        snap["job_key"] = temp_id
        if control.is_cancelled() and (snap.get("status") or "") not in (
            "cancelled",
            "error",
            "failed",
        ):
            snap["status"] = "cancelled"
            snap["last_message"] = "Generation stopped by user."
            logs_snap = list(snap.get("log") or [])
            logs_snap.append("Generation stopped by user.")
            snap["log"] = logs_snap[-500:]
            if state:
                state.status = "cancelled"
                root = settings.output_root / state.project_id
                try:
                    (root / "state.json").write_text(
                        state.model_dump_json(indent=2), encoding="utf-8"
                    )
                except Exception:
                    pass
        _set_job(temp_id, snap, state.project_id)
    except CancelledError as e:
        with _lock:
            j = _jobs.get(temp_id) or {}
            j["status"] = "cancelled"
            j["last_message"] = str(e)
            j["log"] = (j.get("log") or []) + [f"Stopped: {e}"]
            j["job_key"] = temp_id
            _jobs[temp_id] = j
            pid = project_id_holder.get("id") or j.get("project_id")
            if pid:
                _jobs[str(pid)] = j
                try:
                    st = load_state(str(pid), settings)
                    if st:
                        st.status = "cancelled"
                        st.log = list(j.get("log") or st.log)
                        root = settings.output_root / str(pid)
                        (root / "state.json").write_text(
                            st.model_dump_json(indent=2), encoding="utf-8"
                        )
                except Exception:
                    pass
        _persist_queue_safe()
    except Exception as e:
        with _lock:
            prev = _jobs.get(temp_id) or {}
            _jobs[temp_id] = {
                "status": "error",
                "project_id": project_id_holder.get("id") or temp_id,
                "job_key": temp_id,
                "log": logs[-500:] + [str(e)],
                "last_message": str(e),
                "temp": not bool(project_id_holder.get("id")),
                "kind": prev.get("kind") or "generate",
                "generate_payload": prev.get("generate_payload"),
                "resume_payload": prev.get("resume_payload"),
            }
            pid = project_id_holder.get("id")
            if pid:
                _jobs[str(pid)] = _jobs[temp_id]
        _persist_queue_safe()


def _run_resume_job(
    project_id: str, req: ResumeRequest, temp_id: str, control: JobControl
) -> None:
    logs: list[str] = []
    project_id_holder: dict[str, str] = {"id": project_id}

    def on_start(pid: str) -> None:
        project_id_holder["id"] = pid
        control.set_project_id(pid)
        with _lock:
            j = _jobs.get(temp_id) or {"status": "running", "log": [], "last_message": ""}
            j["project_id"] = pid
            j["temp"] = False
            j["status"] = "running"
            j["job_key"] = temp_id
            j["kind"] = j.get("kind") or "resume"
            _jobs[temp_id] = j
            _jobs[pid] = j
        _persist_queue_safe()

    def log(msg: str) -> None:
        logs.append(msg)
        with _lock:
            j = _jobs.get(temp_id) or {}
            j["log"] = logs[-500:]
            j["last_message"] = msg
            cur = (j.get("status") or "").lower()
            if cur not in ("cancelled", "error", "completed", "completed_no_assemble", "failed"):
                if control.is_cancelled():
                    j["status"] = "cancelling"
                else:
                    j["status"] = "running"
            pid = project_id_holder.get("id")
            if pid:
                j["project_id"] = pid
                j["temp"] = False
                _merge_state_into_job(j, pid, mem_logs=logs)
                _jobs[pid] = j
            j["job_key"] = temp_id
            _jobs[temp_id] = j

    try:
        on_start(project_id)
        pipe = ProductionPipeline(settings, control=control)
        state = pipe.resume(project_id, req, log=log, on_start=on_start)
        snap = _job_snapshot(state)
        snap["job_key"] = temp_id
        if control.is_cancelled() and (snap.get("status") or "") not in (
            "cancelled",
            "error",
            "failed",
        ):
            snap["status"] = "cancelled"
        _set_job(temp_id, snap, state.project_id)
    except CancelledError as e:
        with _lock:
            j = _jobs.get(temp_id) or {}
            j["status"] = "cancelled"
            j["last_message"] = str(e)
            j["log"] = (j.get("log") or []) + [f"Stopped: {e}"]
            j["job_key"] = temp_id
            _jobs[temp_id] = j
            _jobs[project_id] = j
        _persist_queue_safe()
    except Exception as e:
        with _lock:
            prev = _jobs.get(temp_id) or {}
            err = {
                "status": "error",
                "project_id": project_id,
                "job_key": temp_id,
                "log": logs[-500:] + [str(e)],
                "last_message": str(e),
                "temp": False,
                "kind": prev.get("kind") or "resume",
                "generate_payload": prev.get("generate_payload"),
                "resume_payload": prev.get("resume_payload") or req.model_dump(),
            }
            _jobs[temp_id] = err
            _jobs[project_id] = err
        _persist_queue_safe()


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is required")

    # Resolve library slug → full style prompt when provided
    if req.style_slug and (not req.style or not req.style.strip()):
        st = get_style(req.style_slug)
        if st:
            req.style = build_style_prompt(st)

    temp_id = f"pending_{uuid.uuid4().hex[:12]}"
    preview = req.prompt.strip().replace("\n", " ")
    payload = req.model_dump()

    def task(control: JobControl) -> None:
        _run_job(req, temp_id, control)

    return _enqueue_job(
        task,
        temp_id,
        "Starting pipeline…",
        prompt_preview=preview,
        kind="generate",
        generate_payload=payload,
    )


@app.post("/api/projects/{project_id}/resume")
async def resume_project(project_id: str, req: ResumeRequest = ResumeRequest()):
    state = load_state(project_id, settings)
    if not state:
        raise HTTPException(404, "Project not found")
    if not state.plan or not state.plan.shots:
        raise HTTPException(400, "Project has no plan — cannot resume")

    temp_id = f"resume_{project_id}_{uuid.uuid4().hex[:8]}"
    res_payload = req.model_dump()

    def task(control: JobControl) -> None:
        _run_resume_job(project_id, req, temp_id, control)

    title = state.plan.title if state.plan else project_id
    out = _enqueue_job(
        task,
        temp_id,
        f"Resuming {project_id}…",
        project_id=project_id,
        prompt_preview=title,
        kind="resume",
        resume_payload=res_payload,
        title=title,
    )
    with _lock:
        j = _jobs.get(temp_id)
        if j:
            j["project_id"] = project_id
            j["temp"] = False
            j["title"] = title
            _jobs[project_id] = j
    _persist_queue_safe()
    return out


@app.post("/api/generate/stop")
async def stop_generate(body: StopBody | None = None):
    """Stop one job, all live workers, and/or clear the queue."""
    body = body or StopBody()
    worker_alive = _any_worker_alive()

    with _lock:
        live = [
            (k, dict(v))
            for k, v in _jobs.items()
            if _is_running_status(v.get("status")) or v.get("status") == "cancelling"
        ]
        queued_n = len(_queue)

    stop_everything = body.stop_all or (not body.job_ref and not body.project_id)

    orphan_pids: list[str] = []
    if not live and not worker_alive and not queued_n and stop_everything:
        # Explicit stop with nothing live — cancel sticky disk projects (manual)
        orphan_pids = _mark_orphaned_projects_stopped()
        msg = "No running generation"
        if orphan_pids:
            msg = f"Cleared stuck job status for {', '.join(orphan_pids[:3])}"
            _persist_queue_safe()
        return {
            "ok": True,
            "message": msg,
            "stopped": bool(orphan_pids),
            "orphaned": orphan_pids,
        }

    if stop_everything:
        with _lock:
            targets = list(_controls.keys())
    else:
        targets = _find_job_keys_for_target(body.job_ref, body.project_id)

    cleared_queued: list[str] = []
    with _lock:
        keep_queue: list[dict[str, Any]] = []
        for item in _queue:
            jk = item.get("job_key")
            j = _jobs.get(jk or "") or {}
            should_drop = False
            if stop_everything or body.clear_queue:
                should_drop = True
            elif body.job_ref and jk == body.job_ref:
                should_drop = True
            elif body.project_id and (
                j.get("project_id") == body.project_id or jk == body.project_id
            ):
                should_drop = True
            if should_drop:
                if jk:
                    cleared_queued.append(jk)
                    jq = _jobs.get(jk) or {}
                    jq["status"] = "cancelled"
                    jq["last_message"] = "Removed from queue"
                    logs = list(jq.get("log") or [])
                    logs.append("Removed from queue before start.")
                    jq["log"] = logs[-500:]
                    _jobs[jk] = jq
                    pid = jq.get("project_id")
                    if pid and pid != jk:
                        _jobs[str(pid)] = jq
            else:
                keep_queue.append(item)
        _queue[:] = keep_queue
        _queue_positions()

    if not targets and not cleared_queued:
        return {
            "ok": True,
            "message": "Nothing to stop",
            "stopped": False,
            "orphaned": orphan_pids,
        }

    stopped_controls: list[JobControl] = []
    for jk in targets:
        with _lock:
            c = _controls.get(jk)
            j = _jobs.get(jk)
        if c:
            c.request_stop()
            stopped_controls.append(c)
        if j:
            with _lock:
                j = dict(_jobs.get(jk) or j)
                if _is_running_status(j.get("status")) or j.get("status") == "cancelling":
                    j["status"] = "cancelling"
                    j["last_message"] = "Stop requested…"
                    logs = list(j.get("log") or [])
                    logs.append("Stop requested — interrupting ComfyUI and winding down…")
                    j["log"] = logs[-500:]
                    _jobs[jk] = j
                    pid = j.get("project_id")
                    if pid:
                        _jobs[str(pid)] = j

    def _interrupt() -> None:
        # Comfy interrupt is global; with parallel jobs it may affect peers.
        ComfyH3Client(settings).interrupt()

    if stopped_controls:
        try:
            _interrupt()
        except Exception:
            pass
        stopped_controls[0].start_interrupt_nudge(_interrupt, interval_sec=2.0, max_sec=120.0)

    parts = []
    if stopped_controls:
        parts.append(f"stopping {len(stopped_controls)} worker(s)")
    if cleared_queued:
        parts.append(f"cleared {len(cleared_queued)} queued")
    msg = "Stop requested — " + (", ".join(parts) if parts else "waiting for wind-down")
    _persist_queue_safe()

    return {
        "ok": True,
        "message": msg,
        "stopped": True,
        "stopped_keys": targets,
        "cleared_queue": cleared_queued,
        "project_id": stopped_controls[0].project_id if stopped_controls else None,
    }


@app.delete("/api/queue/{job_ref}")
async def dequeue_job(job_ref: str):
    """Remove a still-queued job without affecting running workers."""
    with _lock:
        before = len(_queue)
        _queue[:] = [i for i in _queue if i.get("job_key") != job_ref]
        removed = before - len(_queue)
        j = _jobs.get(job_ref)
        if j and (j.get("status") or "").lower() == "queued":
            j["status"] = "cancelled"
            j["last_message"] = "Removed from queue"
            logs = list(j.get("log") or [])
            logs.append("Removed from queue.")
            j["log"] = logs[-500:]
            _jobs[job_ref] = j
        _queue_positions()
    _persist_queue_safe()
    if not removed and not j:
        raise HTTPException(404, "Queue entry not found")
    return {"ok": True, "removed": bool(removed), "job_ref": job_ref}


@app.get("/api/queue")
async def queue_status():
    with _lock:
        items = []
        for i, item in enumerate(_queue):
            jk = item.get("job_key")
            j = dict(_jobs.get(jk) or {})
            items.append(
                {
                    "job_key": jk,
                    "position": i + 1,
                    "project_id": j.get("project_id") or item.get("project_id"),
                    "label": item.get("label") or j.get("label"),
                    "prompt_preview": j.get("prompt_preview") or "",
                    "title": j.get("title"),
                    "status": j.get("status") or "queued",
                }
            )
        running = []
        for jk, t in _workers.items():
            if t and t.is_alive():
                j = dict(_jobs.get(jk) or {})
                running.append(
                    {
                        "job_key": jk,
                        "project_id": j.get("project_id"),
                        "status": j.get("status"),
                        "title": j.get("title"),
                        "prompt_preview": j.get("prompt_preview") or "",
                    }
                )
    return {
        "queue": items,
        "running": running,
        "max_parallel_jobs": _effective_parallel(),
        "workers_alive": len(running),
    }


@app.get("/api/settings/parallel")
async def get_parallel_setting():
    return {
        "max_parallel_jobs": _effective_parallel(),
        "default": settings.max_parallel_jobs,
        "override": _parallel_override,
    }


@app.post("/api/settings/parallel")
async def set_parallel_setting(body: ParallelJobsBody):
    global _parallel_override
    with _lock:
        _parallel_override = int(body.max_parallel_jobs)
    _pump_queue()
    _persist_queue_safe()
    return {
        "ok": True,
        "max_parallel_jobs": _effective_parallel(),
    }


def _mark_orphaned_projects_stopped() -> list[str]:
    """
    Explicit stop-time cleanup: sticky disk statuses with no live worker become cancelled.
    Does NOT run automatically on job list polls (that would kill restart resume).
    """
    if _any_worker_alive():
        return []
    with _lock:
        if _queue:
            return []
        active_pids = {
            str(v.get("project_id"))
            for v in _jobs.values()
            if _is_active_status(v.get("status")) and v.get("project_id")
        }
    marked: list[str] = []
    sticky = {"planning", "assembling", "running", "generating", "reviewing", "cancelling", "interrupted"}
    for p in list_projects(settings):
        st = (p.get("status") or "").lower()
        if st not in sticky:
            continue
        pid = p.get("project_id")
        if not pid or str(pid) in active_pids:
            continue
        state = load_state(str(pid), settings)
        if not state or (state.status or "").lower() not in sticky:
            continue
        state.status = "cancelled"
        state.log = list(state.log or []) + [
            "Stopped: cleared stuck status (no active worker)."
        ]
        try:
            root = settings.output_root / str(pid)
            (root / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
            marked.append(str(pid))
            with _lock:
                j = _jobs.get(str(pid)) or {}
                j.update(
                    {
                        "status": "cancelled",
                        "project_id": str(pid),
                        "log": list(state.log)[-500:],
                        "last_message": state.log[-1] if state.log else "cancelled",
                    }
                )
                _jobs[str(pid)] = j
        except Exception:
            continue
    return marked


@app.get("/api/jobs")
async def jobs():
    """In-memory jobs plus disk logs. Live only when a worker is actually running."""
    worker_alive = _any_worker_alive()

    with _lock:
        active = {k: dict(v) for k, v in _jobs.items()}
        queue_items = []
        for i, item in enumerate(_queue):
            jk = item.get("job_key")
            j = dict(active.get(jk) or {})
            queue_items.append(
                {
                    "job_key": jk,
                    "position": i + 1,
                    "project_id": j.get("project_id") or item.get("project_id"),
                    "label": item.get("label") or j.get("label"),
                    "prompt_preview": j.get("prompt_preview") or "",
                    "title": j.get("title"),
                    "status": j.get("status") or "queued",
                }
            )
        cancel_any = any(c.is_cancelled() for c in _controls.values())
        workers_alive = sum(1 for t in _workers.values() if t and t.is_alive())
        control_pids = {
            (c.project_id or ""): c for c in _controls.values() if c.project_id
        }
        restore_notes = list(_restore_notes)

    # Inject restore messages once into any active job log surface
    if restore_notes:
        for note in restore_notes:
            for key in list(active.keys())[:3]:
                job = dict(active[key])
                logs = list(job.get("log") or [])
                if note not in logs:
                    logs.insert(0, note)
                    job["log"] = logs[-500:]
                    active[key] = job

    if not worker_alive:
        # Soft-mark: no worker underneath a "running" memory entry (crashed process)
        # Keep as interrupted so restart restore can pick it up; do not auto-cancel.
        for key, job in list(active.items()):
            st = (job.get("status") or "").lower()
            if st == "queued":
                if any(q.get("job_key") == key for q in queue_items):
                    continue
            if _is_running_status(job.get("status")) or job.get("status") == "cancelling":
                job = dict(job)
                job["status"] = "interrupted"
                logs = list(job.get("log") or [])
                msg = "Interrupted: worker is no longer running (will auto-resume on restart)."
                if msg not in logs:
                    logs.append(msg)
                job["log"] = logs[-500:]
                job["last_message"] = logs[-1]
                active[key] = job
                with _lock:
                    # only update if still not assigned a live worker
                    if key not in _workers or not (_workers.get(key) and _workers[key].is_alive()):
                        _jobs[key] = job

    projects = list_projects(settings)
    live_from_mem = worker_alive and any(
        _is_running_status(v.get("status")) or v.get("status") == "cancelling"
        for v in active.values()
    )

    if not live_from_mem and projects and worker_alive:
        for p in projects:
            st = (p.get("status") or "").lower()
            pid = p.get("project_id")
            if not pid:
                continue
            if st in {
                "planning",
                "assembling",
                "running",
                "generating",
                "reviewing",
                "cancelling",
            }:
                state = load_state(str(pid), settings)
                if state:
                    ctl = control_pids.get(str(pid))
                    status = "cancelling" if (ctl and ctl.is_cancelled()) else "running"
                    snap = _job_snapshot(state, status=status)
                    active[state.project_id] = snap
                    live_from_mem = True

    for key, job in list(active.items()):
        pid = job.get("project_id")
        if not pid or str(pid).startswith("pending_"):
            continue
        if (job.get("status") or "") not in (
            "running",
            "planning",
            "assembling",
            "cancelling",
            "cancelled",
            "queued",
        ):
            if job.get("log"):
                continue
        state = load_state(str(pid), settings)
        if state and state.log:
            if len(state.log) >= len(job.get("log") or []):
                job = dict(job)
                _merge_state_into_job(job, str(pid))
                job["last_message"] = (
                    (job.get("log") or state.log)[-1]
                    if (job.get("log") or state.log)
                    else ""
                )
                disk_st = (state.status or "").lower()
                mem_st = (job.get("status") or "").lower()
                if mem_st in ("running", "cancelling") and disk_st in (
                    "cancelled",
                    "completed",
                    "completed_no_assemble",
                    "failed",
                    "error",
                ):
                    job["status"] = state.status
                elif mem_st in ("running", "cancelling") and not worker_alive:
                    job["status"] = "cancelled"
                active[key] = job
            else:
                job = dict(job)
                _merge_state_into_job(job, str(pid), mem_logs=list(job.get("log") or []))
                active[key] = job
        elif state:
            job = dict(job)
            _merge_state_into_job(job, str(pid), mem_logs=list(job.get("log") or []))
            active[key] = job

    tabs: list[dict[str, Any]] = []
    seen_tabs: set[str] = set()
    for key, v in active.items():
        st = (v.get("status") or "").lower()
        if st not in (
            "queued",
            "running",
            "planning",
            "assembling",
            "generating",
            "reviewing",
            "cancelling",
        ):
            continue
        jk = v.get("job_key") or key
        pid = v.get("project_id") or jk
        tab_id = str(jk)
        if tab_id in seen_tabs:
            continue
        if key != jk and key == pid and jk in active:
            continue
        seen_tabs.add(tab_id)
        tabs.append(
            {
                "job_key": jk,
                "project_id": pid,
                "status": v.get("status"),
                "title": v.get("title"),
                "prompt_preview": v.get("prompt_preview") or "",
                "queue_position": v.get("queue_position"),
                "last_message": v.get("last_message") or "",
            }
        )

    def _tab_rank(t: dict[str, Any]) -> tuple:
        st = (t.get("status") or "").lower()
        if st == "queued":
            return (1, t.get("queue_position") or 999)
        if st == "cancelling":
            return (0, 1)
        return (0, 0)

    tabs.sort(key=_tab_rank)

    current = None
    for v in active.values():
        if worker_alive and (
            _is_running_status(v.get("status")) or v.get("status") in ("cancelling",)
        ):
            current = v
            break
    if current is None and projects:
        state = load_state(projects[0]["project_id"], settings)
        if state:
            current = _job_snapshot(state)

    return {
        "active": active,
        "current": current,
        "projects": projects,
        "worker_alive": worker_alive,
        "workers_alive": workers_alive,
        "cancel_requested": cancel_any,
        "queue": queue_items,
        "tabs": tabs,
        "max_parallel_jobs": _effective_parallel(),
    }


@app.get("/api/projects")
async def projects():
    return list_projects(settings)


@app.get("/api/projects/{project_id}")
async def project_detail(project_id: str):
    state = load_state(project_id, settings)
    if not state:
        raise HTTPException(404, "Project not found")
    return state.model_dump()


@app.get("/api/projects/{project_id}/file")
async def project_file(project_id: str, path: str):
    """Serve a file under the project directory only."""
    root = (settings.output_root / project_id).resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)):
        raise HTTPException(400, "Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(target)


@app.get("/api/projects/{project_id}/master")
async def project_master(project_id: str):
    state = load_state(project_id, settings)
    if not state or not state.master_path:
        raise HTTPException(404, "Master not available")
    p = Path(state.master_path)
    if not p.exists():
        raise HTTPException(404, "Master file missing")
    return FileResponse(p, media_type="video/mp4", filename=p.name)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
