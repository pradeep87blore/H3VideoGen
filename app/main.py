"""FastAPI app: UI + API for H3VideoGen."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .comfy_h3 import ComfyH3Client
from .config import get_settings
from .job_control import CancelledError, job_control
from .models import DirectorOnlyRequest, GenerateRequest, ResumeRequest
from .pipeline import ProductionPipeline, list_projects, load_state
from .services import (
    comfy_reachable,
    ensure_runtime_services,
    essentials_report,
    ollama_reachable,
)

ROOT = Path(__file__).resolve().parent.parent
settings = get_settings()

app = FastAPI(title="H3 Video Gen", version="0.1.0")
templates = Jinja2Templates(directory=str(ROOT / "web" / "templates"))

# project jobs: project_id -> status info
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
_worker: threading.Thread | None = None

static_dir = ROOT / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "enable_voice": settings.enable_voice,
            "comfy_url": settings.comfy_base_url,
            "critic_threshold": settings.critic_pass_threshold,
        },
    )


@app.get("/api/health")
async def health():
    comfy_ok = comfy_reachable(settings)
    comfy_info: Any = None
    h3_info: Any = None
    err = None
    if comfy_ok:
        try:
            client = ComfyH3Client(settings)
            comfy_info = client.health()
            h3_info = client.h3_capability()
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

    llm_status = LLMRouter(settings).status()
    essentials = essentials_report(settings)
    with _lock:
        running = any((v.get("status") or "") == "running" for v in _jobs.values())
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
        "cancel_requested": job_control.is_cancelled(),
        "essentials": essentials,
        "heartbeat": {
            "interval_sec": 10,
            "checked_at": checked_at,
            "ready_for_generate": essentials.get("ready_for_generate", False),
            "tools": essentials.get("services") or [],
        },
    }


@app.post("/api/services/ensure")
async def services_ensure():
    """Start ComfyUI / Ollama if configured and not already running."""
    lines: list[str] = []

    def log(m: str) -> None:
        lines.append(m)

    report = ensure_runtime_services(settings, log=log, need_comfy=True, need_ollama=True)
    essentials = essentials_report(settings)
    comfy_rep = report.get("comfy") or {}
    return {
        "ok": bool(comfy_rep.get("ok")),
        "report": report,
        "log": lines,
        "comfy_ok": bool(comfy_rep.get("ok")),
        "ollama_ok": ollama_reachable(settings),
        "essentials": essentials,
    }


@app.post("/api/plan")
async def plan_only(req: DirectorOnlyRequest):
    try:
        plan = ProductionPipeline(settings).plan_only(
            req.prompt, req.style, req.target_duration_sec, req.max_shots
        )
        return plan.model_dump()
    except Exception as e:
        raise HTTPException(500, str(e)) from e


_LIVE_STATUSES = frozenset(
    {"planning", "assembling", "running", "generating", "reviewing", "cancelled"}
)


def _job_snapshot(state_or_dict: Any, *, status: str | None = None) -> dict[str, Any]:
    """Normalize a job dict for the UI (logs from memory or disk)."""
    if hasattr(state_or_dict, "model_dump"):
        s = state_or_dict
        chars = []
        if s.plan and s.plan.characters:
            for c in s.plan.characters:
                chars.append(
                    {
                        "id": c.id,
                        "name": c.name,
                        "look": (c.look or "")[:220],
                        "image_path": c.image_path,
                        "sheet_count": sum(
                            1
                            for p in (c.sheet or [])
                            if p.image_path
                        ),
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
            "temp": False,
        }
    return dict(state_or_dict)


def _set_job(key: str, data: dict[str, Any], *aliases: str) -> None:
    with _lock:
        _jobs[key] = data
        for a in aliases:
            if a and a != key:
                _jobs[a] = data


def _is_running_status(status: str | None) -> bool:
    return (status or "").lower() in {
        "running",
        "planning",
        "assembling",
        "generating",
        "reviewing",
    }


def _begin_worker(task_fn, temp_id: str, label: str) -> dict[str, Any]:
    global _worker
    with _lock:
        for j in _jobs.values():
            if _is_running_status(j.get("status")):
                raise HTTPException(409, "A generation job is already running — stop it first")
        if _worker and _worker.is_alive():
            raise HTTPException(409, "A generation worker is still shutting down — wait a moment")

        job_control.reset()
        _jobs[temp_id] = {
            "status": "running",
            "project_id": temp_id,
            "log": [label],
            "last_message": label,
            "temp": True,
        }

    t = threading.Thread(target=task_fn, name="h3-generate", daemon=True)
    job_control.bind_thread(t)
    _worker = t
    t.start()
    return {"ok": True, "message": label, "job_ref": temp_id}


def _run_job(req: GenerateRequest, temp_id: str) -> None:
    logs: list[str] = []
    project_id_holder: dict[str, str] = {}

    def on_start(project_id: str) -> None:
        project_id_holder["id"] = project_id
        job_control.set_project_id(project_id)
        with _lock:
            j = _jobs.get(temp_id) or {
                "status": "running",
                "log": [],
                "last_message": "",
            }
            j["project_id"] = project_id
            j["temp"] = False
            j["status"] = "running"
            _jobs[temp_id] = j
            _jobs[project_id] = j

    def log(msg: str) -> None:
        logs.append(msg)
        with _lock:
            j = _jobs.get(temp_id) or {}
            j["log"] = logs[-500:]
            j["last_message"] = msg
            cur = (j.get("status") or "").lower()
            # Never demote terminal / cancelling statuses back to "running"
            if cur not in ("cancelled", "error", "completed", "completed_no_assemble", "failed"):
                if job_control.is_cancelled():
                    j["status"] = "cancelling"
                else:
                    j["status"] = "running"
            pid = project_id_holder.get("id")
            if pid:
                j["project_id"] = pid
                j["temp"] = False
                # Keep cast list fresh for the UI after director plans
                try:
                    st = load_state(pid, settings)
                    if st and st.plan:
                        if st.plan.title:
                            j["title"] = st.plan.title
                        if st.plan.characters:
                            j["characters"] = [
                                {
                                    "id": c.id,
                                    "name": c.name,
                                    "look": (c.look or "")[:220],
                                    "image_path": c.image_path,
                                    "sheet_count": sum(
                                        1 for p in (c.sheet or []) if p.image_path
                                    ),
                                }
                                for c in st.plan.characters
                            ]
                except Exception:
                    pass
                _jobs[pid] = j
            _jobs[temp_id] = j

    try:
        pipe = ProductionPipeline(settings, control=job_control)
        state = pipe.run(req, log=log, on_start=on_start)
        snap = _job_snapshot(state)
        # If stop was requested mid-run but pipeline didn't surface cancel, force terminal
        if job_control.is_cancelled() and (snap.get("status") or "") not in (
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
            _jobs[temp_id] = j
            pid = project_id_holder.get("id") or j.get("project_id")
            if pid:
                _jobs[str(pid)] = j
                # Persist cancelled so /api/jobs won't resurrect a zombie
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
    except Exception as e:
        with _lock:
            _jobs[temp_id] = {
                "status": "error",
                "project_id": project_id_holder.get("id") or temp_id,
                "log": logs[-500:] + [str(e)],
                "last_message": str(e),
                "temp": not bool(project_id_holder.get("id")),
            }
    finally:
        job_control.set_prompt_id(None)
        job_control.bind_thread(None)


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is required")

    temp_id = f"pending_{id(req)}"

    def task() -> None:
        _run_job(req, temp_id)

    return _begin_worker(task, temp_id, "Starting pipeline…")


def _run_resume_job(project_id: str, req: ResumeRequest, temp_id: str) -> None:
    logs: list[str] = []
    project_id_holder: dict[str, str] = {"id": project_id}

    def on_start(pid: str) -> None:
        project_id_holder["id"] = pid
        job_control.set_project_id(pid)
        with _lock:
            j = _jobs.get(temp_id) or {"status": "running", "log": [], "last_message": ""}
            j["project_id"] = pid
            j["temp"] = False
            j["status"] = "running"
            _jobs[temp_id] = j
            _jobs[pid] = j

    def log(msg: str) -> None:
        logs.append(msg)
        with _lock:
            j = _jobs.get(temp_id) or {}
            j["log"] = logs[-500:]
            j["last_message"] = msg
            cur = (j.get("status") or "").lower()
            if cur not in ("cancelled", "error", "completed", "completed_no_assemble", "failed"):
                if job_control.is_cancelled():
                    j["status"] = "cancelling"
                else:
                    j["status"] = "running"
            pid = project_id_holder.get("id")
            if pid:
                j["project_id"] = pid
                j["temp"] = False
                try:
                    st = load_state(pid, settings)
                    if st and st.plan:
                        if st.plan.title:
                            j["title"] = st.plan.title
                        if st.plan.characters:
                            j["characters"] = [
                                {
                                    "id": c.id,
                                    "name": c.name,
                                    "look": (c.look or "")[:220],
                                    "image_path": c.image_path,
                                    "sheet_count": sum(
                                        1 for p in (c.sheet or []) if p.image_path
                                    ),
                                }
                                for c in st.plan.characters
                            ]
                except Exception:
                    pass
                _jobs[pid] = j
            _jobs[temp_id] = j

    try:
        on_start(project_id)
        pipe = ProductionPipeline(settings, control=job_control)
        state = pipe.resume(project_id, req, log=log, on_start=on_start)
        snap = _job_snapshot(state)
        if job_control.is_cancelled() and (snap.get("status") or "") not in (
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
            _jobs[temp_id] = j
            _jobs[project_id] = j
    except Exception as e:
        with _lock:
            _jobs[temp_id] = {
                "status": "error",
                "project_id": project_id,
                "log": logs[-500:] + [str(e)],
                "last_message": str(e),
                "temp": False,
            }
            _jobs[project_id] = _jobs[temp_id]
    finally:
        job_control.set_prompt_id(None)
        job_control.bind_thread(None)


@app.post("/api/projects/{project_id}/resume")
async def resume_project(project_id: str, req: ResumeRequest = ResumeRequest()):
    state = load_state(project_id, settings)
    if not state:
        raise HTTPException(404, "Project not found")
    if not state.plan or not state.plan.shots:
        raise HTTPException(400, "Project has no plan — cannot resume")

    temp_id = f"resume_{project_id}_{id(req)}"

    def task() -> None:
        _run_resume_job(project_id, req, temp_id)

    out = _begin_worker(task, temp_id, f"Resuming {project_id}…")
    out["project_id"] = project_id
    # Alias under real project id immediately for pollers
    with _lock:
        j = _jobs.get(temp_id)
        if j:
            j["project_id"] = project_id
            j["temp"] = False
            _jobs[project_id] = j
    return out


@app.post("/api/generate/stop")
async def stop_generate():
    """Stop the active generation job (pipeline + ComfyUI interrupt)."""
    worker_alive = bool(_worker and _worker.is_alive()) or job_control.worker_alive()

    with _lock:
        live = [
            (k, dict(v))
            for k, v in _jobs.items()
            if _is_running_status(v.get("status")) or v.get("status") == "cancelling"
        ]

    # Always clear phantom "running" projects left on disk when no worker exists
    orphan_pids = _mark_orphaned_projects_stopped()

    if not live and not worker_alive:
        msg = "No running generation"
        if orphan_pids:
            msg = f"Cleared stuck job status for {', '.join(orphan_pids[:3])}"
        return {
            "ok": True,
            "message": msg,
            "stopped": bool(orphan_pids),
            "orphaned": orphan_pids,
        }

    job_control.request_stop()

    def _interrupt() -> None:
        ComfyH3Client(settings, control=job_control).interrupt()

    try:
        _interrupt()
    except Exception:
        pass
    job_control.start_interrupt_nudge(_interrupt, interval_sec=2.0, max_sec=120.0)

    with _lock:
        for k, v in list(_jobs.items()):
            if _is_running_status(v.get("status")) or v.get("status") == "cancelling":
                v = dict(v)
                v["status"] = "cancelling"
                v["last_message"] = "Stop requested…"
                logs = list(v.get("log") or [])
                logs.append("Stop requested — interrupting ComfyUI and winding down…")
                v["log"] = logs[-500:]
                _jobs[k] = v

    return {
        "ok": True,
        "message": "Stop requested — waiting for pipeline / Comfy to wind down",
        "stopped": True,
        "project_id": job_control.project_id,
    }


def _mark_orphaned_projects_stopped() -> list[str]:
    """
    Projects left as planning/running on disk after process exit show as live forever.
    When no worker is alive, mark those non-terminal states cancelled so UI unlocks.
    """
    if job_control.worker_alive() or (_worker and _worker.is_alive()):
        return []
    marked: list[str] = []
    sticky = {"planning", "assembling", "running", "generating", "reviewing", "cancelling"}
    for p in list_projects(settings):
        st = (p.get("status") or "").lower()
        if st not in sticky:
            continue
        pid = p.get("project_id")
        if not pid:
            continue
        state = load_state(str(pid), settings)
        if not state or (state.status or "").lower() not in sticky:
            continue
        state.status = "cancelled"
        state.log = list(state.log or []) + [
            "Stopped: orphaned job (no active worker — previous run interrupted or server restarted)."
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
    worker_alive = bool(_worker and _worker.is_alive()) or job_control.worker_alive()

    # Heal sticky statuses left over from kills/restarts (Stop button used to noop here)
    if not worker_alive:
        _mark_orphaned_projects_stopped()

    with _lock:
        active = {k: dict(v) for k, v in _jobs.items()}

    # Demote memory jobs that claim live but have no worker
    if not worker_alive:
        for key, job in list(active.items()):
            if _is_running_status(job.get("status")) or job.get("status") == "cancelling":
                job = dict(job)
                job["status"] = "cancelled"
                logs = list(job.get("log") or [])
                logs.append("Stopped: worker is no longer running.")
                job["log"] = logs[-500:]
                job["last_message"] = logs[-1]
                active[key] = job
                with _lock:
                    _jobs[key] = job

    projects = list_projects(settings)
    live_from_mem = worker_alive and any(
        _is_running_status(v.get("status")) or v.get("status") == "cancelling"
        for v in active.values()
    )

    # Only recover as live when a worker is actually running
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
                    status = "cancelling" if job_control.is_cancelled() else "running"
                    snap = _job_snapshot(state, status=status)
                    active[state.project_id] = snap
                    live_from_mem = True
                    break

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
        ):
            if job.get("log"):
                continue
        state = load_state(str(pid), settings)
        if state and state.log:
            if len(state.log) >= len(job.get("log") or []):
                job = dict(job)
                job["log"] = list(state.log)
                job["last_message"] = state.log[-1]
                if state.master_path:
                    job["master_path"] = state.master_path
                if state.plan:
                    job["title"] = state.plan.title
                    if state.plan.characters:
                        job["characters"] = [
                            {
                                "id": c.id,
                                "name": c.name,
                                "look": (c.look or "")[:220],
                                "image_path": c.image_path,
                                "sheet_count": sum(
                                    1 for p in (c.sheet or []) if p.image_path
                                ),
                            }
                            for c in state.plan.characters
                        ]
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
        "cancel_requested": job_control.is_cancelled(),
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
