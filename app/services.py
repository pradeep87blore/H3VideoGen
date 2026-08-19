"""Ensure local runtime services (ComfyUI, Ollama) are up."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Settings

LogFn = Callable[[str], None]

_comfy_proc: subprocess.Popen | None = None
_ollama_proc: subprocess.Popen | None = None


def _emit(log: LogFn | None, msg: str) -> None:
    if log:
        log(msg)


def comfy_reachable(settings: Settings, timeout: float = 3.0) -> bool:
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{settings.comfy_base_url.rstrip('/')}/system_stats")
            return r.status_code < 500
    except Exception:
        return False


_h3_status_cache: dict[str, Any] = {"ts": 0.0, "value": None, "key": ""}
_H3_STATUS_TTL_SEC = 20.0


def comfy_h3_status(
    settings: Settings,
    timeout: float = 8.0,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Probe MiniMax H3 nodes on the configured Comfy base URL (short, cached for health)."""
    import time as _time

    cache_key = f"{settings.comfy_base_url}|{timeout}"
    now = _time.time()
    if (
        use_cache
        and _h3_status_cache.get("key") == cache_key
        and _h3_status_cache.get("value") is not None
        and now - float(_h3_status_cache.get("ts") or 0) < _H3_STATUS_TTL_SEC
    ):
        return dict(_h3_status_cache["value"])

    if not comfy_reachable(settings, timeout=min(2.0, timeout)):
        out = {
            "ok": False,
            "reachable": False,
            "missing_nodes": ["MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"],
            "error": f"Unreachable at {settings.comfy_base_url}",
        }
        _h3_status_cache.update({"ts": now, "value": out, "key": cache_key})
        return dict(out)
    from .comfy_h3 import ComfyH3Client

    out = ComfyH3Client(settings).h3_capability(timeout=timeout)
    _h3_status_cache.update({"ts": now, "value": out, "key": cache_key})
    return dict(out)


def ollama_reachable(settings: Settings, timeout: float = 3.0) -> bool:
    if not settings.local_llm_enabled:
        return False
    base = settings.local_llm_base_url.rstrip("/")
    # OpenAI-compat: .../v1 → hit .../v1/models or root tags
    urls = [f"{base}/models"]
    if base.endswith("/v1"):
        root = base[: -len("/v1")]
        urls.append(f"{root}/api/tags")
    try:
        with httpx.Client(timeout=timeout) as client:
            for u in urls:
                try:
                    r = client.get(u)
                    if r.status_code < 500:
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False


def _windows_new_console_flags() -> int:
    if sys.platform != "win32":
        return 0
    # CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP
    return 0x00000010 | 0x00000200


def _comfy_paths(settings: Settings) -> tuple[Path, Path]:
    root = Path(settings.comfyui_root) if settings.comfyui_root else Path(settings.ai_root) / "ComfyUI"
    if settings.comfyui_python:
        py = Path(settings.comfyui_python)
    else:
        if sys.platform == "win32":
            py = root / ".venv" / "Scripts" / "python.exe"
        else:
            py = root / ".venv" / "bin" / "python"
        if not py.exists():
            py = Path(sys.executable)
    return root, py


def _pids_listening_on_port(port: int) -> list[int]:
    """Best-effort: PIDs that currently own TCP listen on port (Windows + psutil-free)."""
    pids: set[int] = set()
    if sys.platform == "win32":
        try:
            # netstat -ano: ... LISTENING <pid>
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "tcp"],
                text=True,
                errors="replace",
                timeout=15,
            )
            needle = f":{port}"
            for line in out.splitlines():
                if "LISTENING" not in line.upper():
                    continue
                if needle not in line:
                    continue
                # match :8188 as end of local address (avoid :81880)
                parts = line.split()
                if len(parts) < 5:
                    continue
                local = parts[1] if parts[0].upper().startswith("TCP") else parts[0]
                if not local.endswith(needle) and f"{needle} " not in f"{local} ":
                    # e.g. 127.0.0.1:8188
                    if not local.rsplit(":", 1)[-1] == str(port):
                        continue
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    pass
        except Exception:
            pass
        # PowerShell fallback
        if not pids:
            try:
                ps = (
                    f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                    f"-ErrorAction SilentlyContinue).OwningProcess"
                )
                out = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command", ps],
                    text=True,
                    errors="replace",
                    timeout=20,
                )
                for tok in out.replace(",", " ").split():
                    tok = tok.strip()
                    if tok.isdigit():
                        pids.add(int(tok))
            except Exception:
                pass
    else:
        try:
            out = subprocess.check_output(
                ["ss", "-ltnp"],
                text=True,
                errors="replace",
                timeout=10,
            )
            for line in out.splitlines():
                if f":{port} " in line or line.rstrip().endswith(f":{port}"):
                    # users:(("python",pid=123,...
                    if "pid=" in line:
                        for piece in line.split("pid=")[1:]:
                            num = "".join(ch for ch in piece if ch.isdigit())
                            if num:
                                pids.add(int(num))
                                break
        except Exception:
            pass
    return sorted(p for p in pids if p > 0)


def stop_process_on_comfy_port(settings: Settings, log: LogFn | None = None) -> dict[str, Any]:
    """Stop whatever is holding COMFYUI_PORT so the H3 install can bind."""
    global _comfy_proc
    port = int(settings.comfyui_port)
    pids = _pids_listening_on_port(port)
    # Also terminate a process we started earlier
    if _comfy_proc and _comfy_proc.poll() is None:
        try:
            pids = sorted(set(pids) | {_comfy_proc.pid})
        except Exception:
            pass

    if not pids:
        return {"ok": True, "status": "port_free", "killed": []}

    _emit(log, f"Stopping non-H3 / conflicting process(es) on port {port}: {pids}")
    killed: list[int] = []
    for pid in pids:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    timeout=30,
                )
            else:
                os.kill(pid, 15)
            killed.append(pid)
        except Exception as exc:
            _emit(log, f"Could not stop pid {pid}: {exc}")

    # Wait until port answers no longer
    deadline = time.time() + 25
    while time.time() < deadline:
        if not comfy_reachable(settings, timeout=1.0) and not _pids_listening_on_port(port):
            break
        time.sleep(0.5)

    if _comfy_proc and _comfy_proc.poll() is not None:
        _comfy_proc = None

    still = _pids_listening_on_port(port)
    if still:
        return {
            "ok": False,
            "status": "port_busy",
            "killed": killed,
            "remaining": still,
            "error": f"Port {port} still held by PIDs {still}",
        }
    return {"ok": True, "status": "stopped", "killed": killed}


def start_comfyui(settings: Settings, log: LogFn | None = None) -> dict[str, Any]:
    """Launch ComfyUI in a new process if not already reachable."""
    global _comfy_proc
    if comfy_reachable(settings):
        return {"ok": True, "status": "already_running", "url": settings.comfy_base_url}

    root, py = _comfy_paths(settings)
    main_py = root / "main.py"
    if not main_py.exists():
        return {
            "ok": False,
            "status": "missing",
            "error": f"ComfyUI main.py not found at {main_py}",
            "hint": "Set COMFYUI_ROOT to your ComfyUI install.",
        }
    if not py.exists():
        return {
            "ok": False,
            "status": "missing_python",
            "error": f"Python not found at {py}",
            "hint": "Set COMFYUI_PYTHON to ComfyUI's venv python.",
        }

    extra = [a for a in (settings.comfyui_extra_args or []) if a]
    cmd = [
        str(py),
        str(main_py),
        "--listen",
        settings.comfyui_host,
        "--port",
        str(settings.comfyui_port),
        *extra,
    ]
    wait_budget = max(15, min(settings.comfy_start_timeout_sec, settings.essentials_wait_sec))
    _emit(log, f"Starting ComfyUI (wait up to {wait_budget}s): {' '.join(cmd[:4])}…")
    try:
        kwargs: dict[str, Any] = {
            "cwd": str(root),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = _windows_new_console_flags()
        else:
            kwargs["start_new_session"] = True

        _comfy_proc = subprocess.Popen(cmd, **kwargs)
    except Exception as exc:
        return {"ok": False, "status": "start_failed", "error": str(exc)}

    deadline = time.time() + wait_budget
    next_tick = time.time() + 15
    while time.time() < deadline:
        if comfy_reachable(settings, timeout=2.0):
            _emit(log, f"ComfyUI is up at {settings.comfy_base_url}")
            return {
                "ok": True,
                "status": "started",
                "url": settings.comfy_base_url,
                "pid": _comfy_proc.pid if _comfy_proc else None,
            }
        if _comfy_proc and _comfy_proc.poll() is not None:
            return {
                "ok": False,
                "status": "exited",
                "error": f"ComfyUI process exited early code={_comfy_proc.returncode}",
            }
        if time.time() >= next_tick:
            left = int(deadline - time.time())
            _emit(log, f"Still waiting for ComfyUI… (~{left}s left before giving up)")
            next_tick = time.time() + 15
        time.sleep(1.5)

    return {
        "ok": False,
        "status": "timeout",
        "error": (
            f"ComfyUI did not answer {settings.comfy_base_url} within {wait_budget}s. "
            "Start Comfy manually or check COMFYUI_ROOT / its console for errors."
        ),
        "pid": _comfy_proc.pid if _comfy_proc else None,
    }


def start_ollama(settings: Settings, log: LogFn | None = None) -> dict[str, Any]:
    """Launch Ollama serve if local LLM is enabled and unreachable."""
    global _ollama_proc
    if not settings.local_llm_enabled:
        return {"ok": True, "status": "disabled"}
    if ollama_reachable(settings):
        return {"ok": True, "status": "already_running"}

    # Only auto-start for default Ollama ports / hostnames
    base = settings.local_llm_base_url.lower()
    if "11434" not in base and "ollama" not in base:
        return {
            "ok": False,
            "status": "skipped",
            "error": "LOCAL_LLM_BASE_URL doesn't look like Ollama; not auto-starting.",
        }

    exe = settings.ollama_cmd.strip() or "ollama"
    resolved = shutil.which(exe) or exe
    if sys.platform == "win32" and not Path(resolved).exists() and not shutil.which(exe):
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(r"C:\Program Files\Ollama\ollama.exe"),
        ]
        for c in candidates:
            if c.exists():
                resolved = str(c)
                break

    cmd = [resolved, "serve"]
    _emit(log, f"Starting Ollama: {cmd[0]} serve …")
    try:
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = _windows_new_console_flags()
        else:
            kwargs["start_new_session"] = True
        _ollama_proc = subprocess.Popen(cmd, **kwargs)
    except Exception as exc:
        return {"ok": False, "status": "start_failed", "error": str(exc)}

    deadline = time.time() + max(10, settings.ollama_start_timeout_sec)
    while time.time() < deadline:
        if ollama_reachable(settings, timeout=2.0):
            _emit(log, "Ollama is reachable")
            return {
                "ok": True,
                "status": "started",
                "pid": _ollama_proc.pid if _ollama_proc else None,
            }
        if _ollama_proc and _ollama_proc.poll() is not None:
            # On some installs `ollama serve` hands off and exits; re-check once more
            if ollama_reachable(settings):
                return {"ok": True, "status": "started_handoff"}
            return {
                "ok": False,
                "status": "exited",
                "error": f"ollama serve exited code={_ollama_proc.returncode}",
            }
        time.sleep(1.0)

    return {
        "ok": False,
        "status": "timeout",
        "error": f"Ollama not reachable within {settings.ollama_start_timeout_sec}s",
    }


def ensure_h3_comfy(settings: Settings, log: LogFn | None = None) -> dict[str, Any]:
    """
    Ensure the Comfy instance at COMFYUI_HOST:PORT has MiniMax H3 nodes.
    If another Comfy (without H3) already owns the port, optionally replace it
    with COMFYUI_ROOT (default E:/AI/ComfyUI).
    """
    if comfy_reachable(settings):
        cap = comfy_h3_status(settings)
        if cap.get("ok"):
            _emit(log, f"ComfyUI H3-ready at {settings.comfy_base_url}")
            return {
                "ok": True,
                "status": "already_running",
                "url": settings.comfy_base_url,
                "h3": cap,
            }

        missing = cap.get("missing_nodes") or []
        _emit(
            log,
            f"ComfyUI is up at {settings.comfy_base_url} but missing H3 nodes: {missing}",
        )
        if not settings.comfy_replace_non_h3:
            return {
                "ok": False,
                "status": "wrong_comfy",
                "url": settings.comfy_base_url,
                "h3": cap,
                "error": (
                    f"Comfy at {settings.comfy_base_url} lacks MiniMax H3 nodes "
                    f"({', '.join(missing)}). Stop that process and start "
                    f"{settings.comfyui_root}, or set COMFY_REPLACE_NON_H3=true."
                ),
            }
        if not settings.auto_start_comfy:
            return {
                "ok": False,
                "status": "wrong_comfy",
                "h3": cap,
                "error": (
                    "Wrong ComfyUI on the port (no H3 nodes) and AUTO_START_COMFY=false. "
                    f"Start {settings.comfyui_root} yourself."
                ),
            }

        stop = stop_process_on_comfy_port(settings, log=log)
        if not stop.get("ok"):
            return {
                "ok": False,
                "status": "replace_failed",
                "h3": cap,
                "stop": stop,
                "error": stop.get("error") or "Could not free Comfy port",
            }
        started = start_comfyui(settings, log=log)
        if not started.get("ok"):
            return {**started, "h3": cap, "stop": stop}

        cap2 = comfy_h3_status(settings)
        if settings.comfy_require_h3_nodes and not cap2.get("ok"):
            return {
                "ok": False,
                "status": "started_without_h3",
                "url": settings.comfy_base_url,
                "h3": cap2,
                "stop": stop,
                "error": (
                    f"Started Comfy from {settings.comfyui_root} but H3 nodes still missing: "
                    f"{cap2.get('missing_nodes')}. Update ComfyUI to a build with nodes_minimax_h3."
                ),
            }
        _emit(log, f"ComfyUI H3 nodes OK after replace -> {settings.comfy_base_url}")
        return {**started, "status": "replaced", "h3": cap2, "stop": stop}

    # Not reachable
    if settings.auto_start_comfy:
        started = start_comfyui(settings, log=log)
        if not started.get("ok"):
            return started
        cap = comfy_h3_status(settings)
        if settings.comfy_require_h3_nodes and not cap.get("ok"):
            return {
                "ok": False,
                "status": "started_without_h3",
                "url": settings.comfy_base_url,
                "h3": cap,
                "error": (
                    f"ComfyUI started but lacks H3 nodes: {cap.get('missing_nodes')}. "
                    f"Check COMFYUI_ROOT={settings.comfyui_root}."
                ),
            }
        return {**started, "h3": cap}

    return {
        "ok": False,
        "status": "not_running",
        "error": "ComfyUI down and AUTO_START_COMFY=false",
    }


def ensure_runtime_services(
    settings: Settings,
    log: LogFn | None = None,
    *,
    need_comfy: bool = True,
    need_ollama: bool = True,
) -> dict[str, Any]:
    """
    Bring up ComfyUI / Ollama when configured to auto-start.
    Returns a report; does not raise unless caller's choice.
    """
    report: dict[str, Any] = {"comfy": None, "ollama": None}

    if need_ollama and settings.auto_start_ollama and settings.local_llm_enabled:
        if ollama_reachable(settings):
            report["ollama"] = {"ok": True, "status": "already_running"}
            _emit(log, "Ollama already running")
        else:
            report["ollama"] = start_ollama(settings, log=log)
    elif need_ollama:
        report["ollama"] = {
            "ok": ollama_reachable(settings),
            "status": "already_running" if ollama_reachable(settings) else "not_started",
        }

    if need_comfy:
        report["comfy"] = ensure_h3_comfy(settings, log=log)

    return report


def invalidate_service_caches() -> None:
    """Drop cached health probes so the next check hits live services."""
    global _h3_status_cache
    _h3_status_cache = {"ts": 0.0, "value": None, "key": ""}


def heal_runtime_services(
    settings: Settings,
    log: LogFn | None = None,
    *,
    need_comfy: bool = True,
    need_ollama: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """
    Self-heal dependency stack (ComfyUI / Ollama) without waiting on the user.

    Unlike ensure_runtime_services, this can free a hung port that no longer
    answers HTTP, then (re)start the correct MiniMax-H3 Comfy install.
    Safe to call mid-generation when a dependency becomes unreachable.
    """
    invalidate_service_caches()
    tag = (reason or "runtime").strip() or "runtime"
    _emit(log, f"Self-heal ({tag}): restoring runtime services…")
    report: dict[str, Any] = {
        "reason": tag,
        "comfy": None,
        "ollama": None,
        "actions": [],
    }

    if need_comfy:
        up = comfy_reachable(settings, timeout=2.0)
        if not up and settings.auto_start_comfy:
            try:
                pids = _pids_listening_on_port(settings.comfyui_port)
            except Exception:
                pids = []
            if pids:
                _emit(
                    log,
                    f"Self-heal: port {settings.comfyui_port} held by {pids} but not answering — freeing…",
                )
                stop = stop_process_on_comfy_port(settings, log=log)
                report["actions"].append({"action": "free_comfy_port", "result": stop})
            report["comfy"] = ensure_h3_comfy(settings, log=log)
        elif not up:
            report["comfy"] = ensure_h3_comfy(settings, log=log)
        else:
            cap = comfy_h3_status(settings, use_cache=False)
            if cap.get("ok") or not settings.comfy_require_h3_nodes:
                report["comfy"] = {
                    "ok": True,
                    "status": "healthy",
                    "url": settings.comfy_base_url,
                    "h3": cap,
                }
                _emit(log, f"Self-heal: ComfyUI healthy at {settings.comfy_base_url}")
            else:
                # Wrong install on the port / missing H3 — replace if allowed
                report["comfy"] = ensure_h3_comfy(settings, log=log)
        invalidate_service_caches()

    if need_ollama and settings.local_llm_enabled:
        if ollama_reachable(settings, timeout=2.0):
            report["ollama"] = {"ok": True, "status": "healthy"}
            _emit(log, "Self-heal: Ollama healthy")
        elif settings.auto_start_ollama:
            report["ollama"] = start_ollama(settings, log=log)
        else:
            report["ollama"] = {
                "ok": False,
                "status": "not_running",
                "error": "Ollama down and AUTO_START_OLLAMA=false",
            }

    return report


def ensure_comfy_or_raise(settings: Settings, log: LogFn | None = None) -> None:
    report = heal_runtime_services(
        settings, log=log, need_comfy=True, need_ollama=False, reason="ensure_or_raise"
    )
    comfy = report.get("comfy") or {}
    if not comfy.get("ok") and not comfy_reachable(settings):
        raise RuntimeError(
            comfy.get("error")
            or f"ComfyUI not reachable at {settings.comfy_base_url}"
        )


def ffmpeg_available(settings: Settings) -> bool:
    path = settings.ffmpeg_path or "ffmpeg"
    if Path(path).is_file():
        return True
    if shutil.which(path) is not None:
        return True
    # Shared AI_ROOT portable install (bootstrap places it here)
    _exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    for cand in (
        Path(settings.ai_root) / "FFmpeg" / "bin" / _exe,
        Path(settings.ai_root) / "FFmpeg" / _exe,
    ):
        if cand.is_file():
            return True
    try:
        ff_root = Path(settings.ai_root) / "FFmpeg"
        if ff_root.is_dir():
            for found in ff_root.rglob(_exe):
                if found.is_file():
                    return True
    except Exception:
        pass
    return False


def resolve_ffmpeg(settings: Settings) -> str:
    """Best ffmpeg executable path for subprocess calls."""
    path = settings.ffmpeg_path or "ffmpeg"
    if Path(path).is_file():
        return str(Path(path))
    w = shutil.which(path)
    if w:
        return w
    _exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    for cand in (
        Path(settings.ai_root) / "FFmpeg" / "bin" / _exe,
        Path(settings.ai_root) / "FFmpeg" / _exe,
    ):
        if cand.is_file():
            return str(cand)
    try:
        ff_root = Path(settings.ai_root) / "FFmpeg"
        if ff_root.is_dir():
            for found in ff_root.rglob(_exe):
                if found.is_file():
                    return str(found)
    except Exception:
        pass
    return path


def essentials_report(settings: Settings) -> dict[str, Any]:
    """
    Snapshot of tools needed for full generation.
    `blocking` items must be healthy before Generate can succeed.
    `warnings` are soft (pipeline can degrade / fall back).
    """
    services: list[dict[str, Any]] = []
    blocking: list[str] = []
    warnings: list[str] = []

    comfy_ok = comfy_reachable(settings)
    h3 = comfy_h3_status(settings) if comfy_ok else {
        "ok": False,
        "reachable": False,
        "missing_nodes": ["MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"],
    }
    h3_ok = bool(h3.get("ok"))
    comfy_ready = comfy_ok and (h3_ok or not settings.comfy_require_h3_nodes)
    if comfy_ok and h3_ok:
        comfy_detail = f"{settings.comfy_base_url} (H3 nodes OK)"
        comfy_fix = None
    elif comfy_ok and not h3_ok:
        miss = ", ".join(h3.get("missing_nodes") or [])
        comfy_detail = f"{settings.comfy_base_url} but missing H3 nodes: {miss}"
        comfy_fix = (
            "Another ComfyUI is bound to this port without MiniMax H3. "
            f"Stop it and start {settings.comfyui_root}, or enable AUTO_START_COMFY + "
            "COMFY_REPLACE_NON_H3 (default true) so H3VideoGen can replace it."
        )
    else:
        comfy_detail = f"Not reachable at {settings.comfy_base_url}"
        comfy_fix = (
            "Start ComfyUI with H3 models, or enable AUTO_START_COMFY "
            f"(COMFYUI_ROOT={settings.comfyui_root}). Wait limit "
            f"{min(settings.comfy_start_timeout_sec, settings.essentials_wait_sec)}s."
        )
    services.append(
        {
            "id": "comfyui",
            "name": "ComfyUI (MiniMax H3)",
            "ok": comfy_ready,
            "required": True,
            "detail": comfy_detail,
            "fix": comfy_fix,
            "h3": h3,
        }
    )
    if not comfy_ok:
        blocking.append(
            f"ComfyUI is not running at {settings.comfy_base_url} — required for video generation."
        )
    elif settings.comfy_require_h3_nodes and not h3_ok:
        blocking.append(
            f"ComfyUI at {settings.comfy_base_url} is missing MiniMax H3 nodes "
            f"({', '.join(h3.get('missing_nodes') or [])}). "
            "Use the E:/AI/ComfyUI install (or update Comfy)."
        )

    ff_ok = ffmpeg_available(settings)
    services.append(
        {
            "id": "ffmpeg",
            "name": "FFmpeg",
            "ok": ff_ok,
            "required": True,
            "detail": settings.ffmpeg_path if ff_ok else f"Not found: {settings.ffmpeg_path}",
            "fix": None if ff_ok else "Install FFmpeg and set FFMPEG_PATH / FFPROBE_PATH in .env.",
        }
    )
    if not ff_ok:
        blocking.append(f"FFmpeg not found ({settings.ffmpeg_path}) — required for frames/master.")

    gemini_ok = bool(settings.gemini_api_key and settings.gemini_api_key.strip())
    services.append(
        {
            "id": "gemini",
            "name": "Gemini API key",
            "ok": gemini_ok,
            "required": False,
            "detail": "set" if gemini_ok else "GEMINI_API_KEY missing",
            "fix": None
            if gemini_ok
            else "Set GEMINI_API_KEY in .env for best director/critic (offline/local fallback otherwise).",
        }
    )
    if not gemini_ok:
        warnings.append("Gemini key missing — director/critic will fall back to local LLM or offline.")

    ollama_ok = ollama_reachable(settings) if settings.local_llm_enabled else True
    if settings.local_llm_enabled:
        vision_detail = ""
        vision_ok = True
        if ollama_ok:
            try:
                from .llm.local_openai import LocalOpenAIBackend

                back = LocalOpenAIBackend(settings)
                models = back.list_models()
                has_vision = any(m.get("vision") for m in models)
                try:
                    vname = back.resolve_model(images=True)
                except Exception:
                    vname = settings.local_llm_vision_model or ""
                vision_ok = bool(has_vision or (vname and back._name_suggests_vision(vname)))
                vision_detail = (
                    f"text={settings.local_llm_model} vision={vname or 'unset'}"
                    if vision_ok
                    else f"text={settings.local_llm_model}; no vision model (pull qwen2.5vl)"
                )
            except Exception as exc:
                vision_ok = False
                vision_detail = f"model probe failed: {exc}"
        services.append(
            {
                "id": "ollama",
                "name": "Local LLM (Ollama)",
                "ok": ollama_ok,
                "required": False,
                "detail": (
                    f"{settings.local_llm_base_url} ({vision_detail})"
                    if ollama_ok
                    else f"Not reachable at {settings.local_llm_base_url}"
                ),
                "fix": None
                if ollama_ok
                else "Start Ollama (`ollama serve`) or enable AUTO_START_OLLAMA — used when Gemini is down.",
            }
        )
        if ollama_ok and not vision_ok:
            warnings.append(
                "Ollama has no vision model — critic frame QA needs "
                f"`ollama pull {settings.local_llm_vision_model or 'qwen2.5vl'}` "
                "and LOCAL_LLM_VISION_MODEL for cast exclusivity when Gemini is down."
            )
        if not ollama_ok and not gemini_ok:
            blocking.append(
                "Neither Gemini nor local LLM (Ollama) is available — director will use offline templates only."
            )
        elif not ollama_ok:
            warnings.append(
                f"Ollama not running at {settings.local_llm_base_url} — Gemini is the only strong planner/critic."
            )

    wait_sec = max(30, min(settings.essentials_wait_sec, 600))
    return {
        "ok": len(blocking) == 0,
        "ready_for_generate": comfy_ready and ff_ok,
        "services": services,
        "blocking": blocking,
        "warnings": warnings,
        "wait_limit_sec": wait_sec,
        "prompt": _essentials_prompt(blocking, warnings, wait_sec),
    }


def _essentials_prompt(blocking: list[str], warnings: list[str], wait_sec: int) -> str | None:
    if not blocking and not warnings:
        return None
    lines = ["Prerequisites check:"]
    if blocking:
        lines.append("BLOCKING (fix these before Generate):")
        lines.extend(f"  • {b}" for b in blocking)
        lines.append(
            f"Self-heal will auto-start ComfyUI/Ollama (≤{wait_sec // 60} min / {wait_sec}s) and keep the job going."
        )
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  • {w}" for w in warnings)
    return "\n".join(lines)
