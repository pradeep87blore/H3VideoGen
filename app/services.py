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
    _emit(log, f"Starting ComfyUI: {' '.join(cmd[:4])} … (cwd={root})")
    try:
        kwargs: dict[str, Any] = {
            "cwd": str(root),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = _windows_new_console_flags()
            # Don't inherit handles; survive parent exit as own window/process
        else:
            kwargs["start_new_session"] = True

        _comfy_proc = subprocess.Popen(cmd, **kwargs)
    except Exception as exc:
        return {"ok": False, "status": "start_failed", "error": str(exc)}

    deadline = time.time() + max(15, settings.comfy_start_timeout_sec)
    while time.time() < deadline:
        if comfy_reachable(settings, timeout=2.0):
            _emit(log, f"ComfyUI is up at {settings.comfy_base_url}")
            return {
                "ok": True,
                "status": "started",
                "url": settings.comfy_base_url,
                "pid": _comfy_proc.pid if _comfy_proc else None,
            }
        # Early exit if process died
        if _comfy_proc and _comfy_proc.poll() is not None:
            return {
                "ok": False,
                "status": "exited",
                "error": f"ComfyUI process exited early code={_comfy_proc.returncode}",
            }
        time.sleep(1.5)

    return {
        "ok": False,
        "status": "timeout",
        "error": (
            f"ComfyUI did not answer {settings.comfy_base_url} within "
            f"{settings.comfy_start_timeout_sec}s (still starting? check its console)."
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
        if comfy_reachable(settings):
            report["comfy"] = {"ok": True, "status": "already_running", "url": settings.comfy_base_url}
            _emit(log, f"ComfyUI already running at {settings.comfy_base_url}")
        elif settings.auto_start_comfy:
            report["comfy"] = start_comfyui(settings, log=log)
        else:
            report["comfy"] = {
                "ok": False,
                "status": "not_running",
                "error": "ComfyUI down and AUTO_START_COMFY=false",
            }

    return report


def ensure_comfy_or_raise(settings: Settings, log: LogFn | None = None) -> None:
    report = ensure_runtime_services(settings, log=log, need_comfy=True, need_ollama=False)
    comfy = report.get("comfy") or {}
    if not comfy.get("ok") and not comfy_reachable(settings):
        raise RuntimeError(
            comfy.get("error")
            or f"ComfyUI not reachable at {settings.comfy_base_url}"
        )
