"""
Ensure AI_ROOT prerequisites: folders, ComfyUI, MiniMax H3 models, FFmpeg, Ollama models.

Respects the shared layout:
  E:/AI/Models/...           (weights)
  E:/AI/ComfyUI/...          (runtime + optional junctions -> Models)
  E:/AI/FFmpeg/...           (optional portable tools)

Runs on app launch (background) and via `python run.py bootstrap`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Settings, get_settings

LogFn = Callable[[str], None]

HF_REPO = "Comfy-Org/MiniMax-H3"
COMFY_GIT = "https://github.com/comfyanonymous/ComfyUI.git"

# Portable FFmpeg (Windows essentials build)
FFMPEG_WIN_URL = (
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
)

# Status shared with the API / UI
_status_lock = threading.Lock()
_status: dict[str, Any] = {
    "running": False,
    "done": False,
    "phase": "idle",
    "message": "",
    "log": [],
    "items": [],
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_thread: threading.Thread | None = None


def bootstrap_status() -> dict[str, Any]:
    with _status_lock:
        return dict(_status)


def _set_status(**kwargs: Any) -> None:
    with _status_lock:
        _status.update(kwargs)
        if "message" in kwargs and kwargs["message"]:
            log = list(_status.get("log") or [])
            log.append(str(kwargs["message"]))
            _status["log"] = log[-200:]


def _emit(log: LogFn | None, msg: str) -> None:
    _set_status(message=msg)
    if log:
        log(msg)
    else:
        print(f"[bootstrap] {msg}", flush=True)


@dataclass
class BootstrapReport:
    ok: bool
    items: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "items": self.items,
            "actions": self.actions,
            "errors": self.errors,
            "message": self.message,
        }


def models_root(settings: Settings) -> Path:
    """Shared weight folder (E:/AI/Models preferred over ComfyUI/models)."""
    shared = Path(settings.ai_root) / "Models"
    if shared.is_dir() or settings.auto_install_use_shared_models:
        return shared
    return Path(settings.comfyui_root) / "models"


def model_subdir(settings: Settings, kind: str) -> Path:
    return models_root(settings) / kind


def h3_model_specs(settings: Settings) -> list[dict[str, str]]:
    """Files required for MiniMax H3 relative to HF repo folder keys."""
    return [
        {
            "id": "h3_unet",
            "name": "H3 FL2VA diffusion (T2V/I2V)",
            "repo_file": f"diffusion_models/{settings.h3_unet}",
            "dest": str(model_subdir(settings, "diffusion_models") / settings.h3_unet),
            "kind": "diffusion_models",
        },
        {
            "id": "h3_unet_r2v",
            "name": "H3 Ref2VA diffusion (R2V)",
            "repo_file": f"diffusion_models/{settings.h3_unet_r2v}",
            "dest": str(model_subdir(settings, "diffusion_models") / settings.h3_unet_r2v),
            "kind": "diffusion_models",
        },
        {
            "id": "h3_clip",
            "name": "H3 text encoder (CLIP)",
            "repo_file": f"text_encoders/{settings.h3_clip}",
            "dest": str(model_subdir(settings, "text_encoders") / settings.h3_clip),
            "kind": "text_encoders",
        },
        {
            "id": "h3_video_vae",
            "name": "H3 video VAE",
            "repo_file": f"vae/{settings.h3_video_vae}",
            "dest": str(model_subdir(settings, "vae") / settings.h3_video_vae),
            "kind": "vae",
        },
        {
            "id": "h3_audio_vae",
            "name": "H3 audio VAE",
            "repo_file": f"vae/{settings.h3_audio_vae}",
            "dest": str(model_subdir(settings, "vae") / settings.h3_audio_vae),
            "kind": "vae",
        },
    ]


def _file_ok(path: Path, *, min_bytes: int = 1_000_000) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _min_bytes_for(spec_id: str) -> int:
    if "audio" in spec_id:
        return 50_000_000  # ~580MB expected
    if "video_vae" in spec_id or spec_id.endswith("video_vae"):
        return 100_000_000
    if "vae" in spec_id:
        return 50_000_000
    return 1_000_000_000  # diffusion + clip ≈ multi-GB


def _hf_download_url(repo_file: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{repo_file}"


def _download_file(
    url: str,
    dest: Path,
    *,
    log: LogFn | None = None,
    timeout: float = 120.0,
    headers: dict[str, str] | None = None,
) -> None:
    """Download with simple resume (Range) and progress logs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(dest) + ".partial")

    existing = part.stat().st_size if part.is_file() else 0
    hdrs = {"User-Agent": "H3VideoGen/1.0"}
    if headers:
        hdrs.update(headers)
    if existing > 0:
        hdrs["Range"] = f"bytes={existing}-"

    _emit(
        log,
        f"Downloading {dest.name} ..."
        + (f" (resume {existing // (1024**2)} MiB)" if existing else ""),
    )

    with httpx.stream("GET", url, headers=hdrs, follow_redirects=True, timeout=timeout) as r:
        if r.status_code == 416 and existing > 0:
            if dest.exists():
                dest.unlink(missing_ok=True)
            part.replace(dest)
            return
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")

        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb" and part.exists():
            part.unlink(missing_ok=True)
            existing = 0

        total = None
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            try:
                total = int(cr.rsplit("/", 1)[-1])
            except ValueError:
                total = None
        elif r.headers.get("Content-Length"):
            try:
                cl = int(r.headers["Content-Length"])
                total = cl + (existing if mode == "ab" else 0)
            except ValueError:
                total = None

        written = existing
        last_log = time.time()
        with part.open(mode) as f:
            for chunk in r.iter_bytes(1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_log >= 8.0:
                    if total:
                        pct = 100.0 * written / max(total, 1)
                        _emit(
                            log,
                            f"  ... {dest.name}: {written // (1024**2)} / "
                            f"{total // (1024**2)} MiB ({pct:.1f}%)",
                        )
                    else:
                        _emit(log, f"  ... {dest.name}: {written // (1024**2)} MiB")
                    last_log = now

    if dest.exists():
        dest.unlink(missing_ok=True)
    part.replace(dest)
    _emit(log, f"Downloaded {dest.name} ({dest.stat().st_size // (1024**2)} MiB)")


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    log: LogFn | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    _emit(log, f"$ {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        env=merged,
        timeout=timeout,
        check=False,
    )


def ensure_ai_layout(settings: Settings, log: LogFn | None = None) -> list[str]:
    """Create standard AI_ROOT folders."""
    root = Path(settings.ai_root)
    created: list[str] = []
    for sub in (
        "",
        "Models",
        "Models/diffusion_models",
        "Models/text_encoders",
        "Models/vae",
        "Models/checkpoints",
        "Models/clip",
        "Models/loras",
        "ComfyUI",
        "FFmpeg",
        "Outputs",
        "Temp",
        "Logs",
    ):
        p = root / sub if sub else root
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            created.append(str(p))
    if created:
        _emit(log, f"Created folders under {root}: {len(created)}")
    return created


def ensure_extra_model_paths(settings: Settings, log: LogFn | None = None) -> None:
    """Point ComfyUI at AI_ROOT/Models via extra_model_paths.yaml."""
    comfy = Path(settings.comfyui_root)
    if not comfy.is_dir():
        return
    yaml_path = comfy / "extra_model_paths.yaml"
    models = models_root(settings).as_posix().rstrip("/") + "/"
    # Only rewrite if missing or our marker, preserve other custom configs
    body = ""
    if yaml_path.is_file():
        body = yaml_path.read_text(encoding="utf-8", errors="replace")
        low = body.lower()
        # Preserve third-party managers; only rewrite our own file or empty/example stubs
        if "h3wrapper" in low and "h3videogen" not in low:
            _emit(log, f"Leaving existing extra_model_paths.yaml ({yaml_path})")
            return
        if "h3videogen" not in low and "base_path" in low and "h3wrapper" not in low:
            # Unknown custom config — do not clobber
            _emit(log, f"Leaving custom extra_model_paths.yaml ({yaml_path})")
            return
    content = (
        "# Managed by H3VideoGen - auto-install\n"
        "h3videogen:\n"
        f"    base_path: {models}\n"
        "    checkpoints: checkpoints/\n"
        "    clip: clip/\n"
        "    clip_vision: clip_vision/\n"
        "    configs: configs/\n"
        "    controlnet: controlnet/\n"
        "    diffusion_models: diffusion_models/\n"
        "    embeddings: embeddings/\n"
        "    loras: loras/\n"
        "    text_encoders: text_encoders/\n"
        "    unet: unet/\n"
        "    upscale_models: upscale_models/\n"
        "    vae: vae/\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    _emit(log, f"Wrote {yaml_path} -> {models}")


def _ensure_junction(link: Path, target: Path, log: LogFn | None = None) -> None:
    """Windows directory junction so ComfyUI/models/<kind> -> AI/Models/<kind>."""
    target.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        if not link.exists():
            try:
                link.symlink_to(target, target_is_directory=True)
                _emit(log, f"Symlink {link} -> {target}")
            except OSError as exc:
                _emit(log, f"Symlink failed {link}: {exc}")
        return

    try:
        if link.exists() and link.resolve() == target.resolve():
            return
    except Exception:
        pass
    # Existing non-empty real directory - leave (may already be a junction)
    if link.is_dir():
        try:
            if any(link.iterdir()):
                return
            link.rmdir()
        except OSError:
            return
    link.parent.mkdir(parents=True, exist_ok=True)
    r = _run(["cmd", "/c", "mklink", "/J", str(link), str(target)], log=log)
    if r.returncode != 0:
        _emit(log, f"Junction failed: {(r.stderr or r.stdout or '')[:200]}")
    else:
        _emit(log, f"Junction {link} -> {target}")


def ensure_model_junctions(settings: Settings, log: LogFn | None = None) -> None:
    if not settings.auto_install_use_shared_models:
        return
    comfy_models = Path(settings.comfyui_root) / "models"
    if not Path(settings.comfyui_root).is_dir():
        return
    comfy_models.mkdir(parents=True, exist_ok=True)
    for kind in ("diffusion_models", "text_encoders", "vae", "checkpoints", "clip", "loras"):
        _ensure_junction(comfy_models / kind, model_subdir(settings, kind), log=log)


def ensure_comfyui(settings: Settings, log: LogFn | None = None) -> dict[str, Any]:
    """Clone ComfyUI if missing; optionally create venv + pip install requirements."""
    root = Path(settings.comfyui_root)
    main_py = root / "main.py"
    out: dict[str, Any] = {"ok": False, "path": str(root), "actions": []}

    if main_py.is_file():
        _emit(log, f"ComfyUI OK at {root}")
        out["ok"] = True
        out["status"] = "present"
        return out

    if not settings.auto_install_comfy:
        out["error"] = f"ComfyUI missing at {root} and AUTO_INSTALL_COMFY=false"
        return out

    git = shutil.which("git")
    if not git:
        out["error"] = "git not found on PATH - cannot clone ComfyUI"
        _emit(log, out["error"])
        return out

    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and not main_py.exists():
        # Empty/partial folder - clone into place
        if any(root.iterdir()):
            # non-empty incomplete: clone to temp name then? try git init pull
            _emit(log, f"ComfyUI folder incomplete at {root}; attempting git clone into temp...")
            tmp = parent / f"{root.name}_clone_tmp"
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            r = _run([git, "clone", "--depth", "1", COMFY_GIT, str(tmp)], log=log, timeout=600)
            if r.returncode != 0:
                out["error"] = (r.stderr or r.stdout or "git clone failed")[:500]
                return out
            # merge main files if needed
            for item in tmp.iterdir():
                dest = root / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            r = _run([git, "clone", "--depth", "1", COMFY_GIT, str(root)], log=log, timeout=600)
            if r.returncode != 0:
                out["error"] = (r.stderr or r.stdout or "git clone failed")[:500]
                return out
    else:
        r = _run([git, "clone", "--depth", "1", COMFY_GIT, str(root)], log=log, timeout=600)
        if r.returncode != 0:
            out["error"] = (r.stderr or r.stdout or "git clone failed")[:500]
            _emit(log, out["error"])
            return out

    out["actions"].append("cloned_comfyui")
    _emit(log, f"Cloned ComfyUI -> {root}")

    if settings.auto_install_comfy_deps and main_py.is_file():
        py = _ensure_comfy_venv(settings, log=log)
        if py:
            req = root / "requirements.txt"
            if req.is_file():
                r = _run(
                    [str(py), "-m", "pip", "install", "-r", str(req)],
                    cwd=root,
                    log=log,
                    timeout=1800,
                )
                if r.returncode != 0:
                    _emit(log, f"Comfy pip install had errors (may still work): {(r.stderr or '')[-400:]}")
                else:
                    out["actions"].append("comfy_deps")
                    _emit(log, "ComfyUI Python deps installed")

    out["ok"] = main_py.is_file()
    out["status"] = "installed" if out["ok"] else "failed"
    if not out["ok"]:
        out["error"] = "ComfyUI main.py still missing after clone"
    return out


def _ensure_comfy_venv(settings: Settings, log: LogFn | None = None) -> Path | None:
    root = Path(settings.comfyui_root)
    if sys.platform == "win32":
        py = root / ".venv" / "Scripts" / "python.exe"
    else:
        py = root / ".venv" / "bin" / "python"
    if py.is_file():
        return py
    base = sys.executable
    r = _run([base, "-m", "venv", str(root / ".venv")], log=log, timeout=120)
    if r.returncode != 0 or not py.is_file():
        _emit(log, "Could not create ComfyUI .venv")
        return None
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"], log=log, timeout=180)
    return py


def ensure_h3_models(settings: Settings, log: LogFn | None = None) -> dict[str, Any]:
    """Download missing MiniMax H3 weights into Models root."""
    results: list[dict[str, Any]] = []
    actions: list[str] = []
    errors: list[str] = []
    token = (settings.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else None

    for spec in h3_model_specs(settings):
        dest = Path(spec["dest"])
        item = {"id": spec["id"], "name": spec["name"], "path": str(dest), "ok": False}
        floor = _min_bytes_for(spec["id"])
        if _file_ok(dest, min_bytes=floor):
            item["ok"] = True
            item["status"] = "present"
            results.append(item)
            continue
        if not settings.auto_install_models:
            item["status"] = "missing"
            item["error"] = "AUTO_INSTALL_MODELS=false"
            errors.append(f"{spec['name']} missing")
            results.append(item)
            continue
        url = _hf_download_url(spec["repo_file"])
        try:
            _download_file(
                url,
                dest,
                log=log,
                headers=headers,
                timeout=float(settings.auto_install_download_timeout_sec),
            )
            item["ok"] = _file_ok(dest, min_bytes=floor)
            item["status"] = "downloaded" if item["ok"] else "failed"
            if item["ok"]:
                actions.append(f"downloaded:{spec['id']}")
            else:
                errors.append(f"Download incomplete: {spec['name']}")
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)[:300]
            errors.append(f"{spec['name']}: {exc}")
            _emit(log, f"Failed to download {spec['name']}: {exc}")
        results.append(item)

    ok = all(r.get("ok") for r in results)
    return {"ok": ok, "items": results, "actions": actions, "errors": errors}


def find_ffmpeg_candidates(settings: Settings) -> list[Path]:
    paths: list[Path] = []
    _exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    for p in (
        Path(settings.ffmpeg_path) if settings.ffmpeg_path else None,
        Path(settings.ai_root) / "FFmpeg" / "bin" / _exe,
        Path(settings.ai_root) / "FFmpeg" / _exe,
    ):
        if p:
            paths.append(p)
    which = shutil.which("ffmpeg")
    if which:
        paths.append(Path(which))
    # nested gyan layout
    ff_root = Path(settings.ai_root) / "FFmpeg"
    if ff_root.is_dir():
        for found in ff_root.rglob(_exe):
            paths.append(found)
            break
    return paths


def ensure_ffmpeg(settings: Settings, log: LogFn | None = None) -> dict[str, Any]:
    from .services import ffmpeg_available

    if ffmpeg_available(settings):
        return {"ok": True, "status": "present", "path": settings.ffmpeg_path}

    for cand in find_ffmpeg_candidates(settings):
        if cand.is_file():
            _emit(log, f"Found FFmpeg at {cand}")
            return {"ok": True, "status": "found", "path": str(cand)}

    if not settings.auto_install_ffmpeg or sys.platform != "win32":
        return {
            "ok": False,
            "status": "missing",
            "error": "FFmpeg not found (install or set FFMPEG_PATH)",
        }

    dest_root = Path(settings.ai_root) / "FFmpeg"
    dest_root.mkdir(parents=True, exist_ok=True)
    zip_path = Path(settings.ai_root) / "Temp" / "ffmpeg-essentials.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download_file(FFMPEG_WIN_URL, zip_path, log=log, timeout=600)
        _emit(log, "Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_root)
        # find extracted binary
        _exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        for found in dest_root.rglob(_exe):
            probe = found.parent / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
            _emit(log, f"FFmpeg installed at {found}")
            return {
                "ok": True,
                "status": "installed",
                "path": str(found),
                "ffprobe": str(probe) if probe.is_file() else None,
            }
        return {"ok": False, "status": "error", "error": f"{_exe} not in archive"}
    except Exception as exc:
        _emit(log, f"FFmpeg install failed: {exc}")
        return {"ok": False, "status": "error", "error": str(exc)[:400]}


def ensure_ollama_models(settings: Settings, log: LogFn | None = None) -> dict[str, Any]:
    if not settings.local_llm_enabled or not settings.auto_install_ollama_models:
        return {"ok": True, "status": "skipped"}

    exe = settings.ollama_cmd.strip() or "ollama"
    resolved = shutil.which(exe) or exe
    if sys.platform == "win32" and not Path(resolved).exists() and not shutil.which(exe):
        for c in (
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(r"C:\Program Files\Ollama\ollama.exe"),
        ):
            if c.exists():
                resolved = str(c)
                break
    if not Path(resolved).exists() and not shutil.which(exe):
        return {"ok": True, "status": "ollama_not_installed", "detail": "skip model pull"}

    # Start serve? leave to services auto-start
    models = []
    for m in (settings.local_llm_model, settings.local_llm_vision_model):
        m = (m or "").strip()
        if m and m not in models:
            models.append(m)

    pulled: list[str] = []
    errors: list[str] = []
    for m in models:
        # Check if present
        r = _run([resolved, "list"], log=log, timeout=60)
        present = m.lower() in (r.stdout or "").lower() if r.returncode == 0 else False
        if present:
            _emit(log, f"Ollama model present: {m}")
            continue
        _emit(log, f"Pulling Ollama model {m} ...")
        r2 = _run([resolved, "pull", m], log=log, timeout=float(settings.auto_install_download_timeout_sec))
        if r2.returncode == 0:
            pulled.append(m)
        else:
            errors.append(f"ollama pull {m}: {(r2.stderr or r2.stdout or '')[:200]}")
    return {"ok": not errors, "pulled": pulled, "errors": errors, "status": "done"}


def scan_prereqs(settings: Settings | None = None) -> BootstrapReport:
    """Read-only check of what is missing (no downloads)."""
    settings = settings or get_settings()
    items: list[dict[str, Any]] = []
    errors: list[str] = []

    # AI root
    ai = Path(settings.ai_root)
    items.append(
        {
            "id": "ai_root",
            "name": f"AI root ({ai})",
            "ok": ai.is_dir(),
            "detail": str(ai),
            "action": "ready" if ai.is_dir() else "install",
        }
    )

    # ComfyUI
    comfy_ok = (Path(settings.comfyui_root) / "main.py").is_file()
    items.append(
        {
            "id": "comfyui",
            "name": "ComfyUI",
            "ok": comfy_ok,
            "detail": str(settings.comfyui_root),
            "action": "ready" if comfy_ok else "install",
        }
    )
    if not comfy_ok:
        errors.append(f"ComfyUI missing at {settings.comfyui_root}")

    # Models
    for spec in h3_model_specs(settings):
        dest = Path(spec["dest"])
        ok = _file_ok(dest, min_bytes=_min_bytes_for(spec["id"]))
        items.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "ok": ok,
                "detail": str(dest),
                "action": "ready" if ok else "download",
            }
        )
        if not ok:
            errors.append(f"Missing model: {spec['name']}")

    # FFmpeg
    from .services import ffmpeg_available

    ff_ok = ffmpeg_available(settings) or any(p.is_file() for p in find_ffmpeg_candidates(settings))
    items.append(
        {
            "id": "ffmpeg",
            "name": "FFmpeg",
            "ok": ff_ok,
            "detail": settings.ffmpeg_path,
            "action": "ready" if ff_ok else "install",
        }
    )
    if not ff_ok:
        errors.append("FFmpeg not found")

    ok = not any(not i["ok"] for i in items if i["id"] != "ffmpeg") and ff_ok
    # ffmpeg required for assemble
    return BootstrapReport(
        ok=all(i["ok"] for i in items),
        items=items,
        errors=errors,
        message="All prerequisites present" if not errors else f"{len(errors)} issue(s)",
    )


def ensure_all_prereqs(
    settings: Settings | None = None,
    log: LogFn | None = None,
    *,
    force: bool = False,
) -> BootstrapReport:
    """
    Create folders, install ComfyUI if missing, download H3 models,
    FFmpeg, and Ollama models as configured.
    """
    settings = settings or get_settings()
    if not settings.auto_install_prereqs and not force:
        rep = scan_prereqs(settings)
        rep.message = "AUTO_INSTALL_PREREQS=false - scan only"
        return rep

    actions: list[str] = []
    errors: list[str] = []
    items: list[dict[str, Any]] = []

    _set_status(running=True, done=False, phase="layout", error=None, started_at=time.time())
    try:
        created = ensure_ai_layout(settings, log=log)
        if created:
            actions.append("layout")

        _set_status(phase="comfyui")
        comfy = ensure_comfyui(settings, log=log)
        items.append(
            {
                "id": "comfyui",
                "name": "ComfyUI",
                "ok": bool(comfy.get("ok")),
                "detail": comfy.get("path") or str(settings.comfyui_root),
                "action": comfy.get("status") or "",
            }
        )
        actions.extend(comfy.get("actions") or [])
        if not comfy.get("ok"):
            errors.append(comfy.get("error") or "ComfyUI install failed")

        ensure_extra_model_paths(settings, log=log)
        ensure_model_junctions(settings, log=log)

        _set_status(phase="models")
        models = ensure_h3_models(settings, log=log)
        for it in models.get("items") or []:
            items.append(
                {
                    "id": it["id"],
                    "name": it.get("name") or it["id"],
                    "ok": bool(it.get("ok")),
                    "detail": it.get("path") or "",
                    "action": it.get("status") or "",
                }
            )
        actions.extend(models.get("actions") or [])
        errors.extend(models.get("errors") or [])

        _set_status(phase="ffmpeg")
        ff = ensure_ffmpeg(settings, log=log)
        items.append(
            {
                "id": "ffmpeg",
                "name": "FFmpeg",
                "ok": bool(ff.get("ok")),
                "detail": ff.get("path") or "",
                "action": ff.get("status") or "",
            }
        )
        if ff.get("status") == "installed":
            actions.append("ffmpeg")
        if not ff.get("ok"):
            errors.append(ff.get("error") or "FFmpeg missing")

        _set_status(phase="ollama")
        ol = ensure_ollama_models(settings, log=log)
        items.append(
            {
                "id": "ollama_models",
                "name": "Ollama models",
                "ok": bool(ol.get("ok")),
                "detail": ", ".join(ol.get("pulled") or []) or ol.get("status", ""),
                "action": ol.get("status") or "",
            }
        )
        if ol.get("pulled"):
            actions.append("ollama_models")
        errors.extend(ol.get("errors") or [])

        # Write a small state file for next launch skip optimization
        state_path = Path(settings.ai_root) / "Logs" / "h3videogen_bootstrap.json"
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "ts": time.time(),
                        "ok": len(errors) == 0,
                        "actions": actions,
                        "items": items,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

        report = BootstrapReport(
            ok=len(errors) == 0,
            items=items,
            actions=actions,
            errors=errors,
            message=(
                "Prerequisites ready"
                if not errors
                else f"Bootstrap finished with {len(errors)} issue(s)"
            ),
        )
        _set_status(
            running=False,
            done=True,
            phase="done",
            message=report.message,
            items=items,
            error="; ".join(errors) if errors else None,
            finished_at=time.time(),
        )
        _emit(log, report.message)
        return report
    except Exception as exc:
        _set_status(
            running=False,
            done=True,
            phase="error",
            error=str(exc),
            finished_at=time.time(),
            message=f"Bootstrap failed: {exc}",
        )
        raise


def start_bootstrap_background(settings: Settings | None = None) -> bool:
    """Fire-and-forget installer on a daemon thread. Returns False if already running."""
    global _thread
    settings = settings or get_settings()
    if not settings.auto_install_prereqs:
        return False
    with _status_lock:
        if _status.get("running"):
            return False
        if _thread and _thread.is_alive():
            return False

    def _run_bg() -> None:
        try:
            ensure_all_prereqs(settings, log=None)
        except Exception as exc:
            traceback_msg = str(exc)
            _set_status(running=False, done=True, error=traceback_msg, phase="error")

    t = threading.Thread(target=_run_bg, name="h3-bootstrap", daemon=True)
    _thread = t
    t.start()
    return True


def run_bootstrap_blocking(settings: Settings | None = None, log: LogFn | None = None) -> BootstrapReport:
    """Blocking bootstrap for CLI / launch.bat."""
    settings = settings or get_settings()
    return ensure_all_prereqs(settings, log=log, force=bool(settings.auto_install_prereqs))
