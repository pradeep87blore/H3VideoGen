"""ComfyUI HTTP client for MiniMax H3 T2V + R2V (reference-to-video)."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import httpx

from .config import Settings
from .job_control import CancelledError, JobControl


class ComfyError(RuntimeError):
    pass


class ComfyH3Client:
    def __init__(self, settings: Settings, control: JobControl | None = None):
        self.settings = settings
        self.base = settings.comfy_base_url.rstrip("/")
        self.control = control

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.base}/system_stats")
            r.raise_for_status()
            return r.json()

    def interrupt(self) -> None:
        """Ask ComfyUI to stop the current graph and clear the queue."""
        with httpx.Client(timeout=10.0) as client:
            try:
                client.post(f"{self.base}/interrupt")
            except Exception:
                pass
            try:
                client.post(f"{self.base}/queue", json={"clear": True})
            except Exception:
                pass
            pid = self.control.prompt_id if self.control else None
            if pid:
                try:
                    client.post(f"{self.base}/queue", json={"delete": [pid]})
                except Exception:
                    pass

    # ─── upload refs into Comfy input folder ─────────────────────────────

    def upload_image(self, path: Path, *, subfolder: str = "H3VideoGen") -> str:
        """Upload local image; returns filename usable by LoadImage."""
        path = Path(path)
        if not path.exists():
            raise ComfyError(f"Reference image missing: {path}")
        with path.open("rb") as f:
            files = {
                "image": (path.name, f, "image/png" if path.suffix.lower() == ".png" else "image/jpeg"),
            }
            data = {"overwrite": "true", "subfolder": subfolder, "type": "input"}
            with httpx.Client(timeout=120.0) as client:
                r = client.post(f"{self.base}/upload/image", files=files, data=data)
                if r.status_code >= 400:
                    raise ComfyError(f"Image upload failed {r.status_code}: {r.text[:1500]}")
                body = r.json()
        name = body.get("name") or path.name
        sub = body.get("subfolder") or subfolder
        # LoadImage expects relative path under input; include subfolder if present
        return f"{sub}/{name}" if sub else name

    # ─── workflows ───────────────────────────────────────────────────────

    def _loader_nodes(self, *, unet_name: str) -> dict[str, Any]:
        s = self.settings
        return {
            "6": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": unet_name, "weight_dtype": "default"},
            },
            "13": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": s.h3_clip,
                    "type": "minimax",
                    "device": "default",
                },
            },
            "11": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": s.h3_video_vae},
            },
            "24": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": s.h3_audio_vae},
            },
            "17": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "res_multistep"},
            },
            "9": {
                "class_type": "BasicScheduler",
                "inputs": {
                    "model": ["6", 0],
                    # beta/normal outperform simple for R2V; fine for T2V too
                    "scheduler": "beta" if unet_name == s.h3_unet_r2v else "simple",
                    "steps": s.default_steps,
                    "denoise": 1.0,
                },
            },
            "16": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["6", 0], "conditioning": ["104", 0]},
            },
            "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
            "14": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["15", 0],
                    "guider": ["16", 0],
                    "sampler": ["17", 0],
                    "sigmas": ["9", 0],
                    "latent_image": ["104", 1],
                },
            },
            "10": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["14", 0], "vae": ["11", 0]},
            },
            "23": {
                "class_type": "VAEDecodeAudio",
                "inputs": {"samples": ["14", 0], "vae": ["24", 0]},
            },
            "91": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["10", 0],
                    "audio": ["23", 0],
                    "fps": float(s.default_fps),
                },
            },
            "92": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["91", 0],
                    "filename_prefix": "video/H3VideoGen/out",
                    "format": "auto",
                    "codec": "auto",
                },
            },
        }

    def build_t2v_workflow(
        self,
        prompt: str,
        *,
        length: int,
        seed: int,
        filename_prefix: str,
        width: int | None = None,
        height: int | None = None,
        first_frame_comfy_name: str | None = None,
    ) -> dict[str, Any]:
        s = self.settings
        width = width or s.default_width
        height = height or s.default_height
        wf = self._loader_nodes(unet_name=s.h3_unet)
        wf["9"]["inputs"]["scheduler"] = "simple"
        wf["15"]["inputs"]["noise_seed"] = seed
        wf["92"]["inputs"]["filename_prefix"] = filename_prefix
        cond_inputs: dict[str, Any] = {
            "clip": ["13", 0],
            "vae": ["11", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
        }
        if first_frame_comfy_name:
            wf["200"] = {
                "class_type": "LoadImage",
                "inputs": {"image": first_frame_comfy_name},
            }
            cond_inputs["first_frame"] = ["200", 0]
        wf["104"] = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": cond_inputs,
        }
        return wf

    def build_r2v_workflow(
        self,
        prompt: str,
        *,
        length: int,
        seed: int,
        filename_prefix: str,
        ref_comfy_names: Sequence[str],
        width: int | None = None,
        height: int | None = None,
        ref_image_size: str | None = None,
    ) -> dict[str, Any]:
        """Reference-to-video. ref_comfy_names order → <Picture 1..N>."""
        s = self.settings
        width = width or s.default_width
        height = height or s.default_height
        ref_image_size = ref_image_size or s.h3_ref_image_size
        names = [n for n in ref_comfy_names if n][:9]
        if not names:
            raise ComfyError("R2V requires at least one reference image")

        wf = self._loader_nodes(unet_name=s.h3_unet_r2v)
        wf["15"]["inputs"]["noise_seed"] = seed
        wf["92"]["inputs"]["filename_prefix"] = filename_prefix

        ref_inputs: dict[str, Any] = {
            "clip": ["13", 0],
            "vae": ["11", 0],
            "audio_vae": ["24", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
            "ref_image_size": ref_image_size,
        }
        for i, name in enumerate(names):
            node_id = str(200 + i)
            wf[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": name},
            }
            # Autogrow keys as used by MiniMaxH3ReferenceToVideo
            ref_inputs[f"ref_images.ref_image_{i}"] = [node_id, 0]

        wf["104"] = {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": ref_inputs,
        }
        return wf

    def wait(self, prompt_id: str) -> dict[str, Any]:
        t0 = time.time()
        timeout = self.settings.comfyui_timeout_sec
        # Fail fast if Comfy is down/unreachable (don't burn the full hour timeout)
        consecutive_connect_failures = 0
        max_connect_failures = 8  # ~few seconds of hard refusals
        # Short HTTP timeouts so cancel flags are observed often
        with httpx.Client(timeout=8.0) as client:
            while True:
                if self.control and self.control.is_cancelled():
                    self.interrupt()
                    raise CancelledError("Generation cancelled while waiting on ComfyUI")
                try:
                    r = client.get(f"{self.base}/history/{prompt_id}")
                    r.raise_for_status()
                    hist = r.json()
                    consecutive_connect_failures = 0
                except CancelledError:
                    raise
                except Exception as exc:
                    if self.control and self.control.is_cancelled():
                        self.interrupt()
                        raise CancelledError("Generation cancelled while waiting on ComfyUI") from exc
                    consecutive_connect_failures += 1
                    # Connection refused / DNS etc. — Comfy is not running
                    msg = str(exc).lower()
                    hard_down = any(
                        s in msg
                        for s in (
                            "10061",
                            "actively refused",
                            "connection refused",
                            "connecterror",
                            "name or service not known",
                            "failed to establish",
                            "no connection could be made",
                        )
                    )
                    if hard_down and consecutive_connect_failures >= max_connect_failures:
                        raise ComfyError(
                            f"ComfyUI unreachable while waiting for {prompt_id}: {exc}"
                        ) from exc
                    if consecutive_connect_failures >= 40:
                        # ~15–30s of repeated failures with short timeouts
                        raise ComfyError(
                            f"ComfyUI repeatedly failed while waiting for {prompt_id}: {exc}"
                        ) from exc
                    time.sleep(0.4)
                    continue
                if prompt_id in hist:
                    item = hist[prompt_id]
                    status = item.get("status") or {}
                    s = status.get("status_str")
                    if s == "error":
                        # Interrupted graphs often surface as error
                        if self.control and self.control.is_cancelled():
                            raise CancelledError("Generation cancelled while waiting on ComfyUI")
                        raise ComfyError(f"Comfy error: {status}")
                    if s == "success" or status.get("completed"):
                        if self.control and self.control.is_cancelled():
                            raise CancelledError("Generation cancelled after Comfy finished")
                        return item
                    # interrupted / cancelled status strings from some builds
                    if s in ("interrupted", "cancelled", "canceled"):
                        raise CancelledError("ComfyUI job interrupted")
                if time.time() - t0 > timeout:
                    raise ComfyError(f"Timeout waiting for {prompt_id}")
                # ~0.25s cadence for responsive Stop
                for _ in range(5):
                    if self.control and self.control.is_cancelled():
                        self.interrupt()
                        raise CancelledError("Generation cancelled while waiting on ComfyUI")
                    time.sleep(0.25)

    def queue(self, workflow: dict[str, Any], client_id: str | None = None) -> str:
        if self.control:
            self.control.check()
        payload = {"prompt": workflow, "client_id": client_id or str(uuid.uuid4())}
        with httpx.Client(timeout=30.0) as client:
            if self.control and self.control.is_cancelled():
                raise CancelledError("Generation cancelled before queue")
            r = client.post(f"{self.base}/prompt", json=payload)
            if r.status_code >= 400:
                raise ComfyError(f"Queue failed {r.status_code}: {r.text[:2000]}")
            data = r.json()
        if data.get("node_errors"):
            raise ComfyError(json.dumps(data["node_errors"])[:2000])
        pid = data.get("prompt_id")
        if not pid:
            raise ComfyError(f"No prompt_id: {data}")
        if self.control:
            self.control.set_prompt_id(pid)
            self.control.check()
        return pid

    def resolve_output_path(self, history_item: dict[str, Any]) -> Path:
        outputs = history_item.get("outputs") or {}
        filename = None
        subfolder = "video"
        for node_out in outputs.values():
            for entry in (node_out.get("images") or []) + (node_out.get("gifs") or []):
                filename = entry.get("filename")
                subfolder = entry.get("subfolder") or "video"
                if filename:
                    break
            if filename:
                break
        root = Path(self.settings.comfy_output_root)
        candidates: list[Path] = []
        if filename:
            candidates.extend(
                [
                    root / subfolder / filename,
                    root / "video" / subfolder / filename,
                    root / "video" / filename,
                ]
            )
            video_root = root / "video"
            if video_root.exists():
                candidates.extend(video_root.rglob(filename))
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        video_dir = root / "video"
        if video_dir.exists():
            mp4s = sorted(video_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mp4s:
                return mp4s[0]
        raise ComfyError("Could not locate output mp4 in Comfy history/outputs")

    def generate(
        self,
        prompt: str,
        *,
        length: int,
        seed: int,
        filename_prefix: str,
        mode: str = "t2v",
        ref_image_paths: Sequence[Path] | None = None,
        first_frame_path: Path | None = None,
        project_tag: str = "refs",
    ) -> tuple[Path, str, str]:
        """
        Generate a clip.
        Returns (path, prompt_id, mode_used) where mode_used is 'r2v' or 't2v'.
        """
        if self.control:
            self.control.check()
        mode = (mode or "t2v").lower()
        refs = [Path(p) for p in (ref_image_paths or []) if p and Path(p).exists()]

        if mode == "r2v" and refs:
            uploaded = [
                self.upload_image(p, subfolder=f"H3VideoGen/{project_tag}") for p in refs[:9]
            ]
            wf = self.build_r2v_workflow(
                prompt,
                length=length,
                seed=seed,
                filename_prefix=filename_prefix,
                ref_comfy_names=uploaded,
            )
            used = "r2v"
        else:
            first_name = None
            if first_frame_path and Path(first_frame_path).exists():
                first_name = self.upload_image(
                    Path(first_frame_path), subfolder=f"H3VideoGen/{project_tag}"
                )
            wf = self.build_t2v_workflow(
                prompt,
                length=length,
                seed=seed,
                filename_prefix=filename_prefix,
                first_frame_comfy_name=first_name,
            )
            used = "t2v"

        prompt_id = self.queue(wf)
        item = self.wait(prompt_id)
        path = self.resolve_output_path(item)
        if self.control:
            self.control.set_prompt_id(None)
        return path, prompt_id, used
