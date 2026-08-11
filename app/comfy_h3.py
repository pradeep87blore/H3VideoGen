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

# Native MiniMax H3 nodes (ComfyUI ≥0.30.0 / comfy_extras.nodes_minimax_h3)
REQUIRED_H3_NODES = (
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
)
# Optional but expected on the same install
OPTIONAL_H3_NODES = (
    "EmptyMiniMaxH3LatentAV",
)


class ComfyError(RuntimeError):
    pass


class ComfyNeedsResubmit(ComfyError):
    """Comfy restarted or lost the queued prompt — caller should re-submit the graph."""


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

    def object_info(self, timeout: float = 60.0) -> dict[str, Any]:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{self.base}/object_info")
            r.raise_for_status()
            return r.json()

    def h3_capability(self, timeout: float = 12.0) -> dict[str, Any]:
        """
        Whether this Comfy instance can run MiniMax H3 graphs.
        A different app on the same port often answers /system_stats but lacks H3 nodes.
        Keep timeout modest so health checks cannot freeze the API for a full minute.
        """
        try:
            oi = self.object_info(timeout=timeout)
        except Exception as exc:
            return {
                "ok": False,
                "reachable": False,
                "missing_nodes": list(REQUIRED_H3_NODES),
                "present_nodes": [],
                "has_minimax_clip": False,
                "error": str(exc),
            }

        present = [n for n in REQUIRED_H3_NODES if n in oi]
        missing = [n for n in REQUIRED_H3_NODES if n not in oi]
        optional_present = [n for n in OPTIONAL_H3_NODES if n in oi]
        has_minimax_clip = False
        clip = oi.get("CLIPLoader") or {}
        try:
            types = (clip.get("input") or {}).get("required", {}).get("type")
            # shape: (["stable_diffusion", ...],) or list
            opts: list[str] = []
            if isinstance(types, (list, tuple)) and types:
                opts = list(types[0]) if isinstance(types[0], (list, tuple)) else list(types)
            has_minimax_clip = "minimax" in opts
        except Exception:
            has_minimax_clip = False

        ok = not missing
        return {
            "ok": ok,
            "reachable": True,
            "missing_nodes": missing,
            "present_nodes": present,
            "optional_nodes": optional_present,
            "has_minimax_clip": has_minimax_clip,
            "node_count": len(oi),
            "hint": (
                None
                if ok
                else (
                    "This ComfyUI process is missing MiniMax H3 native nodes. "
                    "Point COMFYUI_ROOT at a Comfy ≥0.30 install with nodes_minimax_h3 "
                    "(default E:/AI/ComfyUI) and free the port or set COMFY_REPLACE_NON_H3=true."
                )
            ),
        }

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

    def wait_until_ready(self, timeout_sec: float = 180.0, log: Any = None) -> None:
        """Poll until Comfy answers; self-heal by starting/replacing H3 instance if needed."""
        deadline = time.time() + max(15.0, timeout_sec)
        last_err: Exception | None = None
        last_heal = 0.0
        while time.time() < deadline:
            if self.control and self.control.is_cancelled():
                raise CancelledError("Cancelled while waiting for ComfyUI")
            try:
                self.health()
                return
            except Exception as exc:
                last_err = exc
            now = time.time()
            # First failure immediately; retry heal every ~45s while still down
            if last_heal <= 0.0 or (now - last_heal) >= 45.0:
                try:
                    from .services import heal_runtime_services, invalidate_service_caches

                    invalidate_service_caches()
                    heal_runtime_services(
                        self.settings,
                        log=log,
                        need_comfy=True,
                        need_ollama=False,
                        reason="comfy_not_ready",
                    )
                    last_heal = now
                except Exception as exc:
                    last_err = exc
                    last_heal = now
            time.sleep(2.0)
        raise ComfyError(f"ComfyUI not ready within {timeout_sec:.0f}s: {last_err}")

    def _is_connection_error(self, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(
            s in msg
            for s in (
                "10054",
                "10061",
                "connection reset",
                "connection refused",
                "forcibly closed",
                "connecterror",
                "readerror",
                "timed out",
                "timeout",
                "server disconnected",
                "remote end closed",
                "network is unreachable",
                "unreachable",
                "comfyui unreachable",
                "repeatedly failed",
            )
        ) or isinstance(exc, (httpx.TransportError, httpx.TimeoutException))

    def _heal_comfy(self, log: Any = None) -> None:
        from .services import heal_runtime_services, invalidate_service_caches

        invalidate_service_caches()
        report = heal_runtime_services(
            self.settings,
            log=log,
            need_comfy=True,
            need_ollama=False,
            reason="mid_generation",
        )
        comfy = (report or {}).get("comfy") or {}
        if not comfy.get("ok"):
            # still try a short wait in case it is mid-boot
            try:
                self.health()
            except Exception as exc:
                raise ComfyError(
                    comfy.get("error") or f"Self-heal failed for ComfyUI: {exc}"
                ) from exc

    def upload_image(self, path: Path, *, subfolder: str = "H3VideoGen") -> str:
        """Upload local image; returns filename usable by LoadImage."""
        path = Path(path)
        if not path.exists():
            raise ComfyError(f"Reference image missing: {path}")

        last_err: Exception | None = None
        for attempt in range(1, 5):
            if self.control:
                self.control.check()
            try:
                with path.open("rb") as f:
                    files = {
                        "image": (
                            path.name,
                            f,
                            "image/png" if path.suffix.lower() == ".png" else "image/jpeg",
                        ),
                    }
                    data = {"overwrite": "true", "subfolder": subfolder, "type": "input"}
                    with httpx.Client(timeout=120.0) as client:
                        r = client.post(f"{self.base}/upload/image", files=files, data=data)
                        if r.status_code >= 400:
                            raise ComfyError(f"Image upload failed {r.status_code}: {r.text[:1500]}")
                        body = r.json()
                name = body.get("name") or path.name
                sub = body.get("subfolder") or subfolder
                return f"{sub}/{name}" if sub else name
            except CancelledError:
                raise
            except Exception as exc:
                last_err = exc
                if not self._is_connection_error(exc) and not isinstance(exc, ComfyError):
                    raise ComfyError(str(exc)) from exc
                if (
                    not self._is_connection_error(exc)
                    and isinstance(exc, ComfyError)
                    and "upload failed" in str(exc).lower()
                ):
                    raise
                if attempt >= 4:
                    break
                try:
                    self._heal_comfy()
                except Exception:
                    pass
                self.wait_until_ready(timeout_sec=120.0)
                time.sleep(min(8, attempt * 2))
        raise ComfyError(f"Image upload failed after retries: {last_err}")


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
        consecutive_connect_failures = 0
        max_connect_failures = 6
        heal_done = False
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
                            "10054",
                            "connection reset",
                            "forcibly closed",
                            "server disconnected",
                        )
                    )
                    soft_down = hard_down or self._is_connection_error(exc)

                    if soft_down and consecutive_connect_failures >= max_connect_failures:
                        # Self-heal: restart/repair Comfy. After a real restart the
                        # prompt_id is gone — resubmit. After a brief blip, continue.
                        try:
                            if not heal_done:
                                self._heal_comfy()
                                heal_done = True
                                self.wait_until_ready(timeout_sec=120.0)
                                consecutive_connect_failures = 0
                                # Probe once: if the old prompt is still known, keep waiting
                                try:
                                    with httpx.Client(timeout=8.0) as probe:
                                        pr = probe.get(f"{self.base}/history/{prompt_id}")
                                        pr.raise_for_status()
                                        if prompt_id in (pr.json() or {}):
                                            continue
                                except Exception:
                                    pass
                                raise ComfyNeedsResubmit(
                                    f"ComfyUI recovered while waiting for {prompt_id}; "
                                    "prompt was lost — resubmitting job"
                                )
                            raise ComfyNeedsResubmit(
                                f"ComfyUI still unstable while waiting for {prompt_id}: {exc}"
                            ) from exc
                        except ComfyNeedsResubmit:
                            raise
                        except Exception as heal_exc:
                            raise ComfyError(
                                f"ComfyUI unreachable for {prompt_id} and self-heal failed: {heal_exc}"
                            ) from heal_exc

                    if consecutive_connect_failures >= 50:
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
                        if self.control and self.control.is_cancelled():
                            raise CancelledError("Generation cancelled while waiting on ComfyUI")
                        raise ComfyError(f"Comfy error: {status}")
                    if s == "success" or status.get("completed"):
                        if self.control and self.control.is_cancelled():
                            raise CancelledError("Generation cancelled after Comfy finished")
                        return item
                    if s in ("interrupted", "cancelled", "canceled"):
                        raise CancelledError("ComfyUI job interrupted")
                elif heal_done:
                    # Comfy came back but this prompt_id is gone
                    raise ComfyNeedsResubmit(
                        f"Prompt {prompt_id} missing after Comfy recover — resubmitting"
                    )
                if time.time() - t0 > timeout:
                    raise ComfyError(f"Timeout waiting for {prompt_id}")
                for _ in range(5):
                    if self.control and self.control.is_cancelled():
                        self.interrupt()
                        raise CancelledError("Generation cancelled while waiting on ComfyUI")
                    time.sleep(0.25)

    def queue(self, workflow: dict[str, Any], client_id: str | None = None) -> str:
        if self.control:
            self.control.check()
        payload = {"prompt": workflow, "client_id": client_id or str(uuid.uuid4())}
        last_err: Exception | None = None
        for attempt in range(1, 5):
            if self.control:
                self.control.check()
            try:
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
            except CancelledError:
                raise
            except ComfyError as exc:
                if not self._is_connection_error(exc) or attempt >= 4:
                    raise
                last_err = exc
                try:
                    self._heal_comfy()
                except Exception:
                    pass
                self.wait_until_ready(timeout_sec=180.0)
                time.sleep(min(8, attempt * 2))
            except Exception as exc:
                last_err = exc
                if not self._is_connection_error(exc) or attempt >= 4:
                    raise ComfyError(f"Queue failed: {exc}") from exc
                try:
                    self._heal_comfy()
                except Exception:
                    pass
                self.wait_until_ready(timeout_sec=180.0)
                time.sleep(min(8, attempt * 2))
        raise ComfyError(f"Queue failed after retries: {last_err}")


    def _output_search_roots(self) -> list[Path]:
        """Possible folders where SaveVideo writes (depends on Comfy extra paths)."""
        s = self.settings
        roots: list[Path] = []
        for r in (
            Path(s.comfy_output_root),
            Path(s.comfyui_root) / "output",
            Path(s.ai_root) / "Outputs",
            Path(s.ai_root) / "ComfyUI" / "output",
        ):
            if r not in roots:
                roots.append(r)
        return roots

    def download_output_media(
        self,
        filename: str,
        *,
        subfolder: str = "",
        file_type: str = "output",
        dest: Path | None = None,
    ) -> Path:
        """Fetch a Comfy output via /view (reliable when local paths differ)."""
        params = {
            "filename": filename,
            "subfolder": subfolder.replace("\\", "/").strip("/"),
            "type": file_type,
        }
        with httpx.Client(timeout=120.0) as client:
            r = client.get(f"{self.base}/view", params=params)
            if r.status_code >= 400:
                raise ComfyError(
                    f"Comfy /view failed {r.status_code} for {filename} sub={subfolder}: {r.text[:500]}"
                )
            data = r.content
        if not data or len(data) < 64:
            raise ComfyError(f"Empty Comfy view payload for {filename}")
        if dest is None:
            dest = Path(self.settings.output_root) / "_comfy_fetch" / filename
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def resolve_output_path(self, history_item: dict[str, Any]) -> Path:
        outputs = history_item.get("outputs") or {}
        filename = None
        subfolder = "video"
        file_type = "output"
        for node_out in outputs.values():
            for entry in (node_out.get("images") or []) + (node_out.get("gifs") or []):
                filename = entry.get("filename")
                subfolder = entry.get("subfolder") or "video"
                file_type = entry.get("type") or "output"
                if filename:
                    break
            if filename:
                break
        if not filename:
            raise ComfyError("Comfy history has no video filename in outputs")

        # Normalize Windows-style subfolders from Comfy history
        sub = str(subfolder).replace("\\", "/").strip("/")

        candidates: list[Path] = []
        for root in self._output_search_roots():
            candidates.extend(
                [
                    root / sub / filename,
                    root / "video" / sub / filename,
                    root / filename,
                ]
            )
            # Also try each path segment as relative under root
            if sub:
                parts = Path(sub)
                candidates.append(root / parts / filename)

        seen: set[str] = set()
        for c in candidates:
            key = str(c.resolve()) if c.exists() else str(c)
            if key in seen:
                continue
            seen.add(key)
            if c.exists() and c.is_file() and c.stat().st_size > 64:
                return c

        # Rglob only same basename under known roots, prefer newest
        matches: list[Path] = []
        for root in self._output_search_roots():
            if not root.exists():
                continue
            try:
                matches.extend(p for p in root.rglob(filename) if p.is_file())
            except Exception:
                continue
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]

        # Last resort: pull bytes from Comfy HTTP API
        try:
            return self.download_output_media(
                filename,
                subfolder=sub,
                file_type=file_type,
                dest=Path(self.settings.output_root) / "_comfy_fetch" / sub.replace("/", "_") / filename,
            )
        except ComfyError:
            raise ComfyError(
                f"Could not locate output {filename} (subfolder={sub}) under "
                f"{[str(r) for r in self._output_search_roots()]}"
            )


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
        Retries the whole submit when Comfy drops the connection before the job starts.
        """
        last_err: Exception | None = None
        for attempt in range(1, 6):
            if self.control:
                self.control.check()
            try:
                self.wait_until_ready(timeout_sec=120.0 if attempt > 1 else 30.0)
                mode_l = (mode or "t2v").lower()
                refs = [Path(p) for p in (ref_image_paths or []) if p and Path(p).exists()]

                if mode_l == "r2v" and refs:
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
            except CancelledError:
                raise
            except Exception as exc:
                last_err = exc
                # Missing nodes / bad graph should not be retried forever
                if isinstance(exc, ComfyError) and (
                    "missing_node" in str(exc).lower()
                    or "node_errors" in str(exc).lower()
                    or "queue failed 400" in str(exc).lower()
                ) and not isinstance(exc, ComfyNeedsResubmit):
                    raise
                needs_resubmit = isinstance(exc, ComfyNeedsResubmit)
                conn = self._is_connection_error(exc) or (
                    isinstance(exc, ComfyError) and self._is_connection_error(exc)
                )
                if not needs_resubmit and not conn:
                    raise
                if attempt >= 5:
                    break
                try:
                    self._heal_comfy()
                except Exception:
                    pass
                time.sleep(min(15, attempt * 3))
                self.wait_until_ready(timeout_sec=180.0)
        raise ComfyError(f"Generate failed after retries: {last_err}") from last_err
