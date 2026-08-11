"""CLI entrypoints for H3 Video Gen."""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

import uvicorn

from app.config import get_settings
from app.job_control import job_control
from app.models import DirectorOnlyRequest, GenerateRequest
from app.pipeline import ProductionPipeline


def _load_json_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"JSON job file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {p}: {e}") from e
    if not isinstance(data, dict):
        raise SystemExit(f"JSON job file must be an object/dict: {p}")
    return data


def _merge_job_payload(
    *,
    json_path: str | None,
    prompt: str | None,
    style: str | None,
    duration: float | None,
    shots: int | None,
    retakes: int | None,
    no_assemble: bool,
    seed: int | None = None,
    h3_mode: str | None = None,
    narrative_mode: str | None = None,
) -> dict[str, Any]:
    """Build GenerateRequest kwargs from optional JSON + CLI overrides."""
    data: dict[str, Any] = {}
    if json_path:
        data = _load_json_file(json_path)

    # Accept a few alias keys used in automation docs
    aliases = {
        "target_duration": "target_duration_sec",
        "duration": "target_duration_sec",
        "duration_sec": "target_duration_sec",
        "max_retakes_per_shot": "max_retakes",
        "retakes": "max_retakes",
        "shots": "max_shots",
        "max_shot": "max_shots",
        "story": "prompt",
        "concept": "prompt",
        "visual_style": "style",
        "mode": "narrative_mode",
        "narrative": "narrative_mode",
    }
    for src, dst in aliases.items():
        if src in data and dst not in data:
            data[dst] = data[src]

    if prompt is not None:
        data["prompt"] = prompt
    if style is not None:
        data["style"] = style
    if duration is not None:
        data["target_duration_sec"] = duration
    if shots is not None:
        data["max_shots"] = shots
    if retakes is not None:
        data["max_retakes"] = retakes
    if no_assemble:
        data["auto_assemble"] = False
    if seed is not None:
        data["seed_base"] = seed
    if h3_mode is not None:
        data["h3_mode"] = h3_mode
    if narrative_mode is not None:
        data["narrative_mode"] = narrative_mode

    if not str(data.get("prompt") or "").strip():
        raise SystemExit(
            "Story prompt is required. Use --prompt, or --json with a \"prompt\" field."
        )
    return data


def cmd_serve() -> None:
    s = get_settings()

    def _request_shutdown(*_args: object) -> None:
        # Let any active generation wind down; also interrupt Comfy if mid-sample.
        job_control.request_stop()
        try:
            from app.comfy_h3 import ComfyH3Client

            ComfyH3Client(s, control=job_control).interrupt()
        except Exception:
            pass

    # Windows: SIGINT (Ctrl+C). SIGTERM only partially supported — still register.
    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, _request_shutdown)
        except Exception:
            pass

    # timeout_graceful_shutdown lets daemon workers exit without waiting forever
    config = uvicorn.Config(
        "app.main:app",
        host=s.host,
        port=s.port,
        reload=False,
        log_level="info",
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)

    # Chain: our stop + uvicorn's default lifecycle
    original_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum: int, frame: object) -> None:
        _request_shutdown()
        server.should_exit = True
        if callable(original_handler) and original_handler not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            try:
                original_handler(signum, frame)  # type: ignore[misc]
            except Exception:
                pass

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:
        pass

    print(f"H3 Video Gen → http://{s.host}:{s.port}  (Ctrl+C to stop)", flush=True)
    server.run()


def cmd_generate(args: argparse.Namespace) -> int:
    payload = _merge_job_payload(
        json_path=args.json,
        prompt=args.prompt,
        style=args.style,
        duration=args.duration,
        shots=args.shots,
        retakes=args.retakes,
        no_assemble=args.no_assemble,
        seed=args.seed,
        h3_mode=args.h3_mode,
        narrative_mode=getattr(args, "narrative_mode", None),
    )
    req = GenerateRequest.model_validate(payload)

    def log(msg: str) -> None:
        print(msg, flush=True)

    def _stop(*_a: object) -> None:
        job_control.request_stop()
        print("Stop requested…", flush=True)

    try:
        signal.signal(signal.SIGINT, _stop)
    except Exception:
        pass

    job_control.reset()
    state = ProductionPipeline(control=job_control).run(req, log=log)
    print(
        json.dumps(
            {
                "project_id": state.project_id,
                "status": state.status,
                "master": state.master_path,
            },
            indent=2,
        )
    )
    return 0 if state.status.startswith("completed") else 1


def cmd_resume(args: argparse.Namespace) -> int:
    from app.models import ResumeRequest

    def log(msg: str) -> None:
        print(msg, flush=True)

    def _stop(*_a: object) -> None:
        job_control.request_stop()
        print("Stop requested…", flush=True)

    try:
        signal.signal(signal.SIGINT, _stop)
    except Exception:
        pass

    job_control.reset()
    req = ResumeRequest(
        max_retakes=args.retakes,
        auto_assemble=not args.no_assemble,
        seed_base=args.seed,
        redo_failed=not args.skip_failed,
        h3_mode=args.h3_mode,
    )
    state = ProductionPipeline(control=job_control).resume(args.project_id, req, log=log)
    print(
        json.dumps(
            {
                "project_id": state.project_id,
                "status": state.status,
                "master": state.master_path,
            },
            indent=2,
        )
    )
    return 0 if state.status.startswith("completed") else 1


def cmd_plan(args: argparse.Namespace) -> int:
    payload = _merge_job_payload(
        json_path=args.json,
        prompt=args.prompt,
        style=args.style,
        duration=args.duration,
        shots=args.shots,
        retakes=None,
        no_assemble=False,
        narrative_mode=getattr(args, "narrative_mode", None),
    )
    req = DirectorOnlyRequest.model_validate(
        {
            "prompt": payload["prompt"],
            "style": payload.get("style", "Premium 3D animated cinematic short"),
            "target_duration_sec": payload.get("target_duration_sec", 60.0),
            "max_shots": payload.get("max_shots", 12),
            "narrative_mode": payload.get("narrative_mode", "character"),
        }
    )
    plan = ProductionPipeline().plan_only(
        req.prompt,
        req.style,
        req.target_duration_sec,
        req.max_shots,
        narrative_mode=req.narrative_mode,
    )
    print(plan.model_dump_json(indent=2))
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Install missing ComfyUI / H3 models / FFmpeg / Ollama models under AI_ROOT."""
    from app.prereq_install import run_bootstrap_blocking, scan_prereqs

    if getattr(args, "scan_only", False):
        rep = scan_prereqs()
        print(json.dumps(rep.to_dict(), indent=2))
        return 0 if rep.ok else 1

    print("Checking / installing prerequisites under AI_ROOT ...", flush=True)
    rep = run_bootstrap_blocking(log=lambda m: print(m, flush=True))
    print(json.dumps(rep.to_dict(), indent=2), flush=True)
    return 0 if rep.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H3 Video Gen — Gemini director/critic + MiniMax H3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Start web UI")
    p_serve.set_defaults(func=lambda a: cmd_serve() or 0)

    p_boot = sub.add_parser(
        "bootstrap",
        help="Install missing tools/models into AI_ROOT (E:/AI by default)",
    )
    p_boot.add_argument(
        "--scan",
        dest="scan_only",
        action="store_true",
        help="Only report what is missing (no install)",
    )
    p_boot.set_defaults(func=cmd_bootstrap)

    p_gen = sub.add_parser(
        "generate",
        help="Run full pipeline from CLI flags and/or a JSON job file",
    )
    p_gen.add_argument(
        "--json",
        "-j",
        dest="json",
        default=None,
        help="Path to job JSON (prompt, style, target_duration_sec, max_shots, max_retakes, …)",
    )
    p_gen.add_argument("--prompt", default=None, help="Story / concept (overrides JSON)")
    p_gen.add_argument("--style", default=None, help="Visual style (overrides JSON)")
    p_gen.add_argument("--duration", type=float, default=None, help="Target duration seconds")
    p_gen.add_argument("--shots", type=int, default=None, help="Max shots")
    p_gen.add_argument("--retakes", type=int, default=None, help="Max retakes per shot")
    p_gen.add_argument("--seed", type=int, default=None, help="Seed base")
    p_gen.add_argument("--h3-mode", dest="h3_mode", default=None, choices=["r2v", "t2v", "auto"])
    p_gen.add_argument(
        "--narrative-mode",
        dest="narrative_mode",
        default=None,
        choices=["character", "documentary", "explainer"],
        help="character | documentary | explainer",
    )
    p_gen.add_argument("--no-assemble", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    p_resume = sub.add_parser("resume", help="Resume an unfinished project from disk")
    p_resume.add_argument("project_id", help="outputs/<project_id>")
    p_resume.add_argument("--retakes", type=int, default=2)
    p_resume.add_argument("--seed", type=int, default=42)
    p_resume.add_argument("--h3-mode", dest="h3_mode", default=None, choices=["r2v", "t2v", "auto"])
    p_resume.add_argument("--no-assemble", action="store_true")
    p_resume.add_argument(
        "--skip-failed",
        action="store_true",
        help="Do not re-render shots that already failed critic",
    )
    p_resume.set_defaults(func=cmd_resume)

    p_plan = sub.add_parser("plan", help="Director plan only (no GPU render)")
    p_plan.add_argument("--json", "-j", dest="json", default=None, help="Path to job JSON")
    p_plan.add_argument("--prompt", default=None)
    p_plan.add_argument("--style", default=None)
    p_plan.add_argument("--duration", type=float, default=None)
    p_plan.add_argument("--shots", type=int, default=None)
    p_plan.add_argument(
        "--narrative-mode",
        dest="narrative_mode",
        default=None,
        choices=["character", "documentary", "explainer"],
    )
    p_plan.set_defaults(func=cmd_plan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
