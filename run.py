"""CLI entrypoints for H3 Video Gen."""
from __future__ import annotations

import argparse
import json
import signal
import sys

import uvicorn

from app.config import get_settings
from app.job_control import job_control
from app.models import GenerateRequest
from app.pipeline import ProductionPipeline


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
    req = GenerateRequest(
        prompt=args.prompt,
        style=args.style,
        target_duration_sec=args.duration,
        max_shots=args.shots,
        max_retakes=args.retakes,
        auto_assemble=not args.no_assemble,
    )

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
    plan = ProductionPipeline().plan_only(args.prompt, args.style, args.duration, args.shots)
    print(plan.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="H3 Video Gen — Gemini director/critic + MiniMax H3")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Start web UI")
    p_serve.set_defaults(func=lambda a: cmd_serve() or 0)

    p_gen = sub.add_parser("generate", help="Run full pipeline from CLI")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--style", default="Premium 3D animated cinematic YouTube short")
    p_gen.add_argument("--duration", type=float, default=60.0)
    p_gen.add_argument("--shots", type=int, default=12)
    p_gen.add_argument("--retakes", type=int, default=2)
    p_gen.add_argument("--no-assemble", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    p_resume = sub.add_parser("resume", help="Resume an unfinished project from disk")
    p_resume.add_argument("project_id", help="outputs/<project_id>")
    p_resume.add_argument("--retakes", type=int, default=2)
    p_resume.add_argument("--seed", type=int, default=42)
    p_resume.add_argument("--no-assemble", action="store_true")
    p_resume.add_argument(
        "--skip-failed",
        action="store_true",
        help="Do not re-render shots that already failed critic",
    )
    p_resume.set_defaults(func=cmd_resume)

    p_plan = sub.add_parser("plan", help="Director plan only (no GPU render)")
    p_plan.add_argument("--prompt", required=True)
    p_plan.add_argument("--style", default="Premium 3D animated cinematic YouTube short")
    p_plan.add_argument("--duration", type=float, default=60.0)
    p_plan.add_argument("--shots", type=int, default=12)
    p_plan.set_defaults(func=cmd_plan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
