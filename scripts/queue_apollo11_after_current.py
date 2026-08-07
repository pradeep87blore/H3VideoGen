"""Wait for any live H3VideoGen job to finish, then start Apollo 11 generate via API."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
JOB_PATH = ROOT / "jobs" / "apollo11_hyperreal_60s.json"
BASE = "http://127.0.0.1:7860"
POLL_SEC = 30


def api_get(path: str) -> dict:
    r = httpx.get(f"{BASE}{path}", timeout=30.0)
    r.raise_for_status()
    return r.json()


def is_live(status: str | None) -> bool:
    return str(status or "").lower() in {
        "running",
        "planning",
        "assembling",
        "generating",
        "reviewing",
        "cancelling",
    }


def live_job(payload: dict) -> dict | None:
    if not payload.get("worker_alive"):
        return None
    current = payload.get("current") or {}
    if is_live(current.get("status")):
        return current
    for job in (payload.get("active") or {}).values():
        if is_live(job.get("status")):
            return job
    return None


def main() -> int:
    if not JOB_PATH.exists():
        print(f"Missing job file: {JOB_PATH}", flush=True)
        return 2
    body = json.loads(JOB_PATH.read_text(encoding="utf-8"))
    print(f"Queue: will start {JOB_PATH.name} after current job finishes…", flush=True)
    print(f"Target: {body.get('target_duration_sec')}s / max_shots={body.get('max_shots')}", flush=True)

    while True:
        try:
            payload = api_get("/api/jobs")
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] UI unreachable ({exc}); retry in {POLL_SEC}s", flush=True)
            time.sleep(POLL_SEC)
            continue

        job = live_job(payload)
        if job:
            last = (job.get("last_message") or "")[:140]
            print(
                f"[{time.strftime('%H:%M:%S')}] waiting — "
                f"{job.get('project_id')} {job.get('status')} "
                f"{job.get('title') or ''} | {last}",
                flush=True,
            )
            time.sleep(POLL_SEC)
            continue

        print(f"[{time.strftime('%H:%M:%S')}] No live job — starting Apollo 11…", flush=True)
        break

    # POST generate
    try:
        r = httpx.post(f"{BASE}/api/generate", json=body, timeout=60.0)
        print(f"HTTP {r.status_code}: {r.text[:800]}", flush=True)
        if r.status_code >= 400:
            return 1
    except Exception as exc:
        print(f"Start failed: {exc}", flush=True)
        return 1

    # Confirm appears in jobs list
    time.sleep(2)
    try:
        payload = api_get("/api/jobs")
        job = live_job(payload) or payload.get("current")
        if job:
            print(
                f"Apollo job running: project_id={job.get('project_id')} "
                f"status={job.get('status')} title={job.get('title')}",
                flush=True,
            )
    except Exception as exc:
        print(f"Post-check: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
