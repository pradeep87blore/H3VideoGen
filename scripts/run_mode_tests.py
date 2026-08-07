"""Queue and run emu war + bitcoin jobs after current GPU job finishes (CLI)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:7860"
JOBS = [
    ROOT / "jobs" / "emu_war_doc_25s.json",
    ROOT / "jobs" / "bitcoin_explainer_25s.json",
]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
POLL = 45


def ui_live() -> bool:
    try:
        r = httpx.get(f"{BASE}/api/jobs", timeout=15.0)
        r.raise_for_status()
        j = r.json()
        if not j.get("worker_alive"):
            return False
        cur = j.get("current") or {}
        st = str(cur.get("status") or "").lower()
        return st in {
            "running",
            "planning",
            "assembling",
            "generating",
            "reviewing",
            "cancelling",
        }
    except Exception:
        return False


def main() -> int:
    print("Waiting for UI worker idle (Apollo etc.) before CLI tests…", flush=True)
    while ui_live():
        print(f"[{time.strftime('%H:%M:%S')}] UI still running a job…", flush=True)
        time.sleep(POLL)
    print("UI idle — running mode tests via CLI.", flush=True)

    for job in JOBS:
        if not job.exists():
            print("Missing", job, flush=True)
            return 2
        print(f"\n=== GENERATE {job.name} ===", flush=True)
        cp = subprocess.run(
            [
                str(PY),
                str(ROOT / "run.py"),
                "generate",
                "--json",
                str(job),
            ],
            cwd=str(ROOT),
        )
        print(f"Exit {cp.returncode} for {job.name}", flush=True)
        if cp.returncode != 0:
            print("Continuing to next job despite non-zero exit…", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
