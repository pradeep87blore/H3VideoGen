"""Durable job queue — survives process exit and restarts."""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings


QUEUE_VERSION = 1


def queue_file_path(settings: Settings) -> Path:
    root = Path(settings.output_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / "job_queue.json"


def empty_queue_document() -> dict[str, Any]:
    return {
        "version": QUEUE_VERSION,
        "saved_at": None,
        "items": [],
        "parallel_override": None,
    }


def load_queue_document(settings: Settings) -> dict[str, Any]:
    path = queue_file_path(settings)
    if not path.exists():
        return empty_queue_document()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return empty_queue_document()
        items = data.get("items")
        if not isinstance(items, list):
            data["items"] = []
        data.setdefault("version", QUEUE_VERSION)
        return data
    except Exception:
        return empty_queue_document()


def save_queue_document(settings: Settings, document: dict[str, Any]) -> Path:
    """Atomically write queue document under OUTPUT_ROOT/job_queue.json."""
    path = queue_file_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": QUEUE_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "items": list(document.get("items") or []),
        "parallel_override": document.get("parallel_override"),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".job_queue_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except Exception:
            pass
        raise
    return path


def clear_queue_document(settings: Settings) -> None:
    save_queue_document(settings, empty_queue_document())


def durable_item(
    *,
    job_key: str,
    kind: str,
    label: str = "",
    project_id: str | None = None,
    prompt_preview: str = "",
    title: str | None = None,
    status: str = "queued",
    generate_payload: dict[str, Any] | None = None,
    resume_payload: dict[str, Any] | None = None,
    enqueued_at: float | None = None,
) -> dict[str, Any]:
    """Normalize one durable queue row (JSON-serializable)."""
    return {
        "job_key": job_key,
        "kind": kind if kind in ("generate", "resume") else "generate",
        "label": label or "",
        "project_id": project_id,
        "prompt_preview": (prompt_preview or "")[:200],
        "title": title,
        "status": status or "queued",
        "generate_payload": generate_payload,
        "resume_payload": resume_payload,
        "enqueued_at": enqueued_at if enqueued_at is not None else time.time(),
    }
