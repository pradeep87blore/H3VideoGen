"""Shared JSON extraction helpers for LLM outputs."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output (tolerates fences / leading prose)."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty LLM response; expected JSON object")

    # Strip common markdown fences
    if "```" in raw:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.I)
        if fenced:
            raw = fenced.group(1).strip()
        else:
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])

    last_err: Exception | None = None
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data[0]
            raise ValueError(f"JSON root must be object, got {type(data).__name__}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc

    # Mild repairs common in small local models
    repaired = candidates[-1] if candidates else raw
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*]", "]", repaired)
    try:
        data = json.loads(repaired)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        last_err = exc

    raise ValueError(
        f"Could not parse JSON from LLM output: {last_err}; head={raw[:200]!r}"
    )
