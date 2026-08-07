"""Probe configured Gemini model IDs with a minimal live generate call."""
from __future__ import annotations

import time

from app.config import get_settings
from google import genai
from google.genai import types


def main() -> None:
    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)

    candidates: list[str] = []
    for m in [
        s.gemini_director_model,
        s.gemini_critic_model,
        *s.gemini_model_fallbacks,
        *s.gemini_image_models,
        "gemini-3-flash-preview",
        "gemini-2.0-flash",
        "gemini-3.6-flash",
    ]:
        m = (m or "").strip()
        if m and m not in candidates:
            candidates.append(m)

    print(f"Probing {len(candidates)} models...\n")

    listed: list[str] = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", None) or str(m)
            name = name.replace("models/", "")
            if "gemini" in name.lower() and "embed" not in name.lower():
                listed.append(name)
        print(f"API list returned {len(listed)} gemini-ish models (sample):")
        for n in listed[:30]:
            print(f"  - {n}")
        if len(listed) > 30:
            print(f"  ... +{len(listed) - 30} more")
        print()
    except Exception as e:
        print(f"List models failed: {type(e).__name__}: {e}\n")

    listed_set = set(listed)
    results: list[tuple[str, str, str, str]] = []

    for model in candidates:
        t0 = time.time()
        try:
            r = client.models.generate_content(
                model=model,
                contents='Reply with JSON only: {"ok": true}',
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            text = (r.text or "").strip().replace("\n", " ")[:120]
            ms = int((time.time() - t0) * 1000)
            status = "OK"
            detail = f"{ms}ms  response={text!r}"
        except Exception as e:
            ms = int((time.time() - t0) * 1000)
            err = str(e)
            low = err.lower()
            if "404" in err or "NOT_FOUND" in err or "not found" in low or "no longer available" in low:
                status = "NOT_FOUND"
            elif "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in low:
                status = "QUOTA"
            elif "503" in err or "UNAVAILABLE" in err or "high demand" in low:
                status = "OVERLOADED"
            elif "403" in err or "permission" in low:
                status = "FORBIDDEN"
            else:
                status = "ERROR"
            detail = f"{ms}ms  {err[:240].replace(chr(10), ' ')}"

        in_list = "yes" if model in listed_set else ("no" if listed else "?")
        results.append((status, model, in_list, detail))
        print(f"[{status:10}] {model:40} list={in_list:3}  {detail}")

    print("\n=== SUMMARY ===")
    for status in ("OK", "OVERLOADED", "QUOTA", "NOT_FOUND", "FORBIDDEN", "ERROR"):
        group = [r for r in results if r[0] == status]
        if group:
            print(f"{status} ({len(group)}):")
            for _, model, _, _ in group:
                print(f"  - {model}")


if __name__ == "__main__":
    main()
