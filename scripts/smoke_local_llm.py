"""Smoke-test local LLM text + vision fallback paths."""
from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.llm.json_util import extract_json
from app.llm.local_openai import LocalOpenAIBackend
from app.llm.router import LLMRouter

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    get_settings.cache_clear()
    s = get_settings()
    b = LocalOpenAIBackend(s)
    print("catalog", b.list_models(force=True))
    print("text model", b.resolve_model(images=False))
    print("vision model", b.resolve_model(images=True))

    t, p = b.generate_text(
        system="Return JSON only.",
        user='Return {"ok": true, "role": "director"}',
        temperature=0.1,
    )
    print("TEXT", p, extract_json(t))

    img = ROOT / "outputs/20260807_032554_e42bc957/character_board/C03_front_full.jpg"
    if not img.exists():
        from PIL import Image

        img = ROOT / "_probe_frame.jpg"
        Image.new("RGB", (256, 256), (40, 120, 200)).save(img)

    t2, p2 = b.generate_text(
        system="You are a harsh film critic. Return JSON only.",
        user=(
            "Look at the image. Return JSON: "
            '{"verdict":"PASS or RETAKE","overall_score":7.0,'
            '"summary":"one sentence","cast_seen":["names if any"]}'
        ),
        temperature=0.2,
        images=[img],
    )
    print("VISION", p2)
    print("VISION head", t2[:400])
    print("VISION json", extract_json(t2))

    st = LLMRouter(s).status()
    print(
        "router",
        st.get("local_llm_resolved_text_model"),
        st.get("local_llm_resolved_vision_model"),
        st.get("local_llm_models"),
    )


if __name__ == "__main__":
    main()
