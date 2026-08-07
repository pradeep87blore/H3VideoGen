"""AI video styles reference library — prompts + procedural thumbnails."""
from __future__ import annotations

import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = ROOT / "ai_video_styles_reference_library.json"
THUMB_DIR = ROOT / "web" / "static" / "style_thumbs"
THUMB_W, THUMB_H = 320, 200


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "style"


@lru_cache(maxsize=1)
def load_library() -> dict[str, Any]:
    if not LIBRARY_PATH.exists():
        return {"version": "0", "title": "empty", "total_styles": 0, "styles": []}
    return json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))


def list_styles() -> list[dict[str, Any]]:
    data = load_library()
    return list(data.get("styles") or [])


def get_style(slug: str) -> dict[str, Any] | None:
    want = _slugify(slug)
    for s in list_styles():
        if _slugify(str(s.get("slug") or s.get("name") or "")) == want:
            return s
    return None


def build_style_prompt(style: dict[str, Any]) -> str:
    """Compose a production-ready style bible string for the director / H3."""
    name = (style.get("name") or "Style").strip()
    category = (style.get("category") or "").strip()
    sample = (style.get("sample_prompt") or "").strip()
    desc = (style.get("description") or "").strip()
    traits = [str(t).strip() for t in (style.get("visual_characteristics") or []) if str(t).strip()]
    lighting = (style.get("lighting") or "").strip()
    materials = (style.get("materials") or "").strip()
    palette = (style.get("color_palette") or "").strip()
    camera = (style.get("camera_style") or "").strip()
    motion = (style.get("animation_style") or "").strip()
    negatives = [str(n).strip() for n in (style.get("negative_keywords") or []) if str(n).strip()]

    parts: list[str] = [
        f"VISUAL STYLE LOCK: {name}" + (f" ({category})" if category else "") + ".",
    ]
    if sample:
        parts.append(sample.rstrip(".") + ".")
    elif desc:
        parts.append(desc)
    else:
        parts.append(f"Render the entire production consistently in {name} style.")

    if traits:
        parts.append("Traits: " + ", ".join(traits) + ".")
    if lighting:
        parts.append(f"Lighting: {lighting.rstrip('.')}.")
    if materials:
        parts.append(f"Materials: {materials.rstrip('.')}.")
    if palette:
        parts.append(f"Palette: {palette.rstrip('.')}.")
    if camera:
        parts.append(f"Camera: {camera.rstrip('.')}.")
    if motion:
        parts.append(f"Motion: {motion.rstrip('.')}.")

    parts.append(
        "Keep a coherent single art direction across every shot and character. "
        "YouTube-polished framing, high visual consistency, no on-screen text or subtitles baked into image, "
        "no mixed competing art styles."
    )
    if negatives:
        parts.append("Avoid: " + ", ".join(negatives) + ".")
    return " ".join(parts)


# --- Thumbnail painting -----------------------------------------------------

# Per-slug accent/base palettes for recognizable thumbnails
_PALETTE: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "photorealistic": ((40, 55, 72), (120, 145, 160), (220, 210, 195)),
    "cinematic": ((12, 14, 28), (90, 40, 50), (230, 190, 120)),
    "pixar-style-3d": ((55, 130, 200), (255, 180, 90), (250, 240, 230)),
    "dreamworks-style": ((30, 90, 140), (70, 170, 110), (255, 220, 140)),
    "stylized-3d": ((90, 60, 160), (255, 120, 160), (120, 230, 200)),
    "claymation": ((180, 90, 60), (230, 160, 100), (90, 140, 80)),
    "stop-motion": ((60, 50, 45), (180, 130, 80), (220, 200, 160)),
    "lego-animation": ((200, 30, 40), (30, 90, 200), (250, 200, 30)),
    "low-poly": ((40, 90, 70), (120, 180, 100), (240, 210, 100)),
    "voxel-blocky": ((50, 50, 80), (100, 180, 80), (180, 100, 60)),
    "wireframe": ((10, 20, 30), (0, 200, 220), (40, 60, 80)),
    "rag-doll-plush": ((160, 70, 100), (240, 180, 190), (120, 90, 140)),
    "felt-fabric": ((170, 50, 60), (80, 120, 70), (240, 220, 180)),
    "crochet-knitted": ((90, 130, 150), (220, 160, 140), (250, 235, 220)),
    "origami-paper": ((240, 245, 250), (90, 150, 220), (230, 120, 90)),
    "paper-cutout": ((250, 245, 235), (200, 70, 60), (40, 70, 120)),
    "pop-up-book": ((245, 230, 200), (180, 40, 50), (50, 100, 70)),
    "wooden-toy": ((90, 55, 30), (180, 120, 60), (230, 200, 140)),
    "puppet-marionette": ((50, 35, 30), (160, 90, 60), (200, 170, 120)),
    "sand-art": ((180, 140, 80), (120, 90, 50), (240, 220, 170)),
    "paper-quilling": ((250, 245, 240), (220, 80, 100), (80, 140, 200)),
    "stained-glass": ((30, 20, 40), (200, 50, 80), (40, 120, 200)),
    "watercolor-painting": ((240, 245, 250), (100, 160, 220), (220, 120, 140)),
    "oil-painting": ((40, 35, 50), (160, 80, 40), (200, 170, 80)),
    "ink-drawing": ((245, 245, 240), (20, 20, 25), (80, 80, 90)),
    "pencil-sketch": ((235, 232, 225), (90, 90, 95), (50, 50, 55)),
    "colored-pencil": ((245, 240, 230), (220, 80, 60), (60, 120, 200)),
    "anime": ((250, 200, 210), (80, 140, 220), (255, 255, 255)),
    "studio-ghibli-inspired": ((120, 180, 140), (255, 200, 120), (90, 150, 210)),
    "comic-book": ((250, 240, 50), (220, 30, 40), (20, 20, 30)),
    "manga": ((245, 245, 245), (20, 20, 20), (180, 180, 200)),
    "flat-vector": ((40, 120, 200), (255, 160, 50), (240, 245, 250)),
    "whiteboard-animation": ((250, 250, 250), (30, 30, 40), (50, 120, 200)),
    "fantasy-painting": ((40, 30, 80), (180, 90, 200), (255, 200, 80)),
    "dark-fantasy": ((15, 10, 25), (80, 30, 50), (160, 140, 90)),
    "cyberpunk-neon": ((10, 5, 30), (255, 40, 160), (40, 230, 255)),
    "steampunk": ((40, 30, 20), (160, 100, 40), (200, 160, 80)),
    "solarpunk": ((40, 120, 80), (180, 220, 100), (255, 200, 80)),
    "isometric": ((70, 130, 180), (240, 180, 80), (90, 180, 140)),
    "blueprint-style": ((10, 40, 90), (80, 160, 255), (200, 220, 255)),
    "ancient-mythological-indian-art": ((140, 40, 40), (220, 160, 40), (40, 80, 50)),
    "ukiyo-e": ((250, 240, 220), (40, 90, 140), (200, 60, 50)),
    "chinese-ink-wash": ((240, 238, 230), (30, 30, 35), (100, 110, 120)),
    "miniature-painting": ((50, 70, 40), (160, 120, 60), (200, 180, 100)),
    "fortnite-style": ((40, 200, 220), (255, 80, 160), (255, 220, 40)),
    "zelda-inspired": ((40, 100, 60), (220, 180, 60), (80, 160, 200)),
    "genshin-inspired": ((100, 160, 220), (255, 160, 180), (250, 230, 200)),
    "league-of-legends-splash-art": ((30, 20, 50), (200, 80, 40), (255, 200, 80)),
    "diablo-style": ((25, 10, 10), (160, 30, 20), (100, 80, 40)),
    "pixel-art": ((40, 40, 80), (80, 200, 120), (240, 100, 80)),
    "retro-16-bit": ((20, 20, 60), (80, 200, 80), (220, 80, 160)),
    "isometric-rpg": ((50, 80, 100), (180, 140, 80), (100, 180, 120)),
    "concept-art": ((50, 60, 90), (120, 140, 160), (220, 160, 80)),
    "matte-painting": ((30, 50, 90), (180, 120, 70), (240, 200, 120)),
    "sci-fi-illustration": ((10, 20, 50), (40, 180, 220), (200, 80, 255)),
    "neon-line-art": ((5, 5, 15), (255, 40, 180), (40, 255, 220)),
    "holographic": ((20, 30, 50), (180, 100, 255), (80, 255, 220)),
    "liquid-paint": ((30, 20, 40), (220, 40, 100), (40, 160, 220)),
    "smoke-art": ((30, 30, 40), (140, 140, 160), (220, 220, 230)),
    "fractal-art": ((10, 5, 30), (255, 60, 140), (60, 200, 255)),
    "kaleidoscope": ((40, 10, 60), (255, 80, 120), (80, 220, 200)),
    "surreal-dreamscape": ((40, 20, 80), (255, 140, 180), (100, 180, 255)),
    "ice-sculpture": ((180, 220, 240), (100, 180, 220), (240, 250, 255)),
    "crystal-world": ((20, 40, 90), (160, 80, 255), (100, 255, 220)),
    "double-exposure": ((20, 40, 30), (80, 140, 100), (200, 180, 120)),
    "abstract-geometry": ((20, 20, 40), (255, 80, 100), (80, 200, 255)),
}


def _palette_for(slug: str, name: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if slug in _PALETTE:
        return _PALETTE[slug]
    rng = random.Random(hash(slug + name) & 0xFFFFFFFF)
    base = (rng.randint(20, 80), rng.randint(20, 80), rng.randint(30, 100))
    mid = (rng.randint(80, 200), rng.randint(60, 180), rng.randint(60, 200))
    hi = (rng.randint(180, 255), rng.randint(160, 255), rng.randint(140, 255))
    return base, mid, hi


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _mix(
    c0: tuple[int, int, int], c1: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    return (
        int(_lerp(c0[0], c1[0], t)),
        int(_lerp(c0[1], c1[1], t)),
        int(_lerp(c0[2], c1[2], t)),
    )


def _gradient(
    img: Image.Image,
    c0: tuple[int, int, int],
    c1: tuple[int, int, int],
    *,
    vertical: bool = True,
) -> None:
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            t = y / max(1, h - 1) if vertical else x / max(1, w - 1)
            px[x, y] = _mix(c0, c1, t)


def _draw_mountains(
    draw: ImageDraw.ImageDraw,
    w: int,
    h: int,
    color: tuple[int, int, int],
    seed: int,
    *,
    y0: float = 0.45,
) -> None:
    rng = random.Random(seed)
    pts = [(0, h)]
    n = 8
    for i in range(n + 1):
        x = int(w * i / n)
        peak = h * (y0 + rng.uniform(-0.08, 0.12) + 0.08 * math.sin(i * 1.2))
        pts.append((x, int(peak)))
    pts.append((w, h))
    draw.polygon(pts, fill=color)


def _draw_sun(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    r: int,
    color: tuple[int, int, int],
) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def _draw_tree(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    scale: float,
    trunk: tuple[int, int, int],
    leaves: tuple[int, int, int],
) -> None:
    tw = max(2, int(6 * scale))
    th = int(28 * scale)
    draw.rectangle([x - tw // 2, y - th, x + tw // 2, y], fill=trunk)
    r = int(18 * scale)
    draw.ellipse([x - r, y - th - r, x + r, y - th + r // 2], fill=leaves)


def _pixelate(img: Image.Image, block: int = 10) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, w // block), max(1, h // block)), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.NEAREST)


def _wireframe_overlay(draw: ImageDraw.ImageDraw, w: int, h: int, color: tuple[int, int, int]) -> None:
    for i in range(0, w, 24):
        draw.line([(i, 0), (i, h)], fill=color, width=1)
    for j in range(0, h, 24):
        draw.line([(0, j), (w, j)], fill=color, width=1)
    # perspective diamond
    cx, cy = w // 2, int(h * 0.55)
    for s in (40, 70, 100):
        draw.polygon(
            [(cx, cy - s), (cx + s, cy), (cx, cy + s // 2), (cx - s, cy)],
            outline=color,
        )


def _draw_figure(
    draw: ImageDraw.ImageDraw,
    cx: int,
    base_y: int,
    scale: float,
    body: tuple[int, int, int],
    head: tuple[int, int, int] | None = None,
) -> None:
    head = head or body
    r = int(14 * scale)
    draw.ellipse([cx - r, base_y - int(70 * scale) - r, cx + r, base_y - int(70 * scale) + r], fill=head)
    draw.rounded_rectangle(
        [cx - int(12 * scale), base_y - int(55 * scale), cx + int(12 * scale), base_y - int(10 * scale)],
        radius=int(6 * scale),
        fill=body,
    )
    # legs
    draw.line(
        [(cx - int(6 * scale), base_y - int(10 * scale)), (cx - int(10 * scale), base_y)],
        fill=body,
        width=max(2, int(4 * scale)),
    )
    draw.line(
        [(cx + int(6 * scale), base_y - int(10 * scale)), (cx + int(10 * scale), base_y)],
        fill=body,
        width=max(2, int(4 * scale)),
    )


def _font(size: int = 16) -> ImageFont.ImageFont:
    for path in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_style_thumb(style: dict[str, Any]) -> Image.Image:
    """Paint a small distinctive thumbnail for a library style."""
    name = str(style.get("name") or "Style")
    slug = _slugify(str(style.get("slug") or name))
    category = str(style.get("category") or "")
    base, mid, hi = _palette_for(slug, name)
    seed = hash(slug) & 0xFFFFFFFF
    rng = random.Random(seed)

    img = Image.new("RGB", (THUMB_W, THUMB_H), base)
    draw = ImageDraw.Draw(img)
    w, h = THUMB_W, THUMB_H

    # Background treatment by category / slug keywords
    key = slug
    _gradient(img, base, mid if "dark" not in key else (max(0, base[0] - 10), max(0, base[1] - 10), max(0, base[2] - 5)))
    draw = ImageDraw.Draw(img)

    if "wireframe" in key:
        _draw_mountains(draw, w, h, mid, seed)
        _wireframe_overlay(draw, w, h, hi)
    elif "pixel" in key or "16-bit" in key or "retro" in key or "voxel" in key:
        _draw_mountains(draw, w, h, mid, seed)
        _draw_sun(draw, int(w * 0.78), int(h * 0.22), 22, hi)
        _draw_tree(draw, int(w * 0.3), int(h * 0.78), 1.1, _mix(base, mid, 0.4), hi)
        _draw_figure(draw, int(w * 0.55), int(h * 0.82), 1.0, hi, mid)
        img = _pixelate(img, 12 if "pixel" in key else 8)
        draw = ImageDraw.Draw(img)
    elif "lego" in key:
        _gradient(img, (30, 40, 60), (80, 100, 140))
        draw = ImageDraw.Draw(img)
        colors = [(200, 30, 40), (30, 90, 200), (250, 200, 30), (40, 160, 70)]
        for i, col in enumerate(colors):
            x0 = 30 + i * 70
            draw.rounded_rectangle([x0, 60, x0 + 55, 150], radius=6, fill=col)
            for dx in (10, 30):
                draw.ellipse([x0 + dx, 48, x0 + dx + 14, 62], fill=_mix(col, (255, 255, 255), 0.25))
    elif "low-poly" in key:
        pts_sets = []
        for _ in range(14):
            pts = [(rng.randint(0, w), rng.randint(0, h)) for _ in range(3)]
            col = _mix(base, hi, rng.random())
            draw.polygon(pts, fill=col)
            pts_sets.append(pts)
    elif "anime" in key or "manga" in key or "genshin" in key:
        _draw_sun(draw, int(w * 0.75), int(h * 0.25), 28, hi)
        _draw_mountains(draw, w, h, mid, seed, y0=0.55)
        # big eyes figure
        cx = int(w * 0.4)
        _draw_figure(draw, cx, int(h * 0.88), 1.2, mid, hi)
        draw.ellipse([cx - 10, int(h * 0.42), cx - 2, int(h * 0.52)], fill=(20, 20, 40))
        draw.ellipse([cx + 2, int(h * 0.42), cx + 10, int(h * 0.52)], fill=(20, 20, 40))
    elif "ghibli" in key:
        _gradient(img, (140, 200, 230), (250, 220, 160), vertical=True)
        draw = ImageDraw.Draw(img)
        _draw_mountains(draw, w, h, (90, 160, 100), seed, y0=0.5)
        _draw_tree(draw, 80, int(h * 0.8), 1.4, (100, 70, 40), (60, 140, 70))
        _draw_tree(draw, 220, int(h * 0.78), 1.0, (100, 70, 40), (70, 150, 80))
        _draw_figure(draw, 160, int(h * 0.85), 0.9, (230, 200, 170), (250, 230, 200))
    elif "watercolor" in key or "oil" in key or "paint" in key or "liquid" in key:
        for _ in range(40):
            x, y = rng.randint(0, w), rng.randint(0, h)
            r = rng.randint(12, 50)
            col = _mix(mid, hi, rng.random())
            draw.ellipse([x - r, y - r, x + r, y + r], fill=col)
        img = img.filter(ImageFilter.GaussianBlur(radius=3 if "water" in key else 1.2))
        draw = ImageDraw.Draw(img)
    elif "ink" in key or "pencil" in key or "sketch" in key or "chinese-ink" in key:
        _gradient(img, hi if sum(hi) > 500 else (245, 242, 235), _mix(hi, base, 0.15))
        draw = ImageDraw.Draw(img)
        for _ in range(35):
            x0, y0 = rng.randint(0, w), rng.randint(0, h)
            x1, y1 = x0 + rng.randint(-40, 40), y0 + rng.randint(-30, 30)
            draw.line([(x0, y0), (x1, y1)], fill=base if sum(base) < 200 else (30, 30, 35), width=rng.randint(1, 3))
        _draw_figure(draw, int(w * 0.48), int(h * 0.88), 1.1, (25, 25, 30))
    elif "comic" in key:
        _gradient(img, hi, mid)
        draw = ImageDraw.Draw(img)
        # action burst
        cx, cy = w // 2, h // 2
        for i in range(12):
            ang = i * math.pi / 6
            x1 = cx + int(math.cos(ang) * 20)
            y1 = cy + int(math.sin(ang) * 20)
            x2 = cx + int(math.cos(ang) * 90)
            y2 = cy + int(math.sin(ang) * 70)
            draw.polygon([(cx, cy), (x1, y1), (x2, y2)], fill=_mix(mid, base, i % 2))
        _draw_figure(draw, cx, int(h * 0.85), 1.0, base, mid)
    elif "cyberpunk" in key or "neon" in key or "holographic" in key:
        _gradient(img, (5, 5, 20), base)
        draw = ImageDraw.Draw(img)
        for i in range(6):
            y = 30 + i * 28
            col = hi if i % 2 == 0 else mid
            draw.line([(20, y), (w - 20, y + rng.randint(-8, 8))], fill=col, width=2)
        # city blocks
        for i in range(8):
            x = 20 + i * 38
            bh = rng.randint(40, 100)
            draw.rectangle([x, h - 20 - bh, x + 28, h - 20], fill=_mix(base, mid, 0.4), outline=hi)
        _draw_figure(draw, int(w * 0.5), h - 22, 0.85, hi, mid)
    elif "stained-glass" in key or "kaleidoscope" in key or "fractal" in key:
        cx, cy = w // 2, h // 2
        for i in range(16):
            a0 = i * math.pi / 8
            a1 = (i + 1) * math.pi / 8
            pts = [
                (cx, cy),
                (cx + int(math.cos(a0) * 120), cy + int(math.sin(a0) * 90)),
                (cx + int(math.cos(a1) * 120), cy + int(math.sin(a1) * 90)),
            ]
            draw.polygon(pts, fill=_mix(mid, hi, (i % 4) / 3), outline=base)
    elif "felt" in key or "plush" in key or "crochet" in key or "rag-doll" in key:
        _gradient(img, mid, hi)
        draw = ImageDraw.Draw(img)
        # soft clouds / fabric bumps
        for _ in range(18):
            x, y = rng.randint(0, w), rng.randint(0, h)
            r = rng.randint(15, 40)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=_mix(mid, hi, rng.random()))
        _draw_figure(draw, int(w * 0.5), int(h * 0.88), 1.15, base, hi)
        # stitch marks
        for _ in range(20):
            x, y = rng.randint(10, w - 10), rng.randint(10, h - 10)
            draw.line([(x, y), (x + 4, y + 2)], fill=_mix(base, (0, 0, 0), 0.3), width=1)
    elif "origami" in key or "paper" in key or "cutout" in key or "pop-up" in key or "quilling" in key:
        _gradient(img, (250, 248, 242), (230, 225, 215))
        draw = ImageDraw.Draw(img)
        layers = [mid, hi, base]
        for i, col in enumerate(layers):
            y = 40 + i * 35
            draw.polygon(
                [(30, y + 40), (w // 2, y), (w - 30, y + 40), (w // 2, y + 70)],
                fill=col,
            )
        _draw_figure(draw, int(w * 0.5), int(h * 0.9), 0.8, mid, hi)
    elif "blueprint" in key:
        _gradient(img, (8, 30, 70), (20, 60, 120))
        draw = ImageDraw.Draw(img)
        for i in range(0, w, 16):
            draw.line([(i, 0), (i, h)], fill=(40, 100, 180), width=1)
        for j in range(0, h, 16):
            draw.line([(0, j), (w, j)], fill=(40, 100, 180), width=1)
        _draw_figure(draw, int(w * 0.5), int(h * 0.85), 1.0, hi)
        draw.ellipse([40, 40, 100, 90], outline=hi, width=2)
    elif "steampunk" in key or "wooden" in key:
        _gradient(img, base, mid)
        draw = ImageDraw.Draw(img)
        for i, (cx, cy, r) in enumerate([(80, 70, 40), (200, 100, 55), (250, 50, 25)]):
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=hi, width=3)
            draw.ellipse([cx - r // 3, cy - r // 3, cx + r // 3, cy + r // 3], fill=hi)
            for a in range(0, 360, 45):
                rad = math.radians(a)
                draw.line(
                    [
                        (cx + int(math.cos(rad) * r * 0.4), cy + int(math.sin(rad) * r * 0.4)),
                        (cx + int(math.cos(rad) * r * 0.9), cy + int(math.sin(rad) * r * 0.9)),
                    ],
                    fill=hi,
                    width=2,
                )
    elif "clay" in key or "stop-motion" in key or "puppet" in key:
        _gradient(img, mid, hi)
        draw = ImageDraw.Draw(img)
        _draw_mountains(draw, w, h, base, seed, y0=0.6)
        _draw_figure(draw, int(w * 0.45), int(h * 0.9), 1.3, mid, hi)
        # clay lumps
        for _ in range(8):
            x, y = rng.randint(20, w - 20), rng.randint(int(h * 0.6), h - 10)
            r = rng.randint(8, 18)
            draw.ellipse([x - r, y - r // 2, x + r, y + r // 2], fill=_mix(mid, base, 0.5))
    elif "ice" in key or "crystal" in key:
        _gradient(img, hi, mid)
        draw = ImageDraw.Draw(img)
        for _ in range(10):
            cx, cy = rng.randint(30, w - 30), rng.randint(30, h - 30)
            s = rng.randint(20, 50)
            draw.polygon(
                [(cx, cy - s), (cx + s // 2, cy), (cx, cy + s // 2), (cx - s // 2, cy)],
                fill=_mix(hi, mid, rng.random()),
                outline=(255, 255, 255),
            )
    elif "ukiyo" in key or "indian" in key or "mythological" in key:
        _gradient(img, hi, mid)
        draw = ImageDraw.Draw(img)
        _draw_mountains(draw, w, h, base, seed, y0=0.5)
        _draw_sun(draw, int(w * 0.7), int(h * 0.28), 30, mid)
        _draw_figure(draw, int(w * 0.4), int(h * 0.88), 1.1, base, mid)
        # wave / arch pattern
        for i in range(5):
            y = int(h * 0.65) + i * 8
            draw.arc([20, y - 20, w - 20, y + 40], 0, 180, fill=hi, width=2)
    elif "fantasy" in key or "zelda" in key or "diablo" in key:
        _gradient(img, base, mid)
        draw = ImageDraw.Draw(img)
        _draw_mountains(draw, w, h, _mix(base, mid, 0.5), seed, y0=0.4)
        _draw_sun(draw, int(w * 0.75), int(h * 0.2), 18, hi)
        _draw_figure(draw, int(w * 0.4), int(h * 0.88), 1.0, mid, hi)
        # magic sparkles
        for _ in range(25):
            x, y = rng.randint(0, w), rng.randint(0, int(h * 0.6))
            draw.point((x, y), fill=hi)
    else:
        # Default scenic vignette + figure
        _draw_sun(draw, int(w * 0.78), int(h * 0.22), 26, hi)
        _draw_mountains(draw, w, h, mid, seed)
        _draw_tree(draw, int(w * 0.22), int(h * 0.78), 1.0, _mix(base, mid, 0.5), _mix(mid, hi, 0.3))
        _draw_figure(draw, int(w * 0.55), int(h * 0.88), 1.0, hi, mid)

    # Soft vignette
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0, int(h * 0.62), w, h], fill=(0, 0, 0, 110))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Name label
    font = _font(15)
    label = name if len(name) <= 28 else name[:26] + "…"
    # measure
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
    except Exception:
        tw = len(label) * 8
    tx = max(10, (w - tw) // 2)
    ty = h - 28
    # shadow
    draw.text((tx + 1, ty + 1), label, font=font, fill=(0, 0, 0))
    draw.text((tx, ty), label, font=font, fill=(255, 250, 240))

    # Category chip (tiny)
    cat_short = category.split("/")[0].strip() if category else ""
    if cat_short:
        cfont = _font(10)
        draw.text((10, 8), cat_short[:22], font=cfont, fill=(255, 255, 255))

    return img


def ensure_thumbnails(*, force: bool = False) -> int:
    """
    Ensure every library style has a thumb under web/static/style_thumbs.

    Prefer pre-built crops from the reference sheet (see
    scripts/extract_style_thumbs_from_sheet.py). Procedural fallbacks only
    fill *missing* files when force=False; force=True re-fills missing still,
    but never overwrites reference-sheet crops unless force=True *and*
    env H3_REGEN_STYLE_THUMBS=1.
    """
    import os

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    allow_overwrite = force and os.environ.get("H3_REGEN_STYLE_THUMBS") == "1"
    written = 0
    for style in list_styles():
        slug = _slugify(str(style.get("slug") or style.get("name") or "style"))
        path = THUMB_DIR / f"{slug}.jpg"
        if path.exists() and path.stat().st_size > 500 and not allow_overwrite:
            continue
        # Prefer re-export from sheet if available
        sheet = ROOT / "assets" / "ai_video_styles_reference_sheet.png"
        if sheet.exists() and allow_overwrite:
            # leave to extract script; skip procedural
            pass
        img = render_style_thumb(style)
        img.save(path, format="JPEG", quality=88, optimize=True)
        written += 1
    return written


def styles_for_api() -> dict[str, Any]:
    ensure_thumbnails(force=False)
    items = []
    categories: list[str] = []
    seen_cat: set[str] = set()
    for s in list_styles():
        slug = _slugify(str(s.get("slug") or s.get("name") or "style"))
        cat = str(s.get("category") or "Other")
        if cat not in seen_cat:
            seen_cat.add(cat)
            categories.append(cat)
        items.append(
            {
                "slug": slug,
                "name": s.get("name") or slug,
                "category": cat,
                "description": (s.get("description") or "")[:280],
                "style_prompt": build_style_prompt(s),
                "sample_prompt": s.get("sample_prompt") or "",
                "thumb_url": f"/static/style_thumbs/{slug}.jpg",
                "negative_keywords": s.get("negative_keywords") or [],
                "ideal_for": s.get("ideal_for") or [],
            }
        )
    lib = load_library()
    return {
        "version": lib.get("version"),
        "title": lib.get("title"),
        "total": len(items),
        "categories": categories,
        "styles": items,
    }
