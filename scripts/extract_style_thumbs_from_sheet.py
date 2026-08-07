"""Crop style thumbs using a fixed pixel grid on the reference sheet.

Coordinates measured on the 1024×682 AI VIDEO STYLES REFERENCE SHEET.
Per-row left/right tuned by light-frame edge alignment (see scripts/_frame_grid_search.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "ai_video_styles_reference_sheet.png"
OUT_DIR = ROOT / "web" / "static" / "style_thumbs"
QA = ROOT / "assets" / "style_crop_qa"

BASE_W, BASE_H = 1024, 682
COLS = 11
ROWS = 6
INSET_FRAC = 0.12

# (grid_left, grid_right, y_top, y_bot) — image-only, labels excluded
ROW_GRID = [
    (35, 1005, 46, 118),
    (35, 1010, 155, 226),
    (40, 1010, 262, 332),
    (40, 1012, 368, 436),
    (35, 1002, 480, 545),
    (36, 1002, 580, 642),
]

ROW_SLUGS = [
    ["photorealistic", "cinematic", "pixar-style-3d", "dreamworks-style", "stylized-3d", "claymation", "stop-motion", "lego-animation", "low-poly", "voxel-blocky", "wireframe"],
    ["rag-doll-plush", "felt-fabric", "crochet-knitted", "origami-paper", "paper-cutout", "pop-up-book", "wooden-toy", "puppet-marionette", "sand-art", "paper-quilling", "stained-glass"],
    ["watercolor-painting", "oil-painting", "ink-drawing", "pencil-sketch", "colored-pencil", "anime", "studio-ghibli-inspired", "comic-book", "manga", "flat-vector", "whiteboard-animation"],
    ["fantasy-painting", "dark-fantasy", "cyberpunk-neon", "steampunk", "solarpunk", "isometric", "blueprint-style", "ancient-mythological-indian-art", "ukiyo-e", "chinese-ink-wash", "miniature-painting"],
    ["fortnite-style", "zelda-inspired", "genshin-inspired", "league-of-legends-splash-art", "diablo-style", "pixel-art", "retro-16-bit", "isometric-rpg", "concept-art", "matte-painting", "sci-fi-illustration"],
    ["neon-line-art", "holographic", "liquid-paint", "smoke-art", "fractal-art", "kaleidoscope", "surreal-dreamscape", "ice-sculpture", "crystal-world", "double-exposure", "abstract-geometry"],
]


def finish(crop: Image.Image, size: int = 320) -> Image.Image:
    w, h = crop.size
    m = max(1, int(min(w, h) * 0.04))
    crop = crop.crop((m, m, w - m, h - m))
    side = min(crop.size)
    l = (crop.width - side) // 2
    t = (crop.height - side) // 2
    sq = crop.crop((l, t, l + side, t + side)).resize((size, size), Image.Resampling.LANCZOS)
    sq = ImageEnhance.Contrast(sq).enhance(1.05)
    sq = ImageEnhance.Color(sq).enhance(1.04)
    sq = sq.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=28, fill=255)
    bg = Image.new("RGB", (size, size), (14, 16, 22))
    bg.paste(sq, mask=mask)
    return bg


def main() -> None:
    im0 = Image.open(SRC).convert("RGB")
    sx = im0.width / BASE_W
    sy = im0.height / BASE_H
    work_sc = 3
    im = im0.resize((im0.width * work_sc, im0.height * work_sc), Image.Resampling.LANCZOS)

    def map_box(x0, y0, x1, y1):
        return (
            int(x0 * sx * work_sc),
            int(y0 * sy * work_sc),
            int(x1 * sx * work_sc),
            int(y1 * sy * work_sc),
        )

    dbg = im.copy()
    draw = ImageDraw.Draw(dbg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA.mkdir(exist_ok=True)
    qa_set = {
        "photorealistic", "cinematic", "pixar-style-3d", "dreamworks-style",
        "claymation", "lego-animation", "low-poly", "wireframe",
        "felt-fabric", "crochet-knitted", "origami-paper", "stained-glass",
        "anime", "studio-ghibli-inspired", "comic-book",
        "cyberpunk-neon", "steampunk", "ukiyo-e", "solarpunk",
        "pixel-art", "fortnite-style", "neon-line-art", "kaleidoscope",
    }

    items = []
    for r in range(ROWS):
        left, right, y0, y1 = ROW_GRID[r]
        cell_w = (right - left) / COLS
        inset = cell_w * INSET_FRAC
        for c in range(COLS):
            x0 = left + c * cell_w + inset
            x1 = left + (c + 1) * cell_w - inset
            box = map_box(x0, y0, x1, y1)
            draw.rectangle(box, outline=(0, 255, 90), width=2)
            slug = ROW_SLUGS[r][c]
            thumb = finish(im.crop(box))
            thumb.save(OUT_DIR / f"{slug}.jpg", quality=94, optimize=True)
            items.append({"slug": slug, "base_box": [x0, y0, x1, y1], "row": r})
            if slug in qa_set:
                thumb.save(QA / f"{slug}.jpg", quality=94)
                print("QA", slug)

    dbg.resize((1400, int(1400 * im.height / im.width))).save(
        ROOT / "assets" / "style_sheet_detected_grid.jpg", quality=92
    )
    (ROOT / "assets" / "style_grid_boxes.json").write_text(
        json.dumps({"base": [BASE_W, BASE_H], "row_grid": ROW_GRID, "inset_frac": INSET_FRAC, "items": items}, indent=2),
        encoding="utf-8",
    )

    # Preview mosaics: row 0 and a 6x11 overview
    for ri, label in [(0, "row0"), (4, "row4"), (5, "row5")]:
        slugs = ROW_SLUGS[ri]
        thumbs = [Image.open(OUT_DIR / f"{s}.jpg").convert("RGB") for s in slugs]
        h = 90
        thumbs = [t.resize((90, 90)) for t in thumbs]
        mos = Image.new("RGB", (90 * len(thumbs), h), (20, 20, 24))
        x = 0
        for t in thumbs:
            mos.paste(t, (x, 0))
            x += 90
        mos.save(ROOT / "assets" / f"style_{label}_mosaic.jpg", quality=92)

    print(f"wrote {len(items)} thumbs")


if __name__ == "__main__":
    main()
