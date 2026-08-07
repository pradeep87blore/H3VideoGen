"""Debug / retune crop geometry for the style reference sheet."""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "assets" / "ai_video_styles_reference_sheet.png"
im = Image.open(src).convert("RGB")
print("orig", im.size)
# work at higher res for accuracy
scale = 3
im = im.resize((im.width * scale, im.height * scale), Image.Resampling.LANCZOS)
w, h = im.size
px = im.load()

def mean_lum(x0, y0, x1, y1, step=2):
    s = n = 0
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            r, g, b = px[x, y]
            s += (r + g + b) / 3
            n += 1
    return s / max(1, n)

# vertical luminance profile (center band)
print("\nVertical profile (center x):")
for y in range(0, h, max(1, h // 80)):
    m = mean_lum(w // 3, y, 2 * w // 3, min(y + 4, h))
    print(f"  y={y:4d} ({100*y/h:5.1f}%) lum={m:6.1f}")

print("\nHorizontal profile (mid body):")
y_mid = int(h * 0.28)
for x in range(0, w, max(1, w // 60)):
    m = mean_lum(x, y_mid, min(x + 4, w), y_mid + int(h * 0.08))
    print(f"  x={x:4d} ({100*x/w:5.1f}%) lum={m:6.1f}")

# Save annotated guess with manual fractions
# Typical poster fractions (tuned by inspection of 1024x682 sheet)
header_frac = 0.085
footer_frac = 0.045
left_cat_frac = 0.095
right_pad = 0.012
top = int(h * header_frac)
bottom = int(h * (1 - footer_frac))
left = int(w * left_cat_frac)
right = int(w * (1 - right_pad))
cols, rows = 11, 6
# each row contains image (~70%) + label (~30%)
# image is the target
cw = (right - left) / cols
rh = (bottom - top) / rows
img_frac = 0.68  # portion of row that is the image tile
gap_x = 0.06  # relative gap inside cell for rounded margin
gap_y = 0.04

dbg = im.copy()
draw = ImageDraw.Draw(dbg)
draw.rectangle([left, top, right, bottom], outline=(255, 0, 0), width=3)
for r in range(rows):
    for c in range(cols):
        x0 = left + c * cw
        y0 = top + r * rh
        # image box
        ix0 = int(x0 + cw * gap_x)
        ix1 = int(x0 + cw * (1 - gap_x))
        iy0 = int(y0 + rh * gap_y)
        iy1 = int(y0 + rh * img_frac)
        draw.rectangle([ix0, iy0, ix1, iy1], outline=(0, 255, 0), width=2)

out = ROOT / "assets" / "style_sheet_grid_debug.jpg"
dbg.resize((1200, int(1200 * h / w)), Image.Resampling.LANCZOS).save(out, quality=90)
print("wrote", out)
