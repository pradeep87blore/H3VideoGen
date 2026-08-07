from PIL import Image
from pathlib import Path

im0 = Image.open("assets/ai_video_styles_reference_sheet.png").convert("RGB")
sc = 3
im = im0.resize((im0.width * sc, im0.height * sc), Image.Resampling.LANCZOS)
w, h = im.size
y_bands = [
    (140, 415),
    (469, 743),
    (796, 1060),
    (1114, 1370),
    (1428, 1670),
    (1733, 1960),
]
qa = Path("assets/style_crop_qa")
qa.mkdir(exist_ok=True)
trials = [
    ("A", 0.092, 0.988),
    ("B", 0.078, 0.990),
    ("C", 0.068, 0.992),
    ("D", 0.100, 0.985),
    ("E", 0.085, 0.990),
    ("F", 0.110, 0.980),
    ("G", 0.120, 0.978),
]
for name, lf, rf in trials:
    left = int(w * lf)
    right = int(w * rf)
    pitch = (right - left) / 11
    for ci, label in [(0, "c0"), (1, "c1"), (7, "c7")]:
        xa = int(left + ci * pitch + pitch * 0.08)
        xb = int(left + (ci + 1) * pitch - pitch * 0.08)
        ya, yb = y_bands[0]
        crop = im.crop((xa, ya, xb, yb)).resize((200, 200))
        crop.save(qa / f"tune_{name}_{label}.jpg", quality=90)
    print(name, "pitch", round(pitch), "left", left, "right", right)
print("done")
