"""FFmpeg helpers: frame extract, probe, assemble master."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .config import Settings


class MediaError(RuntimeError):
    pass


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise MediaError(e.stderr or e.stdout or str(e)) from e


def probe(settings: Settings, video: Path) -> dict[str, Any]:
    cp = run(
        [
            settings.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate",
            "-of",
            "json",
            str(video),
        ]
    )
    return json.loads(cp.stdout)


def extract_frames(settings: Settings, video: Path, out_dir: Path, times: list[float] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    times = times or [0.5, 2.5, 4.0]
    paths: list[Path] = []
    for i, t in enumerate(times):
        out = out_dir / f"frame_{i:02d}_{t:.1f}s.jpg"
        try:
            run(
                [
                    settings.ffmpeg_path,
                    "-y",
                    "-ss",
                    str(t),
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(out),
                ]
            )
            if out.exists() and out.stat().st_size > 1000:
                paths.append(out)
        except MediaError:
            continue
    if not paths:
        # fallback first frame
        out = out_dir / "frame_00.jpg"
        run(
            [
                settings.ffmpeg_path,
                "-y",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(out),
            ]
        )
        paths.append(out)
    return paths


def make_title_card(settings: Settings, path: Path, title: str, subtitle: str, seconds: float = 3.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Escape drawtext special chars lightly
    safe_title = title.replace(":", "\\:").replace("'", "")[:80]
    safe_sub = subtitle.replace(":", "\\:").replace("'", "")[:100]
    font = "C\\\\:/Windows/Fonts/georgia.ttf"
    vf = (
        f"drawtext=fontfile={font}:text='{safe_title}':"
        f"fontsize=56:fontcolor=0xf5e6c8:x=(w-text_w)/2:y=(h-text_h)/2-36,"
        f"drawtext=fontfile={font}:text='{safe_sub}':"
        f"fontsize=26:fontcolor=0xc4b59a:x=(w-text_w)/2:y=(h-text_h)/2+36"
    )
    run(
        [
            settings.ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x0b0a09:s={settings.default_width}x{settings.default_height}:d={seconds}:r={settings.default_fps}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=32000:cl=stereo",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-t",
            str(seconds),
            str(path),
        ]
    )
    return path


def assemble_master(
    settings: Settings,
    clips: list[Path],
    master_path: Path,
    *,
    title: str = "",
    subtitle: str = "",
    add_cards: bool = True,
) -> Path:
    master_path.parent.mkdir(parents=True, exist_ok=True)
    work = master_path.parent / "norm"
    work.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    if add_cards and title:
        open_card = work / "title_open.mp4"
        make_title_card(settings, open_card, title, subtitle or "AI short · local pipeline", 3.5)
        parts.append(open_card)

    for i, clip in enumerate(clips):
        out = work / f"part_{i:03d}.mp4"
        run(
            [
                settings.ffmpeg_path,
                "-y",
                "-i",
                str(clip),
                "-vf",
                f"scale={settings.default_width}:{settings.default_height}:force_original_aspect_ratio=decrease,"
                f"pad={settings.default_width}:{settings.default_height}:(ow-iw)/2:(oh-ih)/2,fps={settings.default_fps},format=yuv420p",
                "-af",
                "aformat=sample_rates=32000:channel_layouts=stereo",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(out),
            ]
        )
        parts.append(out)

    if add_cards and title:
        end_card = work / "title_end.mp4"
        make_title_card(settings, end_card, "Thanks for watching", title[:60], 3.0)
        parts.append(end_card)

    concat_list = work / "concat.txt"
    # Use absolute POSIX-ish paths for ffmpeg concat on Windows
    lines = []
    for p in parts:
        ap = p.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{ap}'")
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run(
        [
            settings.ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(master_path),
        ]
    )
    return master_path
