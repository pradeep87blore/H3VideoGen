"""MiniMax H3 models shared with pipeline schemas."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ShotStatus(str, Enum):
    pending = "pending"
    generating = "generating"
    reviewing = "reviewing"
    passed = "passed"
    retake = "retake"
    failed = "failed"
    skipped = "skipped"


class CriticVerdict(str, Enum):
    pass_ = "PASS"
    retake = "RETAKE"
    reject = "REJECT"


class CharacterSheetPose(BaseModel):
    """One multi-view still on a character design sheet."""

    pose_id: str  # front_full | three_quarter | side | closeup_face | action | back | …
    label: str = ""
    image_path: Optional[str] = None
    prompt: str = ""


class CharacterDesign(BaseModel):
    """Locked identity sheet for R2V <Picture n> references."""

    id: str = "C01"
    name: str
    look: str = ""
    board_prompt: str = ""
    # Primary still (usually front_full or first available sheet pose)
    image_path: Optional[str] = None
    picture_index: Optional[int] = None  # deprecated single-tag index; sheet-driven now
    # Multi-view stills for stronger identity (H3 R2V allows ≤9 images total)
    sheet: list[CharacterSheetPose] = Field(default_factory=list)


class ShotPlan(BaseModel):
    id: str
    name: str
    beat: str = ""
    duration_sec: float = 5.0
    length_frames: int = 124
    visual_prompt: str
    camera: str = ""
    audio_notes: str = ""
    character_presence: str = ""
    # Which character sheet ids appear on screen (maps to <Picture n> tags)
    ref_character_ids: list[str] = Field(default_factory=list)
    seed: Optional[int] = None


class ProductionPlan(BaseModel):
    title: str
    logline: str = ""
    target_duration_sec: float = 60.0
    aspect_ratio: str = "16:9"
    style_bible: str
    character_lock: str = ""
    characters: list[CharacterDesign] = Field(default_factory=list)
    color_grade: str = ""
    audio_bed: str = ""
    youtube_notes: str = ""
    shots: list[ShotPlan] = Field(default_factory=list)
    raw_director_notes: str = ""


class CriticReview(BaseModel):
    shot_id: str
    take: int = 1
    verdict: CriticVerdict
    overall_score: float = Field(ge=0, le=10)
    youtube_ready: bool = False
    scores: dict[str, float] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    retake_instructions: str = ""
    revised_prompt: str = ""
    summary: str = ""


class ShotRecord(BaseModel):
    plan: ShotPlan
    status: ShotStatus = ShotStatus.pending
    takes: list[dict[str, Any]] = Field(default_factory=list)
    final_video: Optional[str] = None
    final_frame: Optional[str] = None
    reviews: list[CriticReview] = Field(default_factory=list)
    error: Optional[str] = None


class ProjectState(BaseModel):
    project_id: str
    user_prompt: str
    style: str
    plan: Optional[ProductionPlan] = None
    shots: list[ShotRecord] = Field(default_factory=list)
    master_path: Optional[str] = None
    character_board_dir: Optional[str] = None
    h3_mode: str = "r2v"
    status: str = "created"
    log: list[str] = Field(default_factory=list)
    created_at: str = ""


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Story / concept in plain language")
    style: str = Field(
        default="Premium 3D animated fairy-tale, cinematic lighting, YouTube-ready",
        description="Visual style direction separate from the story prompt",
    )
    target_duration_sec: float = Field(default=60.0, ge=15, le=180)
    max_shots: int = Field(default=12, ge=2, le=24)
    max_retakes: Optional[int] = None
    auto_assemble: bool = True
    seed_base: int = 42
    # r2v (reference consistency) | t2v (text only) | auto
    h3_mode: Optional[str] = None


class ResumeRequest(BaseModel):
    """Resume an existing project from the first unfinished shot."""

    max_retakes: Optional[int] = None
    auto_assemble: bool = True
    seed_base: int = 42
    h3_mode: Optional[str] = None
    # Re-run failed/cancelled shots; skip shots that already passed with video on disk
    redo_failed: bool = True


class DirectorOnlyRequest(BaseModel):
    prompt: str
    style: str = "Premium 3D animated cinematic short"
    target_duration_sec: float = 60.0
    max_shots: int = 12
