"""MiniMax H3 models shared with pipeline schemas."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class NarrativeMode(str, Enum):
    """Story structure / production strategy."""

    character = "character"  # fairy tales, ensemble cast, identity sheets
    documentary = "documentary"  # history / events / places
    explainer = "explainer"  # concepts, analogies, educational


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
    image_path: Optional[str] = None
    picture_index: Optional[int] = None
    sheet: list[CharacterSheetPose] = Field(default_factory=list)
    sheet_started_at: Optional[str] = None
    sheet_finished_at: Optional[str] = None
    sheet_duration_sec: Optional[float] = None
    sheet_status: str = "pending"
    sheet_source: str = ""


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
    ref_character_ids: list[str] = Field(default_factory=list)
    narration_line: str = ""
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
    narrative_mode: str = "character"
    narration_script: str = ""


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
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: Optional[float] = None


class StageTiming(BaseModel):
    key: str
    label: str
    started_at: str = ""
    ended_at: Optional[str] = None
    duration_sec: Optional[float] = None
    status: str = "pending"
    detail: str = ""


class ProjectState(BaseModel):
    project_id: str
    user_prompt: str
    style: str
    plan: Optional[ProductionPlan] = None
    shots: list[ShotRecord] = Field(default_factory=list)
    master_path: Optional[str] = None
    character_board_dir: Optional[str] = None
    h3_mode: str = "r2v"
    narrative_mode: str = "character"
    narration_path: Optional[str] = None
    status: str = "created"
    log: list[str] = Field(default_factory=list)
    created_at: str = ""
    job_started_at: str = ""
    job_finished_at: Optional[str] = None
    stage_timings: list[StageTiming] = Field(default_factory=list)


def normalize_narrative_mode(value: str | NarrativeMode | None) -> str:
    if value is None or value == "":
        return NarrativeMode.character.value
    s = str(value).strip().lower()
    aliases = {
        "char": NarrativeMode.character.value,
        "story": NarrativeMode.character.value,
        "fairy": NarrativeMode.character.value,
        "doc": NarrativeMode.documentary.value,
        "history": NarrativeMode.documentary.value,
        "event": NarrativeMode.documentary.value,
        "education": NarrativeMode.explainer.value,
        "explain": NarrativeMode.explainer.value,
        "concept": NarrativeMode.explainer.value,
    }
    s = aliases.get(s, s)
    if s not in {m.value for m in NarrativeMode}:
        return NarrativeMode.character.value
    return s


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Story / concept in plain language")
    style: str = Field(
        default="Premium 3D animated fairy-tale, cinematic lighting, YouTube-ready",
        description="Visual style direction separate from the story prompt",
    )
    target_duration_sec: float = Field(default=60.0, ge=10, le=180)
    max_shots: int = Field(default=12, ge=2, le=24)
    max_retakes: Optional[int] = None
    auto_assemble: bool = True
    seed_base: int = 42
    h3_mode: Optional[str] = None
    narrative_mode: str = "character"

    @field_validator("narrative_mode", mode="before")
    @classmethod
    def _mode(cls, v: Any) -> str:
        return normalize_narrative_mode(v)


class ResumeRequest(BaseModel):
    max_retakes: Optional[int] = None
    auto_assemble: bool = True
    seed_base: int = 42
    h3_mode: Optional[str] = None
    narrative_mode: Optional[str] = None
    redo_failed: bool = True


class DirectorOnlyRequest(BaseModel):
    prompt: str
    style: str = "Premium 3D animated cinematic short"
    target_duration_sec: float = 60.0
    max_shots: int = 12
    narrative_mode: str = "character"

    @field_validator("narrative_mode", mode="before")
    @classmethod
    def _mode(cls, v: Any) -> str:
        return normalize_narrative_mode(v)
