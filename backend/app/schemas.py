"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


# ---- System ----
class PreflightItem(BaseModel):
    name: str
    ok: bool
    detail: str


class PreflightResult(BaseModel):
    ok: bool
    items: list[PreflightItem]


# ---- Dataset ----
class DatasetCreate(BaseModel):
    name: str
    concept: str
    repeat: int = 10
    trigger_word: str = ""
    base_model: str = ""


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    concept: Optional[str] = None
    repeat: Optional[int] = None
    trigger_word: Optional[str] = None
    base_model: Optional[str] = None


class DatasetRead(BaseModel):
    id: int
    name: str
    concept: str
    repeat: int
    trigger_word: str
    base_model: str
    image_count: int
    status: str
    caption_status: str = "idle"
    caption_detail: str = ""
    created_at: datetime


class DatasetImportResult(BaseModel):
    dataset: DatasetRead
    imported: int
    captioned: int
    detail: str


class TagScore(BaseModel):
    tag: str
    confidence: float


class QualityIssue(BaseModel):
    code: str
    label: str
    severity: str  # "warn" | "bad"


class ImageQuality(BaseModel):
    level: str  # "ok" | "warn" | "bad"
    issues: list[QualityIssue] = []


class QualityCheckResult(BaseModel):
    total: int
    ok: int
    warn: int
    bad: int


class ImageItem(BaseModel):
    filename: str
    caption: str
    thumbnail_url: str
    image_url: str
    # WD14 per-tag confidence scores (None unless WD14 tagging was run).
    tag_scores: Optional[list[TagScore]] = None
    # Heuristic image-quality analysis (None unless a quality check was run).
    quality: Optional[ImageQuality] = None


class CaptionUpdate(BaseModel):
    filename: str
    caption: str


# ---- Caption ----
class AutoCaptionRequest(BaseModel):
    threshold: float = 0.35
    inject_trigger: bool = True
    # "auto" = choose by base model style; or force "wd14" / "blip"
    method: str = "auto"
    # Character LoRA: drop body/face tags so those traits bake into the trigger.
    exclude_body_face: bool = False
    # Extra keywords to drop from WD14 tags (case-insensitive substring match).
    exclude_tags: list[str] = []


class AutoCaptionResult(BaseModel):
    ok: bool
    method: str  # "wd14" | "blip" | "manual_fallback"
    captioned: int
    detail: str


# ---- Jobs ----
class JobCreate(BaseModel):
    name: str
    dataset_id: int
    base_model: Optional[str] = None
    backend: str = "local_mps"
    params: dict[str, Any] = {}


class JobUpdate(BaseModel):
    """Partial update for a not-yet-running job. All fields optional."""

    name: Optional[str] = None
    dataset_id: Optional[int] = None
    base_model: Optional[str] = None
    backend: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class JobRead(BaseModel):
    id: int
    name: str
    dataset_id: int
    base_model: str
    backend: str
    params: dict[str, Any]
    status: str
    progress: float
    current_step: int
    total_step: int
    latest_loss: Optional[float]
    error: Optional[str]
    created_at: datetime
    queued_at: Optional[datetime] = None
    finished_at: Optional[datetime]
    # True when a resumable kohya `-state` checkpoint exists on disk, so a
    # paused job can actually continue. Drives the pause-vs-stop UI.
    has_checkpoint: bool = False


# ---- Models ----
class LoraModelRead(BaseModel):
    id: int
    job_id: int
    name: str
    epoch: int
    base_model: str
    file_size: int
    created_at: datetime


# ---- Remote hosts (cloud GPU) ----
class RemoteHostCreate(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str = "root"
    auth_type: str = "key"  # "key" | "password"
    password: str = ""
    private_key_path: str = ""
    workdir: str = "~/loralab"
    kohya_dir: str = "~/sd-scripts"
    python_cmd: str = "python"
    base_models_dir: str = ""
    rvc_dir: str = "~/Retrieval-based-Voice-Conversion-WebUI"


class RemoteHostUpdate(BaseModel):
    name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    auth_type: Optional[str] = None
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    workdir: Optional[str] = None
    kohya_dir: Optional[str] = None
    python_cmd: Optional[str] = None
    base_models_dir: Optional[str] = None
    rvc_dir: Optional[str] = None


class RemoteHostRead(BaseModel):
    """Credentials are never returned; only whether a secret is set."""

    id: int
    name: str
    host: str
    port: int
    username: str
    auth_type: str
    has_password: bool
    private_key_path: str
    workdir: str
    kohya_dir: str
    python_cmd: str
    base_models_dir: str
    rvc_dir: str
    created_at: datetime


class RemoteTestResult(BaseModel):
    ok: bool
    detail: str


# ---- Voice / SVC (RVC) ----
class VoiceDatasetCreate(BaseModel):
    name: str
    speaker: str
    sample_rate: int = 40000


class VoiceDatasetUpdate(BaseModel):
    name: Optional[str] = None
    speaker: Optional[str] = None
    sample_rate: Optional[int] = None


class VoiceDatasetRead(BaseModel):
    id: int
    name: str
    speaker: str
    sample_rate: int
    clip_count: int
    total_seconds: float
    status: str
    created_at: datetime


class AudioClip(BaseModel):
    filename: str
    seconds: float
    size_bytes: int
    audio_url: str


class VoiceJobCreate(BaseModel):
    name: str
    dataset_id: int
    backend: str = "local_rvc"
    params: dict[str, Any] = {}


class VoiceJobRead(BaseModel):
    id: int
    name: str
    dataset_id: int
    backend: str
    params: dict[str, Any]
    status: str
    progress: float
    current_step: int
    total_step: int
    error: Optional[str]
    created_at: datetime
    finished_at: Optional[datetime]


class VoiceModelRead(BaseModel):
    id: int
    job_id: int
    name: str
    speaker: str
    epoch: int
    sample_rate: int
    has_index: bool
    file_size: int
    created_at: datetime
