"""Database models: Dataset, TrainingJob, LoraModel."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Dataset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    concept: str
    repeat: int = 10
    trigger_word: str = ""
    base_model: str = ""  # chosen base checkpoint filename; "" = platform default
    image_count: int = 0
    status: str = "draft"  # draft | captioned | ready
    # Auto-caption task lifecycle (survives page reloads / backend restarts):
    # idle | running | done | failed. Lets the UI restore a "captioning…" state.
    caption_status: str = "idle"
    caption_detail: str = ""  # last caption run result / error message
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    dataset_id: int = Field(foreign_key="dataset.id")
    base_model: str = ""
    backend: str = "local_mps"
    params: str = "{}"  # JSON-encoded training parameters
    status: str = "pending"  # pending | queued | running | succeeded | failed | stopped | paused
    progress: float = 0.0
    current_step: int = 0
    total_step: int = 0
    latest_loss: Optional[float] = None
    pid: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    queued_at: Optional[datetime] = None  # set when placed in the run queue
    finished_at: Optional[datetime] = None


class LoraModel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="trainingjob.id")
    name: str
    epoch: int = 0
    base_model: str = ""  # base checkpoint this LoRA was trained on
    file_path: str = ""
    file_size: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RemoteHost(SQLModel, table=True):
    """A cloud / remote CUDA host reachable over SSH for off-box training."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    host: str
    port: int = 22
    username: str = "root"
    auth_type: str = "key"  # "key" | "password"
    # Credentials are stored locally (single-user SQLite). Never returned to UI.
    password: str = ""
    private_key_path: str = ""  # e.g. ~/.ssh/id_ed25519
    # Remote layout / runtime
    workdir: str = "~/loralab"  # remote base dir for datasets/jobs
    kohya_dir: str = "~/sd-scripts"  # remote kohya checkout
    python_cmd: str = "python"  # remote interpreter / launcher prefix
    base_models_dir: str = ""  # remote dir holding base checkpoints (optional)
    rvc_dir: str = "~/Retrieval-based-Voice-Conversion-WebUI"  # remote RVC checkout
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Voice cloning / SVC vertical (RVC engine). Kept as separate tables so the
# image LoRA flow above is untouched.
# ---------------------------------------------------------------------------
class VoiceDataset(SQLModel, table=True):
    """A collection of audio clips of one target voice for SVC training."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    speaker: str  # target voice / singer name (used as model identity)
    sample_rate: int = 40000  # RVC target sr: 32000 | 40000 | 48000
    clip_count: int = 0
    total_seconds: float = 0.0
    status: str = "draft"  # draft | ready
    created_at: datetime = Field(default_factory=datetime.utcnow)


class VoiceJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    dataset_id: int = Field(foreign_key="voicedataset.id")
    backend: str = "local_rvc"
    params: str = "{}"  # JSON-encoded RVC training params
    status: str = "pending"  # pending | running | succeeded | failed | stopped
    progress: float = 0.0
    current_step: int = 0
    total_step: int = 0
    pid: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None


class VoiceModel(SQLModel, table=True):
    """A produced RVC voice model (.pth weight + optional .index file)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="voicejob.id")
    name: str
    speaker: str = ""
    epoch: int = 0
    sample_rate: int = 40000
    file_path: str = ""  # .pth weight
    index_path: str = ""  # .index feature retrieval file (optional)
    file_size: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
