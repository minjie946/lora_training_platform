"""Global configuration and workspace path management."""
from __future__ import annotations

import os
from pathlib import Path

# backend/app/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = Path(os.environ.get("LORA_WORKSPACE", BACKEND_DIR / "workspace"))

DATASETS_DIR = WORKSPACE_DIR / "datasets"
MODELS_BASE_DIR = WORKSPACE_DIR / "models" / "base"
JOBS_DIR = WORKSPACE_DIR / "jobs"
DB_PATH = WORKSPACE_DIR / "app.db"

# Voice / SVC (RVC) vertical workspace.
VOICE_DATASETS_DIR = WORKSPACE_DIR / "voice" / "datasets"
VOICE_JOBS_DIR = WORKSPACE_DIR / "voice" / "jobs"

# External training engines live under <repo>/engines/ to keep the repo root tidy.
ENGINES_DIR = Path(os.environ.get("ENGINES_DIR", BACKEND_DIR.parent / "engines"))

# Path to the kohya_ss / sd-scripts checkout (must contain train_network.py).
KOHYA_DIR = Path(os.environ.get("KOHYA_DIR", ENGINES_DIR / "sd-scripts"))

# Path to the RVC (Retrieval-based-Voice-Conversion-WebUI) checkout used for SVC.
RVC_DIR = Path(
    os.environ.get("RVC_DIR", ENGINES_DIR / "Retrieval-based-Voice-Conversion-WebUI")
)

# RVC has heavy deps that conflict with kohya's pinned torch, so it gets its own
# virtualenv. Local SVC training is launched with this interpreter (falls back
# to RVC_DIR/.venv/bin/python).
RVC_PYTHON = Path(os.environ.get("RVC_PYTHON", RVC_DIR / ".venv" / "bin" / "python"))

# Default base model filename expected under MODELS_BASE_DIR.
DEFAULT_BASE_MODEL = os.environ.get("DEFAULT_BASE_MODEL", "animagine-xl-4.0-opt.safetensors")

THUMBNAIL_SIZE = (256, 256)
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def ensure_dirs() -> None:
    """Create all workspace directories if missing."""
    for d in (
        WORKSPACE_DIR,
        DATASETS_DIR,
        MODELS_BASE_DIR,
        JOBS_DIR,
        VOICE_DATASETS_DIR,
        VOICE_JOBS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
