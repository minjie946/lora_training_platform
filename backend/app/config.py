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

# ---------------------------------------------------------------------------
# Image tools (微博图片管理): self-contained inside this project. The two
# standalone scripts live under backend/app/image_tools/ (version-controlled),
# while their runtime data — downloaded photos, model caches, cookie — live under
# the workspace (git-ignored). Their heavy CV deps (opencv/insightface/ultralytics)
# are declared as PEP 723 inline deps, so we launch them via `uv run --script`
# (which provisions an ephemeral env) rather than importing them into this venv.
IMAGE_TOOLS_DIR = Path(
    os.environ.get("IMAGE_TOOLS_DIR", BACKEND_DIR / "app" / "image_tools")
)
# Runtime data root (git-ignored) for the image tools.
IMAGE_TOOLS_WORKSPACE = WORKSPACE_DIR / "image_tools"
# Where downloaded albums land + get filtered.
IMAGE_TOOLS_OUT_DIR = Path(
    os.environ.get("IMAGE_TOOLS_OUT_DIR", IMAGE_TOOLS_WORKSPACE / "photos")
)
IMAGE_TOOLS_COOKIE = Path(
    os.environ.get("IMAGE_TOOLS_COOKIE", IMAGE_TOOLS_WORKSPACE / "cookie.txt")
)
# Xiaohongshu (小红书) uses a separate login cookie (needs a1 / web_session / webId).
IMAGE_TOOLS_XHS_COOKIE = Path(
    os.environ.get("IMAGE_TOOLS_XHS_COOKIE", IMAGE_TOOLS_WORKSPACE / "xhs_cookie.txt")
)
# Persistent model caches for the single-person filter (shared across runs so we
# never re-download InsightFace ~600MB / YOLOv8n ~6MB). Passed to the script via
# INSIGHTFACE_ROOT / YOLO_WEIGHTS env vars.
IMAGE_TOOLS_INSIGHTFACE_ROOT = (
    IMAGE_TOOLS_WORKSPACE / "models_insightface"
)
IMAGE_TOOLS_YOLO_WEIGHTS = (
    IMAGE_TOOLS_WORKSPACE / "models_yolo" / "yolov8n.pt"
)
# `uv` launcher used to run the PEP 723 scripts. Absolute path is safest since
# the backend may run under a service manager without the user's PATH.
UV_BIN = os.environ.get("UV_BIN", "uv")

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
        IMAGE_TOOLS_OUT_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
