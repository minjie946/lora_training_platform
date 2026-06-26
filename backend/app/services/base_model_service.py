"""Base model discovery + classification.

Scans MODELS_BASE_DIR for *.safetensors checkpoints and infers, from the file
name, two properties the rest of the platform needs:

- is_sdxl: whether to launch sdxl_train_network.py vs train_network.py
- style:   "anime" or "realistic", which decides the captioning strategy
           (anime -> WD14 booru tags, realistic -> BLIP natural-language caption)

Drop a new .safetensors into MODELS_BASE_DIR and it shows up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import DEFAULT_BASE_MODEL, MODELS_BASE_DIR

# Substrings that hint a photoreal / realistic base model. Everything else is
# treated as anime/illustration (the common case for booru-tagged training).
_REALISTIC_HINTS = (
    "real",
    "realistic",
    "photo",
    "epicreal",
    "juggernaut",
    "rev",  # realisticVision "rev"
    "chilloutmix",
    "majicmix",
    "dreamshaper",
    "absolutereality",
)

_SDXL_HINTS = ("xl", "sdxl", "illustrious", "pony", "noobai", "animagine-xl")


@dataclass
class BaseModelInfo:
    filename: str
    label: str
    is_sdxl: bool
    style: str  # "anime" | "realistic"
    size_bytes: int
    is_default: bool


def classify_is_sdxl(filename: str) -> bool:
    low = filename.lower()
    return any(h in low for h in _SDXL_HINTS)


def classify_style(filename: str) -> str:
    low = filename.lower()
    if any(h in low for h in _REALISTIC_HINTS):
        return "realistic"
    return "anime"


def _label(filename: str) -> str:
    # Strip extension for a friendlier display label.
    return filename.rsplit(".", 1)[0]


def list_base_models() -> list[BaseModelInfo]:
    if not MODELS_BASE_DIR.exists():
        return []
    out: list[BaseModelInfo] = []
    for p in sorted(MODELS_BASE_DIR.glob("*.safetensors")):
        out.append(
            BaseModelInfo(
                filename=p.name,
                label=_label(p.name),
                is_sdxl=classify_is_sdxl(p.name),
                style=classify_style(p.name),
                size_bytes=p.stat().st_size,
                is_default=(p.name == DEFAULT_BASE_MODEL),
            )
        )
    return out


def get_base_model(filename: str) -> Optional[BaseModelInfo]:
    for m in list_base_models():
        if m.filename == filename:
            return m
    return None


def style_of(filename: str) -> str:
    """Style for a given base model filename, defaulting to anime when unknown."""
    info = get_base_model(filename)
    return info.style if info else classify_style(filename or "")


def is_sdxl_of(filename: str) -> bool:
    info = get_base_model(filename)
    return info.is_sdxl if info else classify_is_sdxl(filename or "")
