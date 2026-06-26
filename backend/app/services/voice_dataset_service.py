"""Voice (SVC/RVC) dataset filesystem service: audio clip storage and listing.

Layout: <VOICE_DATASETS_DIR>/<dataset_id>/audio/<clip>.<ext>
The whole audio/ directory is what gets fed to RVC's trainset preprocessing.
"""
from __future__ import annotations

import contextlib
import re
import shutil
import wave
from pathlib import Path

from ..config import ALLOWED_AUDIO_EXTS, VOICE_DATASETS_DIR

AUDIO_DIRNAME = "audio"


def dataset_root(dataset_id: int) -> Path:
    return VOICE_DATASETS_DIR / str(dataset_id)


def audio_dir(dataset_id: int) -> Path:
    return dataset_root(dataset_id) / AUDIO_DIRNAME


def ensure_audio_dir(dataset_id: int) -> Path:
    d = audio_dir(dataset_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _wav_seconds(path: Path) -> float:
    """Best-effort clip duration (stdlib wave for .wav; 0 otherwise)."""
    if path.suffix.lower() != ".wav":
        return 0.0
    with contextlib.suppress(Exception):
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate:
                return round(frames / float(rate), 2)
    return 0.0


def save_clip(dataset_id: int, filename: str, data: bytes) -> str:
    d = ensure_audio_dir(dataset_id)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        raise ValueError(f"不支持的音频格式: {ext}")
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", Path(filename).stem) or "clip"
    dest = d / f"{stem}{ext}"
    idx = 1
    while dest.exists():
        dest = d / f"{stem}_{idx}{ext}"
        idx += 1
    dest.write_bytes(data)
    return dest.name


def list_clips(dataset_id: int) -> list[dict]:
    d = audio_dir(dataset_id)
    if not d.exists():
        return []
    items: list[dict] = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in ALLOWED_AUDIO_EXTS:
            continue
        items.append(
            {
                "filename": p.name,
                "seconds": _wav_seconds(p),
                "size_bytes": p.stat().st_size,
                "audio_url": f"/api/voice/datasets/{dataset_id}/clips/{p.name}/raw",
            }
        )
    return items


def count_clips(dataset_id: int) -> int:
    d = audio_dir(dataset_id)
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in ALLOWED_AUDIO_EXTS)


def total_seconds(dataset_id: int) -> float:
    d = audio_dir(dataset_id)
    if not d.exists():
        return 0.0
    return round(
        sum(_wav_seconds(p) for p in d.iterdir() if p.suffix.lower() in ALLOWED_AUDIO_EXTS),
        2,
    )


def get_clip_path(dataset_id: int, filename: str) -> Path | None:
    d = audio_dir(dataset_id)
    if not d.exists():
        return None
    p = (d / filename).resolve()
    if not str(p).startswith(str(d.resolve())) or not p.exists():
        return None
    return p


def delete_clip(dataset_id: int, filename: str) -> bool:
    p = get_clip_path(dataset_id, filename)
    if not p:
        return False
    p.unlink(missing_ok=True)
    return True


def delete_dataset_files(dataset_id: int) -> None:
    root = dataset_root(dataset_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
