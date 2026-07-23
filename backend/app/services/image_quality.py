"""Heuristic + face-detection image quality analysis for training datasets.

Answers "is this image suitable as LoRA training data?" using only PIL/numpy/
OpenCV (all already installed — no new dependencies). Each check is a cheap,
explainable heuristic; results are advisory only and never block training.

Levels: "ok" (no issues) < "warn" (worth reviewing) < "bad" (likely harmful).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# --- thresholds -------------------------------------------------------------
_MIN_EDGE_WARN = 512   # short edge below this → warn
_MIN_EDGE_BAD = 384    # short edge below this → bad
_ASPECT_MAX = 2.5      # w/h or h/w beyond this → warn (crop loses subject)
_BLUR_WARN = 100.0     # Laplacian variance below this → warn (soft)
_BLUR_BAD = 50.0       # below this → bad (clearly blurry)
# Exposure is judged by pixel clipping, not raw mean — a subject on a white/
# black background must NOT count as over/under-exposed. Only flag when a very
# large fraction of pixels is clipped (frame is blown out / crushed / near-blank).
_DARK_CLIP_FRAC = 0.6   # fraction of near-black pixels above this → underexposed
_BRIGHT_CLIP_FRAC = 0.92  # fraction of near-white pixels above this → blown/blank
_FACE_MIN_AREA_RATIO = 0.04  # largest face smaller than 4% of image → warn

_SEVERITY_RANK = {"ok": 0, "warn": 1, "bad": 2}

# OpenCV Haar face cascade, loaded once (lazily) and reused.
_cascade = None
_cascade_tried = False


def _get_cascade():
    global _cascade, _cascade_tried
    if _cascade_tried:
        return _cascade
    _cascade_tried = True
    try:
        import cv2  # type: ignore

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(path)
        _cascade = clf if not clf.empty() else None
    except Exception:  # noqa: BLE001
        _cascade = None
    return _cascade


def _issue(code: str, label: str, severity: str) -> dict[str, str]:
    return {"code": code, "label": label, "severity": severity}


def analyze_image(path: Path) -> dict[str, Any]:
    """Analyze a single image file. Never raises; on error returns level 'ok'."""
    issues: list[dict[str, str]] = []
    metrics: dict[str, Any] = {}
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            gray = np.asarray(im.convert("L"), dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        return {"level": "ok", "issues": [], "metrics": {"error": str(e)}}

    metrics["width"] = w
    metrics["height"] = h
    short_edge = min(w, h)
    metrics["short_edge"] = short_edge

    # 1. Resolution
    if short_edge < _MIN_EDGE_BAD:
        issues.append(_issue("low_res", f"分辨率过低（短边 {short_edge}px）", "bad"))
    elif short_edge < _MIN_EDGE_WARN:
        issues.append(_issue("low_res", f"分辨率偏低（短边 {short_edge}px）", "warn"))

    # 2. Extreme aspect ratio
    ratio = max(w, h) / max(1, min(w, h))
    metrics["aspect"] = round(ratio, 2)
    if ratio > _ASPECT_MAX:
        issues.append(_issue("aspect", f"宽高比极端（{ratio:.1f}:1，裁剪易丢主体）", "warn"))

    # 3. Blur — variance of the Laplacian on the grayscale image
    blur_var = _laplacian_variance(gray)
    metrics["blur_var"] = round(blur_var, 1)
    if blur_var < _BLUR_BAD:
        issues.append(_issue("blur", "图片模糊（清晰度很低）", "bad"))
    elif blur_var < _BLUR_WARN:
        issues.append(_issue("blur", "图片偏模糊", "warn"))

    # 4. Exposure — judged by pixel clipping so a plain background doesn't
    # count as over/under-exposed. Only a mostly-clipped (blown/crushed) frame
    # trips this.
    mean = float(gray.mean())
    metrics["brightness"] = round(mean, 1)
    dark_frac = float(np.mean(gray < 16))
    bright_frac = float(np.mean(gray > 250))
    metrics["dark_frac"] = round(dark_frac, 3)
    metrics["bright_frac"] = round(bright_frac, 3)
    if dark_frac > _DARK_CLIP_FRAC:
        issues.append(_issue("dark", "画面大面积死黑（曝光不足或近乎空白）", "warn"))
    elif bright_frac > _BRIGHT_CLIP_FRAC:
        issues.append(_issue("bright", "画面几乎全白（过曝或近乎空白）", "warn"))

    # 5 & 6. Face presence / count / size (advisory only for character LoRA)
    _analyze_faces(path, w, h, issues, metrics)

    level = "ok"
    for it in issues:
        if _SEVERITY_RANK[it["severity"]] > _SEVERITY_RANK[level]:
            level = it["severity"]
    return {"level": level, "issues": issues, "metrics": metrics}


def _laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian (higher = sharper)."""
    try:
        import cv2  # type: ignore

        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:  # noqa: BLE001
        # Pure-numpy fallback: 4-neighbour discrete Laplacian.
        lap = (
            -4 * gray
            + np.roll(gray, 1, 0)
            + np.roll(gray, -1, 0)
            + np.roll(gray, 1, 1)
            + np.roll(gray, -1, 1)
        )
        return float(lap.var())


def _analyze_faces(
    path: Path, w: int, h: int, issues: list[dict[str, str]], metrics: dict[str, Any]
) -> None:
    clf = _get_cascade()
    if clf is None:
        return  # OpenCV unavailable — skip face checks silently
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        faces = clf.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    except Exception:  # noqa: BLE001
        return

    count = len(faces)
    metrics["face_count"] = int(count)
    if count == 0:
        issues.append(_issue("no_face", "未检测到清晰正脸", "warn"))
        return
    if count >= 2:
        issues.append(_issue("multi_face", f"检测到多张人脸（{count}）", "warn"))

    # Largest face area relative to the whole image.
    largest = max(int(fw) * int(fh) for (_, _, fw, fh) in faces)
    ratio = largest / float(max(1, w * h))
    metrics["face_area_ratio"] = round(ratio, 4)
    if ratio < _FACE_MIN_AREA_RATIO:
        issues.append(_issue("small_face", "人脸占比过小（主体太远）", "warn"))
