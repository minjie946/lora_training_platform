"""Image tools router (微博图片管理): album pull + single-person filter.

Both actions kick off a background task (see services.image_manager) and the UI
polls task status + logs. Also exposes the list of downloaded/filtered
directories so the filter tab can pick a target and show category breakdowns.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from sqlmodel import select

from ..config import ALLOWED_IMAGE_EXTS, IMAGE_TOOLS_COOKIE
from ..db import get_session
from ..models import ImageTask
from ..schemas import (
    ImageBrowseResult,
    ImageCookieRead,
    ImageCookieUpdate,
    ImageDirEntry,
    ImageFilterRequest,
    ImagePreviewRequest,
    ImagePreviewResult,
    ImagePullRequest,
    ImagePullSelectedRequest,
    ImageSettingsRead,
    ImageSettingsUpdate,
    ImageTaskRead,
    XhsPreviewRequest,
    XhsPullRequest,
    XhsPullSelectedRequest,
)
from ..services import image_manager

router = APIRouter(prefix="/api/images", tags=["images"])

# Category subdirs the filter script produces inside a target directory.
_FILTER_CATEGORIES = ("single", "single_lowq", "multi", "poster", "collage", "animal")


def _to_read(task: ImageTask) -> ImageTaskRead:
    try:
        params = json.loads(task.params or "{}")
    except Exception:  # noqa: BLE001
        params = {}
    # Strip internal launch spec (command/env) — not for the UI.
    params = {k: v for k, v in params.items() if not k.startswith("_")}
    prog = (
        image_manager.parse_progress(task.id, task.status)
        if task.kind == "pull"
        else {"progress": 0.0, "done": 0, "total": 0}
    )
    return ImageTaskRead(
        id=task.id,
        kind=task.kind,
        target=task.target,
        out_dir=task.out_dir,
        params=params,
        status=task.status,
        detail=task.detail,
        created_at=task.created_at,
        finished_at=task.finished_at,
        progress=prog["progress"],
        done=prog["done"],
        total=prog["total"],
    )


def _count_images(directory) -> int:
    n = 0
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Environment / status
# --------------------------------------------------------------------------- #
@router.get("/config")
def image_config():
    """Report whether the tools are wired up (cookie present, out dir)."""
    return {
        "cookie_present": IMAGE_TOOLS_COOKIE.exists(),
        "cookie_path": str(IMAGE_TOOLS_COOKIE),
        "out_dir": str(image_manager.get_out_dir()),
    }


@router.get("/settings", response_model=ImageSettingsRead)
def get_settings():
    """Current image-tools settings (effective download/pull/filter dir)."""
    return ImageSettingsRead(**image_manager.read_settings())


@router.put("/settings", response_model=ImageSettingsRead)
def update_settings(body: ImageSettingsUpdate):
    """Persist the download/pull/filter root dir. Empty resets to default."""
    try:
        data = image_manager.write_settings(out_dir=body.out_dir)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return ImageSettingsRead(**data)


@router.get("/browse", response_model=ImageBrowseResult)
def browse_dir(path: str = Query("", description="要浏览的目录（空=用户主目录）")):
    """List subdirectories for the settings folder picker."""
    return ImageBrowseResult(**image_manager.browse_dir(path))


@router.post("/pick-dir")
def pick_dir(initial: str = Query("", description="打开时的默认目录")):
    """Open the machine's native folder chooser (local backend only).

    Returns {"path": <picked>} or {"path": null} if the user cancelled.
    """
    try:
        picked = image_manager.pick_dir_native(initial)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"path": picked}


@router.get("/cookie", response_model=ImageCookieRead)
def get_cookie(platform: str = Query("weibo", description="weibo | xhs")):
    """Cookie metadata (masked preview only — never the full credential)."""
    return ImageCookieRead(**image_manager.read_cookie_info(platform))


@router.get("/cookie/raw")
def get_cookie_raw(platform: str = Query("weibo", description="weibo | xhs")):
    """Full cookie text for the edit form to prefill (single-user local tool)."""
    return {"cookie": image_manager.read_cookie_raw(platform)}


@router.put("/cookie", response_model=ImageCookieRead)
def set_cookie(body: ImageCookieUpdate, platform: str = Query("weibo", description="weibo | xhs")):
    """Save/replace the login cookie used by the downloader (weibo or 小红书)."""
    try:
        image_manager.write_cookie(body.cookie, platform)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return ImageCookieRead(**image_manager.read_cookie_info(platform))


@router.get("/proxy")
def proxy_image(url: str = Query(..., description="图片直链 (sinaimg / xhscdn)")):
    """Proxy a weibo sinaimg or 小红书 xhscdn image with the right Referer.

    Both CDNs return 403 unless the request carries the platform's Referer, so a
    browser <img> from our origin can't load previews directly. We fetch it
    server-side and stream the bytes back. Restricted to the two known CDNs.
    """
    import urllib.parse

    host = urllib.parse.urlparse(url).netloc
    is_weibo = host.endswith(".sinaimg.cn") or host == "sinaimg.cn"
    is_xhs = (
        host.endswith(".xhscdn.com")
        or host == "xhscdn.com"
        or host.endswith(".xiaohongshu.com")  # ci.xiaohongshu.com 原图 CDN
    )
    if not (is_weibo or is_xhs):
        raise HTTPException(400, "仅允许代理 sinaimg / xhscdn 图片")
    referer = "https://www.xiaohongshu.com/" if is_xhs else "https://weibo.com/"

    import requests

    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
                "Referer": referer,
            },
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"图片获取失败：{e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, "图片获取失败")
    return Response(
        content=r.content,
        media_type=r.headers.get("Content-Type", "image/jpeg"),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/dirs", response_model=list[ImageDirEntry])
def list_dirs():
    """List downloaded directories under the image-tools output dir."""
    out = image_manager.get_out_dir()
    if not out.exists():
        return []
    entries: list[ImageDirEntry] = []
    for d in sorted(out.iterdir()):
        if not d.is_dir():
            continue
        categories: dict[str, int] = {}
        for cat in _FILTER_CATEGORIES:
            sub = d / cat
            if sub.is_dir():
                categories[cat] = _count_images(sub)
        entries.append(
            ImageDirEntry(
                name=d.name,
                image_count=_count_images(d),
                categories=categories,
            )
        )
    return entries


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
@router.get("/tasks", response_model=list[ImageTaskRead])
def list_tasks(kind: str | None = None):
    with get_session() as session:
        stmt = select(ImageTask).order_by(ImageTask.id.desc())
        if kind:
            stmt = stmt.where(ImageTask.kind == kind)
        return [_to_read(t) for t in session.exec(stmt).all()]


@router.get("/tasks/{task_id}", response_model=ImageTaskRead)
def get_task(task_id: int):
    with get_session() as session:
        task = session.get(ImageTask, task_id)
        if not task:
            raise HTTPException(404, "任务不存在")
        return _to_read(task)


@router.get("/tasks/{task_id}/log")
def get_task_log(task_id: int, tail: int = 400):
    with get_session() as session:
        if not session.get(ImageTask, task_id):
            raise HTTPException(404, "任务不存在")
    return {"log": image_manager.read_log(task_id, tail=tail)}


@router.post("/tasks/{task_id}/stop")
def stop_task(task_id: int):
    if not image_manager.stop_task(task_id):
        raise HTTPException(400, "任务未在运行或不存在")
    return {"ok": True}


@router.post("/tasks/{task_id}/pause")
def pause_task(task_id: int):
    """Pause a running pull task (keep downloaded images, allow resume/download)."""
    if not image_manager.pause_task(task_id):
        raise HTTPException(400, "任务未在运行或不是拉取任务")
    return get_task(task_id)


@router.post("/tasks/{task_id}/resume")
def resume_task(task_id: int):
    """Resume a paused pull task (skips already-downloaded files)."""
    try:
        ok = image_manager.resume_task(task_id)
    except ValueError as e:
        raise HTTPException(409, str(e))
    if not ok:
        raise HTTPException(400, "任务不是已暂停状态，无法继续")
    return get_task(task_id)


@router.post("/tasks/{task_id}/discard")
def discard_task(task_id: int):
    """Stop a pull task and delete the images it downloaded."""
    if not image_manager.discard_task(task_id):
        raise HTTPException(400, "任务不存在或不是拉取任务")
    return {"ok": True}


@router.post("/pull", response_model=ImageTaskRead)
def pull_images(body: ImagePullRequest):
    try:
        task_id = image_manager.start_pull(
            uid=body.uid.strip(),
            album=body.album.strip(),
            workers=body.workers,
            start=body.start,
            end=body.end,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return get_task(task_id)


@router.get("/preview-log")
def get_preview_log(
    platform: str = Query("weibo", description="weibo | xhs"),
    tail: int = 400,
):
    """Live log of the current (synchronous) preview fetch for a platform.

    The preview call blocks until it returns the image list, but the fetch can
    be slow (album paging / a real browser session), so the UI polls this while
    waiting to show progress instead of a bare spinner.
    """
    kind = "xhs" if platform == "xhs" else "weibo"
    return {"log": image_manager.read_preview_log(kind, tail=tail)}


@router.post("/preview", response_model=ImagePreviewResult)
def preview_images(body: ImagePreviewRequest):
    """Fetch the pid list (no download) so the UI can preview + select."""
    try:
        data = image_manager.preview_pids(
            uid=body.uid.strip(),
            album=body.album.strip(),
            start=body.start,
            end=body.end,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return ImagePreviewResult(**data)


@router.post("/pull-selected", response_model=ImageTaskRead)
def pull_selected(body: ImagePullSelectedRequest):
    """Download only the chosen pids into the resolved output directory."""
    try:
        task_id = image_manager.start_pull_selected(
            pids=body.pids,
            out_dir_name=body.out_dir_name,
            workers=body.workers,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return get_task(task_id)


# --------------------------------------------------------------------------- #
# 小红书（XHS）：博主主页全量
# --------------------------------------------------------------------------- #
@router.post("/xhs/preview", response_model=ImagePreviewResult)
def xhs_preview(body: XhsPreviewRequest):
    """Fetch a 小红书 author's full image list (no download) for preview."""
    try:
        data = image_manager.preview_xhs_user(
            user=body.user.strip(),
            max_notes=body.max_notes,
            headed=body.headed,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return ImagePreviewResult(**data)


@router.post("/xhs/pull", response_model=ImageTaskRead)
def xhs_pull(body: XhsPullRequest):
    """Download ALL images of a 小红书 author in the background."""
    try:
        task_id = image_manager.start_pull_xhs(
            user=body.user.strip(),
            workers=body.workers,
            max_notes=body.max_notes,
            headed=body.headed,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return get_task(task_id)


@router.post("/xhs/pull-selected", response_model=ImageTaskRead)
def xhs_pull_selected(body: XhsPullSelectedRequest):
    """Download only the chosen 小红书 image ids for an author."""
    try:
        task_id = image_manager.start_pull_xhs_selected(
            ids=body.ids,
            user=body.user.strip(),
            out_dir_name=body.out_dir_name,
            workers=body.workers,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return get_task(task_id)


@router.post("/filter", response_model=ImageTaskRead)
def filter_images(body: ImageFilterRequest):
    try:
        task_id = image_manager.start_filter(
            directory=body.directory,
            recursive=body.recursive,
            dry_run=body.dry_run,
            min_face=body.min_face,
            text_blocks=body.text_blocks,
            text_area=body.text_area,
            no_text_filter=body.no_text_filter,
            no_animal_filter=body.no_animal_filter,
            no_quality_filter=body.no_quality_filter,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return get_task(task_id)
