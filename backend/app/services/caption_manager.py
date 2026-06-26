"""Background auto-caption task manager.

Captioning can take a while (model download + per-image inference), so we run it
in a daemon thread and persist its lifecycle on the Dataset row
(``caption_status``). That lets the UI restore a "captioning…" state after a
page reload, and lets us reconcile a run that was interrupted by a backend
restart.
"""
from __future__ import annotations

import threading

from ..db import get_session
from ..models import Dataset
from . import caption_service as cap


# In-process guard so the same dataset isn't captioned twice concurrently.
_running: set[int] = set()
_lock = threading.Lock()


def is_running(dataset_id: int) -> bool:
    with _lock:
        return dataset_id in _running


def start_auto_caption(
    dataset_id: int,
    *,
    threshold: float,
    do_inject: bool,
    trigger: str,
    base_model: str,
    method: str,
    exclude_body_face: bool,
    exclude_tags: list[str],
) -> None:
    """Kick off captioning in the background. Raises if already running."""
    with _lock:
        if dataset_id in _running:
            raise ValueError("该数据集正在打标中")
        _running.add(dataset_id)

    _set_status(dataset_id, "running", "打标进行中…")

    thread = threading.Thread(
        target=_run,
        args=(dataset_id,),
        kwargs=dict(
            threshold=threshold,
            do_inject=do_inject,
            trigger=trigger,
            base_model=base_model,
            method=method,
            exclude_body_face=exclude_body_face,
            exclude_tags=exclude_tags,
        ),
        daemon=True,
    )
    thread.start()


def _run(
    dataset_id: int,
    *,
    threshold: float,
    do_inject: bool,
    trigger: str,
    base_model: str,
    method: str,
    exclude_body_face: bool,
    exclude_tags: list[str],
) -> None:
    try:
        _, count, detail = cap.auto_caption(
            dataset_id,
            threshold,
            do_inject,
            trigger,
            base_model=base_model,
            method=method,
            exclude_body_face=exclude_body_face,
            exclude_tags=exclude_tags,
        )
        _set_status(dataset_id, "done", detail, mark_captioned=count > 0)
    except Exception as e:  # noqa: BLE001
        _set_status(dataset_id, "failed", f"打标失败: {e}")
    finally:
        with _lock:
            _running.discard(dataset_id)


def _set_status(
    dataset_id: int, status: str, detail: str, mark_captioned: bool = False
) -> None:
    with get_session() as session:
        ds = session.get(Dataset, dataset_id)
        if not ds:
            return
        ds.caption_status = status
        ds.caption_detail = detail
        if mark_captioned:
            ds.status = "captioned"
        session.add(ds)
        session.commit()


def reconcile_on_startup() -> None:
    """A run marked 'running' in the DB whose thread died with the previous
    process can never finish — flag it failed so the UI stops showing a spinner.
    """
    from sqlmodel import select

    with get_session() as session:
        stale = session.exec(
            select(Dataset).where(Dataset.caption_status == "running")
        ).all()
        for ds in stale:
            ds.caption_status = "failed"
            ds.caption_detail = "打标进程在后端重启时中断，请重新打标"
            session.add(ds)
        if stale:
            session.commit()
