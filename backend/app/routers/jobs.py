"""Jobs router: create/start/stop/query training jobs + SSE log/progress stream."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from ..db import session_dependency
from ..models import Dataset, TrainingJob
from ..schemas import JobCreate, JobRead, JobUpdate
from ..services import config_builder, job_manager
from ..services.backends.registry import get_backend, list_backends

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _to_read(job: TrainingJob) -> JobRead:
    return JobRead(
        id=job.id,
        name=job.name,
        dataset_id=job.dataset_id,
        base_model=job.base_model,
        backend=job.backend,
        params=json.loads(job.params or "{}"),
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        total_step=job.total_step,
        latest_loss=job.latest_loss,
        error=job.error,
        created_at=job.created_at,
        queued_at=job.queued_at,
        finished_at=job.finished_at,
        has_checkpoint=job_manager.has_checkpoint(job.id),
    )


@router.get("/backends")
def get_backends():
    return list_backends()


@router.get("/defaults")
def get_defaults():
    return config_builder.DEFAULT_PARAMS


@router.get("", response_model=list[JobRead])
def list_jobs(session: Session = Depends(session_dependency)):
    jobs = session.exec(select(TrainingJob).order_by(TrainingJob.id.desc())).all()
    return [_to_read(j) for j in jobs]


@router.post("", response_model=JobRead)
def create_job(body: JobCreate, session: Session = Depends(session_dependency)):
    dataset = session.get(Dataset, body.dataset_id)
    if not dataset:
        raise HTTPException(404, "数据集不存在")
    try:
        get_backend(body.backend)
    except ValueError as e:
        raise HTTPException(400, str(e))

    job = TrainingJob(
        name=body.name,
        dataset_id=body.dataset_id,
        base_model=body.base_model or "",
        backend=body.backend,
        params=json.dumps(body.params or {}),
        total_step=config_builder.estimate_total_steps(
            dataset.image_count, dataset.repeat, body.params or {}
        ),
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return _to_read(job)


@router.get("/{job_id}", response_model=JobRead)
def get_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return _to_read(job)


@router.patch("/{job_id}", response_model=JobRead)
def update_job(
    job_id: int, body: JobUpdate, session: Session = Depends(session_dependency)
):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status in ("running", "paused"):
        raise HTTPException(400, "运行中或已暂停的任务不可编辑，请先停止")

    affects_training = False
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "任务名称不能为空")
        job.name = name
    if body.dataset_id is not None and body.dataset_id != job.dataset_id:
        if not session.get(Dataset, body.dataset_id):
            raise HTTPException(404, "数据集不存在")
        job.dataset_id = body.dataset_id
        affects_training = True
    if body.base_model is not None and body.base_model != job.base_model:
        job.base_model = body.base_model
        affects_training = True
    if body.backend is not None and body.backend != job.backend:
        try:
            get_backend(body.backend)
        except ValueError as e:
            raise HTTPException(400, str(e))
        job.backend = body.backend
        affects_training = True
    if body.params is not None:
        new_params = json.dumps(body.params)
        if new_params != job.params:
            job.params = new_params
            affects_training = True

    # Recompute the estimated total steps from the (possibly changed) dataset
    # and params so list/detail views stay accurate before the next run.
    dataset = session.get(Dataset, job.dataset_id)
    if dataset:
        job.total_step = config_builder.estimate_total_steps(
            dataset.image_count, dataset.repeat, json.loads(job.params or "{}")
        )
    # Changing training inputs invalidates any prior run — reset to pending so
    # the next start begins cleanly. A pure rename keeps the existing state.
    if affects_training:
        job.status = "pending"
        job.progress = 0.0
        job.current_step = 0
        job.latest_loss = None
        job.error = None
        job.finished_at = None
        session.add(job)
        session.commit()
        # Old models/logs/checkpoints no longer match the new config — wipe them.
        job_manager.reset_job_artifacts(job_id)
        session.refresh(job)
        return _to_read(job)

    session.add(job)
    session.commit()
    session.refresh(job)
    return _to_read(job)


@router.post("/{job_id}/start", response_model=JobRead)
def start_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status == "running":
        raise HTTPException(400, "任务已在运行")
    if job.status == "queued":
        raise HTTPException(400, "任务已在队列中")

    backend = get_backend(job.backend)
    pf = backend.preflight()
    if not pf.ok:
        raise HTTPException(400, f"环境检查未通过: {pf.detail}")

    try:
        # If another job is already running, this is placed in the queue and
        # will start automatically when the device frees up.
        job_manager.start_or_queue(job_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"启动失败: {e}")
    session.refresh(job)
    return _to_read(job)


@router.post("/{job_id}/dequeue", response_model=JobRead)
def dequeue_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if not job_manager.dequeue_job(job_id):
        raise HTTPException(400, "任务不在队列中")
    session.refresh(job)
    return _to_read(job)


@router.post("/{job_id}/stop", response_model=JobRead)
def stop_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if not job_manager.stop_job(job_id):
        raise HTTPException(400, "任务未在运行")
    session.refresh(job)
    return _to_read(job)


@router.post("/{job_id}/pause", response_model=JobRead)
def pause_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if not job_manager.pause_job(job_id):
        raise HTTPException(400, "只有正在运行的任务可以暂停")
    session.refresh(job)
    return _to_read(job)


@router.post("/{job_id}/resume", response_model=JobRead)
def resume_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    backend = get_backend(job.backend)
    pf = backend.preflight()
    if not pf.ok:
        raise HTTPException(400, f"环境检查未通过: {pf.detail}")
    try:
        job_manager.resume_job(job_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"继续训练失败: {e}")
    session.refresh(job)
    return _to_read(job)


@router.post("/{job_id}/clone", response_model=JobRead)
def clone_job(job_id: int, session: Session = Depends(session_dependency)):
    src = session.get(TrainingJob, job_id)
    if not src:
        raise HTTPException(404, "任务不存在")
    job = TrainingJob(
        name=f"{src.name} (重训)",
        dataset_id=src.dataset_id,
        base_model=src.base_model,
        backend=src.backend,
        params=src.params,
        total_step=src.total_step,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return _to_read(job)


@router.delete("/{job_id}")
def delete_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    job_manager.delete_job(job_id)
    return {"ok": True}


@router.get("/{job_id}/log")
def get_log(job_id: int, tail: int = 200, session: Session = Depends(session_dependency)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return {"log": job_manager.read_log(job_id, tail=tail)}


@router.get("/{job_id}/stream")
async def stream_job(job_id: int):
    """SSE stream of incremental log lines + periodic progress snapshots."""

    async def event_gen():
        log_file = job_manager.log_path(job_id)
        last_size = 0
        while True:
            # progress snapshot
            with job_manager.get_session() as s:
                job = s.get(TrainingJob, job_id)
                if not job:
                    yield f"event: error\ndata: {json.dumps({'msg': 'job not found'})}\n\n"
                    return
                snapshot = {
                    "status": job.status,
                    "progress": job.progress,
                    "current_step": job.current_step,
                    "total_step": job.total_step,
                    "latest_loss": job.latest_loss,
                    "has_checkpoint": job_manager.has_checkpoint(job_id),
                }
                status = job.status
            yield f"event: progress\ndata: {json.dumps(snapshot)}\n\n"

            # incremental log
            if log_file.exists():
                size = log_file.stat().st_size
                if size > last_size:
                    with log_file.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_size)
                        chunk = f.read()
                    last_size = size
                    for line in chunk.splitlines():
                        yield f"event: log\ndata: {json.dumps({'line': line})}\n\n"

            if status in ("succeeded", "failed", "stopped", "paused"):
                yield f"event: done\ndata: {json.dumps({'status': status})}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
