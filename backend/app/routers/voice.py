"""Voice (SVC/RVC) router: datasets, audio clips, jobs, produced voice models.

Self-contained under /api/voice so the image flow is untouched. Reuses the
robust voice_job_manager (start/stop/delete/reconcile) and the voice backend
registry (local RVC + remote-host RVC).
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select

from ..db import session_dependency
from ..models import VoiceDataset, VoiceJob, VoiceModel
from ..schemas import (
    AudioClip,
    VoiceDatasetCreate,
    VoiceDatasetRead,
    VoiceDatasetUpdate,
    VoiceJobCreate,
    VoiceJobRead,
    VoiceModelRead,
)
from ..services import voice_config, voice_dataset_service as vds, voice_job_manager as vjm
from ..services.voice_backends.registry import get_backend, list_backends

router = APIRouter(prefix="/api/voice", tags=["voice"])


# ---- helpers ----
def _ds_or_404(session: Session, dataset_id: int) -> VoiceDataset:
    obj = session.get(VoiceDataset, dataset_id)
    if not obj:
        raise HTTPException(404, "数据集不存在")
    return obj


def _job_to_read(job: VoiceJob) -> VoiceJobRead:
    return VoiceJobRead(
        id=job.id,
        name=job.name,
        dataset_id=job.dataset_id,
        backend=job.backend,
        params=json.loads(job.params or "{}"),
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        total_step=job.total_step,
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )


def _refresh_counts(session: Session, ds: VoiceDataset) -> None:
    ds.clip_count = vds.count_clips(ds.id)
    ds.total_seconds = vds.total_seconds(ds.id)
    if ds.clip_count > 0:
        ds.status = "ready"
    session.add(ds)
    session.commit()


# ---- meta ----
@router.get("/backends")
def get_backends():
    return list_backends()


@router.get("/defaults")
def get_defaults():
    return voice_config.DEFAULT_PARAMS


# ---- datasets ----
@router.get("/datasets", response_model=list[VoiceDatasetRead])
def list_datasets(session: Session = Depends(session_dependency)):
    return session.exec(select(VoiceDataset).order_by(VoiceDataset.id.desc())).all()


@router.post("/datasets", response_model=VoiceDatasetRead)
def create_dataset(body: VoiceDatasetCreate, session: Session = Depends(session_dependency)):
    obj = VoiceDataset(name=body.name, speaker=body.speaker, sample_rate=body.sample_rate)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    vds.ensure_audio_dir(obj.id)
    return obj


@router.get("/datasets/{dataset_id}", response_model=VoiceDatasetRead)
def get_dataset(dataset_id: int, session: Session = Depends(session_dependency)):
    return _ds_or_404(session, dataset_id)


@router.patch("/datasets/{dataset_id}", response_model=VoiceDatasetRead)
def update_dataset(
    dataset_id: int, body: VoiceDatasetUpdate, session: Session = Depends(session_dependency)
):
    obj = _ds_or_404(session, dataset_id)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(obj, k, v)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return obj


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(session_dependency)):
    obj = _ds_or_404(session, dataset_id)
    session.delete(obj)
    session.commit()
    vds.delete_dataset_files(dataset_id)
    return {"ok": True}


# ---- clips ----
@router.get("/datasets/{dataset_id}/clips", response_model=list[AudioClip])
def list_clips(dataset_id: int, session: Session = Depends(session_dependency)):
    _ds_or_404(session, dataset_id)
    return vds.list_clips(dataset_id)


@router.post("/datasets/{dataset_id}/clips", response_model=list[AudioClip])
async def upload_clips(
    dataset_id: int,
    files: list[UploadFile] = File(...),
    session: Session = Depends(session_dependency),
):
    obj = _ds_or_404(session, dataset_id)
    for f in files:
        data = await f.read()
        try:
            vds.save_clip(dataset_id, f.filename, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
    _refresh_counts(session, obj)
    return vds.list_clips(dataset_id)


@router.get("/datasets/{dataset_id}/clips/{filename}/raw")
def get_raw_clip(dataset_id: int, filename: str):
    p = vds.get_clip_path(dataset_id, filename)
    if not p:
        raise HTTPException(404, "音频不存在")
    return FileResponse(p)


@router.delete("/datasets/{dataset_id}/clips/{filename}")
def delete_clip(dataset_id: int, filename: str, session: Session = Depends(session_dependency)):
    obj = _ds_or_404(session, dataset_id)
    if not vds.delete_clip(dataset_id, filename):
        raise HTTPException(404, "音频不存在")
    _refresh_counts(session, obj)
    return {"ok": True}


# ---- jobs ----
@router.get("/jobs", response_model=list[VoiceJobRead])
def list_jobs(session: Session = Depends(session_dependency)):
    jobs = session.exec(select(VoiceJob).order_by(VoiceJob.id.desc())).all()
    return [_job_to_read(j) for j in jobs]


@router.post("/jobs", response_model=VoiceJobRead)
def create_job(body: VoiceJobCreate, session: Session = Depends(session_dependency)):
    ds = session.get(VoiceDataset, body.dataset_id)
    if not ds:
        raise HTTPException(404, "数据集不存在")
    try:
        get_backend(body.backend)
    except ValueError as e:
        raise HTTPException(400, str(e))
    job = VoiceJob(
        name=body.name,
        dataset_id=body.dataset_id,
        backend=body.backend,
        params=json.dumps(body.params or {}),
        total_step=voice_config.estimate_total_steps(ds.clip_count, body.params or {}),
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return _job_to_read(job)


@router.get("/jobs/{job_id}", response_model=VoiceJobRead)
def get_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(VoiceJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return _job_to_read(job)


@router.post("/jobs/{job_id}/start", response_model=VoiceJobRead)
def start_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(VoiceJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if job.status == "running":
        raise HTTPException(400, "任务已在运行")
    backend = get_backend(job.backend)
    pf = backend.preflight()
    if not pf.ok:
        raise HTTPException(400, f"环境检查未通过: {pf.detail}")
    try:
        vjm.start_job(job_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"启动失败: {e}")
    session.refresh(job)
    return _job_to_read(job)


@router.post("/jobs/{job_id}/stop", response_model=VoiceJobRead)
def stop_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(VoiceJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    if not vjm.stop_job(job_id):
        raise HTTPException(400, "任务未在运行")
    session.refresh(job)
    return _job_to_read(job)


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int, session: Session = Depends(session_dependency)):
    job = session.get(VoiceJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    vjm.delete_job(job_id)
    return {"ok": True}


@router.get("/jobs/{job_id}/log")
def get_log(job_id: int, tail: int = 200, session: Session = Depends(session_dependency)):
    job = session.get(VoiceJob, job_id)
    if not job:
        raise HTTPException(404, "任务不存在")
    return {"log": vjm.read_log(job_id, tail=tail)}


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: int):
    async def event_gen():
        log_file = vjm.log_path(job_id)
        last_size = 0
        while True:
            with vjm.get_session() as s:
                job = s.get(VoiceJob, job_id)
                if not job:
                    yield f"event: error\ndata: {json.dumps({'msg': 'job not found'})}\n\n"
                    return
                snapshot = {
                    "status": job.status,
                    "progress": job.progress,
                    "current_step": job.current_step,
                    "total_step": job.total_step,
                }
                status = job.status
            yield f"event: progress\ndata: {json.dumps(snapshot)}\n\n"

            if log_file.exists():
                size = log_file.stat().st_size
                if size > last_size:
                    with log_file.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(last_size)
                        chunk = f.read()
                    last_size = size
                    for line in chunk.splitlines():
                        yield f"event: log\ndata: {json.dumps({'line': line})}\n\n"

            if status in ("succeeded", "failed", "stopped"):
                yield f"event: done\ndata: {json.dumps({'status': status})}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---- models ----
@router.get("/models", response_model=list[VoiceModelRead])
def list_models(job_id: int | None = None, session: Session = Depends(session_dependency)):
    stmt = select(VoiceModel).order_by(VoiceModel.id.desc())
    if job_id is not None:
        stmt = stmt.where(VoiceModel.job_id == job_id)
    rows = session.exec(stmt).all()
    return [
        VoiceModelRead(
            id=m.id,
            job_id=m.job_id,
            name=m.name,
            speaker=m.speaker,
            epoch=m.epoch,
            sample_rate=m.sample_rate,
            has_index=bool(m.index_path),
            file_size=m.file_size,
            created_at=m.created_at,
        )
        for m in rows
    ]


@router.get("/models/{model_id}/download")
def download_model(model_id: int, session: Session = Depends(session_dependency)):
    from pathlib import Path

    m = session.get(VoiceModel, model_id)
    if not m:
        raise HTTPException(404, "模型不存在")
    p = Path(m.file_path)
    if not p.exists():
        raise HTTPException(404, "模型文件已丢失")
    return FileResponse(p, filename=m.name, media_type="application/octet-stream")


@router.delete("/models/{model_id}")
def delete_model(model_id: int, session: Session = Depends(session_dependency)):
    from pathlib import Path

    m = session.get(VoiceModel, model_id)
    if not m:
        raise HTTPException(404, "模型不存在")
    for path in (m.file_path, m.index_path):
        if path:
            Path(path).unlink(missing_ok=True)
    session.delete(m)
    session.commit()
    return {"ok": True}
