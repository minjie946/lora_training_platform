"""Voice (SVC/RVC) job manager: lifecycle, subprocess supervision, log parsing.

Mirrors the image job_manager but for VoiceJob/VoiceModel and the RVC pipeline.
Reuses the same robust stop strategy (kill by PID process group) and startup
reconciliation so a backend reload doesn't leave jobs stuck "running".
"""
from __future__ import annotations

import json
import os
import re
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import VOICE_JOBS_DIR
from ..db import get_session
from ..models import VoiceDataset, VoiceJob, VoiceModel
from . import voice_config, voice_dataset_service
from .voice_backends.base import VoiceLaunchSpec
from .voice_backends.registry import get_backend


class _RunningJob:
    def __init__(self, proc, thread):
        self.proc = proc
        self.thread = thread


_running: dict[int, _RunningJob] = {}
_lock = threading.Lock()

# RVC training logs lines like: "Epoch: 12 ..." / "epoch 12" / "saving ckpt ... epoch12"
_EPOCH_RE = re.compile(r"[Ee]poch[:\s_]*([0-9]+)")


def job_dir(job_id: int) -> Path:
    return VOICE_JOBS_DIR / str(job_id)


def log_path(job_id: int) -> Path:
    return job_dir(job_id) / "train.log"


def exp_dir(job_id: int) -> Path:
    return job_dir(job_id) / "exp"


def weights_dir(job_id: int) -> Path:
    return job_dir(job_id) / "weights"


def _update_job(job_id: int, **fields) -> None:
    with get_session() as session:
        job = session.get(VoiceJob, job_id)
        if not job:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        session.add(job)
        session.commit()


def start_job(job_id: int) -> None:
    with _lock:
        if job_id in _running:
            raise ValueError("任务已在运行")

    with get_session() as session:
        job = session.get(VoiceJob, job_id)
        if not job:
            raise ValueError("任务不存在")
        dataset = session.get(VoiceDataset, job.dataset_id)
        if not dataset:
            raise ValueError("数据集不存在")
        backend = get_backend(job.backend)
        params = json.loads(job.params or "{}")
        # Sample rate must match the dataset's declared sr.
        params.setdefault("sample_rate", dataset.sample_rate)
        dataset_id = job.dataset_id
        clip_count = dataset.clip_count
        exp_name = _safe_name(dataset.speaker or f"voice_{job_id}")
        total = voice_config.estimate_total_steps(clip_count, params)

    trainset = voice_dataset_service.audio_dir(dataset_id)
    ed = exp_dir(job_id)
    wd = weights_dir(job_id)
    ed.mkdir(parents=True, exist_ok=True)
    wd.mkdir(parents=True, exist_ok=True)

    spec = VoiceLaunchSpec(
        job_id=job_id,
        exp_name=exp_name,
        trainset_dir=trainset,
        exp_dir=ed,
        weights_out_dir=wd,
        log_path=log_path(job_id),
        params=params,
    )

    proc = backend.start(spec)
    _update_job(
        job_id,
        status="running",
        pid=proc.pid,
        error=None,
        finished_at=None,
        progress=0.0,
        total_step=total,
    )

    thread = threading.Thread(target=_supervise, args=(job_id, proc, total), daemon=True)
    with _lock:
        _running[job_id] = _RunningJob(proc, thread)
    thread.start()


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s.strip()) or "voice"


def _supervise(job_id: int, proc, total_epoch: int) -> None:
    lp = log_path(job_id)
    lp.parent.mkdir(parents=True, exist_ok=True)
    pos = 0
    best_epoch = 0
    try:
        while True:
            alive = proc.poll() is None
            if lp.exists():
                with lp.open("r", encoding="utf-8", errors="replace") as logf:
                    logf.seek(pos)
                    chunk = logf.read()
                    pos = logf.tell()
                for raw in chunk.splitlines():
                    m = _EPOCH_RE.search(raw)
                    if m:
                        ep = int(m.group(1))
                        if ep > best_epoch:
                            best_epoch = ep
                            fields = {"current_step": ep}
                            if total_epoch:
                                fields["progress"] = round(min(ep / total_epoch, 1.0), 4)
                            _update_job(job_id, **fields)
            if not alive:
                break
            time.sleep(1.0)
    except Exception as e:  # noqa: BLE001
        _update_job(job_id, error=f"日志读取异常: {e}")

    code = proc.wait()
    _finalize(job_id, code)
    with _lock:
        _running.pop(job_id, None)


def _finalize(job_id: int, return_code: int) -> None:
    with get_session() as session:
        job = session.get(VoiceJob, job_id)
        if not job:
            return
        if job.status == "stopped":
            job.finished_at = datetime.utcnow()
            session.add(job)
            session.commit()
            return
        if return_code == 0:
            job.status = "succeeded"
            job.progress = 1.0
        else:
            job.status = "failed"
            if not job.error:
                job.error = f"训练进程退出码 {return_code}"
        job.finished_at = datetime.utcnow()
        session.add(job)
        session.commit()

    if return_code == 0:
        _register_models(job_id)


def _register_models(job_id: int) -> None:
    wd = weights_dir(job_id)
    if not wd.exists():
        return
    with get_session() as session:
        job = session.get(VoiceJob, job_id)
        dataset = session.get(VoiceDataset, job.dataset_id) if job else None
        speaker = dataset.speaker if dataset else ""
        sr = dataset.sample_rate if dataset else 40000
        index_file = next(iter(sorted(wd.glob("*.index"))), None)
        for f in sorted(wd.glob("*.pth")):
            already = session.exec(
                _select_voice_model(job_id, f.name)
            ).first()
            if already:
                continue
            session.add(
                VoiceModel(
                    job_id=job_id,
                    name=f.name,
                    speaker=speaker,
                    epoch=_epoch_from_name(f.name),
                    sample_rate=sr,
                    file_path=str(f),
                    index_path=str(index_file) if index_file else "",
                    file_size=f.stat().st_size,
                )
            )
        session.commit()


def _select_voice_model(job_id: int, name: str):
    from sqlmodel import select

    return select(VoiceModel).where(VoiceModel.job_id == job_id, VoiceModel.name == name)


def _epoch_from_name(name: str) -> int:
    m = re.search(r"[eE](\d+)", name) or re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _kill_pid_group(pid: int, timeout: float = 10.0) -> None:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    except Exception:  # noqa: BLE001
        pgid = pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:  # noqa: BLE001
        pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, Exception):  # noqa: BLE001
            return
        time.sleep(0.3)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


def stop_job(job_id: int) -> bool:
    with _lock:
        rj = _running.get(job_id)

    with get_session() as session:
        job = session.get(VoiceJob, job_id)
        if not job:
            return False
        backend_name = job.backend
        pid = job.pid
        status = job.status

    if rj:
        _update_job(job_id, status="stopped")
        get_backend(backend_name).stop(rj.proc)
        with _lock:
            _running.pop(job_id, None)
        return True

    if _pid_alive(pid):
        _update_job(job_id, status="stopped", finished_at=datetime.utcnow())
        _kill_pid_group(pid)  # type: ignore[arg-type]
        return True

    if status == "running":
        _update_job(
            job_id,
            status="stopped",
            finished_at=datetime.utcnow(),
            error="进程已不存在，已标记为停止",
        )
        return True

    return False


def read_log(job_id: int, tail: Optional[int] = None) -> str:
    lp = log_path(job_id)
    if not lp.exists():
        return ""
    text = lp.read_text(encoding="utf-8", errors="replace")
    if tail:
        return "\n".join(text.splitlines()[-tail:])
    return text


def delete_job(job_id: int) -> bool:
    import shutil

    if is_running(job_id):
        stop_job(job_id)

    from sqlmodel import select

    with get_session() as session:
        job = session.get(VoiceJob, job_id)
        if not job:
            return False
        models = session.exec(select(VoiceModel).where(VoiceModel.job_id == job_id)).all()
        for m in models:
            for path in (m.file_path, m.index_path):
                if path:
                    Path(path).unlink(missing_ok=True)
            session.delete(m)
        session.delete(job)
        session.commit()

    shutil.rmtree(job_dir(job_id), ignore_errors=True)
    return True


def is_running(job_id: int) -> bool:
    with _lock:
        if job_id in _running:
            return True
    with get_session() as session:
        job = session.get(VoiceJob, job_id)
        return bool(job and job.status == "running" and _pid_alive(job.pid))


def reconcile_on_startup() -> None:
    from sqlmodel import select

    with get_session() as session:
        rows = session.exec(select(VoiceJob).where(VoiceJob.status == "running")).all()
        running = [(j.id, j.pid) for j in rows]

    for job_id, pid in running:
        with _lock:
            if job_id in _running:
                continue
        if not _pid_alive(pid):
            _finalize(job_id, return_code=1)
            _update_job(
                job_id,
                error="后端重启时检测到训练进程已结束（异常退出或被系统终止）",
            )
