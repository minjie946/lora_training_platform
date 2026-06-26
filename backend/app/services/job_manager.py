"""Job manager: lifecycle, subprocess supervision, log persistence, progress parsing."""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import DEFAULT_BASE_MODEL, JOBS_DIR, MODELS_BASE_DIR
from ..db import get_session
from ..models import Dataset, LoraModel, TrainingJob
from ..utils.log_parser import parse_line
from . import config_builder, dataset_service
from .backends.registry import get_backend


class _RunningJob:
    def __init__(self, proc, thread):
        self.proc = proc
        self.thread = thread


# In-process registry of currently running jobs (single-machine assumption).
_running: dict[int, _RunningJob] = {}
_lock = threading.Lock()


def job_dir(job_id: int) -> Path:
    return JOBS_DIR / str(job_id)


def log_path(job_id: int) -> Path:
    return job_dir(job_id) / "train.log"


def config_path(job_id: int) -> Path:
    return job_dir(job_id) / "config.toml"


def output_dir(job_id: int) -> Path:
    return job_dir(job_id) / "output"


def _is_sdxl_model(base_model_name: str) -> bool:
    """Heuristic: SDXL checkpoints carry 'xl' in the filename (e.g. animagine-xl)."""
    return "xl" in base_model_name.lower()


def _update_job(job_id: int, **fields) -> None:
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        session.add(job)
        session.commit()


def prepare_job(job_id: int) -> None:
    """Build config.toml for a pending job. Raises on invalid setup."""
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise ValueError("任务不存在")
        dataset = session.get(Dataset, job.dataset_id)
        if not dataset:
            raise ValueError("数据集不存在")
        params = json.loads(job.params or "{}")
        # Base model priority: explicit job override > dataset's choice > default.
        base_model_name = job.base_model or dataset.base_model or DEFAULT_BASE_MODEL
        base_model_path = MODELS_BASE_DIR / base_model_name

        train_data_dir = dataset_service.dataset_root(dataset.id)
        out_dir = output_dir(job_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = config_builder.build_config(
            base_model_path=base_model_path,
            train_data_dir=train_data_dir,
            output_dir=out_dir,
            output_name=f"{dataset.concept}_{job_id}",
            params=params,
            is_sdxl=_is_sdxl_model(base_model_name),
        )
        config_builder.write_config(cfg, config_path(job_id))

        total = config_builder.estimate_total_steps(
            dataset.image_count, dataset.repeat, params
        )
        job.total_step = total
        # Persist the resolved base model so detail/list views can show it.
        job.base_model = base_model_name
        session.add(job)
        session.commit()


def start_job(job_id: int, resume: bool = False) -> None:
    """Launch a prepared job asynchronously.

    When ``resume`` is True, continue from the latest saved kohya ``-state``
    checkpoint (optimizer/scheduler/step) instead of starting from scratch.
    """
    with _lock:
        if job_id in _running:
            raise ValueError("任务已在运行")

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            raise ValueError("任务不存在")
        backend = get_backend(job.backend)

    prepare_job(job_id)

    # prepare_job resolves & persists the effective base model; read it back.
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        base_model_name = (job.base_model if job else "") or DEFAULT_BASE_MODEL
        job_params = json.loads(job.params or "{}") if job else {}

    from .backends.base import LaunchSpec
    from .backends.resource import mps_env

    resume_dir = _latest_state_dir(job_id) if resume else None

    spec = LaunchSpec(
        job_id=job_id,
        config_path=config_path(job_id),
        output_dir=output_dir(job_id),
        log_path=log_path(job_id),
        is_sdxl=_is_sdxl_model(base_model_name),
        resume_state_dir=resume_dir,
        env=mps_env(job_params.get("resource_tier")),
    )

    proc = backend.start(spec)
    # On resume keep the existing progress; a fresh run starts at 0.
    fields = dict(status="running", pid=proc.pid, error=None, finished_at=None)
    if not resume:
        fields["progress"] = 0.0
    _update_job(job_id, **fields)

    thread = threading.Thread(target=_supervise, args=(job_id, proc), daemon=True)
    with _lock:
        _running[job_id] = _RunningJob(proc, thread)
    thread.start()


def _latest_state_dir(job_id: int) -> Optional[Path]:
    """Find the most recent kohya `-state` checkpoint dir for a job, if any.

    kohya writes `<output_name>-state` (and `-stepNNN-state`) into output_dir
    when save_state is on. Pick the newest one by mtime.
    """
    out = output_dir(job_id)
    if not out.exists():
        return None
    states = [p for p in out.glob("*-state") if p.is_dir()]
    if not states:
        return None
    return max(states, key=lambda p: p.stat().st_mtime)


def _supervise(job_id: int, proc) -> None:
    """Tail the job's log file, parse progress, finalize on process exit.

    The training process writes its own stdout/stderr to the log file (so it can
    survive a backend reload), so here we only read that file to update progress.
    """
    lp = log_path(job_id)
    lp.parent.mkdir(parents=True, exist_ok=True)
    best_total = 0
    pos = 0
    try:
        while True:
            alive = proc.poll() is None
            if lp.exists():
                with lp.open("r", encoding="utf-8", errors="replace") as logf:
                    logf.seek(pos)
                    chunk = logf.read()
                    pos = logf.tell()
                for raw in chunk.splitlines():
                    upd = parse_line(raw)
                    if upd.is_empty():
                        continue
                    fields: dict = {}
                    if upd.total_step and upd.total_step > best_total:
                        best_total = upd.total_step
                    if upd.current_step is not None and best_total:
                        fields["current_step"] = upd.current_step
                        fields["total_step"] = best_total
                        fields["progress"] = round(min(upd.current_step / best_total, 1.0), 4)
                    if upd.loss is not None:
                        fields["latest_loss"] = upd.loss
                    if fields:
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
        job = session.get(TrainingJob, job_id)
        if not job:
            return
        # A paused/stopped job was intentionally killed — don't mark it
        # failed just because the process exited non-zero.
        if job.status in ("stopped", "paused"):
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
    out = output_dir(job_id)
    if not out.exists():
        return
    from sqlmodel import select

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        base_model_name = (job.base_model if job else "") or DEFAULT_BASE_MODEL
        for f in sorted(out.glob("*.safetensors")):
            already = session.exec(
                select(LoraModel).where(
                    LoraModel.job_id == job_id, LoraModel.name == f.name
                )
            ).first()
            if already:
                continue
            epoch = _epoch_from_name(f.name)
            session.add(
                LoraModel(
                    job_id=job_id,
                    name=f.name,
                    epoch=epoch,
                    base_model=base_model_name,
                    file_path=str(f),
                    file_size=f.stat().st_size,
                )
            )
        session.commit()


def _epoch_from_name(name: str) -> int:
    import re

    m = re.search(r"-(\d+)\.safetensors$", name)
    return int(m.group(1)) if m else 0


def _pid_alive(pid: Optional[int]) -> bool:
    """True if a PID is alive (signal 0 probe)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours to signal
    except Exception:  # noqa: BLE001
        return False


def _kill_pid_group(pid: int, timeout: float = 10.0) -> None:
    """Kill a process group by PID (training runs in its own session)."""
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
    """Stop a running job.

    Works even after a backend reload wiped the in-memory registry, by falling
    back to the PID persisted in the database and killing its process group.
    Also resolves a stale "running" row whose process is already gone.
    """
    with _lock:
        rj = _running.get(job_id)

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return False
        backend_name = job.backend
        pid = job.pid
        status = job.status

    # Case 1: we still own the process handle in this worker.
    if rj:
        _update_job(job_id, status="stopped")
        get_backend(backend_name).stop(rj.proc)
        with _lock:
            _running.pop(job_id, None)
        return True

    # Case 2: process survived but registry was lost (reload) — kill by PID.
    if _pid_alive(pid):
        _update_job(job_id, status="stopped", finished_at=datetime.utcnow())
        _kill_pid_group(pid)  # type: ignore[arg-type]
        return True

    # Case 3: row says running but the process is already gone — reconcile.
    if status == "running":
        _update_job(
            job_id,
            status="stopped",
            finished_at=datetime.utcnow(),
            error=job_error_if_unset(job_id, "进程已不存在，已标记为停止"),
        )
        return True

    return False


def pause_job(job_id: int) -> bool:
    """Pause a running job: kill the process but mark it 'paused' so it can be
    resumed later from the latest saved checkpoint. Requires save_state, which
    is enabled by default in the generated config.
    """
    with _lock:
        rj = _running.get(job_id)

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return False
        backend_name = job.backend
        pid = job.pid
        status = job.status

    if status not in ("running",):
        return False

    # Mark paused BEFORE killing so _supervise/_finalize won't flip it to failed.
    if rj:
        _update_job(job_id, status="paused")
        get_backend(backend_name).stop(rj.proc)
        with _lock:
            _running.pop(job_id, None)
        return True

    if _pid_alive(pid):
        _update_job(job_id, status="paused", finished_at=None)
        _kill_pid_group(pid)  # type: ignore[arg-type]
        return True

    # Process already gone — still allow resume from whatever checkpoint exists.
    _update_job(job_id, status="paused")
    return True


def resume_job(job_id: int) -> bool:
    """Resume a paused job from its latest checkpoint.

    Raises if there is no saved checkpoint yet: kohya only writes a `-state`
    dir at the END of each epoch, so a job paused before its first epoch
    completed has nothing to resume from. In that case the caller should offer
    to restart from scratch instead of silently doing so.
    """
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return False
        if job.status not in ("paused",):
            raise ValueError("只有已暂停的任务可以继续")

    if _latest_state_dir(job_id) is None:
        raise ValueError(
            "尚无可续训的检查点：需完成至少一个 epoch 才会生成断点。"
            "该任务暂停时还没跑完首个 epoch，无法从中途恢复。"
        )
    start_job(job_id, resume=True)
    return True


def has_checkpoint(job_id: int) -> bool:
    """True if a resumable kohya `-state` checkpoint exists for this job."""
    return _latest_state_dir(job_id) is not None


def job_error_if_unset(job_id: int, msg: str) -> Optional[str]:
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if job and job.error:
            return job.error
    return msg


def is_running(job_id: int) -> bool:
    with _lock:
        if job_id in _running:
            return True
    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        return bool(job and job.status == "running" and _pid_alive(job.pid))


def reconcile_on_startup() -> None:
    """On backend start, fix jobs whose 'running' row no longer has a live process.

    A training subprocess launched in its own session keeps running across a
    backend reload, but the supervisor thread is gone. If the process is dead we
    mark the job failed/stopped; if it's still alive we re-attach a supervisor so
    progress keeps updating and the job finalizes correctly.
    """
    from sqlmodel import select

    with get_session() as session:
        rows = session.exec(
            select(TrainingJob).where(TrainingJob.status == "running")
        ).all()
        running = [(j.id, j.pid) for j in rows]

    for job_id, pid in running:
        with _lock:
            if job_id in _running:
                continue
        if _pid_alive(pid):
            _reattach(job_id, pid)  # type: ignore[arg-type]
        else:
            _finalize(job_id, return_code=1)
            _update_job(
                job_id,
                error="后端重启时检测到训练进程已结束（异常退出或被系统终止）",
            )


def _reattach(job_id: int, pid: int) -> None:
    """Re-attach a supervisor to a still-alive training process after reload."""
    proc = _ExternalProc(pid)
    thread = threading.Thread(target=_supervise, args=(job_id, proc), daemon=True)
    with _lock:
        _running[job_id] = _RunningJob(proc, thread)
    thread.start()


class _ExternalProc:
    """Minimal Popen-like wrapper around a PID we don't directly own.

    Lets _supervise poll/wait on a process inherited across a backend reload.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self._returncode: Optional[int] = None

    def poll(self) -> Optional[int]:
        if self._returncode is not None:
            return self._returncode
        if not _pid_alive(self.pid):
            self._returncode = 0  # exit code unknown; assume clean unless caller overrides
            return self._returncode
        return None

    def wait(self) -> int:
        while self.poll() is None:
            time.sleep(0.5)
        return self._returncode or 0


def read_log(job_id: int, tail: Optional[int] = None) -> str:
    lp = log_path(job_id)
    if not lp.exists():
        return ""
    text = lp.read_text(encoding="utf-8", errors="replace")
    if tail:
        return "\n".join(text.splitlines()[-tail:])
    return text


def delete_job(job_id: int) -> bool:
    """Delete a job: stop it if running, then remove its models, files and row."""
    import shutil

    from sqlmodel import select

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if not job:
            return False

    # Stop a live run first (kills the process group); ignore if already gone.
    if is_running(job_id):
        stop_job(job_id)

    with _lock:
        _running.pop(job_id, None)

    # Remove produced model files + rows.
    with get_session() as session:
        models = session.exec(
            select(LoraModel).where(LoraModel.job_id == job_id)
        ).all()
        for m in models:
            try:
                Path(m.file_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            session.delete(m)
        session.commit()

    # Remove the whole job workspace (config.toml, logs, output dir).
    try:
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass

    with get_session() as session:
        job = session.get(TrainingJob, job_id)
        if job:
            session.delete(job)
            session.commit()
    return True


def reset_job_artifacts(job_id: int) -> None:
    """Wipe a job's produced models, logs and workspace, keeping the row.

    Used when a job's training inputs change on edit: the previous run's
    outputs no longer match the new config, so start clean. The job row itself
    is preserved (the caller resets its status to pending).
    """
    import shutil

    from sqlmodel import select

    # Remove produced model files + rows for this job.
    with get_session() as session:
        models = session.exec(
            select(LoraModel).where(LoraModel.job_id == job_id)
        ).all()
        for m in models:
            try:
                Path(m.file_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            session.delete(m)
        session.commit()

    # Remove the job workspace (config.toml, logs, output dir with checkpoints).
    try:
        shutil.rmtree(job_dir(job_id), ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass
