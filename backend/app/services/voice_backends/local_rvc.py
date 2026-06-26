"""Local Mac (MPS/CPU) RVC backend.

Generates the per-job RVC pipeline script and runs it via RVC's own virtualenv
(RVC has heavy deps that conflict with the backend/kohya env, so it's isolated).
RVC's training largely targets CUDA; on Mac it falls back to CPU/MPS and is
slow, but the platform lets the user try it (and otherwise pick a remote host).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ...config import RVC_DIR, RVC_PYTHON
from .. import voice_config
from .base import VoiceBackend, VoiceLaunchSpec, VoicePreflight


def _rvc_python() -> str:
    """RVC's isolated interpreter, falling back to the current one if absent."""
    return str(RVC_PYTHON) if Path(RVC_PYTHON).exists() else sys.executable


def _terminate_process_tree(pid: int, timeout: float = 10.0) -> None:
    if not pid:
        return
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
        except ProcessLookupError:
            return
        except Exception:  # noqa: BLE001
            return
        time.sleep(0.3)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


def _local_device() -> str:
    try:
        import torch  # type: ignore

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class LocalRvcBackend(VoiceBackend):
    name = "local_rvc"
    label = "本地 Mac (RVC)"

    def preflight(self) -> VoicePreflight:
        problems: list[str] = []
        train_script = RVC_DIR / "infer" / "modules" / "train" / "train.py"
        if not train_script.exists():
            problems.append(
                f"未找到 RVC 训练脚本（{train_script}）。请先克隆 RVC（./start.sh 会自动处理）或设置 RVC_DIR"
            )
        if not Path(RVC_PYTHON).exists():
            problems.append(
                f"未找到 RVC 独立虚拟环境（{RVC_PYTHON}）。请运行 ./start.sh 自动创建并安装 RVC 依赖"
            )
        if problems:
            return VoicePreflight(ok=False, detail="；".join(problems))
        return VoicePreflight(ok=True, detail=f"本地 RVC 就绪（设备：{_local_device()}）")

    def start(self, spec: VoiceLaunchSpec) -> subprocess.Popen:
        script = voice_config.build_pipeline_script(
            rvc_dir=str(RVC_DIR),
            python_cmd=_rvc_python(),
            exp_name=spec.exp_name,
            trainset_dir=str(spec.trainset_dir),
            exp_dir=str(spec.exp_dir),
            weights_out_dir=str(spec.weights_out_dir),
            params=spec.params,
            device=_local_device(),
        )
        script_path = spec.exp_dir / "pipeline.sh"
        voice_config.write_script(script, script_path)

        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["EXP"] = str(spec.exp_dir)
        env.update(spec.env)

        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(spec.log_path, "w", encoding="utf-8", buffering=1)
        return subprocess.Popen(
            ["bash", str(script_path)],
            cwd=str(RVC_DIR),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def stop(self, proc: subprocess.Popen) -> None:
        _terminate_process_tree(proc.pid)
