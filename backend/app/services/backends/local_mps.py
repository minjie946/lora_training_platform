"""Local Mac MPS backend: runs kohya_ss train_network.py via the project venv."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import toml

from ...config import DEFAULT_BASE_MODEL, KOHYA_DIR, MODELS_BASE_DIR
from .base import LaunchSpec, PreflightResult, TrainingBackend


def _terminate_process_tree(pid: int, timeout: float = 10.0) -> None:
    """Stop a training run by killing its whole process group.

    The training process is launched with start_new_session=True, so its PGID
    equals its PID. Killing the group also stops the accelerate-spawned worker
    that actually holds the GPU. Escalates SIGTERM -> SIGKILL.
    """
    if not pid:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return  # already gone
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
            os.killpg(pgid, 0)  # probe: raises when the group is gone
        except ProcessLookupError:
            return
        except Exception:  # noqa: BLE001
            return
        time.sleep(0.3)

    # Still alive after grace period — force kill.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


def _supported_mixed_precision() -> str:
    """Return the best mixed_precision this Mac actually supports.

    accelerate raises if fp16 is used on MPS, and bf16 only works when
    is_bf16_available(True). Fall back to "no" so training can start.
    """
    try:
        import torch  # type: ignore
        from accelerate.utils import is_bf16_available  # type: ignore

        if torch.backends.mps.is_available():
            return "bf16" if is_bf16_available(True) else "no"
    except Exception:  # noqa: BLE001
        pass
    return "no"


def _sanitize_config_for_mps(config_path: Path) -> None:
    """Downgrade an unsupported mixed_precision in config.toml before launch."""
    if not config_path.exists():
        return
    cfg = toml.load(config_path)
    training = cfg.get("training")
    if not isinstance(training, dict):
        return
    requested = str(training.get("mixed_precision", "no"))
    if requested in ("fp16", "bf16"):
        allowed = _supported_mixed_precision()
        if allowed != requested:
            training["mixed_precision"] = allowed
            with config_path.open("w", encoding="utf-8") as f:
                toml.dump(cfg, f)


class LocalMpsBackend(TrainingBackend):
    name = "local_mps"
    label = "本地 Mac (MPS)"

    def preflight(self) -> PreflightResult:
        problems: list[str] = []

        train_script = KOHYA_DIR / "train_network.py"
        if not train_script.exists():
            problems.append(f"未找到 kohya train_network.py（{train_script}）")

        try:
            import torch  # type: ignore

            if not torch.backends.mps.is_available():
                problems.append("MPS 不可用（torch.backends.mps.is_available()=False）")
        except Exception:  # noqa: BLE001
            problems.append("torch 未安装，无法在 MPS 上训练")

        model = MODELS_BASE_DIR / DEFAULT_BASE_MODEL
        if not model.exists():
            problems.append(f"未找到底模（{model}）")

        if problems:
            return PreflightResult(ok=False, detail="；".join(problems))
        return PreflightResult(ok=True, detail="本地 MPS 环境就绪")

    def start(self, spec: LaunchSpec) -> subprocess.Popen:
        script_name = "sdxl_train_network.py" if spec.is_sdxl else "train_network.py"
        train_script = KOHYA_DIR / script_name
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env.update(spec.env)

        # MPS can't use fp16 (and often not bf16) — downgrade before launch.
        _sanitize_config_for_mps(spec.config_path)

        # CPU thread budget comes from the resource tier (LORA_CPU_THREADS),
        # defaulting to 1 to keep the machine responsive. Also bound the math
        # libs so a low tier really stays light.
        cpu_threads = env.get("LORA_CPU_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", cpu_threads)

        # Use the current interpreter (project venv) to run accelerate.
        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_cpu_threads_per_process",
            cpu_threads,
            str(train_script),
            "--config_file",
            str(spec.config_path),
        ]

        # Resume from a saved checkpoint (optimizer/scheduler/step) if requested.
        if spec.resume_state_dir and Path(spec.resume_state_dir).exists():
            cmd += ["--resume", str(spec.resume_state_dir)]

        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Write stdout/stderr to the log file instead of a PIPE so the training
        # process survives a backend reload/restart (a dead PIPE reader would
        # otherwise kill it with SIGPIPE). start_new_session=True puts it in its
        # own process group so we can later kill the whole tree by PGID.
        logf = open(spec.log_path, "w", encoding="utf-8", buffering=1)
        return subprocess.Popen(
            cmd,
            cwd=str(KOHYA_DIR),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    def stop(self, proc: subprocess.Popen) -> None:
        _terminate_process_tree(proc.pid)
