"""TrainingBackend abstraction — the compute Adapter extension point.

Only LocalMpsBackend is implemented this round. Future backends
(local_cuda / remote_server / cloud_gpu / external_api) implement the same
contract and register in registry.py, keeping the platform layer decoupled
from the training script layer.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PreflightResult:
    ok: bool
    detail: str


@dataclass
class LaunchSpec:
    """Everything a backend needs to start a training run."""

    job_id: int
    config_path: Path
    output_dir: Path
    log_path: Path
    is_sdxl: bool = False
    # When set, resume training from this kohya `-state` directory (checkpoint).
    resume_state_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)


class TrainingBackend(ABC):
    name: str = "base"
    label: str = "Base"

    @abstractmethod
    def preflight(self) -> PreflightResult:
        """Check device / base model / dependencies for this backend."""

    @abstractmethod
    def start(self, spec: LaunchSpec) -> subprocess.Popen:
        """Launch training, return the process handle (stdout piped)."""

    def stop(self, proc: subprocess.Popen) -> None:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
