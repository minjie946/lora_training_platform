"""Voice (SVC/RVC) training backends.

Mirrors the image-side backend contract but for the RVC pipeline. A backend
receives a VoiceLaunchSpec and returns a Popen-like handle that the voice job
manager supervises (poll/wait), reusing the same process-group kill strategy.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VoicePreflight:
    ok: bool
    detail: str


@dataclass
class VoiceLaunchSpec:
    """Everything a voice backend needs to start an RVC training run."""

    job_id: int
    exp_name: str            # RVC experiment name (== speaker identity)
    trainset_dir: Path       # local dir of raw audio clips
    exp_dir: Path            # RVC experiment/work dir for this job
    weights_out_dir: Path    # where produced .pth/.index land (local)
    log_path: Path
    params: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


class VoiceBackend(ABC):
    name: str = "base"
    label: str = "Base"

    @abstractmethod
    def preflight(self) -> VoicePreflight:
        ...

    @abstractmethod
    def start(self, spec: VoiceLaunchSpec):
        ...

    def stop(self, proc) -> None:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
