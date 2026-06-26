"""Parse kohya/sd-scripts stdout to extract step / epoch / loss progress."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# tqdm-style: "  35%|███   | 560/1600 [..."
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
# "epoch 3/8"
_EPOCH_RE = re.compile(r"epoch[:\s]+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
# "loss=0.0834" or "avr_loss=0.0834" or "loss: 0.0834"
_LOSS_RE = re.compile(r"(?:avr_)?loss[=:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE)


@dataclass
class ProgressUpdate:
    current_step: Optional[int] = None
    total_step: Optional[int] = None
    epoch: Optional[int] = None
    total_epoch: Optional[int] = None
    loss: Optional[float] = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (
                self.current_step,
                self.total_step,
                self.epoch,
                self.total_epoch,
                self.loss,
            )
        )


def parse_line(line: str) -> ProgressUpdate:
    upd = ProgressUpdate()

    # Only the kohya training loop prefixes its tqdm bar with "steps:".
    # Other bars (e.g. "caching latents ... 0/18") must NOT drive job progress,
    # otherwise the UI briefly shows the image-count denominator before training.
    is_training_bar = "steps:" in line.lower()
    if is_training_bar:
        m = _STEP_RE.search(line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            upd.current_step = cur
            upd.total_step = total

    m = _EPOCH_RE.search(line)
    if m:
        upd.epoch = int(m.group(1))
        upd.total_epoch = int(m.group(2))

    m = _LOSS_RE.search(line)
    if m:
        try:
            upd.loss = float(m.group(1))
        except ValueError:
            pass

    return upd
