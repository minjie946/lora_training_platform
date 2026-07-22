"""Resource-usage tiers for local training.

Maps a coarse tier chosen in the UI (low / balanced / full) to the concrete
knobs that actually reduce how much of the Mac a training run occupies:

- ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` caps how much unified memory the MPS
  allocator may hold. PyTorch also has a *low* watermark (default 1.4) and
  requires ``low <= high``; if we only lower ``high`` the allocator raises
  ``invalid low watermark ratio``. So whenever we cap ``high`` we must also
  pin ``low`` at or below it. The "full" tier leaves both at PyTorch defaults.
- ``LORA_CPU_THREADS`` is read by the local backend to set accelerate's
  ``--num_cpu_threads_per_process`` (and OMP threads), limiting CPU pressure.

Kept backend-agnostic so remote/CUDA backends can add their own mapping later.
"""
from __future__ import annotations

RESOURCE_TIERS = ("low", "balanced", "full")
DEFAULT_TIER = "balanced"


def normalize_tier(tier: str | None) -> str:
    return tier if tier in RESOURCE_TIERS else DEFAULT_TIER


def mps_env(tier: str | None) -> dict[str, str]:
    """Return env overrides for a local-MPS run at the given tier."""
    tier = normalize_tier(tier)
    if tier == "low":
        # Cap MPS at ~half of unified memory, single CPU thread — keeps the
        # machine responsive at the cost of training speed.
        return {
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.5",
            "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.4",
            "LORA_CPU_THREADS": "1",
        }
    if tier == "full":
        # Leave both watermarks at PyTorch defaults (no cap) and allow a couple
        # of CPU threads for max speed.
        return {"LORA_CPU_THREADS": "2"}
    # balanced (default): moderate cap, low kept below high to satisfy PyTorch.
    return {
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.7",
        "LORA_CPU_THREADS": "1",
    }
