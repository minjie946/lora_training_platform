"""Build kohya_ss config.toml from platform job parameters.

Defaults follow the Feishu guide (Mac M4 Pro + Animagine XL 4.0 Opt, stage 4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

# Default training parameters aligned with the guide (stage 4.1 / 4.2).
DEFAULT_PARAMS: dict[str, Any] = {
    "resolution": 768,            # 24G safe default; 48G can use 1024
    "network_dim": 16,            # rank
    "network_alpha": 8,
    "max_train_epochs": 8,
    "train_batch_size": 1,
    "unet_lr": 1e-4,
    "text_encoder_lr": 5e-5,
    "optimizer_type": "AdamW",    # Mac: NOT AdamW8bit
    "lr_scheduler": "cosine_with_restarts",
    "lr_warmup_steps": 100,
    "mixed_precision": "no",       # Mac MPS: accelerate rejects fp16/bf16 on this device
    "gradient_checkpointing": True,
    "save_every_n_epochs": 1,
    # Also snapshot a resumable checkpoint every N steps so a mid-epoch pause
    # can resume without losing the whole epoch. 0 disables step-wise saving
    # (only epoch-end checkpoints). Each snapshot also writes optimizer state
    # (~hundreds of MB), so keep it moderate.
    "save_every_n_steps": 200,
    "seed": 42,
    "min_bucket_reso": 768,
    "max_bucket_reso": 1536,
    "enable_bucket": True,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
}


def merge_params(user_params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_PARAMS)
    merged.update({k: v for k, v in (user_params or {}).items() if v is not None})
    return merged


def estimate_total_steps(image_count: int, repeat: int, params: dict[str, Any]) -> int:
    epochs = int(params.get("max_train_epochs", DEFAULT_PARAMS["max_train_epochs"]))
    batch = max(1, int(params.get("train_batch_size", 1)))
    return image_count * repeat * epochs // batch


def build_config(
    *,
    base_model_path: Path,
    train_data_dir: Path,
    output_dir: Path,
    output_name: str,
    params: dict[str, Any],
    is_sdxl: bool = False,
) -> dict[str, Any]:
    p = merge_params(params)
    # SDXL forbids caching Text Encoder outputs while also training the TE
    # network (sdxl_train_network.py asserts network_train_unet_only).
    # Our guide trains the TE (text_encoder_lr > 0), so disable the cache.
    cache_te = bool(p["cache_text_encoder_outputs"]) and not is_sdxl
    training: dict[str, Any] = {
        "output_dir": str(output_dir),
        "output_name": output_name,
        "max_train_epochs": int(p["max_train_epochs"]),
        "train_batch_size": int(p["train_batch_size"]),
        "unet_lr": float(p["unet_lr"]),
        "text_encoder_lr": float(p["text_encoder_lr"]),
        "optimizer_type": str(p["optimizer_type"]),
        "lr_scheduler": str(p["lr_scheduler"]),
        "lr_warmup_steps": int(p["lr_warmup_steps"]),
        "mixed_precision": str(p["mixed_precision"]),
        "gradient_checkpointing": bool(p["gradient_checkpointing"]),
        "save_every_n_epochs": int(p["save_every_n_epochs"]),
        "seed": int(p["seed"]),
        # Persist full training state (optimizer/scheduler/step) alongside
        # each weight save, so a paused job can resume from where it left
        # off via `--resume <output_dir>/<name>-state`.
        "save_state": True,
    }
    # Step-wise resumable checkpoints (optional): lets a mid-epoch pause resume
    # without replaying the whole epoch. Only emit when > 0.
    save_every_n_steps = int(p.get("save_every_n_steps", 0) or 0)
    if save_every_n_steps > 0:
        training["save_every_n_steps"] = save_every_n_steps
    return {
        "model": {
            "pretrained_model_name_or_path": str(base_model_path),
            "v2": False,
            "v_parameterization": False,
        },
        "dataset": {
            "train_data_dir": str(train_data_dir),
            # kohya parses resolution via args.resolution.split(","), so it must
            # be a string ("size" or "width,height"), never an int.
            "resolution": f"{int(p['resolution'])},{int(p['resolution'])}",
            "enable_bucket": bool(p["enable_bucket"]),
            "min_bucket_reso": int(p["min_bucket_reso"]),
            "max_bucket_reso": int(p["max_bucket_reso"]),
        },
        "network": {
            "network_module": "networks.lora",
            "network_dim": int(p["network_dim"]),
            "network_alpha": int(p["network_alpha"]),
        },
        "training": training,
        "advanced": {
            "cache_latents": bool(p["cache_latents"]),
            "cache_text_encoder_outputs": cache_te,
        },
    }


def write_config(config: dict[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        toml.dump(config, f)
    return dest
