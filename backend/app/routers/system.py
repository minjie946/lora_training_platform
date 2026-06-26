"""System router: environment preflight checks."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys

from fastapi import APIRouter

from ..config import DEFAULT_BASE_MODEL, KOHYA_DIR, MODELS_BASE_DIR
from ..schemas import PreflightItem, PreflightResult
from ..services import base_model_service

router = APIRouter(prefix="/api/system", tags=["system"])


def _check_python() -> PreflightItem:
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    return PreflightItem(
        name="Python",
        ok=ok,
        detail=f"{v.major}.{v.minor}.{v.micro}（建议 3.10）",
    )


def _check_torch_mps() -> PreflightItem:
    if importlib.util.find_spec("torch") is None:
        return PreflightItem(
            name="PyTorch / MPS",
            ok=False,
            detail="torch 未安装，请在训练环境安装 MPS 版 PyTorch",
        )
    try:
        import torch  # type: ignore

        avail = bool(torch.backends.mps.is_available())
        built = bool(torch.backends.mps.is_built())
        return PreflightItem(
            name="PyTorch / MPS",
            ok=avail,
            detail=f"torch {torch.__version__}, mps_available={avail}, mps_built={built}",
        )
    except Exception as e:  # noqa: BLE001
        return PreflightItem(name="PyTorch / MPS", ok=False, detail=f"检测失败: {e}")


def _check_kohya() -> PreflightItem:
    train_script = KOHYA_DIR / "train_network.py"
    ok = train_script.exists()
    detail = f"{train_script}" if ok else f"未找到 train_network.py（期望路径: {KOHYA_DIR}）"
    return PreflightItem(name="kohya_ss (sd-scripts)", ok=ok, detail=detail)


def _check_base_model() -> PreflightItem:
    model_path = MODELS_BASE_DIR / DEFAULT_BASE_MODEL
    ok = model_path.exists()
    if ok:
        size_gb = model_path.stat().st_size / (1024**3)
        detail = f"{model_path.name} ({size_gb:.2f} GB)"
    else:
        detail = f"未找到底模，请放置到 {model_path}"
    return PreflightItem(name="底模 (Base Model)", ok=ok, detail=detail)


@router.get("/preflight", response_model=PreflightResult)
def preflight() -> PreflightResult:
    items = [
        _check_python(),
        _check_torch_mps(),
        _check_kohya(),
        _check_base_model(),
    ]
    return PreflightResult(ok=all(i.ok for i in items), items=items)


@router.get("/base-models")
def base_models() -> dict:
    """List discoverable base checkpoints with inferred type/style."""
    models = base_model_service.list_base_models()
    return {
        "default": DEFAULT_BASE_MODEL,
        "models": [
            {
                "filename": m.filename,
                "label": m.label,
                "is_sdxl": m.is_sdxl,
                "style": m.style,
                "size_bytes": m.size_bytes,
                "is_default": m.is_default,
            }
            for m in models
        ],
    }


def _gpu_stats() -> dict:
    """Apple Silicon GPU utilization & memory via ioreg (no sudo required).

    Returns {} on non-macOS or when the data can't be read.
    """
    if sys.platform != "darwin":
        return {}
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except Exception:  # noqa: BLE001
        return {}

    stats: dict = {}
    util = re.search(r'"Device Utilization %"=(\d+)', out)
    if util:
        stats["utilization"] = int(util.group(1))
    in_use = re.search(r'"In use system memory"=(\d+)', out)
    if in_use:
        stats["used_bytes"] = int(in_use.group(1))
    cores = re.search(r'"gpu-core-count" = (\d+)', out)
    if cores:
        stats["cores"] = int(cores.group(1))
    if stats:
        stats["available"] = True
    return stats


@router.get("/resources")
def resources() -> dict:
    """Live CPU / memory / GPU usage for the resource monitor."""
    data: dict = {"platform": sys.platform}

    try:
        import psutil  # type: ignore

        data["cpu_percent"] = psutil.cpu_percent(interval=None)
        data["cpu_count"] = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        data["mem_total"] = vm.total
        data["mem_used"] = vm.used
        data["mem_percent"] = vm.percent
    except Exception:  # noqa: BLE001
        pass

    # Apple Silicon shares system RAM; ioreg gives GPU utilization + memory
    # without sudo and without importing torch in this process.
    data["gpu"] = _gpu_stats()
    return data
