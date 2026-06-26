"""Build the RVC training pipeline for a voice (SVC) job.

RVC training is a 4-stage pipeline run from the RVC checkout:
  1. preprocess  — slice/normalize the trainset into 3s segments
  2. extract f0  — pitch features (when training a pitch-guided / singing model)
  3. extract hubert features
  4. train       — the actual model training, saving .pth weights per epoch
  (+ a faiss .index is built for retrieval at inference time)

Rather than hard-code one fork's exact CLI (they drift), we emit a single
per-job shell script that calls the standard RVC modules in order with the
resolved parameters. The backend just runs that script and tails its output,
so the commands stay transparent and editable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Defaults tuned for a small singing-voice dataset on consumer hardware.
DEFAULT_PARAMS: dict[str, Any] = {
    "sample_rate": 40000,        # 32000 | 40000 | 48000 (must match dataset)
    "f0": True,                  # pitch-guided — required for singing voice
    "f0_method": "rmvpe",        # rmvpe (best) | harvest | pm | crepe
    "total_epoch": 100,
    "batch_size": 4,
    "save_every_epoch": 25,
    "cache_in_gpu": False,       # off for Mac MPS / low VRAM
    "save_every_weights": True,
    "version": "v2",             # RVC model arch version (v1 | v2)
    "n_process": 2,              # preprocessing parallelism
}


def merge_params(user_params: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_PARAMS)
    merged.update({k: v for k, v in (user_params or {}).items() if v is not None})
    return merged


def estimate_total_steps(clip_count: int, params: dict[str, Any]) -> int:
    """Rough proxy: total epochs (per-step progress is parsed from logs)."""
    return int(params.get("total_epoch", DEFAULT_PARAMS["total_epoch"]))


def _pretrain_paths(rvc_dir: str, version: str, sr: int, f0: bool) -> tuple[str, str]:
    """Conventional RVC pretrained generator/discriminator paths."""
    tag = "f0" if f0 else ""
    sr_k = f"{sr // 1000}k"
    base = f"{rvc_dir}/assets/pretrained_v2" if version == "v2" else f"{rvc_dir}/assets/pretrained"
    g = f"{base}/{tag}G{sr_k}.pth"
    d = f"{base}/{tag}D{sr_k}.pth"
    return g, d


def build_pipeline_script(
    *,
    rvc_dir: str,
    python_cmd: str,
    exp_name: str,
    trainset_dir: str,
    exp_dir: str,
    weights_out_dir: str,
    params: dict[str, Any],
    device: str = "cpu",
) -> str:
    """Return a bash script that runs the full RVC training pipeline.

    Paths are passed as already-resolved strings so this works identically for
    a local run and a remote run (the caller supplies remote paths + python).
    """
    p = merge_params(params)
    sr = int(p["sample_rate"])
    f0 = bool(p["f0"])
    version = str(p["version"])
    n_p = int(p["n_process"])
    if0 = 1 if f0 else 0
    g_path, d_path = _pretrain_paths(rvc_dir, version, sr, f0)

    f0_stage = ""
    if f0:
        f0_stage = (
            f'echo "[loralab] (2/4) 提取基频 f0（{p["f0_method"]}）"\n'
            f'"$PY" "$RVC/infer/modules/train/extract/extract_f0_print.py" '
            f'"$EXP" {n_p} {p["f0_method"]}\n'
        )

    # extract_feature_print.py args: device n_part i_part i_gpu exp_dir version is_half
    script = f"""#!/usr/bin/env bash
set -euo pipefail
RVC="{rvc_dir}"
PY="{python_cmd}"
EXP="{exp_dir}"
TRAINSET="{trainset_dir}"
DEVICE="{device}"

cd "$RVC"
mkdir -p "$EXP" "{weights_out_dir}"

echo "[loralab] (1/4) 预处理训练集（采样率 {sr}）"
"$PY" "$RVC/infer/modules/train/preprocess.py" "$TRAINSET" {sr} {n_p} "$EXP" False 3.0

{f0_stage}echo "[loralab] (3/4) 提取 HuBERT 特征"
"$PY" "$RVC/infer/modules/train/extract_feature_print.py" "$DEVICE" 1 0 0 "$EXP" {version} False

echo "[loralab] (4/4) 开始训练：共 {p['total_epoch']} 轮"
"$PY" "$RVC/infer/modules/train/train.py" \\
  -e "{exp_name}" \\
  -sr {sr // 1000}k \\
  -f0 {if0} \\
  -bs {p['batch_size']} \\
  -te {p['total_epoch']} \\
  -se {p['save_every_epoch']} \\
  -pg "{g_path}" \\
  -pd "{d_path}" \\
  -l {0} \\
  -c {1 if p['cache_in_gpu'] else 0} \\
  -sw {1 if p['save_every_weights'] else 0} \\
  -v {version}

echo "[loralab] 构建检索索引 (.index)"
"$PY" - <<'PYEOF' || echo "[loralab] 索引构建跳过（可在推理端重建）"
import os, sys, glob, traceback
try:
    import numpy as np, faiss
    exp = os.environ.get("EXP", "{exp_dir}")
    feat_dir = os.path.join(exp, "3_feature{('256' if version=='v1' else '768')}")
    npys = [np.load(f) for f in sorted(glob.glob(os.path.join(feat_dir, "*.npy")))]
    if npys:
        big = np.concatenate(npys, 0)
        n_ivf = min(int(16 * np.sqrt(big.shape[0])), big.shape[0] // 39 + 1)
        index = faiss.index_factory(big.shape[1], f"IVF{{max(n_ivf,1)}},Flat")
        index.train(big); index.add(big)
        out = os.path.join("{weights_out_dir}", "added_{exp_name}.index")
        faiss.write_index(index, out)
        print("[loralab] 索引已保存:", out)
except Exception:
    traceback.print_exc()
PYEOF

echo "[loralab] DONE"
"""
    return script


def write_script(content: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    dest.chmod(0o755)
    return dest
