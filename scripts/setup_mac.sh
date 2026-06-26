#!/usr/bin/env bash
# One-shot environment setup / check for the LoRA training platform (Mac M-series).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
KOHYA_DIR="${KOHYA_DIR:-$ROOT_DIR/sd-scripts}"
MODELS_DIR="$BACKEND_DIR/workspace/models/base"

echo "==> LoRA Training Platform setup"
echo "Root: $ROOT_DIR"

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[x] uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
echo "[ok] uv: $(uv --version)"

# 2. Backend venv + deps (Python 3.10, in-project .venv)
echo "==> Syncing backend dependencies via uv (Python 3.10)"
(cd "$BACKEND_DIR" && uv python pin 3.10 >/dev/null 2>&1 || true && uv sync)
echo "[ok] backend .venv ready at $BACKEND_DIR/.venv"

# 3. PyTorch (MPS) — installed into the training env, NOT auto-installed here (large).
echo "==> PyTorch (MPS) check"
if (cd "$BACKEND_DIR" && uv run python -c "import torch" >/dev/null 2>&1); then
  (cd "$BACKEND_DIR" && uv run python -c "import torch; print('[ok] torch', torch.__version__, 'mps', torch.backends.mps.is_available())")
else
  echo "[!] torch not installed in backend env."
  echo "    To enable real training run: (cd backend && uv pip install torch torchvision torchaudio)"
fi

# 4. kohya_ss (sd-scripts)
echo "==> kohya_ss (sd-scripts) check"
if [ -f "$KOHYA_DIR/train_network.py" ]; then
  echo "[ok] found $KOHYA_DIR/train_network.py"
else
  echo "[!] sd-scripts not found at $KOHYA_DIR"
  echo "    Clone with: git clone https://github.com/kohya-ss/sd-scripts.git \"$KOHYA_DIR\""
fi

# 5. Base model
echo "==> Base model check"
mkdir -p "$MODELS_DIR"
if [ -f "$MODELS_DIR/animagine-xl-4.0-opt.safetensors" ]; then
  echo "[ok] base model present"
else
  echo "[!] place animagine-xl-4.0-opt.safetensors into: $MODELS_DIR"
fi

echo "==> Done. Start backend with: (cd backend && uv run uvicorn app.main:app --reload)"
