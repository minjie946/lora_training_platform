#!/usr/bin/env bash
# 一键启动前后端：后端后台运行，前端前台运行，Ctrl+C 时一并清理。
set -euo pipefail

# 以脚本自身所在目录为根，确保从任何位置调用都能找到 backend/ frontend/
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 端口可通过环境变量覆盖：BACKEND_PORT / FRONTEND_PORT
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cleanup() {
  echo ""
  echo "==> 正在停止后端…"
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 返回占用指定端口（LISTEN）的进程 PID 列表（可能为空）
pids_on_port() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null || true
}

# 找到从 $1 起的第一个空闲端口
find_free_port() {
  local port="$1"
  while [[ -n "$(pids_on_port "$port")" ]]; do
    port=$((port + 1))
  done
  echo "$port"
}

# 交互式询问 y/n。$1=提示语，$2=默认值(y/n，用于非交互式终端如管道调用)。
# 结果通过 echo 返回（y 或 n）。
ask_yn() {
  local prompt="$1" default="$2" ans
  # 非 TTY（如 CI/管道）无法交互，直接用默认值，避免脚本挂起。
  if [[ ! -t 0 && ! -e /dev/tty ]]; then
    echo "$default"
    return
  fi
  read -r -p "$prompt" ans </dev/tty || ans=""
  ans="${ans:-$default}"
  echo "$ans"
}

# 尝试清理占用某端口的进程组（先 TERM 后 KILL）。清理成功返回 0，仍占用返回 1。
free_port_or_fail() {
  local port="$1" pids
  pids="$(pids_on_port "$port")"
  [[ -z "$pids" ]] && return 0
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(pids_on_port "$port")"
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
  [[ -z "$(pids_on_port "$port")" ]]
}

# ---- 端口检查：后端端口被占用时询问用户（清理 / 换新端口） ----
echo "==> 检查后端端口 ${BACKEND_PORT}"
BACKEND_PIDS="$(pids_on_port "${BACKEND_PORT}")"
if [[ -n "$BACKEND_PIDS" ]]; then
  # 端口可能被多个进程占用（如 uvicorn 主进程 + reload worker）；
  # BSD ps -p 需要逗号分隔的 PID 列表，用空格会取不到命令名。
  BACKEND_PIDS_CSV="$(echo "$BACKEND_PIDS" | tr '\n' ',' | sed 's/,$//')"
  echo "[warn] 端口 ${BACKEND_PORT} 已被占用（PID: ${BACKEND_PIDS_CSV}）："
  ps -p "$BACKEND_PIDS_CSV" -o pid=,command= 2>/dev/null || true
  # 默认 n（换新端口）更安全，避免误杀他人进程。
  choice="$(ask_yn "是否清理占用端口 ${BACKEND_PORT} 的进程？[y=清理 / n=改用新端口]（默认 n）: " "n")"
  if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "==> 正在清理端口 ${BACKEND_PORT} …"
    if free_port_or_fail "${BACKEND_PORT}"; then
      echo "[ok] 已清理，后端端口 ${BACKEND_PORT} 现在可用"
    else
      echo "[error] 端口 ${BACKEND_PORT} 清理失败（可能无权限），请手动处理或用 BACKEND_PORT=xxxx ./start.sh"
      exit 1
    fi
  else
    NEW_BACKEND_PORT="$(find_free_port $((BACKEND_PORT + 1)))"
    echo "[warn] 保留占用进程，后端改用端口 ${NEW_BACKEND_PORT}"
    BACKEND_PORT="${NEW_BACKEND_PORT}"
  fi
fi
echo "[ok] 后端端口 $BACKEND_PORT 可用"

# ---- 端口检查：前端端口被占用时询问用户（清理 / 换新端口） ----
echo "==> 检查前端端口 $FRONTEND_PORT"
FRONTEND_PIDS="$(pids_on_port "${FRONTEND_PORT}")"
if [[ -n "$FRONTEND_PIDS" ]]; then
  FRONTEND_PIDS_CSV="$(echo "$FRONTEND_PIDS" | tr '\n' ',' | sed 's/,$//')"
  echo "[warn] 端口 ${FRONTEND_PORT} 已被占用（PID: ${FRONTEND_PIDS_CSV}）："
  ps -p "$FRONTEND_PIDS_CSV" -o pid=,command= 2>/dev/null || true
  choice="$(ask_yn "是否清理占用端口 ${FRONTEND_PORT} 的进程？[y=清理 / n=改用新端口]（默认 n）: " "n")"
  if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "==> 正在清理端口 ${FRONTEND_PORT} …"
    if free_port_or_fail "${FRONTEND_PORT}"; then
      echo "[ok] 已清理，前端端口 ${FRONTEND_PORT} 现在可用"
    else
      echo "[warn] 端口 ${FRONTEND_PORT} 清理失败，前端改用 $(find_free_port $((FRONTEND_PORT + 1)))"
      FRONTEND_PORT="$(find_free_port $((FRONTEND_PORT + 1)))"
    fi
  else
    NEW_FRONTEND_PORT="$(find_free_port $((FRONTEND_PORT + 1)))"
    echo "[warn] 保留占用进程，前端改用端口 ${NEW_FRONTEND_PORT}"
    FRONTEND_PORT="${NEW_FRONTEND_PORT}"
  fi
else
  echo "[ok] 前端端口 ${FRONTEND_PORT} 可用"
fi

# ---- 依赖自检：缺失则自动补齐 ----
# 外部训练引擎统一放在 engines/ 子目录，保持仓库根目录整洁。
ENGINES_DIR="${ENGINES_DIR:-$ROOT_DIR/engines}"
mkdir -p "$ENGINES_DIR"

KOHYA_DIR="${KOHYA_DIR:-$ENGINES_DIR/sd-scripts}"
RVC_DIR="${RVC_DIR:-$ENGINES_DIR/Retrieval-based-Voice-Conversion-WebUI}"

# 兼容旧布局：若历史上把引擎 clone 在了仓库根目录，自动迁移进 engines/
migrate_legacy() {
  local legacy="$1" target="$2" name="$3"
  if [[ -d "$legacy" && ! -d "$target" ]]; then
    echo "==> 检测到旧位置的 ${name}（${legacy}），迁移到 ${target}"
    mv "${legacy}" "${target}"
  fi
}
migrate_legacy "$ROOT_DIR/sd-scripts" "$KOHYA_DIR" "kohya_ss"
migrate_legacy "$ROOT_DIR/Retrieval-based-Voice-Conversion-WebUI" "$RVC_DIR" "RVC"

# 图像训练 kohya 通过后端虚拟环境的 python 启动，依赖装进 backend/.venv。
BACKEND_VENV="$ROOT_DIR/backend/.venv"
if [[ ! -x "$BACKEND_VENV/bin/python" ]]; then
  echo "==> 初始化后端虚拟环境 (backend/.venv)"
  (cd "$ROOT_DIR/backend" && uv venv)
fi
BACKEND_PY="$BACKEND_VENV/bin/python"

# ---- kohya_ss（图像 LoRA）：缺失则 clone + 装依赖到后端 venv ----
if [[ ! -f "$KOHYA_DIR/train_network.py" ]]; then
  echo "==> 未检测到 kohya_ss，正在 clone 到 ${KOHYA_DIR}"
  git clone git@github.com:kohya-ss/sd-scripts.git "${KOHYA_DIR}" \
    || git clone https://github.com/kohya-ss/sd-scripts.git "${KOHYA_DIR}"
  if [[ -f "${KOHYA_DIR}/requirements.txt" ]]; then
    echo "==> 安装 kohya_ss 依赖到后端 venv（首次较慢）"
    (cd "${KOHYA_DIR}" && uv pip install --python "${BACKEND_PY}" -r requirements.txt) \
      || echo "[warn] kohya_ss 依赖安装失败，可稍后手动：cd ${KOHYA_DIR} && uv pip install --python ${BACKEND_PY} -r requirements.txt"
  fi
else
  echo "[ok] kohya_ss 已就位：${KOHYA_DIR}"
fi

# ---- RVC（声音克隆 / SVC）：缺失则 clone + 在「独立 venv」装依赖 ----
# RVC 依赖较重，且与 kohya 的 torch 版本冲突，因此用自己的虚拟环境，
# 后端通过该 venv 的 python（RVC_PYTHON）启动本地 SVC 训练。
# 固定 Python 3.10：RVC 钉死的 faiss-cpu==1.7.3 只有 cp37~cp311 的 wheel，
# 用 3.12 会报 "no wheels with a matching Python ABI tag"。
RVC_PY_VERSION="${RVC_PY_VERSION:-3.10}"
RVC_TRAIN_PY="$RVC_DIR/infer/modules/train/train.py"
RVC_VENV="$RVC_DIR/.venv"
RVC_PYTHON="$RVC_VENV/bin/python"

# 只用图像训练时可跳过 RVC 的 clone / venv / 依赖安装：SKIP_RVC=1 ./start.sh
if [[ "${SKIP_RVC:-0}" == "1" ]]; then
  echo "[skip] 已设置 SKIP_RVC=1，跳过 RVC（声音克隆）clone 与依赖安装。仅本地 SVC 训练需要它。"
elif [[ ! -f "$RVC_TRAIN_PY" ]]; then
  echo "==> 未检测到 RVC（声音克隆引擎），正在 clone 到 ${RVC_DIR}"
  git clone git@github.com:RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git "${RVC_DIR}" \
    || git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git "${RVC_DIR}" \
    || echo "[warn] RVC clone 失败，可稍后手动 clone；本地 SVC 训练前需保证 ${RVC_TRAIN_PY} 存在"
else
  echo "[ok] RVC 已就位：${RVC_DIR}"
fi

if [[ "${SKIP_RVC:-0}" != "1" && -f "$RVC_TRAIN_PY" ]]; then
  # 若已有 venv 但版本不受支持（如 3.12），删除重建，避免 faiss-cpu 装不上
  if [[ -x "$RVC_PYTHON" ]]; then
    _cur_ver="$("$RVC_PYTHON" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")"
    if [[ "$_cur_ver" == "3.12" || "$_cur_ver" == "3.13" ]]; then
      echo "[warn] RVC venv 为 Python ${_cur_ver}（faiss-cpu 不支持），重建为 ${RVC_PY_VERSION}"
      rm -rf "${RVC_VENV}"
    fi
  fi
  # 为 RVC 单独建 venv（与后端/kohya 隔离），固定 Python 版本
  if [[ ! -x "$RVC_PYTHON" ]]; then
    echo "==> 为 RVC 创建独立虚拟环境（${RVC_VENV}，Python ${RVC_PY_VERSION}）"
    (cd "${RVC_DIR}" && uv venv --python "${RVC_PY_VERSION}") \
      || echo "[warn] RVC venv 创建失败，可稍后手动：cd ${RVC_DIR} && uv venv --python ${RVC_PY_VERSION}"
  fi
  # 仅当依赖尚未安装时才安装（用 torch 是否可导入做粗略判断），避免每次启动重复解析
  if [[ -x "$RVC_PYTHON" ]] && ! "$RVC_PYTHON" -c "import torch" >/dev/null 2>&1; then
    # Mac 优先用平台专用 requirements（RVC 仓库为 Apple Silicon 提供）
    RVC_REQ="requirements.txt"
    if [[ "$(uname -s)" == "Darwin" ]]; then
      for cand in requirements-dmac.txt requirements-mac.txt; do
        if [[ -f "${RVC_DIR}/${cand}" ]]; then RVC_REQ="${cand}"; break; fi
      done
    fi
    if [[ -f "${RVC_DIR}/${RVC_REQ}" ]]; then
      echo "==> 安装 RVC 依赖到独立 venv（使用 ${RVC_REQ}，依赖较重，首次很慢）"
      (cd "${RVC_DIR}" && uv pip install --python "${RVC_PYTHON}" -r "${RVC_REQ}") \
        || echo "[warn] RVC 依赖安装失败。本地训练前请手动处理：cd ${RVC_DIR} && uv pip install --python ${RVC_PYTHON} -r ${RVC_REQ}；仅做远程训练可忽略。"
    fi
  fi
  echo "     提示：RVC 还需下载预训练模型到 assets/（见其 README）；仅做远程训练可忽略本地依赖与模型。"
fi




echo "==> 检查 WD14 打标依赖 (wdtagger / timm)"
if (cd "$ROOT_DIR/backend" && uv run python -c "import wdtagger, timm" >/dev/null 2>&1); then
  echo "[ok] WD14 (wdtagger + timm) 已安装"
else
  echo "==> 未检测到 WD14，正在安装 wdtagger 与 timm"
  (cd "$ROOT_DIR/backend" && uv pip install wdtagger timm)
fi

echo "==> 启动后端 (http://127.0.0.1:${BACKEND_PORT})"
(cd "$ROOT_DIR/backend" && ENGINES_DIR="${ENGINES_DIR}" KOHYA_DIR="${KOHYA_DIR}" RVC_DIR="${RVC_DIR}" RVC_PYTHON="${RVC_PYTHON}" uv run uvicorn app.main:app --reload --port "${BACKEND_PORT}") &
BACKEND_PID=$!

echo "==> 启动前端 (http://localhost:${FRONTEND_PORT})"
cd "$ROOT_DIR/frontend"
FRONTEND_PORT="$FRONTEND_PORT" BACKEND_PORT="$BACKEND_PORT" npm run dev -- --port "$FRONTEND_PORT"
