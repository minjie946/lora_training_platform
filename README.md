# LoRA 训练平台 (LoraLab)

Web 化的 LoRA 训练管理平台。MVP 聚焦**人物生成 LoRA**，对齐《Mac M4 Pro 动漫人物 LoRA 训练完整指南》
（底模 `animagine-xl-4.0-opt`，工具 `kohya_ss`/sd-scripts，设备 Mac MPS）。

完整流程：创建数据集 → 上传图片 → 打标（触发词置首位）→ 配置参数 → 启动训练 → 实时监控 → 下载产出的 LoRA。

## 技术栈

- 后端：FastAPI + SQLModel(SQLite)，用 **uv** 管理 Python 3.10 项内环境
- 前端：React + TypeScript + Vite
- 训练引擎：封装 kohya_ss（生成 `config.toml` + 子进程 `accelerate launch train_network.py`）
- 算力抽象：`TrainingBackend` Adapter 模式，本期实现 `LocalMpsBackend`，预留云 GPU / 自有服务器 / 外部 API / Windows CUDA

## 目录

```
backend/         FastAPI 后端（uv 管理，.venv 在项内）
  app/
    routers/     system / datasets / jobs / models
    services/    dataset / caption / config_builder / job_manager
      backends/  base / local_mps / registry  ← 算力扩展点
    utils/       log_parser
  workspace/     运行期数据（datasets / models/base / jobs / app.db），不入 git
frontend/        React + Vite（5 个页面）
scripts/         setup_mac.sh
```

## 环境要求

- [uv](https://docs.astral.sh/uv/)（管理后端 Python 3.10 环境）
- Node.js 18+ / npm（前端）
- Apple Silicon Mac（MPS 训练）

## 快速开始

### 1. 准备后端环境与依赖检查

```bash
bash scripts/setup_mac.sh
```

脚本会用 uv 创建后端项内环境（`backend/.venv`），并检查 MPS / kohya / 底模四项。
其中训练所需的三项较重，需手动准备：

```bash
# 1) 安装 MPS 版 PyTorch 到后端环境（不装 xformers / bitsandbytes，Mac 不支持）
cd backend && uv pip install torch torchvision torchaudio

# 2) 克隆 kohya_ss（默认期望在仓库根目录的 sd-scripts/，或用 KOHYA_DIR 指定）
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts && uv pip install -r requirements.txt   # 同样不要装 xformers / bitsandbytes

# 3) 把底模放到指定目录
#    backend/workspace/models/base/animagine-xl-4.0-opt.safetensors
```

准备完成后，可在平台「概览」页或下面命令确认 preflight 四项全绿：

```bash
cd backend && uv run python -c "from app.routers.system import preflight; r=preflight(); print('ok=',r.ok); [print(i.ok, i.name, '|', i.detail) for i in r.items]"
```

> 提示：直接运行根目录的 `./start.sh` 会自动检查并补齐 kohya_ss（缺失时 clone）与 WD14 打标依赖（`wdtagger` + `timm`），再一键启动前后端。WD14 用于自动打标，缺失时打标会降级为「仅注入触发词」。

### 2. 启动后端

```bash
cd backend
uv run uvicorn app.main:app --reload          # http://127.0.0.1:8000
```

可用环境变量：`KOHYA_DIR`、`DEFAULT_BASE_MODEL`、`LORA_WORKSPACE`。

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev                                    # http://localhost:5173
```

前端通过 Vite 代理把 `/api` 转发到后端 8000 端口。

### 4. 走一遍流程

在浏览器打开前端，依次：创建数据集 → 上传 15–40 张图 → 自动打标（触发词置首位）→
配置参数 → 启动训练 → 实时监控进度 / Loss → 下载产出的 LoRA。

> 首次真实训练建议先用 **768 分辨率 + epochs=1** 做冒烟，确认整条链路跑通且不 OOM，再上 1024 正式训练。

## 训练参数默认值

对齐指南阶段四：rank 16 / alpha 8、unet_lr 1e-4、te_lr 5e-5、AdamW、cosine_with_restarts、
gradient_checkpointing、batch 1、fp16、48G→1024 / 24G→768、总步数建议 1200–2000。

## 后续路线

- 测试出图（加载 LoRA 对比，接 diffusers / ComfyUI）
- 新增训练后端验证 Adapter 扩展性
- 音乐 / 分镜脚本 LoRA（新增 backend + 数据集类型 + config_builder）
