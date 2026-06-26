# LoRA 训练平台 — 架构方案与实施计划

## Context（背景与目标）

从零搭建一个 **Web 化的 LoRA 训练管理平台**。最终目标支持三类 LoRA：人物生成、音乐、分镜脚本。三类的基座模型、数据格式、训练管线差异巨大，因此本期 **MVP 聚焦「人物生成 LoRA」**，把"上传数据 → 打标 → 配参 → 训练 → 监控 → 产出模型 → 测试出图"全流程跑通。

MVP 的训练管线严格对齐飞书文档《Mac M4 Pro 动漫人物 LoRA 训练完整指南》：

- 基座模型：`animagine-xl-4.0-opt`（SDXL 动漫人物体系）
- 训练工具：`kohya_ss`（sd-scripts），后端封装其 `train_network.py`
- 运行设备：Mac MPS（`device=mps`、`AdamW`、关闭 xformers/bitsandbytes、开 gradient_checkpointing、batch=1、48G 用 1024 / 24G 用 768）

**已确认的关键决策：**
- 交付物：完整平台（前端 + 后端 + 真实训练）
- 技术栈：React + TypeScript（前端）/ FastAPI（后端，与 PyTorch 同生态）
- 训练引擎：封装 kohya_ss（生成 `config.toml` + 子进程 `accelerate launch`）
- 存储/队列：SQLite + 子进程任务管理器（单机零额外依赖，可后续升级 Postgres + Celery）
- 用户体系：单用户、无登录（后续可加鉴权）
- 算力：本期本地 Mac MPS；架构上用 **Adapter 模式** 预留 自有服务器 / 云 GPU / 外部 API / Windows CUDA 的接入

---

## 架构总览

```
lora_training_platform/
├── backend/                      # FastAPI 后端
│   ├── app/
│   │   ├── main.py               # FastAPI 入口、CORS、路由挂载
│   │   ├── config.py             # 全局配置（工作目录、底模目录、kohya 路径等）
│   │   ├── db.py                 # SQLite 连接 + 初始化（SQLModel/SQLAlchemy）
│   │   ├── models.py             # 数据表：Dataset / TrainingJob / LoraModel
│   │   ├── schemas.py            # Pydantic 请求/响应模型
│   │   ├── routers/
│   │   │   ├── datasets.py       # 数据集 CRUD、图片上传、打标
│   │   │   ├── jobs.py           # 训练任务：创建/启动/停止/查询/日志(SSE)
│   │   │   ├── models.py         # 产出 LoRA 列表/下载/删除
│   │   │   └── system.py         # 环境自检：MPS 可用性、底模是否就绪、kohya 是否安装
│   │   ├── services/
│   │   │   ├── dataset_service.py    # 目录结构(10_concept)、缩略图、txt 标签读写
│   │   │   ├── caption_service.py    # WD14 自动打标 + 触发词注入
│   │   │   ├── config_builder.py     # 表单参数 → config.toml（对齐文档阶段四）
│   │   │   ├── job_manager.py        # 任务队列、进程生命周期、日志落盘、进度解析
│   │   │   └── backends/             # 算力 Adapter（核心扩展点）
│   │   │       ├── base.py           # TrainingBackend 抽象基类
│   │   │       ├── local_mps.py      # 本期实现：本地 Mac MPS 跑 kohya_ss
│   │   │       └── registry.py       # 后端注册表，按 job.backend 选择实现
│   │   └── utils/
│   │       └── log_parser.py     # 从 kohya stdout 解析 step/epoch/loss
│   ├── requirements.txt
│   └── workspace/                # 运行期数据（gitignore）
│       ├── datasets/<id>/10_<concept>/  # 图片 + .txt
│       ├── models/base/                  # 底模 safetensors
│       ├── jobs/<id>/                     # config.toml / train.log / output/
│       └── app.db                         # SQLite
├── frontend/                     # React + TS + Vite
│   ├── src/
│   │   ├── api/                  # 后端接口封装（fetch/axios）
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx     # 概览：任务统计、环境状态
│   │   │   ├── Datasets.tsx      # 数据集列表 + 详情（图片网格 + 标签编辑）
│   │   │   ├── NewTraining.tsx   # 训练向导（选数据集 → 配参 → 提交）
│   │   │   ├── JobDetail.tsx     # 实时日志 + loss 曲线 + 进度
│   │   │   └── Models.tsx        # 产出 LoRA 模型库
│   │   ├── components/           # 复用组件（上传、参数表单、日志窗、统计卡）
│   │   └── App.tsx / main.tsx / router
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   └── setup_mac.sh             # 一键环境检查/安装（venv、torch-mps、kohya、底模提示）
├── .trae/documents/            # 本方案
└── README.md
```

### 算力 Adapter 设计（关键扩展点）

`TrainingBackend` 抽象基类定义统一契约，本期只实现 `LocalMpsBackend`：

```python
class TrainingBackend(ABC):
    name: str
    @abstractmethod
    def preflight(self) -> PreflightResult: ...        # 环境自检（设备/底模/依赖）
    @abstractmethod
    def start(self, job: TrainingJob) -> ProcessHandle: ...  # 启动训练，返回句柄
    @abstractmethod
    def stop(self, handle) -> None: ...                # 停止
    @abstractmethod
    def stream_logs(self, job) -> Iterator[str]: ...   # 增量日志
```

未来扩展只需新增 `local_cuda.py` / `remote_server.py` / `cloud_gpu.py` / `external_api.py` 并在 `registry.py` 注册，前端通过下拉选择后端，**平台层与训练脚本层解耦**。

### 数据模型（SQLite）

- **Dataset**: id, name, concept(概念名), repeat, trigger_word, image_count, status, created_at
- **TrainingJob**: id, name, dataset_id, base_model, backend, params(JSON), status(pending/running/succeeded/failed/stopped), progress, current_step/total_step, pid, created_at, finished_at
- **LoraModel**: id, job_id, name, epoch, file_path, file_size, created_at

---

## 训练流程映射（文档 7 阶段 → 平台功能）

| 文档阶段 | 平台实现 |
|---|---|
| 一 图片准备 | 数据集创建 + 图片上传，自动建 `10_<concept>/` 目录，生成缩略图；分辨率建议提示（48G→1024 / 24G→768） |
| 二 打标 | `caption_service` 调 WD14 自动打标 → 注入触发词到标签首位 → 前端图片网格内联编辑 `.txt`；核心特征不入标的提示 |
| 三 环境安装 | `system` 路由 + `setup_mac.sh`：检测 MPS、Python、kohya、底模；缺失项给出指引 |
| 四 训练配置 | `NewTraining` 表单 → `config_builder` 生成 `config.toml`（rank16/alpha8、unet_lr 1e-4、te_lr 5e-5、AdamW、cosine_with_restarts、gradient_checkpointing 等默认值，含总步数实时估算） |
| 五 启动训练 | `job_manager` 经 `LocalMpsBackend` 执行 `accelerate launch train_network.py`，设 `PYTORCH_ENABLE_MPS_FALLBACK=1`，日志落盘 |
| 五 监控 | `log_parser` 解析 step/epoch/loss → SSE 推前端 → 实时进度条 + loss 曲线 |
| 六 测试出图 | 产出多 epoch 模型列表、下载；**出图推理本期作为占位/后续里程碑**（可选接 ComfyUI/diffusers） |
| 七 迭代 | 任务可基于历史参数"克隆重训"；常见问题诊断对照表内置为前端帮助提示 |

---

## 实施里程碑（建议分阶段交付，每步可独立验证）

**M1 — 后端骨架 + 环境自检**
- FastAPI 工程、SQLite 初始化、配置、`/api/system/preflight`（返回 MPS/底模/kohya 状态）
- `setup_mac.sh`
- 验证：`uvicorn` 启动，`GET /api/system/preflight` 返回真实环境状态

**M2 — 数据集管理**
- 数据集 CRUD、图片上传（建 `10_<concept>/`、缩略图）、标签读写接口
- 验证：上传若干图片，磁盘目录结构正确，能读写 `.txt`

**M3 — 打标服务**
- WD14 自动打标（不可用时降级为手动）、触发词注入、批量/单张编辑
- 验证：对数据集一键打标，标签首位为触发词

**M4 — 训练配置 + 任务编排（核心）**
- `config_builder` 生成 `config.toml`；`TrainingBackend` 抽象 + `LocalMpsBackend`；`job_manager` 子进程生命周期、日志落盘、进度解析
- 验证：提交一个真实小数据集训练任务，能跑起 kohya，日志/进度正确，产出 `output/*.safetensors`

**M5 — 前端全流程打通**
- Vite + React 路由，5 个页面，SSE 实时日志 + loss 曲线，模型库下载
- 验证：浏览器内完整走「建数据集 → 打标 → 配参 → 训练 → 看进度 → 下载模型」，用 MCP 浏览器工具回归

**M6（可选）— 测试出图 / 多算力后端**
- 接 diffusers 或 ComfyUI 做加载 LoRA 出图对比；新增第二个 backend 验证 Adapter 扩展性

---

## 验证方式（端到端）

1. **后端**：`cd backend && uvicorn app.main:app --reload`，用 `curl`/Swagger (`/docs`) 验证各路由
2. **环境自检**：`GET /api/system/preflight` 必须如实返回 `torch.backends.mps.is_available()`、底模文件是否存在、kohya 路径是否有效
3. **训练冒烟**：用 5–10 张测试图 + 768 分辨率 + 少量步数（如 epoch=1）跑一次真实 kohya，确认产出 safetensors
4. **前端**：`cd frontend && npm run dev`，用 integrated_browser MCP 走完整向导并截图核对实时日志/进度/loss 曲线
5. **回归**：任务停止/失败/克隆重训等边界路径

---

## 风险与备注

- **kohya_ss + MPS 兼容性**：sd-scripts 对 MPS 支持随版本变化，需固定可用版本；提供 `PYTORCH_ENABLE_MPS_FALLBACK=1` 兜底
- **训练耗时长**（768 约 1.5–4h），M4/M5 联调建议用最小步数冒烟，避免长等待
- **底模体积大**（SDXL 数 GB），不入 git；由 `setup_mac.sh`/前端提示用户手动放置到 `workspace/models/base/`
- **WD14 依赖**：`wd14-tagger` 在 Mac 上若安装困难，打标降级为纯手动，不阻塞主流程
- 后续音乐/分镜 LoRA：通过新增 backend + 新数据集类型 + 新 config_builder 接入，平台层无需重构
