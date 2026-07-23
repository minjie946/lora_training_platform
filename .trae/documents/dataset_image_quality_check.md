# 数据集图片质量检测功能

## Context（背景）

在做 LoRA 数据集时，用户经常无法快速判断哪些图片不适合作为训练素材——模糊、分辨率过低、曝光异常、极端宽高比、无人脸或多张人脸等问题会拖累训练效果。目前 [DatasetDetail.tsx](file:///Users/bytedance/Documents/github/lora_training_platform/frontend/src/pages/DatasetDetail/DatasetDetail.tsx) 只在有 WD14 标签置信度时对卡片着色，没有对**图片本身质量**的任何检查。

本功能新增「检测质量」按钮：对数据集内全部图片做一次启发式 + 人脸检测分析，把不适合的图片在卡片上标记出来，并给出具体原因（如"太模糊""无人脸""分辨率过低"），帮助用户在训练前清理数据集。

**关键前提（已核实）**：`numpy 2.2.6`、`PIL`、`opencv 4.10.0`（含 Haar 人脸级联 `haarcascade_frontalface_default.xml`）在 venv 中**均已存在**，无需新增任何依赖。

## 设计决策（已与用户确认）

- **检测方式**：启发式（PIL+numpy）+ 人脸检测（OpenCV Haar）。
- **触发时机**：数据集详情页手动「检测质量」按钮触发，后台分析全部图片，结果缓存到磁盘。
- **人脸预期**：按人物 LoRA 处理——无人脸 / 多张人脸 / 人脸占比过小 都标为提醒（本平台聚焦人物/角色训练）。

## 检测项与判定规则

对每张图片计算，产出 `issues`（问题列表）与 `level`（`ok` / `warn` / `bad`）：

| 检测项 | 方法 | 判定 |
|---|---|---|
| 分辨率过低 | PIL `.size` | 短边 < 512 → warn；< 384 → bad |
| 极端宽高比 | w/h | 比例 > 2.5 或 < 0.4 → warn（训练裁剪会丢主体） |
| 模糊 | 灰度图 Laplacian 方差（cv2.Laplacian） | 方差 < 100 → warn；< 50 → bad |
| 曝光异常 | 灰度直方图均值/暗部亮部占比 | 均值 <40 或 >215，或极端占比 → warn |
| 无/多人脸 | cv2 Haar `detectMultiScale` | 0 张 → warn(无人脸)；≥2 张 → warn(多人脸) |
| 人脸过小 | 最大人脸框面积 / 图片面积 | < 4% → warn(主体太小) |

`level` 取所有 issue 中最严重的：任一 bad → `bad`，否则有 warn → `warn`，全无 → `ok`。

## 后端改动

### 1. 新增 `backend/app/services/image_quality.py`
- `analyze_image(path: Path) -> dict`：打开图片，跑上述所有检测，返回 `{level, issues: [{code, label, severity}], metrics: {...}}`。
- 复用 `dataset_service` 已有的 HEIC 处理思路——但检测只针对已落盘的图片（都是 png/jpg 等 `ALLOWED_IMAGE_EXTS`），无需处理 HEIC。
- OpenCV 级联全局单例加载（`cv2.CascadeClassifier`），避免每张图重复初始化。
- 单张失败不抛出，返回 `level="ok", issues=[]` 并记 metric error，保证整体不中断。

### 2. `backend/app/services/dataset_service.py`
- 新增质量结果的读写：仿照 caption 的 `.wdtags.json` 落盘模式，把结果写到图片目录下 `.quality.json`（单文件存整个数据集的 `{filename: result}`），提供：
  - `run_quality_check(dataset_id) -> dict`：遍历 `find_image_dir` 下所有图片，逐张 `analyze_image`，汇总写入 `.quality.json`，返回统计 `{total, ok, warn, bad}`。
  - `read_quality(dataset_id) -> dict | None`：读取缓存。
  - 在 `delete_image` 中同步清理该文件在 `.quality.json` 里的条目。
- `list_images` 的返回项中附带该图的 `quality`（从缓存读，无则 `None`），供前端卡片直接渲染。

### 3. `backend/app/services/caption_manager.py`（复用现有后台任务模式）
- 质量检测可能耗时（人脸检测），参照 caption 的守护线程异步模式做一个轻量后台任务，或——因为纯 CPU、每张 <几百 ms、几十张总计数秒——**先用同步实现**（`run_quality_check` 直接在请求内跑）。若图片多可后续升级为异步。**本期采用同步**，接口直接返回统计结果，前端 loading 态覆盖即可。

### 4. `backend/app/schemas.py`
- 新增 `ImageQuality`（`level: str`, `issues: list[QualityIssue]`）与 `QualityIssue`（`code`, `label`, `severity`）。
- `ImageItem` 增加可选字段 `quality: Optional[ImageQuality] = None`。
- 新增 `QualityCheckResult`（`total, ok, warn, bad`）。

### 5. `backend/app/routers/datasets.py`
- 新增 `POST /{dataset_id}/quality-check`：调用 `ds.run_quality_check`，返回 `QualityCheckResult`。
- `list_images` 已有端点自动带上 quality（因 service 层已附加），无需改签名。

## 前端改动

### 1. `frontend/src/api/client.ts`
- `ImageItem` interface 增加 `quality?: { level: string; issues: { code: string; label: string; severity: string }[] }`。
- 新增 `checkQuality: (id) => http(/api/datasets/${id}/quality-check, { method: 'POST' })`。

### 2. `frontend/src/pages/DatasetDetail/DatasetDetail.tsx`
- 顶部工具区（靠近上传/打标按钮）加「检测质量」按钮，点击后 setLoading → `api.checkQuality` → 重新 `loadImages`。
- 卡片上新增质量角标：复用现有 `conf-badge` 的视觉体系，新增一个 `quality-badge`：
  - `bad` → 橙红 + 外发光（沿用 memory 里 `<45%` 的视觉规范）
  - `warn` → 黄色
  - `ok` → 不显示（避免噪声）
- 角标 hover Tooltip（用上次做的 `.tip` 机制）展示该图所有 issue 的 `label`，如"太模糊、无人脸"。
- 检测完成后顶部用一句 muted 文案汇总：`共 N 张，M 张建议核对`（badge 形式，符合 memory 里"悬浮提示而非内联横幅"的偏好——这里用一次性 toast/muted 行即可）。

### 3. `frontend/src/pages/DatasetDetail/DatasetDetail.css`
- 新增 `.quality-badge` 及 `bad/warn` 配色，与既有 `.conf-badge` 风格一致（Zinc/Indigo，bad 用橙红外发光）。质量角标放卡片**左上**，避免与右上的置信度角标重叠。

## 复用点

- 磁盘缓存模式仿照 `caption_service.read_wdtags` / `.wdtags.json`（[dataset_service.py](file:///Users/bytedance/Documents/github/lora_training_platform/backend/app/services/dataset_service.py) L145、L276）。
- 卡片角标 + hover Tooltip 复用本会话已建的全局 `.tip` 机制与 `.conf-badge` 着色体系（[DatasetDetail.tsx](file:///Users/bytedance/Documents/github/lora_training_platform/frontend/src/pages/DatasetDetail/DatasetDetail.tsx) L284-290）。
- 图片目录定位复用 `ds.find_image_dir` / `get_image_path`。

## 验证方式

1. 后端单元验证：对现有数据集 #1 的图片目录跑 `analyze_image`，确认返回结构正确、OpenCV 人脸检测能命中正脸样图、模糊/低分图被标记。
   ```bash
   cd backend && .venv/bin/python -c "from app.services.image_quality import analyze_image; from pathlib import Path; import glob; [print(p, analyze_image(Path(p))['level'], [i['code'] for i in analyze_image(Path(p))['issues']]) for p in glob.glob('workspace/datasets/1/*_*/*.png')[:10]]"
   ```
2. 起后端 + 前端，进入数据集详情页，点「检测质量」，确认：
   - loading 态出现、完成后卡片出现质量角标；
   - 故意放一张模糊图/纯色图/无人脸图，确认被标 warn/bad 且 hover 显示原因；
   - 删除一张图后 `.quality.json` 对应条目被清理，重新进页面不报错。
3. `tsc --noEmit` 前端类型检查通过；后端 `python -c "import app.main"` 无导入错误。

## 说明与边界

- Haar 人脸检测是经典算法，对**侧脸、遮挡、动漫脸**召回有限。动漫底模数据集可能大量误报"无人脸"——因此人脸类 issue 只作 **warn（提醒）不作 bad（阻断）**，且文案措辞为"未检测到清晰正脸"，避免误导。
- 所有 issue 都不阻止训练，仅作视觉提醒；是否删图由用户决定。
- 若后续数据集图片量很大（数百张），再把同步检测升级为 `caption_manager` 那样的后台守护线程 + 轮询。
