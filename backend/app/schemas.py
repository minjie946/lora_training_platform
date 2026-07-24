"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


# ---- System ----
class PreflightItem(BaseModel):
    name: str
    ok: bool
    detail: str


class PreflightResult(BaseModel):
    ok: bool
    items: list[PreflightItem]


# ---- Dataset ----
class DatasetCreate(BaseModel):
    name: str
    concept: str
    repeat: int = 10
    trigger_word: str = ""
    base_model: str = ""


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    concept: Optional[str] = None
    repeat: Optional[int] = None
    trigger_word: Optional[str] = None
    base_model: Optional[str] = None


class DatasetRead(BaseModel):
    id: int
    name: str
    concept: str
    repeat: int
    trigger_word: str
    base_model: str
    image_count: int
    status: str
    caption_status: str = "idle"
    caption_detail: str = ""
    created_at: datetime


class DatasetImportResult(BaseModel):
    dataset: DatasetRead
    imported: int
    captioned: int
    detail: str


class TagScore(BaseModel):
    tag: str
    confidence: float


class QualityIssue(BaseModel):
    code: str
    label: str
    severity: str  # "warn" | "bad"


class ImageQuality(BaseModel):
    level: str  # "ok" | "warn" | "bad"
    issues: list[QualityIssue] = []


class QualityCheckResult(BaseModel):
    total: int
    ok: int
    warn: int
    bad: int


class ImageItem(BaseModel):
    filename: str
    caption: str
    thumbnail_url: str
    image_url: str
    # WD14 per-tag confidence scores (None unless WD14 tagging was run).
    tag_scores: Optional[list[TagScore]] = None
    # Heuristic image-quality analysis (None unless a quality check was run).
    quality: Optional[ImageQuality] = None


class CaptionUpdate(BaseModel):
    filename: str
    caption: str


# ---- Caption ----
class AutoCaptionRequest(BaseModel):
    threshold: float = 0.35
    inject_trigger: bool = True
    # "auto" = choose by base model style; or force "wd14" / "blip" / "florence2"
    method: str = "auto"
    # Character LoRA: drop body/face tags so those traits bake into the trigger.
    exclude_body_face: bool = False
    # Extra keywords to drop from WD14 tags (case-insensitive substring match).
    exclude_tags: list[str] = []
    # WD14 tagger model: "swinv2-v3" (default) | "eva02-large-v3" | raw HF repo id.
    wd14_model: str = "swinv2-v3"


class AutoCaptionResult(BaseModel):
    ok: bool
    method: str  # "wd14" | "blip" | "manual_fallback"
    captioned: int
    detail: str


# ---- Jobs ----
class JobCreate(BaseModel):
    name: str
    dataset_id: int
    base_model: Optional[str] = None
    backend: str = "local_mps"
    params: dict[str, Any] = {}


class JobUpdate(BaseModel):
    """Partial update for a not-yet-running job. All fields optional."""

    name: Optional[str] = None
    dataset_id: Optional[int] = None
    base_model: Optional[str] = None
    backend: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class JobRead(BaseModel):
    id: int
    name: str
    dataset_id: int
    base_model: str
    backend: str
    params: dict[str, Any]
    status: str
    progress: float
    current_step: int
    total_step: int
    latest_loss: Optional[float]
    error: Optional[str]
    created_at: datetime
    queued_at: Optional[datetime] = None
    finished_at: Optional[datetime]
    # True when a resumable kohya `-state` checkpoint exists on disk, so a
    # paused job can actually continue. Drives the pause-vs-stop UI.
    has_checkpoint: bool = False


# ---- Models ----
class LoraModelRead(BaseModel):
    id: int
    job_id: int
    name: str
    epoch: int
    base_model: str
    file_size: int
    created_at: datetime


# ---- Prompt library (提示词库) ----
class PromptCreate(BaseModel):
    category: str = "其他"
    zh: str
    en: str
    mutex_group: str = ""
    aliases: str = ""


class PromptUpdate(BaseModel):
    category: Optional[str] = None
    zh: Optional[str] = None
    en: Optional[str] = None
    mutex_group: Optional[str] = None
    aliases: Optional[str] = None


class PromptRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    category: str
    zh: str
    en: str
    mutex_group: str
    aliases: str
    created_at: datetime


class PromptSearchRequest(BaseModel):
    """查找一个中文词：命中词库则返回匹配项，未命中则走翻译兜底。"""

    query: str


class TranslatedPrompt(BaseModel):
    """翻译兜底产出的候选提示词（未收录进词库）。"""

    zh: str
    en: str
    source: str  # "dictionary" | "api" | "none"


class PromptSearchResult(BaseModel):
    query: str
    # 词库命中项（可能多个，如别名重叠）。
    matches: list[PromptRead] = []
    # 未命中时的翻译兜底结果（命中时为 None）。
    translated: Optional[TranslatedPrompt] = None


class MutexConflict(BaseModel):
    """一对互斥提示词。"""

    group: str
    a_zh: str
    a_en: str
    b_zh: str
    b_en: str


class MutexCheckRequest(BaseModel):
    """检查一组选中提示词是否存在互斥。传入 prompt id 列表。"""

    ids: list[int] = []
    # 也允许直接传英文提示词（组合场景下可能包含未入库的翻译结果）。
    extra_en: list[str] = []


class CombineRequest(BaseModel):
    """组合场景：选择若干提示词，产出中英文拼接串并检查互斥。"""

    ids: list[int]
    separator: str = ", "


class CombineResult(BaseModel):
    zh: str  # 中文组合串
    en: str  # 英文组合串
    conflicts: list[MutexConflict] = []
    items: list[PromptRead] = []


# ---- Remote hosts (cloud GPU) ----
class RemoteHostCreate(BaseModel):
    name: str
    host: str
    port: int = 22
    username: str = "root"
    auth_type: str = "key"  # "key" | "password"
    password: str = ""
    private_key_path: str = ""
    workdir: str = "~/loralab"
    kohya_dir: str = "~/sd-scripts"
    python_cmd: str = "python"
    base_models_dir: str = ""
    rvc_dir: str = "~/Retrieval-based-Voice-Conversion-WebUI"


class RemoteHostUpdate(BaseModel):
    name: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    auth_type: Optional[str] = None
    password: Optional[str] = None
    private_key_path: Optional[str] = None
    workdir: Optional[str] = None
    kohya_dir: Optional[str] = None
    python_cmd: Optional[str] = None
    base_models_dir: Optional[str] = None
    rvc_dir: Optional[str] = None


class RemoteHostRead(BaseModel):
    """Credentials are never returned; only whether a secret is set."""

    id: int
    name: str
    host: str
    port: int
    username: str
    auth_type: str
    has_password: bool
    private_key_path: str
    workdir: str
    kohya_dir: str
    python_cmd: str
    base_models_dir: str
    rvc_dir: str
    created_at: datetime


class RemoteTestResult(BaseModel):
    ok: bool
    detail: str


# ---- Voice / SVC (RVC) ----
class VoiceDatasetCreate(BaseModel):
    name: str
    speaker: str
    sample_rate: int = 40000


class VoiceDatasetUpdate(BaseModel):
    name: Optional[str] = None
    speaker: Optional[str] = None
    sample_rate: Optional[int] = None


class VoiceDatasetRead(BaseModel):
    id: int
    name: str
    speaker: str
    sample_rate: int
    clip_count: int
    total_seconds: float
    status: str
    created_at: datetime


class AudioClip(BaseModel):
    filename: str
    seconds: float
    size_bytes: int
    audio_url: str


class VoiceJobCreate(BaseModel):
    name: str
    dataset_id: int
    backend: str = "local_rvc"
    params: dict[str, Any] = {}


class VoiceJobRead(BaseModel):
    id: int
    name: str
    dataset_id: int
    backend: str
    params: dict[str, Any]
    status: str
    progress: float
    current_step: int
    total_step: int
    error: Optional[str]
    created_at: datetime
    finished_at: Optional[datetime]


class VoiceModelRead(BaseModel):
    id: int
    job_id: int
    name: str
    speaker: str
    epoch: int
    sample_rate: int
    has_index: bool
    file_size: int
    created_at: datetime


# ---- Image tools (微博图片管理) ----
class ImagePullRequest(BaseModel):
    """Launch a weibo album download. Exactly one of uid/album must be set."""

    uid: str = ""
    album: str = ""  # photo.weibo.com album URL
    workers: int = 6
    start: int = 1  # 1-based inclusive
    end: Optional[int] = None  # inclusive; None = to the end


class ImageFilterRequest(BaseModel):
    """Run the single-person filter over a downloaded directory (relative to
    the image-tools output dir)."""

    directory: str  # e.g. "uid_1234567890"
    recursive: bool = False
    dry_run: bool = False
    min_face: float = 0.5  # min face area %
    text_blocks: int = 5
    text_area: float = 5.0
    no_text_filter: bool = False
    no_animal_filter: bool = False
    no_quality_filter: bool = False  # 关闭 LoRA 训练质量筛选(不细分 single_lowq/)


class ImageSelectRequest(BaseModel):
    """从某目录的 single/ 里精选最适合 LoRA 训练的 Top-N 拷到 single_best/。"""

    directory: str  # e.g. "uid_123" 或 "uid_123/single"
    count: int = 50  # 精选数量
    quality_weight: float = 0.6  # 质量 vs 多样性权重(0~1，越高越偏质量)
    no_diversity: bool = False  # 关闭多样性去重，纯按质量分取 Top-N


class ImageTaskRead(BaseModel):
    id: int
    kind: str  # "pull" | "filter" | "select"
    target: str
    out_dir: str
    params: dict[str, Any]
    status: str
    detail: str
    created_at: datetime
    finished_at: Optional[datetime]
    # Download progress (pull tasks): 0..1 fraction plus raw counts, parsed from
    # the downloader's log. 0 when no download has started.
    progress: float = 0.0
    done: int = 0
    total: int = 0


class ImageDirEntry(BaseModel):
    """A downloaded/filtered directory available for browsing or filtering."""

    name: str  # directory name under the image-tools out dir
    image_count: int
    # Per-category counts produced by the filter (single/multi/poster/...).
    categories: dict[str, int] = {}


class ImageCookieRead(BaseModel):
    present: bool
    length: int = 0
    # A masked preview (head/tail only) so the UI can show it without leaking
    # the full credential.
    preview: str = ""
    updated_at: Optional[datetime] = None
    # Whether the key login fields (SUB=/SUBP=) are present — a rough validity hint.
    looks_valid: bool = False


class ImageCookieUpdate(BaseModel):
    cookie: str


class ImagePreviewRequest(BaseModel):
    """Fetch the pid list (no download) for preview + selective download."""

    uid: str = ""
    album: str = ""
    start: int = 1
    end: Optional[int] = None


class ImagePreviewItem(BaseModel):
    pid: str
    thumb_url: str
    full_url: str


class ImagePreviewResult(BaseModel):
    out_dir_name: str
    uid: str = ""
    album_id: Optional[str] = None
    pids: list[ImagePreviewItem] = []


class ImagePullSelectedRequest(BaseModel):
    pids: list[str]
    out_dir_name: str
    workers: int = 6


# --------------------------------------------------------------------------- #
# 小红书（XHS）：博主主页全量
# --------------------------------------------------------------------------- #
class XhsPreviewRequest(BaseModel):
    """Fetch a 小红书 author's full image list (no download) for preview."""

    user: str  # profile URL or user_id
    max_notes: Optional[int] = None  # cap notes parsed (None = all)
    headed: bool = False  # pop a real browser window to solve captcha + paginate


class XhsPullRequest(BaseModel):
    """Download ALL images of a 小红书 author."""

    user: str
    workers: int = 6
    max_notes: Optional[int] = None
    headed: bool = False


class XhsPullSelectedRequest(BaseModel):
    """Download only the chosen 小红书 image ids for an author."""

    ids: list[str]
    user: str
    out_dir_name: str
    workers: int = 6


class ImageSettingsRead(BaseModel):
    """Effective download/pull/filter root dir + whether it's the default."""

    out_dir: str
    default_out_dir: str
    is_default: bool = True
    exists: bool = False


class ImageSettingsUpdate(BaseModel):
    # Empty/None resets to the default output dir.
    out_dir: Optional[str] = None


class ImageBrowseResult(BaseModel):
    """A directory listing for the settings folder picker."""

    path: str
    parent: Optional[str] = None
    dirs: list[str] = []
