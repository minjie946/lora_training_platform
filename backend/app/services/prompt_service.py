"""提示词库业务逻辑：查找 / 翻译兜底 / 互斥检查 / 组合 / 预置种子数据。

设计要点
--------
* 查找：先按中文名 / 别名精确或包含匹配词库；命中即返回。未命中时走
  "本地词典优先 + 可选在线 API" 的翻译兜底（见 `translate_zh`）。
* 互斥：两个提示词若共享同一个非空 `mutex_group`（如 "hair_color"），则互斥。
  组合 / 检查流程据此标记冲突对。
* 种子：首次启动（词库为空）时预置一批常用 Danbooru 风格标签，开箱即用。

翻译兜底不依赖任何第三方库：本地词典命中直接返回；否则若配置了
`PROMPT_TRANSLATE_URL`（+ 可选 Key），用标准库 urllib 调一次通用 HTTP 接口，
失败或未配置则返回 source="none"，前端提示"未收录，请手动补充"。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from sqlmodel import Session, select

from ..models import Prompt

# --------------------------------------------------------------------------- #
# 本地中文 -> 英文词典（翻译兜底第一优先级，命中率高、零延迟、离线可用）。
# 只需覆盖常见画面描述词；未覆盖的再走在线 API。
# --------------------------------------------------------------------------- #
LOCAL_DICT: dict[str, str] = {
    "微笑": "smile",
    "大笑": "laughing",
    "长发": "long hair",
    "短发": "short hair",
    "黑发": "black hair",
    "金发": "blonde hair",
    "白发": "white hair",
    "红发": "red hair",
    "棕发": "brown hair",
    "双马尾": "twintails",
    "马尾": "ponytail",
    "刘海": "bangs",
    "红眼": "red eyes",
    "蓝眼": "blue eyes",
    "绿眼": "green eyes",
    "站立": "standing",
    "坐着": "sitting",
    "躺着": "lying",
    "全身": "full body",
    "半身": "upper body",
    "特写": "close-up",
    "正面": "front view",
    "侧面": "from side",
    "背面": "from behind",
    "俯视": "from above",
    "仰视": "from below",
    "白天": "daytime",
    "夜晚": "night",
    "室内": "indoors",
    "室外": "outdoors",
    "校服": "school uniform",
    "和服": "kimono",
    "连衣裙": "dress",
    "泳装": "swimsuit",
    "衬衫": "shirt",
    "裙子": "skirt",
    "眼镜": "glasses",
    "帽子": "hat",
    "杰作": "masterpiece",
    "高质量": "best quality",
    "高细节": "highly detailed",
    "景深": "depth of field",
    "简单背景": "simple background",
    "白色背景": "white background",
}

# 预置种子：(分类, 中文, 英文, 互斥组, 别名)
SEED_PROMPTS: list[tuple[str, str, str, str, str]] = [
    # 质量
    ("质量", "杰作", "masterpiece", "", ""),
    ("质量", "最高质量", "best quality", "quality", "高质量"),
    ("质量", "普通质量", "normal quality", "quality", ""),
    ("质量", "高细节", "highly detailed", "", "精细"),
    # 发型（互斥：一个人只有一种发长）
    ("发型", "长发", "long hair", "hair_length", ""),
    ("发型", "短发", "short hair", "hair_length", ""),
    ("发型", "双马尾", "twintails", "", ""),
    ("发型", "刘海", "bangs", "", ""),
    # 发色（互斥）
    ("发色", "黑发", "black hair", "hair_color", ""),
    ("发色", "金发", "blonde hair", "hair_color", "金色头发"),
    ("发色", "白发", "white hair", "hair_color", ""),
    ("发色", "红发", "red hair", "hair_color", ""),
    ("发色", "棕发", "brown hair", "hair_color", ""),
    # 瞳色（互斥）
    ("瞳色", "红眼", "red eyes", "eye_color", ""),
    ("瞳色", "蓝眼", "blue eyes", "eye_color", ""),
    ("瞳色", "绿眼", "green eyes", "eye_color", ""),
    # 表情
    ("表情", "微笑", "smile", "", ""),
    ("表情", "大笑", "laughing", "", ""),
    ("表情", "面无表情", "expressionless", "", ""),
    # 姿态（互斥：站/坐/躺）
    ("姿态", "站立", "standing", "pose", ""),
    ("姿态", "坐着", "sitting", "pose", ""),
    ("姿态", "躺着", "lying", "pose", ""),
    # 构图（互斥：取景范围）
    ("构图", "全身", "full body", "framing", ""),
    ("构图", "半身", "upper body", "framing", ""),
    ("构图", "特写", "close-up", "framing", ""),
    # 视角（互斥）
    ("视角", "正面", "front view", "angle", ""),
    ("视角", "侧面", "from side", "angle", ""),
    ("视角", "背面", "from behind", "angle", ""),
    ("视角", "俯视", "from above", "angle", ""),
    ("视角", "仰视", "from below", "angle", ""),
    # 服装
    ("服装", "校服", "school uniform", "", ""),
    ("服装", "和服", "kimono", "", ""),
    ("服装", "连衣裙", "dress", "", ""),
    ("服装", "泳装", "swimsuit", "", ""),
    # 场景（互斥：室内/室外）
    ("场景", "室内", "indoors", "location", ""),
    ("场景", "室外", "outdoors", "location", ""),
    # 光照 / 时间（互斥）
    ("时间", "白天", "daytime", "time", ""),
    ("时间", "夜晚", "night", "time", ""),
    # 背景
    ("背景", "简单背景", "simple background", "", ""),
    ("背景", "白色背景", "white background", "", ""),
    ("背景", "景深", "depth of field", "", "虚化"),
]


def seed_if_empty(session: Session) -> int:
    """词库为空时写入种子数据。返回写入条数（已有数据则返回 0）。"""
    existing = session.exec(select(Prompt.id).limit(1)).first()
    if existing is not None:
        return 0
    for category, zh, en, group, aliases in SEED_PROMPTS:
        session.add(
            Prompt(category=category, zh=zh, en=en, mutex_group=group, aliases=aliases)
        )
    session.commit()
    return len(SEED_PROMPTS)


def reconcile_on_startup() -> None:
    """启动钩子：确保预置种子存在（首次运行开箱即用）。"""
    from ..db import get_session

    with get_session() as session:
        seed_if_empty(session)


# --------------------------------------------------------------------------- #
# 翻译兜底
# --------------------------------------------------------------------------- #
def translate_zh(text: str) -> tuple[str, str]:
    """把一个中文词译成英文提示词。

    返回 (english, source)，source ∈ {"dictionary", "api", "none"}。
    优先本地词典；其次可选在线 API；都没有则返回 ("", "none")。
    """
    key = text.strip()
    if not key:
        return "", "none"
    if key in LOCAL_DICT:
        return LOCAL_DICT[key], "dictionary"

    api_url = os.environ.get("PROMPT_TRANSLATE_URL", "").strip()
    if not api_url:
        return "", "none"
    try:
        return _call_translate_api(api_url, key), "api"
    except Exception:
        # 网络 / 解析失败时静默降级，不阻塞查找主流程。
        return "", "none"


def _call_translate_api(api_url: str, text: str) -> str:
    """调用一个通用「中->英」翻译 HTTP 接口。

    约定：POST JSON {"q": <中文>, "source": "zh", "target": "en"}，
    返回 JSON 中含 "translatedText" 或 "text" 字段（兼容 LibreTranslate 等）。
    通过 PROMPT_TRANSLATE_KEY 环境变量可选注入 api_key。
    """
    payload: dict[str, str] = {"q": text, "source": "zh", "target": "en", "format": "text"}
    api_key = os.environ.get("PROMPT_TRANSLATE_KEY", "").strip()
    if api_key:
        payload["api_key"] = api_key
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    en = data.get("translatedText") or data.get("text") or ""
    return str(en).strip()


# --------------------------------------------------------------------------- #
# 查找
# --------------------------------------------------------------------------- #
def search(session: Session, query: str) -> tuple[list[Prompt], Optional[tuple[str, str]]]:
    """按中文查找词库。返回 (命中项列表, 翻译兜底)。

    命中时翻译兜底为 None；未命中时命中项为空，翻译兜底为 (english, source)。
    """
    q = query.strip()
    if not q:
        return [], None

    rows = session.exec(select(Prompt)).all()
    exact: list[Prompt] = []
    partial: list[Prompt] = []
    for p in rows:
        alias_list = [a.strip() for a in (p.aliases or "").split(",") if a.strip()]
        names = [p.zh, *alias_list]
        if q in names or q == p.en:
            exact.append(p)
        elif any(q in n or n in q for n in names if n) or (p.en and q.lower() in p.en.lower()):
            partial.append(p)

    matches = exact or partial
    if matches:
        return matches, None
    en, source = translate_zh(q)
    return [], (en, source)


# --------------------------------------------------------------------------- #
# 互斥检查
# --------------------------------------------------------------------------- #
def find_conflicts(prompts: list[Prompt]) -> list[dict]:
    """在一组提示词中找出互斥冲突对。

    同一个非空 mutex_group 内若出现 >=2 个提示词，两两即为冲突。
    """
    by_group: dict[str, list[Prompt]] = {}
    for p in prompts:
        g = (p.mutex_group or "").strip()
        if not g:
            continue
        by_group.setdefault(g, []).append(p)

    conflicts: list[dict] = []
    for group, items in by_group.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                conflicts.append(
                    {
                        "group": group,
                        "a_zh": a.zh,
                        "a_en": a.en,
                        "b_zh": b.zh,
                        "b_en": b.en,
                    }
                )
    return conflicts
