"""Caption service: style-aware auto-tagging + trigger-word injection.

Captioning strategy is chosen by the dataset's base model style:
- anime/illustration models -> WD14 booru-style tags (wdtagger)
- realistic/photo models     -> BLIP natural-language caption (transformers)

Both backends are optional and detected dynamically; if neither is available we
degrade to a manual workflow (trigger-word injection only).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from ..config import ALLOWED_IMAGE_EXTS
from . import base_model_service
from . import dataset_service as ds


# Hosts that must bypass any local MITM proxy (e.g. the corp proxy at
# 127.0.0.1:8899) when downloading model weights. That proxy presents a
# self-signed cert Python's OpenSSL won't trust, so HF downloads fail with
# CERTIFICATE_VERIFY_FAILED even though the sites are directly reachable.
_HF_NO_PROXY_HOSTS = (
    "huggingface.co",
    "hf.co",
    ".hf.co",
    "cdn-lfs.huggingface.co",
    ".xethub.hf.co",
    "cas-bridge.xethub.hf.co",
)


def _ensure_hf_direct() -> None:
    """Append HF hosts to NO_PROXY so model downloads skip the local proxy.

    Idempotent; merges with any existing NO_PROXY value.
    """
    for var in ("NO_PROXY", "no_proxy"):
        existing = [h for h in os.environ.get(var, "").split(",") if h.strip()]
        merged = list(dict.fromkeys(existing + list(_HF_NO_PROXY_HOSTS)))
        os.environ[var] = ",".join(merged)


def _split_tags(caption: str) -> list[str]:
    return [t.strip() for t in caption.split(",") if t.strip()]


# Body / face / figure related tags. When training a *character* LoRA and you
# want these traits baked into the trigger word (so they render consistently),
# these should be REMOVED from captions rather than tagged. Matched loosely as
# substrings against lower-cased, space-normalized WD14 tags.
BODY_FACE_KEYWORDS = [
    # figure / body shape
    "breasts", "flat chest", "cleavage", "curvy", "slim", "petite", "plump",
    "thick", "muscular", "toned", "abs", "wide hips", "hips", "thighs",
    "thigh gap", "waist", "narrow waist", "butt", "ass", "large breasts",
    "medium breasts", "small breasts", "huge breasts", "body", "figure",
    "tall", "short", "slender", "chubby", "skinny", "fit",
    # face / head shape & features (identity-defining)
    "face", "facial", "jaw", "cheek", "chin", "nose", "lips", "eyes",
    "eye", "eyebrows", "forehead", "freckles", "mole", "skin",
    "round face", "oval face", "sharp features", "double eyelid",
]


def _filter_pairs(
    pairs: list[tuple[str, float]], excludes: list[str]
) -> list[tuple[str, float]]:
    """Drop (tag, confidence) pairs whose tag matches any exclude keyword
    (case-insensitive substring)."""
    if not excludes:
        return pairs
    lows = [e.strip().lower() for e in excludes if e.strip()]
    return [(t, c) for (t, c) in pairs if not any(e in t.strip().lower() for e in lows)]


# Sidecar file holding per-tag WD14 confidence scores for the UI. It sits next
# to the image (like the .txt) but is NEVER fed to training — kohya only reads
# the .txt, so scores can't pollute captions.
WDTAGS_SUFFIX = ".wdtags.json"


def _wdtags_path(image_path: Path) -> Path:
    # Build from stem (not with_suffix, which rejects multi-dot suffixes on
    # some Python versions): "img.png" -> "img.wdtags.json".
    return image_path.parent / f"{image_path.stem}{WDTAGS_SUFFIX}"


def _write_wdtags(image_path: Path, pairs: list[tuple[str, float]], threshold: float) -> None:
    payload = {
        "threshold": threshold,
        "tags": [{"tag": t, "confidence": round(c, 4)} for t, c in pairs],
    }
    _wdtags_path(image_path).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def read_wdtags(image_path: Path) -> list[dict] | None:
    """Return the stored WD14 [{tag, confidence}] list for an image, or None.

    The frontend correlates these against the current .txt caption so scores
    stay meaningful even after manual edits.
    """
    sp = _wdtags_path(image_path)
    if not sp.exists():
        return None
    try:
        payload = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    tags = payload.get("tags")
    if not isinstance(tags, list):
        return None
    return tags


def delete_wdtags(image_path: Path) -> None:
    _wdtags_path(image_path).unlink(missing_ok=True)


def _join_tags(tags: list[str]) -> str:
    # de-duplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return ", ".join(out)


def inject_trigger(caption: str, trigger: str) -> str:
    """Ensure the trigger word is the first tag."""
    trigger = trigger.strip()
    if not trigger:
        return caption
    tags = [t for t in _split_tags(caption) if t.lower() != trigger.lower()]
    return _join_tags([trigger, *tags])


# Selectable WD14 tagger models (all timm `hf-hub:` repos, loaded by wdtagger).
# swinv2-v3 is wdtagger's default: fast, good accuracy — kept as our default.
# eva02-large-v3 is the most accurate of the v3 family but much heavier (runs on
# CPU here since wdtagger has no MPS path), so it's opt-in.
WD14_MODELS = {
    "swinv2-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "eva02-large-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
}
DEFAULT_WD14_MODEL = "swinv2-v3"


def _resolve_wd14_repo(wd14_model: str) -> str:
    """Map a short key ('eva02-large-v3') or a raw HF repo id to a repo id."""
    key = (wd14_model or "").strip()
    if not key:
        return WD14_MODELS[DEFAULT_WD14_MODEL]
    if key in WD14_MODELS:
        return WD14_MODELS[key]
    # Allow passing a full repo id directly (forward-compatible with new models).
    return key


# Florence-2 captioner: microsoft's lightweight VLM. The "large" variant gives
# noticeably richer captions than BLIP while staying manageable on-device.
FLORENCE2_MODEL = "microsoft/Florence-2-large"
# Task prompt controlling verbosity: <CAPTION> | <DETAILED_CAPTION> | <MORE_DETAILED_CAPTION>.
FLORENCE2_TASK = "<DETAILED_CAPTION>"


def wd14_available() -> bool:
    return importlib.util.find_spec("wdtagger") is not None


def blip_available() -> bool:
    return (
        importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("torch") is not None
    )


def florence2_available() -> bool:
    # Florence-2 loads with trust_remote_code and needs einops + timm at runtime.
    return (
        importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("einops") is not None
        and importlib.util.find_spec("timm") is not None
    )


def _wd14_tag_image(tagger, image_path: Path, threshold: float) -> list[tuple[str, float]]:
    """Run WD14 on a single image, returning (tag, confidence) pairs.

    Confidence is kept so the UI can flag low-certainty tags for review. When a
    backend/version doesn't expose scores, confidence falls back to 1.0.
    Tolerates several wdtagger API shapes.
    """
    from PIL import Image  # local import, pillow is a hard dep

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        # wdtagger >=0.16 uses general_threshold/character_threshold;
        # older versions used threshold=. Try the new signature, then fall back.
        try:
            result = tagger.tag(im, general_threshold=threshold)
        except TypeError:
            result = tagger.tag(im, threshold=threshold)
    # Result exposes .character_tags / .general_tags (tuples) in new versions,
    # or dicts / a plain dict in older ones.
    pairs: list[tuple[str, float]] = []
    for attr in ("character_tags", "general_tags"):
        val = getattr(result, attr, None)
        if isinstance(val, dict):
            pairs.extend((k, float(v)) for k, v in val.items())
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs.append((str(item[0]), float(item[1])))
                else:
                    pairs.append((str(item), 1.0))
    if not pairs and isinstance(result, dict):
        pairs = [(k, float(v)) for k, v in result.items()]
    return [(t.replace("_", " "), c) for t, c in pairs]


_blip_cache: dict = {}


def _get_blip():
    """Lazily load (and cache) the BLIP captioning model + processor."""
    if "model" in _blip_cache:
        return _blip_cache["processor"], _blip_cache["model"]
    _ensure_hf_direct()  # bypass local MITM proxy for the model download
    import torch  # type: ignore
    from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore

    name = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    _blip_cache.update({"processor": processor, "model": model, "device": device})
    return processor, model


def _blip_caption_image(processor, model, image_path: Path) -> str:
    """Generate a natural-language caption for one image with BLIP."""
    import torch  # type: ignore
    from PIL import Image

    device = _blip_cache.get("device", "cpu")
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        inputs = processor(im, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def _run_wd14(
    images, threshold, do_inject, trigger, excludes=None, wd14_model="",
) -> tuple[str, int, str]:
    import wdtagger  # type: ignore

    _ensure_hf_direct()  # bypass local MITM proxy for the model download
    repo = _resolve_wd14_repo(wd14_model)
    tagger = wdtagger.Tagger(model_repo=repo)
    excludes = excludes or []
    count = 0
    removed_total = 0
    for p in images:
        pairs = _wd14_tag_image(tagger, p, threshold)
        if excludes:
            before = len(pairs)
            pairs = _filter_pairs(pairs, excludes)
            removed_total += before - len(pairs)
        tags = [t for t, _ in pairs]
        caption = _join_tags(tags)
        if do_inject:
            caption = inject_trigger(caption, trigger)
        # Clean tags go to the .txt (training input); confidence scores go to a
        # sidecar for the UI only.
        p.with_suffix(".txt").write_text(caption, encoding="utf-8")
        _write_wdtags(p, pairs, threshold)
        count += 1
    extra = f"，已排除 {removed_total} 处身材/脸型等标签（烘焙进触发词）" if excludes else ""
    model_note = f"（{repo.split('/')[-1]}）"
    return "wd14", count, f"WD14 标签打标完成{model_note}，共 {count} 张{extra}"


def _run_blip(images, do_inject, trigger) -> tuple[str, int, str]:
    processor, model = _get_blip()
    count = 0
    for p in images:
        caption = _blip_caption_image(processor, model, p)
        if do_inject:
            caption = inject_trigger(caption, trigger)
        p.with_suffix(".txt").write_text(caption, encoding="utf-8")
        count += 1
    return "blip", count, f"BLIP 自然语言描述完成（写实风格），共 {count} 张"


_florence_cache: dict = {}


def _get_florence2():
    """Lazily load (and cache) the Florence-2 model + processor."""
    if "model" in _florence_cache:
        return _florence_cache["processor"], _florence_cache["model"]
    _ensure_hf_direct()  # bypass local MITM proxy for the model download
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Florence-2 ships custom modeling code -> trust_remote_code. Use fp32 on
    # mps/cpu (fp16 is unstable outside CUDA). Force eager attention: the model's
    # remote code predates newer transformers' sdpa dispatch and otherwise trips
    # on a missing `_supports_sdpa` attribute (transformers>=4.5x).
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE2_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL, trust_remote_code=True)
    _florence_cache.update({"processor": processor, "model": model, "device": device})
    return processor, model


def _florence2_caption_image(processor, model, image_path: Path) -> str:
    """Generate a detailed natural-language caption for one image with Florence-2."""
    import torch  # type: ignore
    from PIL import Image

    device = _florence_cache.get("device", "cpu")
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        inputs = processor(text=FLORENCE2_TASK, images=im, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
                # Florence-2's remote code indexes the legacy past_key_values tuple
                # shape, which breaks against transformers>=4.5x's Cache class.
                # Disabling the KV cache sidesteps that path (slower but correct).
                use_cache=False,
            )
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            text, task=FLORENCE2_TASK, image_size=(im.width, im.height)
        )
    # post_process_generation returns {task: caption}; fall back to raw text.
    caption = parsed.get(FLORENCE2_TASK) if isinstance(parsed, dict) else None
    return str(caption or text).strip()


def _run_florence2(images, do_inject, trigger) -> tuple[str, int, str]:
    processor, model = _get_florence2()
    count = 0
    for p in images:
        caption = _florence2_caption_image(processor, model, p)
        if do_inject:
            caption = inject_trigger(caption, trigger)
        p.with_suffix(".txt").write_text(caption, encoding="utf-8")
        count += 1
    return "florence2", count, f"Florence-2 自然语言描述完成（写实风格），共 {count} 张"


def _manual_fallback(images, do_inject, trigger, detail_prefix) -> tuple[str, int, str]:
    count = 0
    if do_inject and trigger.strip():
        for p in images:
            cf = p.with_suffix(".txt")
            existing = cf.read_text(encoding="utf-8") if cf.exists() else ""
            cf.write_text(inject_trigger(existing, trigger), encoding="utf-8")
            count += 1
    return "manual_fallback", count, detail_prefix


def auto_caption(
    dataset_id: int,
    threshold: float,
    do_inject: bool,
    trigger: str,
    base_model: str = "",
    method: str = "auto",
    exclude_body_face: bool = False,
    exclude_tags: list[str] | None = None,
    wd14_model: str = "",
) -> tuple[str, int, str]:
    """Caption a dataset.

    method:
      - "auto": choose strategy by the base model's style (default)
      - "wd14": force WD14 booru tags (works for anime AND realistic)
      - "florence2": force Florence-2 natural-language captions (realistic)
      - "blip": force BLIP natural-language captions

    exclude_body_face: drop built-in body/face tags so those identity traits
      bake into the trigger word (recommended for character LoRA). WD14 only.
    exclude_tags: extra user-supplied keywords to drop (WD14 only).
    wd14_model: which WD14 tagger to use ("swinv2-v3" default | "eva02-large-v3"
      | a raw HF repo id). WD14 only.

    Returns (method, captioned_count, detail).
    """
    img_dir = ds.find_image_dir(dataset_id)
    if not img_dir:
        return "manual_fallback", 0, "数据集没有图片目录"

    images = [
        p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in ALLOWED_IMAGE_EXTS
    ]
    if not images:
        return "manual_fallback", 0, "数据集没有图片"

    # Build the exclude list (WD14 only): built-in body/face set + user extras.
    excludes: list[str] = []
    if exclude_body_face:
        excludes.extend(BODY_FACE_KEYWORDS)
    if exclude_tags:
        excludes.extend(exclude_tags)

    # Resolve the effective captioner: explicit method wins, else by style.
    # For realistic auto, prefer Florence-2 when available (richer than BLIP).
    chosen = method if method in ("wd14", "blip", "florence2") else None
    if chosen is None:
        style = base_model_service.style_of(base_model)
        if style == "realistic":
            chosen = "florence2" if florence2_available() else "blip"
        else:
            chosen = "wd14"

    if chosen == "florence2":
        if florence2_available():
            try:
                return _run_florence2(images, do_inject, trigger)
            except Exception as e:  # noqa: BLE001
                return _manual_fallback(
                    images, do_inject, trigger, f"Florence-2 运行失败({e})，降级为手动"
                )
        return _manual_fallback(
            images, do_inject, trigger,
            "未安装 Florence-2 依赖（transformers/torch/einops/timm），降级为手动",
        )

    if chosen == "blip":
        if blip_available():
            try:
                return _run_blip(images, do_inject, trigger)
            except Exception as e:  # noqa: BLE001
                return _manual_fallback(
                    images, do_inject, trigger, f"BLIP 运行失败({e})，降级为手动"
                )
        return _manual_fallback(
            images, do_inject, trigger, "未安装 BLIP 依赖（transformers/torch），降级为手动"
        )

    # chosen == "wd14"
    if wd14_available():
        try:
            return _run_wd14(
                images, threshold, do_inject, trigger,
                excludes=excludes, wd14_model=wd14_model,
            )
        except Exception as e:  # noqa: BLE001
            return _manual_fallback(
                images, do_inject, trigger, f"WD14 运行失败({e})，降级为手动"
            )
    return _manual_fallback(
        images, do_inject, trigger, "未安装 WD14（wdtagger），降级为手动打标；仅注入触发词"
    )


def inject_trigger_all(dataset_id: int, trigger: str) -> int:
    img_dir = ds.find_image_dir(dataset_id)
    if not img_dir:
        return 0
    count = 0
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in ALLOWED_IMAGE_EXTS:
            continue
        cf = p.with_suffix(".txt")
        existing = cf.read_text(encoding="utf-8") if cf.exists() else ""
        cf.write_text(inject_trigger(existing, trigger), encoding="utf-8")
        count += 1
    return count
