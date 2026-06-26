"""Dataset filesystem service: directory layout, image storage, thumbnails, captions."""
from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

# Enable HEIC/HEIF decoding (iPhone photos). Optional dependency; if missing,
# HEIC uploads simply fall back to the "unsupported format" error.
try:
    import pillow_heif  # type: ignore

    try:
        pillow_heif.register_heif_opener()
    except Exception:  # noqa: BLE001
        pass  # decoding still works via pillow_heif.open_heif()
    _HEIF_OK = True
except Exception:  # noqa: BLE001
    _HEIF_OK = False

from ..config import ALLOWED_IMAGE_EXTS, DATASETS_DIR, THUMBNAIL_SIZE

# HEIC/HEIF are accepted for input but transcoded to PNG on save (kohya can't
# read HEIC, and neither can the browser <img> preview).
_HEIC_EXTS = {".heic", ".heif"}

THUMB_DIRNAME = ".thumbnails"


def _safe_concept(concept: str) -> str:
    """kohya folder concept part: keep alnum, dash, underscore."""
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", concept.strip())
    return s or "concept"


def dataset_root(dataset_id: int) -> Path:
    return DATASETS_DIR / str(dataset_id)


def image_dir(dataset_id: int, repeat: int, concept: str) -> Path:
    """kohya convention: <root>/<repeat>_<concept>/ ."""
    return dataset_root(dataset_id) / f"{repeat}_{_safe_concept(concept)}"


def find_image_dir(dataset_id: int) -> Path | None:
    """Locate the existing N_concept subdir (there is exactly one)."""
    root = dataset_root(dataset_id)
    if not root.exists():
        return None
    for child in root.iterdir():
        if child.is_dir() and child.name != THUMB_DIRNAME and re.match(r"^\d+_", child.name):
            return child
    return None


def ensure_image_dir(dataset_id: int, repeat: int, concept: str) -> Path:
    target = image_dir(dataset_id, repeat, concept)
    existing = find_image_dir(dataset_id)
    if existing and existing != target:
        existing.rename(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / THUMB_DIRNAME).mkdir(exist_ok=True)
    return target


def _thumb_path(img_dir: Path, filename: str) -> Path:
    return img_dir / THUMB_DIRNAME / f"{Path(filename).stem}.jpg"


def _save_image_file(img_dir: Path, filename: str, data: bytes, caption: str = "") -> str:
    ext = Path(filename).suffix.lower()
    # Transcode HEIC/HEIF to PNG so kohya and the browser can read it.
    if ext in _HEIC_EXTS:
        if not _HEIF_OK:
            raise ValueError("未安装 HEIC 解码支持（pillow-heif），无法导入 HEIC")
        try:
            # Decode via pillow-heif directly (doesn't rely on the PIL opener
            # being registered in this process). Fall back to Image.open.
            try:
                heif = pillow_heif.open_heif(data, convert_hdr_to_8bit=True)
                im = Image.frombytes(heif.mode, heif.size, heif.data)
            except Exception:  # noqa: BLE001
                im = Image.open(io.BytesIO(data))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            data = buf.getvalue()
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"HEIC 解码失败: {e}")
        ext = ".png"
    if ext not in ALLOWED_IMAGE_EXTS:
        raise ValueError(f"不支持的图片格式: {ext}")
    # Sanitize filename, avoid collisions.
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", Path(filename).stem) or "img"
    dest = img_dir / f"{stem}{ext}"
    idx = 1
    while dest.exists():
        dest = img_dir / f"{stem}_{idx}{ext}"
        idx += 1
    dest.write_bytes(data)
    _make_thumbnail(img_dir, dest)
    # Create an empty / provided caption file alongside.
    caption_file = dest.with_suffix(".txt")
    caption_file.write_text(caption.strip(), encoding="utf-8")
    return dest.name


def save_image(dataset_id: int, repeat: int, concept: str, filename: str, data: bytes) -> str:
    img_dir = ensure_image_dir(dataset_id, repeat, concept)
    return _save_image_file(img_dir, filename, data)


def _make_thumbnail(img_dir: Path, image_path: Path) -> None:
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail(THUMBNAIL_SIZE)
            im.save(_thumb_path(img_dir, image_path.name), "JPEG", quality=80)
    except Exception:  # noqa: BLE001
        pass  # thumbnail is best-effort


def list_images(dataset_id: int) -> list[dict]:
    from . import caption_service as cap  # avoid import cycle at module load

    img_dir = find_image_dir(dataset_id)
    if not img_dir:
        return []
    items: list[dict] = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in ALLOWED_IMAGE_EXTS:
            continue
        caption_file = p.with_suffix(".txt")
        caption = caption_file.read_text(encoding="utf-8") if caption_file.exists() else ""
        items.append(
            {
                "filename": p.name,
                "caption": caption,
                "thumbnail_url": f"/api/datasets/{dataset_id}/images/{p.name}/thumbnail",
                "image_url": f"/api/datasets/{dataset_id}/images/{p.name}/raw",
                # WD14 per-tag confidence scores (None unless WD14 was run).
                "tag_scores": cap.read_wdtags(p),
            }
        )
    return items


def count_images(dataset_id: int) -> int:
    img_dir = find_image_dir(dataset_id)
    if not img_dir:
        return 0
    return sum(1 for p in img_dir.iterdir() if p.suffix.lower() in ALLOWED_IMAGE_EXTS)


def _archive_key(name: str) -> str | None:
    """Normalize a zip member name to a safe, comparable relative stem key."""
    raw = name.replace("\\", "/").strip("/")
    if not raw:
        return None
    p = PurePosixPath(raw)
    if any(part in ("", ".", "..") for part in p.parts):
        return None
    if "__MACOSX" in p.parts or p.name.startswith("._"):
        return None
    if p.suffix == "":
        return None
    parts = list(p.parts)
    parts[-1] = Path(parts[-1]).stem
    return "/".join(parts).lower()


def import_labeled_zip(
    dataset_id: int, repeat: int, concept: str, archive_name: str, data: bytes
) -> dict[str, int]:
    """Import a pre-captioned dataset from a zip archive.

    Expected archive contents:
      - images: png/jpg/jpeg/webp/bmp
      - captions: txt with the same relative path stem as the image
    """
    if not (archive_name or "").lower().endswith(".zip"):
        raise ValueError("目前仅支持导入 .zip 压缩包")
    buf = io.BytesIO(data)
    if not zipfile.is_zipfile(buf):
        raise ValueError("压缩包无效，请上传标准 .zip 文件")
    buf.seek(0)
    img_dir = ensure_image_dir(dataset_id, repeat, concept)
    imported = 0
    captioned = 0
    with zipfile.ZipFile(buf) as zf:
        captions: dict[str, str] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            key = _archive_key(info.filename)
            if not key or Path(info.filename).suffix.lower() != ".txt":
                continue
            try:
                captions[key] = zf.read(info).decode("utf-8-sig", "replace").strip()
            except Exception:  # noqa: BLE001
                captions[key] = ""

        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in ALLOWED_IMAGE_EXTS and ext not in _HEIC_EXTS:
                continue
            key = _archive_key(info.filename)
            if not key:
                continue
            saved = _save_image_file(
                img_dir,
                Path(info.filename).name,
                zf.read(info),
                caption=captions.get(key, ""),
            )
            imported += 1
            if read_caption(dataset_id, saved).strip():
                captioned += 1

    if imported == 0:
        raise ValueError("压缩包内未找到可导入的图片文件")
    return {"imported": imported, "captioned": captioned}


def get_image_path(dataset_id: int, filename: str) -> Path | None:
    img_dir = find_image_dir(dataset_id)
    if not img_dir:
        return None
    p = (img_dir / filename).resolve()
    if not str(p).startswith(str(img_dir.resolve())) or not p.exists():
        return None
    return p


def get_thumbnail_path(dataset_id: int, filename: str) -> Path | None:
    img_dir = find_image_dir(dataset_id)
    if not img_dir:
        return None
    tp = _thumb_path(img_dir, filename)
    if tp.exists():
        return tp
    # fall back to original if thumbnail missing
    return get_image_path(dataset_id, filename)


def read_caption(dataset_id: int, filename: str) -> str:
    p = get_image_path(dataset_id, filename)
    if not p:
        return ""
    cf = p.with_suffix(".txt")
    return cf.read_text(encoding="utf-8") if cf.exists() else ""


def write_caption(dataset_id: int, filename: str, caption: str) -> bool:
    p = get_image_path(dataset_id, filename)
    if not p:
        return False
    p.with_suffix(".txt").write_text(caption.strip(), encoding="utf-8")
    return True


def delete_image(dataset_id: int, filename: str) -> bool:
    from . import caption_service as cap  # avoid import cycle at module load

    p = get_image_path(dataset_id, filename)
    if not p:
        return False
    img_dir = p.parent
    p.unlink(missing_ok=True)
    p.with_suffix(".txt").unlink(missing_ok=True)
    cap.delete_wdtags(p)
    _thumb_path(img_dir, filename).unlink(missing_ok=True)
    return True


def delete_dataset_files(dataset_id: int) -> None:
    root = dataset_root(dataset_id)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
