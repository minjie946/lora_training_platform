"""Datasets router: CRUD, image upload/list/delete, caption read/write."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ..db import session_dependency
from ..models import Dataset
from ..schemas import (
    AutoCaptionRequest,
    CaptionUpdate,
    DatasetCreate,
    DatasetImportResult,
    DatasetRead,
    DatasetUpdate,
    ImageItem,
)
from ..services import caption_manager
from ..services import caption_service as cap
from ..services import dataset_service as ds

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _get_or_404(session: Session, dataset_id: int) -> Dataset:
    obj = session.get(Dataset, dataset_id)
    if not obj:
        raise HTTPException(404, "数据集不存在")
    return obj


@router.get("", response_model=list[DatasetRead])
def list_datasets(session: Session = Depends(session_dependency)):
    return session.exec(select(Dataset).order_by(Dataset.id.desc())).all()


@router.post("", response_model=DatasetRead)
def create_dataset(body: DatasetCreate, session: Session = Depends(session_dependency)):
    obj = Dataset(
        name=body.name,
        concept=body.concept,
        repeat=body.repeat,
        trigger_word=body.trigger_word,
        base_model=body.base_model or "",
    )
    session.add(obj)
    session.commit()
    session.refresh(obj)
    ds.ensure_image_dir(obj.id, obj.repeat, obj.concept)
    return obj


@router.post("/import", response_model=DatasetImportResult)
async def import_dataset(
    name: str = Form(...),
    concept: str = Form(...),
    repeat: int = Form(10),
    trigger_word: str = Form(""),
    base_model: str = Form(""),
    archive: UploadFile = File(...),
    session: Session = Depends(session_dependency),
):
    if not archive.filename:
        raise HTTPException(400, "请上传压缩包")
    obj = Dataset(
        name=name,
        concept=concept,
        repeat=repeat,
        trigger_word=trigger_word,
        base_model=base_model or "",
    )
    session.add(obj)
    session.commit()
    session.refresh(obj)
    try:
        data = await archive.read()
        stats = ds.import_labeled_zip(obj.id, obj.repeat, obj.concept, archive.filename, data)
    except ValueError as e:
        session.delete(obj)
        session.commit()
        ds.delete_dataset_files(obj.id)
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        session.delete(obj)
        session.commit()
        ds.delete_dataset_files(obj.id)
        raise HTTPException(500, f"导入失败: {e}")

    obj.image_count = stats["imported"]
    obj.status = "captioned" if stats["captioned"] > 0 else "ready"
    session.add(obj)
    session.commit()
    session.refresh(obj)
    detail = (
        f"已导入 {stats['imported']} 张图片"
        f"；其中 {stats['captioned']} 张检测到同名 .txt 标注"
    )
    dataset_read = DatasetRead(
        id=obj.id,
        name=obj.name,
        concept=obj.concept,
        repeat=obj.repeat,
        trigger_word=obj.trigger_word,
        base_model=obj.base_model,
        image_count=obj.image_count,
        status=obj.status,
        created_at=obj.created_at,
    )
    return DatasetImportResult(
        dataset=dataset_read,
        imported=stats["imported"],
        captioned=stats["captioned"],
        detail=detail,
    )


@router.get("/{dataset_id}", response_model=DatasetRead)
def get_dataset(dataset_id: int, session: Session = Depends(session_dependency)):
    return _get_or_404(session, dataset_id)


@router.patch("/{dataset_id}", response_model=DatasetRead)
def update_dataset(
    dataset_id: int, body: DatasetUpdate, session: Session = Depends(session_dependency)
):
    obj = _get_or_404(session, dataset_id)
    data = body.model_dump(exclude_unset=True)
    repeat_or_concept_changed = "repeat" in data or "concept" in data
    for k, v in data.items():
        setattr(obj, k, v)
    if repeat_or_concept_changed:
        ds.ensure_image_dir(obj.id, obj.repeat, obj.concept)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return obj


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, session: Session = Depends(session_dependency)):
    obj = _get_or_404(session, dataset_id)
    session.delete(obj)
    session.commit()
    ds.delete_dataset_files(dataset_id)
    return {"ok": True}


# ---- Images ----
@router.get("/{dataset_id}/images", response_model=list[ImageItem])
def list_images(dataset_id: int, session: Session = Depends(session_dependency)):
    _get_or_404(session, dataset_id)
    return ds.list_images(dataset_id)


@router.post("/{dataset_id}/images", response_model=list[ImageItem])
async def upload_images(
    dataset_id: int,
    files: list[UploadFile] = File(...),
    session: Session = Depends(session_dependency),
):
    obj = _get_or_404(session, dataset_id)
    for f in files:
        data = await f.read()
        try:
            ds.save_image(dataset_id, obj.repeat, obj.concept, f.filename, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
    obj.image_count = ds.count_images(dataset_id)
    session.add(obj)
    session.commit()
    return ds.list_images(dataset_id)


@router.get("/{dataset_id}/images/{filename}/thumbnail")
def get_thumbnail(dataset_id: int, filename: str):
    p = ds.get_thumbnail_path(dataset_id, filename)
    if not p:
        raise HTTPException(404, "图片不存在")
    return FileResponse(p)


@router.get("/{dataset_id}/images/{filename}/raw")
def get_raw_image(dataset_id: int, filename: str):
    p = ds.get_image_path(dataset_id, filename)
    if not p:
        raise HTTPException(404, "图片不存在")
    return FileResponse(p)


@router.delete("/{dataset_id}/images/{filename}")
def delete_image(dataset_id: int, filename: str, session: Session = Depends(session_dependency)):
    obj = _get_or_404(session, dataset_id)
    if not ds.delete_image(dataset_id, filename):
        raise HTTPException(404, "图片不存在")
    obj.image_count = ds.count_images(dataset_id)
    session.add(obj)
    session.commit()
    return {"ok": True}


# ---- Captions ----
@router.put("/{dataset_id}/captions")
def update_caption(
    dataset_id: int, body: CaptionUpdate, session: Session = Depends(session_dependency)
):
    _get_or_404(session, dataset_id)
    if not ds.write_caption(dataset_id, body.filename, body.caption):
        raise HTTPException(404, "图片不存在")
    return {"ok": True}


@router.post("/{dataset_id}/auto-caption")
def auto_caption(
    dataset_id: int,
    body: AutoCaptionRequest,
    session: Session = Depends(session_dependency),
):
    obj = _get_or_404(session, dataset_id)
    try:
        caption_manager.start_auto_caption(
            dataset_id,
            threshold=body.threshold,
            do_inject=body.inject_trigger,
            trigger=obj.trigger_word,
            base_model=obj.base_model,
            method=body.method,
            exclude_body_face=body.exclude_body_face,
            exclude_tags=body.exclude_tags,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"ok": True, "caption_status": "running", "detail": "打标已在后台开始"}


@router.get("/{dataset_id}/caption-status")
def caption_status(dataset_id: int, session: Session = Depends(session_dependency)):
    obj = _get_or_404(session, dataset_id)
    return {
        "dataset_id": dataset_id,
        "caption_status": obj.caption_status,
        "detail": obj.caption_detail,
        "status": obj.status,
    }


@router.post("/{dataset_id}/inject-trigger")
def inject_trigger(dataset_id: int, session: Session = Depends(session_dependency)):
    obj = _get_or_404(session, dataset_id)
    count = cap.inject_trigger_all(dataset_id, obj.trigger_word)
    return {"ok": True, "updated": count}
