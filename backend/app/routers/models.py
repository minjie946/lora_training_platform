"""Models router: list / download / delete produced LoRA files."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import session_dependency
from ..models import LoraModel
from ..schemas import LoraModelRead

router = APIRouter(prefix="/api/models", tags=["models"])


class BulkDeleteIn(BaseModel):
    ids: list[int]


@router.get("", response_model=list[LoraModelRead])
def list_models(
    job_id: int | None = None, session: Session = Depends(session_dependency)
):
    stmt = select(LoraModel).order_by(LoraModel.id.desc())
    if job_id is not None:
        stmt = stmt.where(LoraModel.job_id == job_id)
    return session.exec(stmt).all()


@router.get("/{model_id}/download")
def download_model(model_id: int, session: Session = Depends(session_dependency)):
    m = session.get(LoraModel, model_id)
    if not m:
        raise HTTPException(404, "模型不存在")
    p = Path(m.file_path)
    if not p.exists():
        raise HTTPException(404, "模型文件已丢失")
    return FileResponse(p, filename=m.name, media_type="application/octet-stream")


@router.delete("/{model_id}")
def delete_model(model_id: int, session: Session = Depends(session_dependency)):
    m = session.get(LoraModel, model_id)
    if not m:
        raise HTTPException(404, "模型不存在")
    _delete_one(session, m)
    session.commit()
    return {"ok": True}


@router.post("/bulk-delete")
def bulk_delete_models(
    payload: BulkDeleteIn, session: Session = Depends(session_dependency)
):
    if not payload.ids:
        return {"ok": True, "deleted": 0}
    models = session.exec(
        select(LoraModel).where(LoraModel.id.in_(payload.ids))
    ).all()
    for m in models:
        _delete_one(session, m)
    session.commit()
    return {"ok": True, "deleted": len(models)}


def _delete_one(session: Session, m: LoraModel) -> None:
    """Remove the model DB row and its produced *.safetensors file on disk."""
    Path(m.file_path).unlink(missing_ok=True)
    session.delete(m)
