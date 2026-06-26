"""Remote hosts router: CRUD for cloud GPU / SSH hosts + connection test."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db import session_dependency
from ..models import RemoteHost
from ..schemas import (
    RemoteHostCreate,
    RemoteHostRead,
    RemoteHostUpdate,
    RemoteTestResult,
)
from ..services import ssh_service

router = APIRouter(prefix="/api/remotes", tags=["remotes"])


def _to_read(h: RemoteHost) -> RemoteHostRead:
    return RemoteHostRead(
        id=h.id,
        name=h.name,
        host=h.host,
        port=h.port,
        username=h.username,
        auth_type=h.auth_type,
        has_password=bool(h.password),
        private_key_path=h.private_key_path,
        workdir=h.workdir,
        kohya_dir=h.kohya_dir,
        python_cmd=h.python_cmd,
        base_models_dir=h.base_models_dir,
        rvc_dir=h.rvc_dir,
        created_at=h.created_at,
    )


def _conn_of(h: RemoteHost) -> ssh_service.RemoteConn:
    return ssh_service.RemoteConn(
        host=h.host,
        port=h.port,
        username=h.username,
        auth_type=h.auth_type,
        password=h.password,
        private_key_path=h.private_key_path,
    )


@router.get("", response_model=list[RemoteHostRead])
def list_remotes(session: Session = Depends(session_dependency)):
    rows = session.exec(select(RemoteHost).order_by(RemoteHost.id.desc())).all()
    return [_to_read(h) for h in rows]


@router.post("", response_model=RemoteHostRead)
def create_remote(body: RemoteHostCreate, session: Session = Depends(session_dependency)):
    h = RemoteHost(**body.model_dump())
    session.add(h)
    session.commit()
    session.refresh(h)
    return _to_read(h)


@router.patch("/{remote_id}", response_model=RemoteHostRead)
def update_remote(
    remote_id: int, body: RemoteHostUpdate, session: Session = Depends(session_dependency)
):
    h = session.get(RemoteHost, remote_id)
    if not h:
        raise HTTPException(404, "远程主机不存在")
    data = body.model_dump(exclude_unset=True)
    # Empty password on update means "keep existing"; don't wipe it.
    if data.get("password", None) == "":
        data.pop("password", None)
    for k, v in data.items():
        setattr(h, k, v)
    session.add(h)
    session.commit()
    session.refresh(h)
    return _to_read(h)


@router.delete("/{remote_id}")
def delete_remote(remote_id: int, session: Session = Depends(session_dependency)):
    h = session.get(RemoteHost, remote_id)
    if not h:
        raise HTTPException(404, "远程主机不存在")
    session.delete(h)
    session.commit()
    return {"ok": True}


@router.post("/{remote_id}/test", response_model=RemoteTestResult)
def test_remote(remote_id: int, session: Session = Depends(session_dependency)):
    h = session.get(RemoteHost, remote_id)
    if not h:
        raise HTTPException(404, "远程主机不存在")
    ok, detail = ssh_service.test_connection(_conn_of(h))
    return RemoteTestResult(ok=ok, detail=detail)
