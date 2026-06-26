"""Backend registry — maps backend name -> TrainingBackend instance.

Local backends are static. Remote (cloud GPU) backends are created on demand
from RemoteHost rows in the database, one backend per host, named "remote_<id>".
"""
from __future__ import annotations

from .base import TrainingBackend
from .local_mps import LocalMpsBackend

_STATIC: dict[str, TrainingBackend] = {
    LocalMpsBackend.name: LocalMpsBackend(),
}


def _remote_backends() -> dict[str, TrainingBackend]:
    """Build a RemoteSshBackend for each configured host (fresh each call so
    edits/additions are picked up without a restart)."""
    from ...db import get_session
    from ...models import RemoteHost
    from .remote_ssh import RemoteSshBackend
    from sqlmodel import select

    out: dict[str, TrainingBackend] = {}
    try:
        with get_session() as session:
            for host in session.exec(select(RemoteHost)).all():
                b = RemoteSshBackend(host)
                out[b.name] = b
    except Exception:  # noqa: BLE001
        pass
    return out


def get_backend(name: str) -> TrainingBackend:
    backend = _STATIC.get(name)
    if backend is not None:
        return backend
    backend = _remote_backends().get(name)
    if backend is None:
        raise ValueError(f"未知的训练后端: {name}")
    return backend


def list_backends() -> list[dict[str, str]]:
    items = [{"name": b.name, "label": b.label} for b in _STATIC.values()]
    items.extend({"name": b.name, "label": b.label} for b in _remote_backends().values())
    return items
