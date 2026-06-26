"""Voice backend registry — maps backend name -> VoiceBackend instance.

Local RVC is static. Remote RVC backends are created on demand from RemoteHost
rows (one per host, named "remote_rvc_<id>").
"""
from __future__ import annotations

from .base import VoiceBackend
from .local_rvc import LocalRvcBackend

_STATIC: dict[str, VoiceBackend] = {
    LocalRvcBackend.name: LocalRvcBackend(),
}


def _remote_backends() -> dict[str, VoiceBackend]:
    from ...db import get_session
    from ...models import RemoteHost
    from .remote_rvc import RemoteRvcBackend
    from sqlmodel import select

    out: dict[str, VoiceBackend] = {}
    try:
        with get_session() as session:
            for host in session.exec(select(RemoteHost)).all():
                b = RemoteRvcBackend(host)
                out[b.name] = b
    except Exception:  # noqa: BLE001
        pass
    return out


def get_backend(name: str) -> VoiceBackend:
    backend = _STATIC.get(name)
    if backend is not None:
        return backend
    backend = _remote_backends().get(name)
    if backend is None:
        raise ValueError(f"未知的声音训练后端: {name}")
    return backend


def list_backends() -> list[dict[str, str]]:
    items = [{"name": b.name, "label": b.label} for b in _STATIC.values()]
    items.extend({"name": b.name, "label": b.label} for b in _remote_backends().values())
    return items
