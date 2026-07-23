"""SQLite database setup using SQLModel."""
from __future__ import annotations

from sqlmodel import SQLModel, Session, create_engine

from .config import DB_PATH, ensure_dirs

ensure_dirs()

_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    # Import models so they are registered on SQLModel.metadata.
    from . import models  # noqa: F401

    SQLModel.metadata.create_all(_engine)
    _run_lightweight_migrations()


def _run_lightweight_migrations() -> None:
    """Add columns introduced after a table was first created.

    SQLModel.create_all never ALTERs existing tables, so for additive columns we
    apply idempotent ADD COLUMN statements here (SQLite has no IF NOT EXISTS for
    columns, so we check the table info first).
    """
    from sqlalchemy import text

    additions = {
        "dataset": [
            ("base_model", "VARCHAR DEFAULT ''"),
            ("caption_status", "VARCHAR DEFAULT 'idle'"),
            ("caption_detail", "VARCHAR DEFAULT ''"),
        ],
        "loramodel": [("base_model", "VARCHAR DEFAULT ''")],
        "trainingjob": [("queued_at", "DATETIME")],
        "remotehost": [
            ("rvc_dir", "VARCHAR DEFAULT '~/Retrieval-based-Voice-Conversion-WebUI'")
        ],
    }
    with _engine.begin() as conn:
        for table, cols in additions.items():
            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            }
            for name, decl in cols:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {decl}"))


def get_session() -> Session:
    return Session(_engine)


# FastAPI dependency
def session_dependency():
    with Session(_engine) as session:
        yield session
