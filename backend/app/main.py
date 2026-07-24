"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db
from .routers import datasets, images, jobs, models, prompts, remotes, system, voice
from .services.caption_manager import reconcile_on_startup as reconcile_captions_on_startup
from .services.image_manager import reconcile_on_startup as reconcile_images_on_startup
from .services.job_manager import reconcile_on_startup
from .services.prompt_service import reconcile_on_startup as seed_prompts_on_startup
from .services.voice_job_manager import reconcile_on_startup as reconcile_voice_on_startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Recover jobs whose process state was lost across a restart/reload.
    reconcile_on_startup()
    reconcile_voice_on_startup()
    reconcile_captions_on_startup()
    reconcile_images_on_startup()
    seed_prompts_on_startup()
    yield


app = FastAPI(title="LoRA Training Platform", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(datasets.router)
app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(remotes.router)
app.include_router(voice.router)
app.include_router(images.router)
app.include_router(prompts.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
