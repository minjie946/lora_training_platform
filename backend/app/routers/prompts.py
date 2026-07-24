"""提示词库路由：CRUD + 查找（翻译兜底）+ 互斥检查 + 组合。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..db import session_dependency
from ..models import Prompt
from ..schemas import (
    CombineRequest,
    CombineResult,
    MutexCheckRequest,
    PromptCreate,
    PromptRead,
    PromptSearchRequest,
    PromptSearchResult,
    PromptUpdate,
    TranslatedPrompt,
)
from ..services import prompt_service

router = APIRouter(prefix="/api/prompts", tags=["prompts"])


@router.get("", response_model=list[PromptRead])
def list_prompts(
    category: str | None = None, session: Session = Depends(session_dependency)
):
    stmt = select(Prompt).order_by(Prompt.category, Prompt.id)
    if category:
        stmt = stmt.where(Prompt.category == category)
    return session.exec(stmt).all()


@router.get("/categories", response_model=list[str])
def list_categories(session: Session = Depends(session_dependency)):
    rows = session.exec(select(Prompt.category)).all()
    # 去重并保持稳定顺序。
    seen: list[str] = []
    for c in rows:
        if c not in seen:
            seen.append(c)
    return sorted(seen)


@router.post("", response_model=PromptRead)
def create_prompt(body: PromptCreate, session: Session = Depends(session_dependency)):
    if not body.zh.strip() or not body.en.strip():
        raise HTTPException(400, "中文名和英文提示词都不能为空")
    p = Prompt(
        category=body.category.strip() or "其他",
        zh=body.zh.strip(),
        en=body.en.strip(),
        mutex_group=body.mutex_group.strip(),
        aliases=body.aliases.strip(),
    )
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


@router.patch("/{prompt_id}", response_model=PromptRead)
def update_prompt(
    prompt_id: int, body: PromptUpdate, session: Session = Depends(session_dependency)
):
    p = session.get(Prompt, prompt_id)
    if not p:
        raise HTTPException(404, "提示词不存在")
    data = body.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(p, k, (v or "").strip() if isinstance(v, str) else v)
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


@router.delete("/{prompt_id}")
def delete_prompt(prompt_id: int, session: Session = Depends(session_dependency)):
    p = session.get(Prompt, prompt_id)
    if not p:
        raise HTTPException(404, "提示词不存在")
    session.delete(p)
    session.commit()
    return {"ok": True}


@router.post("/search", response_model=PromptSearchResult)
def search_prompt(
    body: PromptSearchRequest, session: Session = Depends(session_dependency)
):
    """输入中文：命中词库返回匹配；未命中走翻译兜底。"""
    matches, translated = prompt_service.search(session, body.query)
    result = PromptSearchResult(query=body.query.strip(), matches=matches)
    if translated is not None:
        en, source = translated
        result.translated = TranslatedPrompt(zh=body.query.strip(), en=en, source=source)
    return result


@router.post("/check", response_model=list)
def check_mutex(body: MutexCheckRequest, session: Session = Depends(session_dependency)):
    """检查选中提示词是否互斥。返回冲突对列表（空列表表示无冲突）。"""
    prompts = _load_selection(session, body.ids)
    return prompt_service.find_conflicts(prompts)


@router.post("/combine", response_model=CombineResult)
def combine_prompts(
    body: CombineRequest, session: Session = Depends(session_dependency)
):
    """组合场景：拼接中英文串并检查互斥。"""
    prompts = _load_selection(session, body.ids)
    sep = body.separator or ", "
    return CombineResult(
        zh=sep.join(p.zh for p in prompts),
        en=sep.join(p.en for p in prompts),
        conflicts=prompt_service.find_conflicts(prompts),
        items=prompts,
    )


def _load_selection(session: Session, ids: list[int]) -> list[Prompt]:
    """按传入 id 顺序取出提示词（忽略不存在的 id）。"""
    if not ids:
        return []
    found = {
        p.id: p for p in session.exec(select(Prompt).where(Prompt.id.in_(ids))).all()
    }
    return [found[i] for i in ids if i in found]
