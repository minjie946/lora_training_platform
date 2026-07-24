import { useEffect, useMemo, useState } from 'react'
import { Search, Plus, Copy, Check, Trash2, Pencil, AlertTriangle, X } from 'lucide-react'
import {
  api,
  Prompt,
  PromptSearchResult,
  CombineResult,
} from '../../api/client'
import PageHeader from '../../components/PageHeader/PageHeader'
import './PromptLibrary.css'

type Tab = 'search' | 'library' | 'combine'

const EMPTY_FORM = { category: '', zh: '', en: '', mutex_group: '', aliases: '' }

export default function PromptLibrary() {
  const [tab, setTab] = useState<Tab>('search')
  const [prompts, setPrompts] = useState<Prompt[]>([])

  const load = () => api.listPrompts().then(setPrompts).catch(() => { })
  useEffect(() => { load() }, [])

  const categories = useMemo(() => {
    const seen: string[] = []
    for (const p of prompts) if (!seen.includes(p.category)) seen.push(p.category)
    return seen.sort()
  }, [prompts])

  return (
    <div className="page">
      <PageHeader
        title="提示词库"
        subtitle="中文查提示词 · 组合 · 互斥检查"
        actions={
          <div className="pl-tabs">
            <button className={`pl-tab${tab === 'search' ? ' active' : ''}`} onClick={() => setTab('search')}>查找</button>
            <button className={`pl-tab${tab === 'library' ? ' active' : ''}`} onClick={() => setTab('library')}>词库</button>
            <button className={`pl-tab${tab === 'combine' ? ' active' : ''}`} onClick={() => setTab('combine')}>组合</button>
          </div>
        }
      />
      <div className="page-body">
        {tab === 'search' && <SearchTab onAdded={load} />}
        {tab === 'library' && <LibraryTab prompts={prompts} categories={categories} reload={load} />}
        {tab === 'combine' && <CombineTab prompts={prompts} categories={categories} />}
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* 复制按钮                                                                     */
/* -------------------------------------------------------------------------- */
function CopyBtn({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setDone(true)
      setTimeout(() => setDone(false), 1200)
    } catch { }
  }
  return (
    <button className="icon-btn" onClick={copy} title="复制">
      {done ? <Check size={14} /> : <Copy size={14} />}
    </button>
  )
}

/* -------------------------------------------------------------------------- */
/* 查找：输入中文 → 命中词库 / 翻译兜底                                          */
/* -------------------------------------------------------------------------- */
function SearchTab({ onAdded }: { onAdded: () => void }) {
  const [q, setQ] = useState('')
  const [result, setResult] = useState<PromptSearchResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const run = async () => {
    const query = q.trim()
    if (!query) return
    setLoading(true); setErr('')
    try {
      setResult(await api.searchPrompt(query))
    } catch (e: any) {
      setErr(e.message || '查找失败')
    } finally {
      setLoading(false)
    }
  }

  const addTranslated = async () => {
    if (!result?.translated) return
    await api.createPrompt({
      category: '其他',
      zh: result.translated.zh,
      en: result.translated.en,
    })
    onAdded()
    run()
  }

  const sourceLabel: Record<string, string> = {
    dictionary: '本地词典', api: '在线翻译', none: '未收录',
  }

  return (
    <div className="pl-search">
      <div className="pl-search-bar">
        <Search size={18} className="pl-search-icon" />
        <input
          className="input"
          placeholder="输入中文，例如：金发、双马尾、微笑…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') run() }}
          autoFocus
        />
        <button className="btn" onClick={run} disabled={loading}>
          {loading && <span className="spinner" />}查找
        </button>
      </div>

      {err && <div className="pl-error">{err}</div>}

      {result && result.matches.length > 0 && (
        <div className="pl-cards">
          {result.matches.map((p) => (
            <div key={p.id} className="pl-card">
              <div className="pl-card-head">
                <span className="badge blue">{p.category}</span>
                {p.mutex_group && <span className="badge violet" title="互斥组">⇄ {p.mutex_group}</span>}
              </div>
              <div className="pl-card-zh">{p.zh}</div>
              <div className="pl-card-en">
                <code>{p.en}</code>
                <CopyBtn text={p.en} />
              </div>
            </div>
          ))}
        </div>
      )}

      {result && result.matches.length === 0 && result.translated && (
        <div className={`pl-fallback ${result.translated.source}`}>
          <div className="pl-fallback-head">
            <AlertTriangle size={16} />
            <span>词库未收录「{result.query}」</span>
            <span className="badge gray">{sourceLabel[result.translated.source] || result.translated.source}</span>
          </div>
          {result.translated.en ? (
            <>
              <div className="pl-card-en big">
                <code>{result.translated.en}</code>
                <CopyBtn text={result.translated.en} />
              </div>
              <button className="btn sm" onClick={addTranslated}>
                <Plus size={14} /> 加入词库
              </button>
            </>
          ) : (
            <div className="muted">没有可用的翻译结果。可在「词库」页手动补充该词的英文提示词，或配置在线翻译接口（PROMPT_TRANSLATE_URL）。</div>
          )}
        </div>
      )}

      {!result && !loading && (
        <div className="empty">输入中文提示词开始查找。命中词库直接返回配对英文，未收录则自动翻译兜底。</div>
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* 词库：按分类浏览 + 增删改                                                     */
/* -------------------------------------------------------------------------- */
function LibraryTab({ prompts, categories, reload }: { prompts: Prompt[]; categories: string[]; reload: () => void }) {
  const [cat, setCat] = useState<string>('')
  const [editing, setEditing] = useState<Prompt | 'new' | null>(null)

  const rows = cat ? prompts.filter((p) => p.category === cat) : prompts

  const remove = async (p: Prompt) => {
    if (!confirm(`确认删除「${p.zh}」？`)) return
    await api.deletePrompt(p.id)
    reload()
  }

  return (
    <div className="pl-library">
      <div className="toolbar">
        <div className="pl-chips">
          <button className={`pl-chip${cat === '' ? ' active' : ''}`} onClick={() => setCat('')}>全部 ({prompts.length})</button>
          {categories.map((c) => (
            <button key={c} className={`pl-chip${cat === c ? ' active' : ''}`} onClick={() => setCat(c)}>
              {c} ({prompts.filter((p) => p.category === c).length})
            </button>
          ))}
        </div>
        <div className="spacer" />
        <button className="btn" onClick={() => setEditing('new')}><Plus size={16} /> 新增提示词</button>
      </div>

      <div className="table-card">
        {rows.length === 0 ? (
          <div className="empty">该分类下暂无提示词。</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>分类</th>
                <th>中文</th>
                <th>英文提示词</th>
                <th>互斥组</th>
                <th>别名</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.id}>
                  <td><span className="badge blue">{p.category}</span></td>
                  <td>{p.zh}</td>
                  <td className="pl-en-cell">
                    <code>{p.en}</code>
                    <CopyBtn text={p.en} />
                  </td>
                  <td>{p.mutex_group ? <span className="badge violet">{p.mutex_group}</span> : <span className="muted">—</span>}</td>
                  <td className="muted">{p.aliases || '—'}</td>
                  <td>
                    <div className="row-actions">
                      <button className="icon-btn" onClick={() => setEditing(p)} title="编辑"><Pencil size={14} /></button>
                      <button className="icon-btn danger" onClick={() => remove(p)} title="删除"><Trash2 size={14} /></button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {editing && (
        <PromptForm
          prompt={editing === 'new' ? null : editing}
          categories={categories}
          onClose={() => setEditing(null)}
          onSaved={() => { setEditing(null); reload() }}
        />
      )}
    </div>
  )
}

function PromptForm({ prompt, categories, onClose, onSaved }: {
  prompt: Prompt | null
  categories: string[]
  onClose: () => void
  onSaved: () => void
}) {
  const [form, setForm] = useState(prompt
    ? { category: prompt.category, zh: prompt.zh, en: prompt.en, mutex_group: prompt.mutex_group, aliases: prompt.aliases }
    : { ...EMPTY_FORM })
  const [err, setErr] = useState('')
  const [saving, setSaving] = useState(false)

  const set = (k: keyof typeof form, v: string) => setForm((f) => ({ ...f, [k]: v }))

  const save = async () => {
    if (!form.zh.trim() || !form.en.trim()) { setErr('中文名和英文提示词都不能为空'); return }
    setSaving(true); setErr('')
    try {
      const body = { ...form, category: form.category.trim() || '其他' }
      if (prompt) await api.updatePrompt(prompt.id, body)
      else await api.createPrompt(body)
      onSaved()
    } catch (e: any) {
      setErr(e.message || '保存失败'); setSaving(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal modal-form" onClick={(e) => e.stopPropagation()}>
        <div className="pl-modal-head">
          <span>{prompt ? '编辑提示词' : '新增提示词'}</span>
          <button className="icon-btn" onClick={onClose}><X size={16} /></button>
        </div>
        <div className="modal-form-body">
          {err && <div className="pl-error">{err}</div>}
          <div className="row">
            <div className="field">
              <label>分类</label>
              <input className="input" list="pl-cat-list" value={form.category}
                placeholder="如 发色 / 服装" onChange={(e) => set('category', e.target.value)} />
              <datalist id="pl-cat-list">
                {categories.map((c) => <option key={c} value={c} />)}
              </datalist>
            </div>
            <div className="field">
              <label>互斥组 <span className="muted">(选填)</span></label>
              <input className="input" value={form.mutex_group}
                placeholder="如 hair_color" onChange={(e) => set('mutex_group', e.target.value)} />
            </div>
          </div>
          <div className="field">
            <label>中文名</label>
            <input className="input" value={form.zh} onChange={(e) => set('zh', e.target.value)} />
          </div>
          <div className="field">
            <label>英文提示词</label>
            <input className="input" value={form.en} onChange={(e) => set('en', e.target.value)} />
          </div>
          <div className="field">
            <label>别名 <span className="muted">(逗号分隔，用于查找命中)</span></label>
            <input className="input" value={form.aliases}
              placeholder="如 金色头发,金色发丝" onChange={(e) => set('aliases', e.target.value)} />
          </div>
          <div className="pl-form-foot">
            <button className="btn ghost" onClick={onClose}>取消</button>
            <button className="btn" onClick={save} disabled={saving}>
              {saving && <span className="spinner" />}保存
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* 组合：勾选提示词 → 中英文组合串 + 互斥检查                                     */
/* -------------------------------------------------------------------------- */
function CombineTab({ prompts, categories }: { prompts: Prompt[]; categories: string[] }) {
  const [selected, setSelected] = useState<number[]>([])
  const [result, setResult] = useState<CombineResult | null>(null)

  // 选择变化时重新组合（顺序即勾选顺序）。
  useEffect(() => {
    if (selected.length === 0) { setResult(null); return }
    let cancelled = false
    api.combinePrompts(selected).then((r) => { if (!cancelled) setResult(r) }).catch(() => { })
    return () => { cancelled = true }
  }, [selected])

  const toggle = (id: number) =>
    setSelected((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])

  // 组内已选的互斥组 → 高亮同组其它项为"冲突候选"。
  const conflictIds = useMemo(() => {
    const ids = new Set<number>()
    if (!result) return ids
    const enToId = new Map(prompts.map((p) => [p.en, p.id]))
    for (const c of result.conflicts) {
      const a = enToId.get(c.a_en); const b = enToId.get(c.b_en)
      if (a != null) ids.add(a); if (b != null) ids.add(b)
    }
    return ids
  }, [result, prompts])

  return (
    <div className="pl-combine">
      <div className="pl-combine-left">
        {categories.map((c) => (
          <div key={c} className="pl-group">
            <div className="pl-group-title">{c}</div>
            <div className="pl-group-items">
              {prompts.filter((p) => p.category === c).map((p) => {
                const on = selected.includes(p.id)
                return (
                  <button
                    key={p.id}
                    className={`pl-pill${on ? ' on' : ''}${on && conflictIds.has(p.id) ? ' conflict' : ''}`}
                    onClick={() => toggle(p.id)}
                    title={p.en}
                  >
                    {p.zh}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="pl-combine-right">
        <div className="card pl-result-card">
          <div className="card-title">已选组合 ({selected.length})</div>

          {result && result.conflicts.length > 0 && (
            <div className="pl-conflicts">
              <div className="pl-conflicts-head"><AlertTriangle size={16} /> 检测到互斥冲突</div>
              {result.conflicts.map((c, i) => (
                <div key={i} className="pl-conflict-row">
                  <span className="badge red">{c.a_zh}</span>
                  <span className="pl-vs">⇄</span>
                  <span className="badge red">{c.b_zh}</span>
                  <span className="muted pl-conflict-group">（同属 {c.group}）</span>
                </div>
              ))}
            </div>
          )}

          {selected.length === 0 ? (
            <div className="empty sm">从左侧勾选提示词，中英文组合会实时展示在这里，并自动检查互斥。</div>
          ) : (
            <>
              <div className="pl-combo-block">
                <div className="pl-combo-label">中文组合 <CopyBtn text={result?.zh || ''} /></div>
                <div className="pl-combo-text">{result?.zh}</div>
              </div>
              <div className="pl-combo-block">
                <div className="pl-combo-label">英文组合 <CopyBtn text={result?.en || ''} /></div>
                <div className="pl-combo-text en"><code>{result?.en}</code></div>
              </div>
              <button className="btn ghost sm pl-clear" onClick={() => setSelected([])}>清空选择</button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
