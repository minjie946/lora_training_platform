import { useEffect, useRef, useState } from 'react'
import {
  Download, Filter, Loader2, StopCircle, FolderOpen, RefreshCw, Users, AlertTriangle,
  KeyRound, CheckCircle2, Pencil, Save, X, Eye, Circle, Settings as SettingsIcon, RotateCcw,
  Folder, CornerLeftUp, Check, Sparkles,
} from 'lucide-react'
import { api, ImageDirEntry, ImageTask, ImageCookie, ImagePreviewResult, ImageSettings, ImageBrowseResult } from '../../api/client'
import Select from '../../components/Select/Select'
import PageHeader from '../../components/PageHeader/PageHeader'
import './Images.css'

type Tab = 'pull' | 'filter' | 'select' | 'settings'

export default function Images() {
  const [tab, setTab] = useState<Tab>('pull')

  return (
    <div className="page">
      <PageHeader title="图片管理" subtitle="微博图片拉取 · 单人筛选 · LoRA 精选" />
      <div className="page-body">

        {/* Tab 切换：图片拉取 / 图片筛选 / LoRA 精选 / 设置 */}
        <div className="tabs">
          <button className={`tab${tab === 'pull' ? ' active' : ''}`} onClick={() => setTab('pull')}>
            <Download size={16} />图片拉取
          </button>
          <button className={`tab${tab === 'filter' ? ' active' : ''}`} onClick={() => setTab('filter')}>
            <Filter size={16} />图片筛选
          </button>
          <button className={`tab${tab === 'select' ? ' active' : ''}`} onClick={() => setTab('select')}>
            <Sparkles size={16} />LoRA 精选
          </button>
          <button className={`tab${tab === 'settings' ? ' active' : ''}`} onClick={() => setTab('settings')}>
            <SettingsIcon size={16} />设置
          </button>
        </div>

        {tab === 'pull' ? <PullPanel />
          : tab === 'filter' ? <FilterPanel />
            : tab === 'select' ? <SelectPanel />
              : <SettingsPanel />}
      </div>
    </div>
  )
}

// ---- 任务日志与进度轮询（拉取/筛选共用） ----
function useTaskRunner(kind: 'pull' | 'filter' | 'select') {
  const [task, setTask] = useState<ImageTask | null>(null)
  const [log, setLog] = useState('')
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  const poll = (id: number) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const [t, l] = await Promise.all([api.imageTask(id), api.imageTaskLog(id)])
        setTask(t)
        setLog(l.log)
        if (t.status !== 'running') stopPolling()
      } catch { /* 网络抖动下次再试 */ }
    }, 2000)
  }

  // 进入页面时恢复：若该类型有正在运行的任务，接管其进度。
  useEffect(() => {
    api.imageTasks(kind).then((tasks) => {
      const running = tasks.find((t) => t.status === 'running') || tasks[0]
      if (running) {
        setTask(running)
        api.imageTaskLog(running.id).then((l) => setLog(l.log)).catch(() => { })
        if (running.status === 'running') poll(running.id)
      }
    }).catch(() => { })
    return stopPolling
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log])

  const begin = (t: ImageTask) => { setTask(t); setLog(''); poll(t.id) }

  // Update to a new task snapshot (e.g. after pause/resume) without clearing the
  // log; re-arm polling if it's running again.
  const attach = (t: ImageTask) => {
    setTask(t)
    if (t.status === 'running') poll(t.id); else stopPolling()
  }
  const clear = () => { stopPolling(); setTask(null); setLog('') }

  return { task, log, logRef, begin, attach, clear }
}

function TaskStatus({ task }: { task: ImageTask | null }) {
  if (!task) return null
  const cls =
    task.status === 'running' ? 'badge violet'
      : task.status === 'paused' ? 'badge amber'
        : task.status === 'done' ? 'badge green'
          : task.status === 'failed' ? 'badge red'
            : 'badge'
  const text =
    task.status === 'running' ? '进行中'
      : task.status === 'paused' ? '已暂停'
        : task.status === 'done' ? '已完成'
          : task.status === 'failed' ? '失败'
            : '已停止'
  // Show a progress bar for pull tasks once the download has a known total.
  const showBar = task.kind === 'pull' && task.total > 0
  const pct = Math.round((task.progress || 0) * 100)
  return (
    <div className="task-status">
      <div className="task-status-head">
        <span className={cls}>
          {task.status === 'running' && <Loader2 size={12} className="spin" />} {text}
        </span>
        {showBar && <span className="muted count">已下载 {task.done}/{task.total} 张（{pct}%）</span>}
        {task.detail && <span className="task-status-detail">{task.detail}</span>}
      </div>
      {showBar && (
        <div className="pull-progress">
          <div
            className={`pull-progress-bar${task.status === 'paused' ? ' paused' : ''}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Cookie 管理卡片（查看状态 + 手动编辑保存）
// --------------------------------------------------------------------------- //
function CookieCard({ platform = 'weibo' }: { platform?: 'weibo' | 'xhs' }) {
  const [info, setInfo] = useState<ImageCookie | null>(null)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')

  const isXhs = platform === 'xhs'
  const platformLabel = isXhs ? '小红书' : '微博'
  const fieldsHint = isXhs ? 'a1、web_session、webId' : 'SUB=、SUBP='

  const load = () => api.imageCookie(platform).then(setInfo).catch(() => setInfo(null))
  useEffect(() => { setEditing(false); setDraft(''); setMsg(''); load() }, [platform])

  const startEdit = async () => {
    setMsg(''); setEditing(true)
    // 预填当前 cookie 便于在原值上修改（本地单用户工具）。
    try { const r = await api.imageCookieRaw(platform); setDraft(r.cookie) } catch { setDraft('') }
  }
  const cancel = () => { setEditing(false); setDraft(''); setMsg('') }

  const save = async () => {
    if (!draft.trim()) { setMsg('请粘贴 Cookie 内容'); return }
    setSaving(true); setMsg('')
    try {
      const r = await api.setImageCookie(draft.trim(), platform)
      setInfo(r)
      setEditing(false)
      setDraft('')
      setMsg(r.looks_valid ? 'Cookie 已保存' : `Cookie 已保存，但未检测到登录字段(${fieldsHint})，可能无效`)
    } catch (e: any) { setMsg(e.message) } finally { setSaving(false) }
  }

  const present = info?.present
  const valid = info?.looks_valid
  const updated = info?.updated_at ? new Date(info.updated_at).toLocaleString() : ''

  return (
    <div className="cookie-card">
      <div className="cookie-head">
        <span className="cookie-title"><KeyRound size={15} className="icon-accent" /> {platformLabel} Cookie</span>
        {!editing && (
          present
            ? <span className={`badge ${valid ? 'green' : 'amber'}`}>
              {valid ? <><CheckCircle2 size={12} /> 已配置</> : <><AlertTriangle size={12} /> 可能无效</>}
            </span>
            : <span className="badge red"><AlertTriangle size={12} /> 未配置</span>
        )}
        <span className="spacer" />
        {!editing && (
          <button className="btn ghost sm" onClick={startEdit}>
            <Pencil size={14} />{present ? '修改' : '配置'}
          </button>
        )}
      </div>

      {!editing ? (
        <div className="cookie-body">
          {present ? (
            <span className="muted cookie-preview">
              <code>{info?.preview}</code>
              <span className="cookie-meta">共 {info?.length} 字符{updated && ` · 更新于 ${updated}`}</span>
            </span>
          ) : (
            <span className="err-text">拉取需要{platformLabel}登录 Cookie，请点「配置」粘贴浏览器登录后的 Cookie。</span>
          )}
        </div>
      ) : (
        <div className="cookie-edit">
          <textarea
            className="input cookie-textarea"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={`粘贴浏览器登录${platformLabel}后复制的整段 Cookie（含 ${fieldsHint} 等字段）`}
            rows={4}
            autoFocus
          />
          <div className="cookie-actions">
            <button className="btn sm" onClick={save} disabled={saving}>
              {saving ? <><span className="spinner" />保存中…</> : <><Save size={14} />保存</>}
            </button>
            <button className="btn ghost sm" onClick={cancel} disabled={saving}><X size={14} />取消</button>
          </div>
        </div>
      )}
      {msg && <div className="cookie-msg muted">{msg}</div>}
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Tab 1：图片拉取
// --------------------------------------------------------------------------- //
function PullPanel() {
  const [platform, setPlatform] = useState<'weibo' | 'xhs'>('weibo')
  const [mode, setMode] = useState<'uid' | 'album'>('uid')
  const [uid, setUid] = useState('')
  const [album, setAlbum] = useState('')
  const [xhsUser, setXhsUser] = useState('')      // 小红书博主主页链接
  const [maxNotes, setMaxNotes] = useState<string>('')  // 小红书解析笔记上限（留空=全部）
  const [xhsHeaded, setXhsHeaded] = useState(false)  // 小红书有头模式：弹浏览器手动过验证码翻页
  const [workers, setWorkers] = useState(6)
  const [start, setStart] = useState(1)
  const [end, setEnd] = useState<string>('')
  const [err, setErr] = useState('')
  // 预览态：抓到的 pid 列表 + 选中集合 + 目标目录名
  const [previewing, setPreviewing] = useState(false)
  const [preview, setPreview] = useState<ImagePreviewResult | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  // 预览抓取是同步阻塞的（相册翻页 / 小红书开真实浏览器逐篇解析），较慢，
  // 因此边等边轮询后端预览日志，实时展示进度而非只转圈。
  const [previewLog, setPreviewLog] = useState('')
  const previewPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const previewLogRef = useRef<HTMLPreElement>(null)
  const { task, log, logRef, begin, attach, clear } = useTaskRunner('pull')

  const running = task?.status === 'running'
  const paused = task?.status === 'paused'
  const isXhs = platform === 'xhs'

  const stopPreviewPoll = () => {
    if (previewPollRef.current) { clearInterval(previewPollRef.current); previewPollRef.current = null }
  }
  useEffect(() => stopPreviewPoll, [])
  useEffect(() => {
    if (previewLogRef.current) previewLogRef.current.scrollTop = previewLogRef.current.scrollHeight
  }, [previewLog])

  const startPreviewPoll = () => {
    stopPreviewPoll()
    setPreviewLog('')
    const plat = isXhs ? 'xhs' : 'weibo'
    previewPollRef.current = setInterval(async () => {
      try { const r = await api.previewLog(plat); setPreviewLog(r.log) } catch { /* 下次再试 */ }
    }, 1500)
  }

  const doPreview = async () => {
    setErr(''); setPreview(null)
    try {
      let r: ImagePreviewResult
      if (isXhs) {
        if (!xhsUser.trim()) { setErr('请输入小红书博主主页链接'); return }
        setPreviewing(true)
        startPreviewPoll()
        r = await api.xhsPreview({
          user: xhsUser.trim(),
          max_notes: maxNotes.trim() ? Number(maxNotes) : null,
          headed: xhsHeaded,
        })
      } else {
        if (mode === 'uid' && !uid.trim()) { setErr('请输入用户 UID'); return }
        if (mode === 'album' && !album.trim()) { setErr('请输入相册链接'); return }
        setPreviewing(true)
        startPreviewPoll()
        r = await api.previewImages({
          uid: mode === 'uid' ? uid.trim() : '',
          album: mode === 'album' ? album.trim() : '',
          start,
          end: end.trim() ? Number(end) : null,
        })
      }
      setPreview(r)
      // 默认全选
      setSelected(new Set(r.pids.map((p) => p.pid)))
      if (r.pids.length === 0) setErr('未抓取到图片，请检查链接或 Cookie')
    } catch (e: any) { setErr(e.message) } finally { setPreviewing(false); stopPreviewPoll() }
  }

  const toggle = (pid: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(pid)) next.delete(pid); else next.add(pid)
      return next
    })
  }
  const selectAll = () => setPreview((p) => { if (p) setSelected(new Set(p.pids.map((x) => x.pid))); return p })
  const selectNone = () => setSelected(new Set())

  const downloadSelected = async () => {
    setErr('')
    if (!preview || selected.size === 0) { setErr('请至少选择一张图片'); return }
    try {
      const chosen = preview.pids.filter((p) => selected.has(p.pid)).map((p) => p.pid)
      const t = isXhs
        ? await api.xhsPullSelected({
          ids: chosen,
          user: xhsUser.trim(),
          out_dir_name: preview.out_dir_name,
          workers,
        })
        : await api.pullSelected({
          pids: chosen,
          out_dir_name: preview.out_dir_name,
          workers,
        })
      begin(t)
    } catch (e: any) { setErr(e.message) }
  }

  // 直接下载：跳过预览/勾选，按当前输入抓取并全部下载。
  const directDownload = async () => {
    setErr(''); setPreview(null)
    try {
      let t
      if (isXhs) {
        if (!xhsUser.trim()) { setErr('请输入小红书博主主页链接'); return }
        t = await api.xhsPull({
          user: xhsUser.trim(),
          workers,
          max_notes: maxNotes.trim() ? Number(maxNotes) : null,
          headed: xhsHeaded,
        })
      } else {
        if (mode === 'uid' && !uid.trim()) { setErr('请输入用户 UID'); return }
        if (mode === 'album' && !album.trim()) { setErr('请输入相册链接'); return }
        t = await api.pullImages({
          uid: mode === 'uid' ? uid.trim() : '',
          album: mode === 'album' ? album.trim() : '',
          workers,
          start,
          end: end.trim() ? Number(end) : null,
        })
      }
      begin(t)
    } catch (e: any) { setErr(e.message) }
  }

  const stop = async () => { if (task) { try { await api.stopImageTask(task.id) } catch { } } }
  const pause = async () => {
    if (!task) return
    try { const t = await api.pauseImageTask(task.id); attach(t) } catch (e: any) { setErr(e.message) }
  }
  const resume = async () => {
    if (!task) return
    try { const t = await api.resumeImageTask(task.id); attach(t) } catch (e: any) { setErr(e.message) }
  }
  const discard = async () => {
    if (!task) return
    if (!window.confirm('确定放弃并删除本次已下载的图片？此操作不可恢复。')) return
    try { await api.discardImageTask(task.id); clear() } catch (e: any) { setErr(e.message) }
  }

  return (
    <div className="card">
      {/* 平台切换：微博 / 小红书 */}
      <div className="seg" style={{ marginBottom: 12 }}>
        <button className={`seg-btn${!isXhs ? ' active' : ''}`}
          onClick={() => setPlatform('weibo')} disabled={running || previewing}>微博</button>
        <button className={`seg-btn${isXhs ? ' active' : ''}`}
          onClick={() => setPlatform('xhs')} disabled={running || previewing}>小红书</button>
      </div>

      {isXhs && (
        <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          小红书为博主主页全量抓取。请粘贴<strong>带 xsec_token 的完整主页链接</strong>（在小红书网页版点进博主主页后，直接复制地址栏 URL，形如 <code>/user/profile/xxx?xsec_token=...&xsec_source=pc_feed</code>）——缺少 token 会触发风控。需配置含 a1 / web_session / webId 的 Cookie；首次使用请先在终端执行一次
          <code> uv run --script backend/app/image_tools/xhs_user_downloader.py --install-browser </code>
          安装浏览器内核。抓取用真实登录会话逐篇解析，较慢，请耐心等待。默认无头只能拿到首屏（约一页）笔记；要抓<strong>全部</strong>笔记请勾选下方「有头模式」，会弹出浏览器窗口，出现验证码时手动完成后即自动翻页收集全部。
        </p>
      )}

      {!isXhs && (
        <div>
          <div className="seg">
            <button className={`seg-btn${mode === 'uid' ? ' active' : ''}`} onClick={() => setMode('uid')}>按用户 UID</button>
            <button className={`seg-btn${mode === 'album' ? ' active' : ''}`} onClick={() => setMode('album')}>按相册链接</button>
          </div>
        </div>
      )}

      {isXhs ? (
        <div className="form-grid">
          <label className="imgfield wide">
            <span className="imgfield-label">博主主页链接（需带 xsec_token）</span>
            <input className="input" value={xhsUser} onChange={(e) => setXhsUser(e.target.value)}
              placeholder="https://www.xiaohongshu.com/user/profile/xxx?xsec_token=...&xsec_source=pc_feed"
              disabled={running || previewing} />
          </label>
          <label className="imgfield sm">
            <span className="imgfield-label">并发数</span>
            <input className="input" type="number" min={1} max={16} value={workers}
              onChange={(e) => setWorkers(Number(e.target.value))} disabled={running} />
          </label>
          <label className="imgfield sm">
            <span className="imgfield-label">解析笔记数（留空=全部）</span>
            <input className="input" type="number" min={1} value={maxNotes}
              onChange={(e) => setMaxNotes(e.target.value)} placeholder="全部"
              disabled={running || previewing} />
          </label>
          <label className="check" style={{ gridColumn: '1 / -1' }}>
            <input type="checkbox" checked={xhsHeaded}
              onChange={(e) => setXhsHeaded(e.target.checked)} disabled={running || previewing} />
            有头模式（弹出浏览器手动过验证码，翻页拿全部笔记）
          </label>
        </div>
      ) : (
        <div className="form-grid">
          {mode === 'uid' ? (
            <label className="imgfield">
              <span className="imgfield-label">用户 UID</span>
              <input className="input" value={uid} onChange={(e) => setUid(e.target.value)}
                placeholder="如 1234567890" disabled={running || previewing} />
            </label>
          ) : (
            <label className="imgfield wide">
              <span className="imgfield-label">相册链接</span>
              <input className="input" value={album} onChange={(e) => setAlbum(e.target.value)}
                placeholder="https://photo.weibo.com/1234567890/albums/detail/album_id/9876543210" disabled={running || previewing} />
            </label>
          )}
          <label className="imgfield sm">
            <span className="imgfield-label">并发数</span>
            <input className="input" type="number" min={1} max={16} value={workers}
              onChange={(e) => setWorkers(Number(e.target.value))} disabled={running} />
          </label>
          <label className="imgfield sm">
            <span className="imgfield-label">起始（第几张）</span>
            <input className="input" type="number" min={1} value={start}
              onChange={(e) => setStart(Number(e.target.value))} disabled={running || previewing} />
          </label>
          <label className="imgfield sm">
            <span className="imgfield-label">结束（留空=到底）</span>
            <input className="input" type="number" min={1} value={end}
              onChange={(e) => setEnd(e.target.value)} placeholder="全部" disabled={running || previewing} />
          </label>
        </div>
      )}

      <div className="toolbar" style={{ marginTop: 4 }}>
        <button className="btn" onClick={doPreview} disabled={previewing || running || paused}>
          {previewing ? (<><span className="spinner" />抓取中…</>) : (<><Eye size={16} />预览图片</>)}
        </button>
        <button className="btn ghost" onClick={directDownload} disabled={previewing || running || paused}>
          <Download size={16} />直接下载
        </button>
        <span className="muted" style={{ fontSize: 12 }}>预览后可勾选下载；或直接下载当前范围全部</span>
        <span className="spacer" />
        {err && <span className="err-text">{err}</span>}
      </div>

      {/* 预览抓取实时日志：预览过程较慢，边等边显示后端进度 */}
      {(previewing || (previewLog && !preview)) && (
        <div style={{ marginTop: 4 }}>
          <div className="task-status-head">
            <span className="badge violet">
              {previewing && <Loader2 size={12} className="spin" />} 抓取图片列表中
            </span>
            <span className="muted count">
              {isXhs ? '小红书开真实浏览器逐篇解析，较慢，请耐心等待' : '正在翻页抓取图片列表'}
            </span>
          </div>
          <pre className="log-box" ref={previewLogRef}>{previewLog || '等待输出…'}</pre>
        </div>
      )}

      {/* 下载进度/日志放在预览网格之前，开始下载后无需滚到底部查看 */}
      <TaskStatus task={task} />
      {(running || paused) && (
        <div className="toolbar" style={{ marginTop: 4 }}>
          <span className="spacer" />
          {running && (
            <button className="btn ghost sm" onClick={pause} title="暂停下载，已拉取的图片会保留">
              <StopCircle size={15} />暂停
            </button>
          )}
          {paused && (
            <button className="btn sm" onClick={resume} title="从断点继续，已下载的自动跳过">
              <Download size={15} />继续下载
            </button>
          )}
          {running && (
            <button className="btn danger sm" onClick={stop}><StopCircle size={15} />停止</button>
          )}
          {paused && (
            <button className="btn danger sm" onClick={discard} title="放弃并删除本次已下载的图片">
              放弃并清空
            </button>
          )}
        </div>
      )}
      {paused && (
        <p className="muted" style={{ fontSize: 12, marginTop: 4 }}>
          已暂停。可「继续下载」补齐剩余图片，或直接前往「图片筛选 / 数据集」使用已拉取到 <code>{task?.out_dir}</code> 的图片。
        </p>
      )}
      {(log || running || paused) && (
        <pre className="log-box" ref={logRef}>{log || '等待输出…'}</pre>
      )}

      {preview && preview.pids.length > 0 && (
        <div className="preview-block">
          <div className="preview-bar">
            <span className="muted">共 {preview.pids.length} 张，已选 <strong>{selected.size}</strong> 张 → 目录 <code>{preview.out_dir_name}</code></span>
            <span className="spacer" />
            <button className="btn ghost sm" onClick={selectAll} disabled={running}>全选</button>
            <button className="btn ghost sm" onClick={selectNone} disabled={running}>全不选</button>
            {running ? (
              <button className="btn danger sm" onClick={stop}><StopCircle size={15} />停止下载</button>
            ) : (
              <button className="btn sm" onClick={downloadSelected} disabled={selected.size === 0}>
                <Download size={15} />下载选中（{selected.size}）
              </button>
            )}
          </div>
          <div className="preview-grid">
            {preview.pids.map((p) => {
              const on = selected.has(p.pid)
              return (
                <div
                  key={p.pid}
                  className={`preview-cell${on ? ' on' : ''}`}
                  onClick={() => !running && toggle(p.pid)}
                  title={on ? '点击取消' : '点击选中'}
                >
                  <img src={api.imageProxyUrl(p.thumb_url)} alt={p.pid} loading="lazy" />
                  <span className="preview-check">{on ? <CheckCircle2 size={18} /> : <Circle size={18} />}</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Tab 2：图片筛选
// --------------------------------------------------------------------------- //
function FilterPanel() {
  const [dirs, setDirs] = useState<ImageDirEntry[]>([])
  const [directory, setDirectory] = useState('')
  const [recursive, setRecursive] = useState(false)
  const [dryRun, setDryRun] = useState(false)
  const [minFace, setMinFace] = useState(0.5)
  const [noText, setNoText] = useState(false)
  const [noAnimal, setNoAnimal] = useState(false)
  const [noQuality, setNoQuality] = useState(false)
  const [err, setErr] = useState('')
  const { task, log, logRef, begin } = useTaskRunner('filter')

  const running = task?.status === 'running'

  const loadDirs = () => {
    api.imageDirs().then((d) => {
      setDirs(d)
      setDirectory((cur) => cur || (d[0]?.name ?? ''))
    }).catch(() => { })
  }
  useEffect(loadDirs, [])
  // 任务完成后刷新目录（分类统计会变化）。
  useEffect(() => { if (task && task.status !== 'running') loadDirs() }, [task?.status])

  const run = async () => {
    setErr('')
    if (!directory) { setErr('请选择要筛选的目录'); return }
    try {
      const t = await api.filterImages({
        directory,
        recursive,
        dry_run: dryRun,
        min_face: minFace,
        no_text_filter: noText,
        no_animal_filter: noAnimal,
        no_quality_filter: noQuality,
      })
      begin(t)
    } catch (e: any) { setErr(e.message) }
  }

  const stop = async () => { if (task) { try { await api.stopImageTask(task.id) } catch { } } }

  const selected = dirs.find((d) => d.name === directory)

  return (
    <div className="card">
      <div className="form-grid">
        <label className="imgfield wide">
          <span className="imgfield-label"><FolderOpen size={13} /> 待筛选目录</span>
          <div className="dir-row">
            <Select
              value={directory}
              onChange={(v) => setDirectory(String(v))}
              disabled={running}
              placeholder={dirs.length === 0 ? '（暂无已下载的目录）' : '选择目录'}
              options={dirs.map((d) => ({ value: d.name, label: `${d.name}（${d.image_count} 张）` }))}
            />
            <button className="btn ghost sm" onClick={loadDirs} title="刷新目录列表"><RefreshCw size={14} /></button>
          </div>
        </label>
        <label className="imgfield sm">
          <span className="imgfield-label">最小人脸占比 %</span>
          <input className="input" type="number" step={0.1} min={0} value={minFace}
            onChange={(e) => setMinFace(Number(e.target.value))} disabled={running} />
        </label>
      </div>

      <div className="check-row">
        <label className="check"><input type="checkbox" checked={recursive} onChange={(e) => setRecursive(e.target.checked)} disabled={running} />递归子目录</label>
        <label className="check"><input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} disabled={running} />仅试运行（不移动文件）</label>
        <label className="check"><input type="checkbox" checked={noText} onChange={(e) => setNoText(e.target.checked)} disabled={running} />关闭海报/文字检测</label>
        <label className="check"><input type="checkbox" checked={noAnimal} onChange={(e) => setNoAnimal(e.target.checked)} disabled={running} />关闭动物检测</label>
        <label className="check"><input type="checkbox" checked={noQuality} onChange={(e) => setNoQuality(e.target.checked)} disabled={running} />关闭训练质量筛选（不细分 single_lowq）</label>
      </div>

      {selected && Object.keys(selected.categories).length > 0 && (
        <div className="cat-bar">
          <Users size={14} className="icon-accent" />
          {Object.entries(selected.categories).map(([k, v]) => (
            <span key={k} className="badge">{catLabel(k)} {v}</span>
          ))}
        </div>
      )}

      <div className="toolbar" style={{ marginTop: 4 }}>
        {running ? (
          <button className="btn danger" onClick={stop}><StopCircle size={16} />停止筛选</button>
        ) : (
          <button className="btn" onClick={run} disabled={!directory}><Filter size={16} />开始筛选</button>
        )}
        <span className="spacer" />
        {err && <span className="err-text">{err}</span>}
      </div>

      <p className="muted hint">
        用 InsightFace 人脸检测挑出单人照，再对单人照做一道 LoRA 训练质量筛选（分辨率 / 清晰度 / 人脸大小与完整 / 曝光 / 正脸角度 / 遮挡），
        达标的进 single/（可直接训练），不达标的进 single_lowq/（日志会标注原因）；多人、海报/文字、拼图、纯动物分别归档，无人脸的留原处。
        首次运行会自动下载模型（约 300MB），请耐心等待。
      </p>

      <TaskStatus task={task} />
      {(log || running) && (
        <pre className="log-box" ref={logRef}>{log || '等待输出…'}</pre>
      )}
    </div>
  )
}

function catLabel(k: string): string {
  return ({
    single: '单人·可训练', single_best: '单人·精选', single_lowq: '单人·质量不足', multi: '多人', poster: '海报/文字', collage: '拼图', animal: '动物',
  } as Record<string, string>)[k] || k
}

// --------------------------------------------------------------------------- //
// Tab 3：LoRA 精选（可选任意系统目录，从中挑最适合训练的 Top-N 拷到 single_best/）
// --------------------------------------------------------------------------- //
function SelectPanel() {
  const [directory, setDirectory] = useState('')
  const [count, setCount] = useState(50)
  const [qualityWeight, setQualityWeight] = useState(0.6)
  const [noDiversity, setNoDiversity] = useState(false)
  const [err, setErr] = useState('')
  const [picking, setPicking] = useState(false)   // 内置目录浏览弹窗（原生不可用时回退）
  const [nativeBusy, setNativeBusy] = useState(false)
  const { task, log, logRef, begin } = useTaskRunner('select')

  const running = task?.status === 'running'

  // 优先调起系统原生文件夹选择框；不可用时回退到内置目录浏览弹窗。
  const chooseNative = async () => {
    setNativeBusy(true); setErr('')
    try {
      const r = await api.pickImageDir(directory || '')
      if (r.path) setDirectory(r.path)
    } catch {
      setPicking(true)
    } finally { setNativeBusy(false) }
  }

  const run = async () => {
    setErr('')
    if (!directory.trim()) { setErr('请选择要检测的目录'); return }
    if (count < 1) { setErr('精选数量至少为 1'); return }
    try {
      const t = await api.selectLoraBest({
        directory: directory.trim(),
        count,
        quality_weight: qualityWeight,
        no_diversity: noDiversity,
      })
      begin(t)
    } catch (e: any) { setErr(e.message) }
  }
  const stop = async () => { if (task) { try { await api.stopImageTask(task.id) } catch { } } }

  return (
    <div className="card">
      <label className="imgfield wide">
        <span className="imgfield-label"><FolderOpen size={13} /> 待检测目录</span>
        <div className="dir-row">
          <input
            className="input"
            style={{ flex: 1 }}
            value={directory}
            onChange={(e) => setDirectory(e.target.value)}
            placeholder="点右侧「选择文件夹」调起系统选择，或直接输入绝对路径"
            disabled={running}
          />
          <button className="btn ghost sm" onClick={chooseNative} disabled={running || nativeBusy}>
            {nativeBusy ? <><span className="spinner" />选择中…</> : <><FolderOpen size={14} />选择文件夹</>}
          </button>
        </div>
      </label>

      <div className="form-grid">
        <label className="imgfield sm">
          <span className="imgfield-label">精选数量</span>
          <input className="input" type="number" min={1} step={1} value={count}
            onChange={(e) => setCount(Number(e.target.value))} disabled={running} />
        </label>
        <label className="imgfield sm">
          <span className="imgfield-label">质量权重 {qualityWeight.toFixed(2)}</span>
          <input type="range" min={0} max={1} step={0.05} value={qualityWeight}
            onChange={(e) => setQualityWeight(Number(e.target.value))} disabled={running} />
        </label>
      </div>

      <div className="check-row">
        <label className="check"><input type="checkbox" checked={noDiversity}
          onChange={(e) => setNoDiversity(e.target.checked)} disabled={running} />关闭多样性去重（纯按质量分取 Top-N）</label>
      </div>

      <div className="toolbar" style={{ marginTop: 4 }}>
        {running ? (
          <button className="btn danger" onClick={stop}><StopCircle size={16} />停止精选</button>
        ) : (
          <button className="btn" onClick={run} disabled={!directory.trim()}><Sparkles size={16} />开始精选</button>
        )}
        <span className="spacer" />
        {err && <span className="err-text">{err}</span>}
      </div>

      <p className="muted hint">
        选择任意目录进行检测：若目录下存在 single/（单人筛选产出）则以它为输入，否则直接检测该目录内的图片。
        按清晰度 / 正脸 / 人脸大小 / 分辨率 / 曝光 / 置信度加权打分，再用人脸 embedding 做最远点采样去重，
        挑出最适合训练的 Top-N 拷贝到同级 single_best/（不改动原图）。质量权重越高越偏高质量、越低越偏多样覆盖；重跑会覆盖上次精选结果。
        首次运行会自动下载模型（约 300MB），请耐心等待。
      </p>

      <TaskStatus task={task} />
      {(log || running) && (
        <pre className="log-box" ref={logRef}>{log || '等待输出…'}</pre>
      )}

      {picking && (
        <DirPicker
          initial={directory || ''}
          onCancel={() => setPicking(false)}
          onPick={(p) => { setDirectory(p); setPicking(false) }}
        />
      )}
    </div>
  )
}

// --------------------------------------------------------------------------- //
// Tab 3：设置（Cookie + 下载/拉取/筛选根目录）
// --------------------------------------------------------------------------- //
function SettingsPanel() {
  return (
    <div className="card">
      <CookieCard platform="weibo" />
      <CookieCard platform="xhs" />
      <OutDirCard />
    </div>
  )
}

// 下载/拉取目录设置：拉取下载、筛选都基于这个根目录。
function OutDirCard() {
  const [cfg, setCfg] = useState<ImageSettings | null>(null)
  const [draft, setDraft] = useState('')
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState('')
  const [picking, setPicking] = useState(false)   // 打开的是后端浏览弹窗（原生失败时的兜底）
  const [nativeBusy, setNativeBusy] = useState(false)

  const load = () => api.imageSettings().then((s) => { setCfg(s); setDraft(s.out_dir) }).catch(() => setCfg(null))
  useEffect(() => { load() }, [])

  const startEdit = () => { setMsg(''); setEditing(true); setDraft(cfg?.out_dir ?? '') }
  const cancel = () => { setEditing(false); setMsg(''); setDraft(cfg?.out_dir ?? '') }

  // 优先调起系统原生文件夹选择框（后端在本机时可用）；不可用则回退到内置浏览弹窗。
  const chooseNative = async () => {
    setNativeBusy(true); setMsg('')
    try {
      const r = await api.pickImageDir(draft || cfg?.out_dir || '')
      if (r.path) setDraft(r.path)   // 用户取消时 path 为 null，保持原值
    } catch {
      setPicking(true)   // 原生选择不可用（非本机/无 GUI），回退到目录浏览弹窗
    } finally { setNativeBusy(false) }
  }

  const save = async (reset = false, value?: string) => {
    setSaving(true); setMsg('')
    try {
      const s = await api.setImageSettings(reset ? null : (value ?? draft).trim())
      setCfg(s); setDraft(s.out_dir); setEditing(false)
      setMsg(reset ? '已恢复默认目录' : '目录已保存')
    } catch (e: any) { setMsg(e.message) } finally { setSaving(false) }
  }

  return (
    <div className="cookie-card">
      <div className="cookie-head">
        <span className="cookie-title"><FolderOpen size={15} className="icon-accent" /> 下载 / 拉取 / 筛选目录</span>
        {!editing && cfg && (
          <span className={`badge ${cfg.is_default ? '' : 'green'}`}>
            {cfg.is_default ? '默认' : '自定义'}
          </span>
        )}
        {!editing && cfg && !cfg.exists && (
          <span className="badge amber"><AlertTriangle size={12} /> 目录不存在</span>
        )}
        <span className="spacer" />
        {!editing && (
          <button className="btn ghost sm" onClick={startEdit}><Pencil size={14} />修改</button>
        )}
      </div>

      {!editing ? (
        <div className="cookie-body">
          <span className="muted cookie-preview">
            <code>{cfg?.out_dir || '（加载中…）'}</code>
            <span className="cookie-meta">拉取下载与图片筛选都在此根目录下进行；默认：{cfg?.default_out_dir}</span>
          </span>
        </div>
      ) : (
        <div className="cookie-edit">
          <div className="dir-row">
            <input
              className="input"
              style={{ flex: 1 }}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="点右侧「选择文件夹」调起系统选择，或直接输入绝对路径（保存时会自动创建）"
              autoFocus
            />
            <button className="btn ghost sm" onClick={chooseNative} disabled={saving || nativeBusy}>
              {nativeBusy ? <><span className="spinner" />选择中…</> : <><FolderOpen size={14} />选择文件夹</>}
            </button>
          </div>
          <div className="cookie-actions">
            <button className="btn sm" onClick={() => save(false)} disabled={saving || !draft.trim()}>
              {saving ? <><span className="spinner" />保存中…</> : <><Save size={14} />保存</>}
            </button>
            <button className="btn ghost sm" onClick={() => save(true)} disabled={saving}>
              <RotateCcw size={14} />恢复默认
            </button>
            <button className="btn ghost sm" onClick={cancel} disabled={saving}><X size={14} />取消</button>
          </div>
        </div>
      )}
      {msg && <div className="cookie-msg muted">{msg}</div>}

      {picking && (
        <DirPicker
          initial={draft || cfg?.out_dir || ''}
          onCancel={() => setPicking(false)}
          onPick={(p) => { setDraft(p); setPicking(false) }}
        />
      )}
    </div>
  )
}

// 服务端目录选择器：浏览后端所在机器的文件系统并选一个目录。
function DirPicker({ initial, onCancel, onPick }: {
  initial: string
  onCancel: () => void
  onPick: (path: string) => void
}) {
  const [data, setData] = useState<ImageBrowseResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const browse = (path: string) => {
    setLoading(true); setErr('')
    api.browseImageDir(path)
      .then(setData)
      .catch((e: any) => setErr(e.message))
      .finally(() => setLoading(false))
  }
  useEffect(() => { browse(initial) }, [])

  const cur = data?.path ?? ''

  return (
    <div className="dp-overlay" onClick={onCancel}>
      <div className="dp-modal" onClick={(e) => e.stopPropagation()}>
        <div className="dp-head">
          <span className="cookie-title"><FolderOpen size={15} className="icon-accent" /> 选择目录</span>
          <span className="spacer" />
          <button className="btn ghost sm" onClick={onCancel}><X size={14} /></button>
        </div>

        <div className="dp-path">
          <button
            className="btn ghost sm"
            onClick={() => data?.parent && browse(data.parent)}
            disabled={!data?.parent || loading}
            title="上一级"
          >
            <CornerLeftUp size={14} />
          </button>
          <code className="dp-cur">{cur || '…'}</code>
        </div>

        <div className="dp-list">
          {loading ? (
            <div className="muted dp-empty"><span className="spinner" /> 加载中…</div>
          ) : err ? (
            <div className="err-text dp-empty">{err}</div>
          ) : data && data.dirs.length > 0 ? (
            data.dirs.map((name) => (
              <button key={name} className="dp-item" onClick={() => browse(`${cur}/${name}`)}>
                <Folder size={15} className="icon-accent" />
                <span className="dp-name">{name}</span>
              </button>
            ))
          ) : (
            <div className="muted dp-empty">（此目录下没有子文件夹）</div>
          )}
        </div>

        <div className="cookie-actions dp-foot">
          <button className="btn sm" onClick={() => cur && onPick(cur)} disabled={!cur}>
            <Check size={14} />选择当前目录
          </button>
          <button className="btn ghost sm" onClick={onCancel}>取消</button>
        </div>
      </div>
    </div>
  )
}
