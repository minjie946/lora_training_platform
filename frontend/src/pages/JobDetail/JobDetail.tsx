import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, Job, LoraModel, ResourceStats } from '../../api/client'
import { statusBadge, jobStepText, formatBytes, inferPhase } from '../../components/helpers'
import './JobDetail.css'

function LossChart({ data }: { data: number[] }) {
  if (data.length < 2) return <div className="muted" style={{ fontSize: 12 }}>暂无足够的 loss 数据</div>
  const w = 600, h = 120, pad = 6
  const max = Math.max(...data), min = Math.min(...data)
  const range = max - min || 1
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - 2 * pad)
    const y = pad + (1 - (v - min) / range) * (h - 2 * pad)
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <svg className="loss-chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <polyline fill="none" stroke="#818cf8" strokeWidth="2" points={pts} />
    </svg>
  )
}

export default function JobDetail() {
  const { id } = useParams()
  const jobId = Number(id)
  const [job, setJob] = useState<Job | null>(null)
  const [log, setLog] = useState<string[]>([])
  const [losses, setLosses] = useState<number[]>([])
  const [models, setModels] = useState<LoraModel[]>([])
  const [err, setErr] = useState('')
  const logRef = useRef<HTMLDivElement>(null)
  const esRef = useRef<EventSource | null>(null)

  const refresh = () => {
    api.getJob(jobId).then(setJob).catch(() => { })
    api.listModels(jobId).then(setModels).catch(() => { })
  }

  useEffect(() => {
    refresh()
    api.jobLog(jobId).then((r) => setLog(r.log ? r.log.split('\n') : [])).catch(() => { })
  }, [jobId])

  // open SSE only while running
  useEffect(() => {
    if (!job) return
    if (job.status === 'running' && !esRef.current) {
      const es = new EventSource(`/api/jobs/${jobId}/stream`)
      esRef.current = es
      es.addEventListener('progress', (e: any) => {
        const d = JSON.parse(e.data)
        setJob((prev) => (prev ? { ...prev, ...d } : prev))
        if (d.latest_loss != null) setLosses((l) => [...l.slice(-300), d.latest_loss])
      })
      es.addEventListener('log', (e: any) => {
        const d = JSON.parse(e.data)
        setLog((l) => [...l.slice(-2000), d.line])
      })
      es.addEventListener('done', () => { es.close(); esRef.current = null; refresh() })
      es.onerror = () => { es.close(); esRef.current = null }
    }
    return () => {
      if (esRef.current && job.status !== 'running') {
        esRef.current.close(); esRef.current = null
      }
    }
  }, [job?.status, jobId])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log])

  const start = async () => {
    setErr('')
    try { await api.startJob(jobId); setLog([]); setLosses([]); refresh() }
    catch (e: any) { setErr(e.message) }
  }
  const stop = async () => { try { await api.stopJob(jobId); refresh() } catch (e: any) { setErr(e.message) } }
  const pause = async () => { setErr(''); try { await api.pauseJob(jobId); refresh() } catch (e: any) { setErr(e.message) } }
  const resume = async () => { setErr(''); try { await api.resumeJob(jobId); refresh() } catch (e: any) { setErr(e.message) } }
  const clone = async () => { const j = await api.cloneJob(jobId); window.location.href = `/jobs/${j.id}` }
  const remove = async () => {
    const running = job?.status === 'running'
    const msg = running
      ? `任务「${job?.name}」正在训练中，删除会先停止训练再删除其日志和产出模型，确定？`
      : `确定删除任务「${job?.name}」？将一并删除其日志和产出模型，且不可恢复。`
    if (!window.confirm(msg)) return
    try { await api.deleteJob(jobId); window.location.href = '/jobs' }
    catch (e: any) { setErr(e.message) }
  }

  if (!job) return <p className="muted">加载中…</p>
  const b = statusBadge(job.status)
  const phase = inferPhase(job.status, log)

  return (
    <div>
      <div className="toolbar">
        <h1 className="page-title">{job.name}</h1>
        <span className={b.cls}>{b.text}</span>
        <span className="spacer" />
        <Link className="btn ghost sm" to="/jobs">返回</Link>
        {job.status !== 'running' && job.status !== 'paused' && (
          <Link className="btn sm ghost" to={`/jobs/${jobId}/edit`}>编辑</Link>
        )}
        {job.status !== 'running' && job.status !== 'paused' && <button className="btn sm" onClick={start}>启动训练</button>}
        {job.status === 'paused' && <button className="btn sm" onClick={resume}>继续训练</button>}
        {job.status === 'running' && job.has_checkpoint && (
          <button className="btn sm ghost" onClick={pause} title="从最近的检查点暂停，之后可继续">暂停</button>
        )}
        {job.status === 'running' && !job.has_checkpoint && (
          <span className="muted" style={{ fontSize: 12 }} title="尚未生成检查点，暂停后无法续训，只能停止后从头再来">
            暂不可暂停（等首个检查点）
          </span>
        )}
        {job.status === 'running' && <button className="btn sm danger" onClick={stop}>停止</button>}
        <button className="btn sm ghost" onClick={clone}>克隆重训</button>
        <button className="btn sm danger" onClick={remove}>删除</button>
      </div>

      {err && <p className="badge red">{err}</p>}
      {job.error && <p className="badge red">错误：{job.error}</p>}

      <div className="card overview-card" style={{ marginBottom: 20 }}>
        {/* 阶段提示：让用户清楚“现在在干嘛”，区分准备/缓存/训练 */}
        <div className="phase-row">
          <span className={`phase-pill phase-${phase.key}`}>{phase.label}</span>
          <span className="muted phase-hint">{phase.hint}</span>
        </div>

        {/* 训练进度（仅统计真正的训练步，不含缓存预处理） */}
        <div className="progress" style={{ margin: '12px 0 6px' }}>
          <div className="bar" style={{ width: `${job.progress * 100}%` }} />
        </div>
        <div className="overview-metrics">
          <span><b>{(job.progress * 100).toFixed(1)}%</b> 训练进度</span>
          <span><b>{jobStepText(job)}</b></span>
          <span>最新 Loss <b>{job.latest_loss != null ? job.latest_loss.toFixed(4) : '—'}</b></span>
          <span className="muted">底模 {job.base_model ? job.base_model.replace(/\.safetensors$/, '') : '默认'}</span>
          <span className="muted">后端 {job.backend}</span>
        </div>

        {/* 资源占用：紧凑内联迷你条 */}
        <ResourceMonitor active={job.status === 'running'} />
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="toolbar" style={{ marginBottom: 0 }}>
          <strong>Loss 曲线</strong>
          <span className="muted" style={{ fontSize: 12 }}>
            Loss 衡量模型预测与训练图的差距，越低越拟合；理想是整体震荡下降后趋于平稳
          </span>
        </div>
        <div style={{ marginTop: 10 }}><LossChart data={losses} /></div>
      </div>

      <LogPanel log={log} jobName={job.name} logRef={logRef} />

      <div>
        <strong style={{ display: 'block', marginBottom: 12 }}>产出模型</strong>
        {models.length === 0 ? (
          <div className="card"><div className="empty">训练成功后会在此列出各 epoch 的 LoRA 权重。</div></div>
        ) : (
          <div className="table-card">
            <table>
              <thead><tr><th>文件</th><th>Epoch</th><th>大小</th><th></th></tr></thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id}>
                    <td>{m.name}</td>
                    <td>{m.epoch}</td>
                    <td>{formatBytes(m.file_size)}</td>
                    <td><a className="btn sm" href={api.modelDownloadUrl(m.id)}>下载</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function LogPanel({
  log,
  jobName,
  logRef,
}: {
  log: string[]
  jobName: string
  logRef: React.RefObject<HTMLDivElement>
}) {
  const [query, setQuery] = useState('')
  const [copied, setCopied] = useState(false)

  const q = query.trim().toLowerCase()
  const filtered = useMemo(
    () => (q ? log.filter((line) => line.toLowerCase().includes(q)) : log),
    [log, q],
  )

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(filtered.join('\n'))
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard 不可用时忽略 */
    }
  }

  const exportLog = () => {
    const blob = new Blob([filtered.join('\n')], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    const safeName = (jobName || 'training').replace(/[^\w\-一-鿿]+/g, '_')
    const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '')
    a.href = url
    a.download = `${safeName}_log_${ts}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="toolbar" style={{ marginBottom: 0 }}>
        <strong>训练日志</strong>
        <span className="spacer" />
        <input
          className="input"
          style={{ maxWidth: 220 }}
          placeholder="搜索日志…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="btn sm ghost" onClick={copy} disabled={filtered.length === 0}>
          {copied ? '已复制' : '复制'}
        </button>
        <button className="btn sm ghost" onClick={exportLog} disabled={filtered.length === 0}>
          导出
        </button>
      </div>
      {q && (
        <p className="muted" style={{ fontSize: 12, margin: '8px 0 0' }}>
          匹配 {filtered.length} / {log.length} 行
        </p>
      )}
      <div className="log-window" ref={logRef} style={{ marginTop: 10 }}>
        {log.length === 0 ? (
          <span className="muted">暂无日志</span>
        ) : filtered.length === 0 ? (
          <span className="muted">没有匹配“{query}”的日志行</span>
        ) : (
          filtered.map((line, i) => <HighlightLine key={i} line={line} term={q} />)
        )}
      </div>
    </div>
  )
}

function HighlightLine({ line, term }: { line: string; term: string }) {
  if (!term) return <div>{line || '\u00a0'}</div>
  const lower = line.toLowerCase()
  const parts: React.ReactNode[] = []
  let i = 0
  let key = 0
  while (i < line.length) {
    const idx = lower.indexOf(term, i)
    if (idx === -1) {
      parts.push(line.slice(i))
      break
    }
    if (idx > i) parts.push(line.slice(i, idx))
    parts.push(<mark key={key++} className="log-hit">{line.slice(idx, idx + term.length)}</mark>)
    i = idx + term.length
  }
  return <div>{parts}</div>
}

function meterColor(pct: number): string {
  if (pct >= 90) return 'var(--err)'
  if (pct >= 70) return 'var(--warn)'
  return 'var(--accent)'
}

function MiniMeter({
  label,
  pct,
  title,
}: {
  label: string
  pct: number | null
  title?: string
}) {
  const safe = pct == null ? 0 : Math.max(0, Math.min(100, pct))
  return (
    <div className="mini-meter" title={title}>
      <span className="mini-label">{label}</span>
      <div className="mini-track">
        <div className="mini-fill" style={{ width: `${safe}%`, background: meterColor(safe) }} />
      </div>
      <span className="mini-val">{pct == null ? '—' : `${safe.toFixed(0)}%`}</span>
    </div>
  )
}

function ResourceMonitor({ active }: { active: boolean }) {
  const [res, setRes] = useState<ResourceStats | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let alive = true
    const tick = () => {
      api
        .resources()
        .then((r) => { if (alive) { setRes(r); setFailed(false) } })
        .catch(() => { if (alive) setFailed(true) })
    }
    tick()
    const interval = active ? 2000 : 5000
    const t = setInterval(tick, interval)
    return () => { alive = false; clearInterval(t) }
  }, [active])

  if (failed && !res) return null

  const cpu = res?.cpu_percent ?? null
  const mem = res?.mem_percent ?? null
  const gpu = res?.gpu
  const gpuUtil = gpu?.available ? gpu.utilization ?? null : null

  const memTitle =
    res?.mem_used != null && res?.mem_total != null
      ? `内存 ${formatBytes(res.mem_used)} / ${formatBytes(res.mem_total)}`
      : '内存'
  const cpuTitle = res?.cpu_count ? `CPU · ${res.cpu_count} 核` : 'CPU'
  const gpuTitle = gpu?.available
    ? `GPU(MPS)` +
    (gpu.cores ? ` · ${gpu.cores} 核` : '') +
    (gpu.used_bytes != null ? ` · 显存 ${formatBytes(gpu.used_bytes)}` : '')
    : 'GPU 不可用'

  return (
    <div className="mini-monitor">
      <MiniMeter label="CPU" pct={cpu} title={cpuTitle} />
      <MiniMeter label="内存" pct={mem} title={memTitle} />
      <MiniMeter label="GPU" pct={gpuUtil} title={gpuTitle} />
    </div>
  )
}
