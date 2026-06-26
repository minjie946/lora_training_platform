import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, Dataset, Job, PreflightResult } from '../../api/client'
import './Dashboard.css'

export default function Dashboard() {
  const [pf, setPf] = useState<PreflightResult | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [err, setErr] = useState('')

  useEffect(() => {
    api.preflight().then(setPf).catch((e) => setErr(e.message))
    api.listDatasets().then(setDatasets).catch(() => {})
    api.listJobs().then(setJobs).catch(() => {})
  }, [])

  const running = jobs.filter((j) => j.status === 'running').length
  const succeeded = jobs.filter((j) => j.status === 'succeeded').length

  return (
    <div>
      <h1 className="page-title">概览</h1>
      <p className="page-sub">人物生成 LoRA 训练平台 · 本地 Mac (MPS)</p>

      <div className="stat-grid" style={{ marginBottom: 24 }}>
        <div className="stat"><div className="num">{datasets.length}</div><div className="label">数据集</div></div>
        <div className="stat"><div className="num">{jobs.length}</div><div className="label">训练任务</div></div>
        <div className="stat"><div className="num">{running}</div><div className="label">训练中</div></div>
        <div className="stat"><div className="num">{succeeded}</div><div className="label">已完成</div></div>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="toolbar">
          <strong>环境自检</strong>
          <span className="spacer" />
          {pf && (
            <span className={pf.ok ? 'badge green' : 'badge amber'}>
              {pf.ok ? '环境就绪' : '需要配置'}
            </span>
          )}
        </div>
        {err && <p className="badge red">{err}</p>}
        {pf?.items.map((it) => (
          <div className="preflight-item" key={it.name}>
            <span className={`dot ${it.ok ? 'ok' : 'bad'}`} />
            <strong style={{ width: 180 }}>{it.name}</strong>
            <span className="muted">{it.detail}</span>
          </div>
        ))}
        {!pf && !err && <p className="muted">检测中…</p>}
      </div>

      <div className="card">
        <div className="toolbar">
          <strong>快速开始</strong>
          <span className="spacer" />
          <Link className="btn sm" to="/datasets">创建数据集</Link>
          <Link className="btn sm ghost" to="/jobs/new">新建训练</Link>
        </div>
        <p className="muted">
          流程：创建数据集 → 上传图片并打标（触发词置首位）→ 配置训练参数 → 启动训练 → 监控进度 → 下载产出的 LoRA。
        </p>
      </div>
    </div>
  )
}
