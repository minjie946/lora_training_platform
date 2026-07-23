import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, Job } from '../../api/client'
import { statusBadge, jobStepText } from '../../components/helpers'
import './Jobs.css'

export default function Jobs() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [deleting, setDeleting] = useState<number | null>(null)

  const load = () => api.listJobs().then(setJobs).catch(() => { })
  useEffect(() => {
    load()
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [])

  const remove = async (job: Job) => {
    const running = job.status === 'running'
    const msg = running
      ? `任务「${job.name}」正在训练中，删除会先停止训练再删除其日志和产出模型，确定？`
      : `确定删除任务「${job.name}」？将一并删除其日志和产出模型，且不可恢复。`
    if (!window.confirm(msg)) return
    setDeleting(job.id)
    try {
      await api.deleteJob(job.id)
      setJobs((prev) => prev.filter((j) => j.id !== job.id))
    } catch (e: any) {
      alert(`删除失败：${e.message}`)
    } finally {
      setDeleting(null)
    }
  }

  const pause = async (job: Job) => {
    try { await api.pauseJob(job.id); load() } catch (e: any) { alert(`暂停失败：${e.message}`) }
  }
  const resume = async (job: Job) => {
    try { await api.resumeJob(job.id); load() } catch (e: any) { alert(`继续失败：${e.message}`) }
  }
  const dequeue = async (job: Job) => {
    try { await api.dequeueJob(job.id); load() } catch (e: any) { alert(`取消排队失败：${e.message}`) }
  }

  return (
    <div>
      <div className="toolbar">
        <h1 className="page-title">训练任务</h1>
        <span className="spacer" />
        <Link className="btn" to="/jobs/new">+ 新建训练</Link>
      </div>
      <div className="table-card">
        {jobs.length === 0 ? (
          <div className="empty">还没有训练任务。</div>
        ) : (
          <table>
            <thead>
              <tr><th>名称</th><th>状态</th><th>底模</th><th>进度</th><th>步数</th><th>Loss</th><th></th></tr>
            </thead>
            <tbody>
              {jobs.map((j) => {
                const b = statusBadge(j.status)
                return (
                  <tr key={j.id}>
                    <td><Link className="linkish" to={`/jobs/${j.id}`}>{j.name}</Link></td>
                    <td><span className={b.cls}>{b.text}</span></td>
                    <td className="muted">{j.base_model ? j.base_model.replace(/\.safetensors$/, '') : '—'}</td>
                    <td style={{ width: 160 }}>
                      <div className="progress"><div className="bar" style={{ width: `${j.progress * 100}%` }} /></div>
                    </td>
                    <td>{jobStepText(j)}</td>
                    <td>{j.latest_loss != null ? j.latest_loss.toFixed(4) : '—'}</td>
                    <td>
                      <div className="row-actions">
                        <Link className="btn sm ghost" to={`/jobs/${j.id}`}>详情</Link>
                        {j.status !== 'running' && j.status !== 'paused' && j.status !== 'queued' && (
                          <Link className="btn sm ghost" to={`/jobs/${j.id}/edit`}>编辑</Link>
                        )}
                        {j.status === 'queued' && (
                          <button className="btn sm ghost" onClick={() => dequeue(j)} title="从队列中移除，不再自动开始">取消排队</button>
                        )}
                        {j.status === 'running' && j.has_checkpoint && (
                          <button className="btn sm ghost" onClick={() => pause(j)} title="从最近的检查点暂停，之后可继续">暂停</button>
                        )}
                        {j.status === 'paused' && (
                          <button className="btn sm" onClick={() => resume(j)}>继续</button>
                        )}
                        <button
                          className="icon-btn danger"
                          title="删除任务"
                          disabled={deleting === j.id}
                          onClick={() => remove(j)}
                        >
                          {deleting === j.id ? <span className="spinner" /> : <IconTrash />}
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function IconTrash() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18" />
      <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
    </svg>
  )
}
