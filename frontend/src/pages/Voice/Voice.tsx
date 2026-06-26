import { useEffect, useRef, useState } from 'react'
import { api, AudioClip, VoiceDataset, VoiceJob, VoiceModel } from '../../api/client'
import { statusBadge, formatBytes } from '../../components/helpers'
import Select from '../../components/Select/Select'
import './Voice.css'

const SR_OPTIONS = [
  { value: 40000, label: '40000 Hz（推荐）' },
  { value: 48000, label: '48000 Hz（高音质，更慢）' },
  { value: 32000, label: '32000 Hz（更省资源）' },
]

const F0_OPTIONS = [
  { value: 'rmvpe', label: 'rmvpe（推荐，最准）' },
  { value: 'harvest', label: 'harvest（较慢）' },
  { value: 'crepe', label: 'crepe' },
  { value: 'pm', label: 'pm（最快）' },
]

export default function Voice() {
  const [datasets, setDatasets] = useState<VoiceDataset[]>([])
  const [jobs, setJobs] = useState<VoiceJob[]>([])
  const [models, setModels] = useState<VoiceModel[]>([])
  const [backends, setBackends] = useState<{ name: string; label: string }[]>([])
  const [sel, setSel] = useState<number | null>(null)
  const [clips, setClips] = useState<AudioClip[]>([])
  const [creating, setCreating] = useState(false)
  const [err, setErr] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const load = () => {
    api.listVoiceDatasets().then(setDatasets).catch(() => { })
    api.listVoiceJobs().then(setJobs).catch(() => { })
    api.listVoiceModels().then(setModels).catch(() => { })
  }
  useEffect(() => {
    load()
    api.voiceBackends().then(setBackends).catch(() => { })
    const t = setInterval(() => {
      api.listVoiceJobs().then(setJobs).catch(() => { })
    }, 3000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (sel != null) api.listClips(sel).then(setClips).catch(() => setClips([]))
    else setClips([])
  }, [sel])

  const selected = datasets.find((d) => d.id === sel) || null

  return (
    <div className="voice-page">
      <div className="toolbar">
        <h1 className="page-title">声音克隆 / SVC</h1>
        <span className="spacer" />
        <button className="btn" onClick={() => { setErr(''); setCreating(true) }}>+ 新建声音数据集</button>
      </div>

      <div className="voice-split">
        {/* 左：数据集列表 */}
        <aside className="voice-aside card">
          <div className="card-title">声音数据集</div>
          {datasets.length === 0 ? (
            <div className="empty sm">还没有声音数据集。<br />点击右上角新建。</div>
          ) : (
            <div className="voice-ds-list">
              {datasets.map((d) => {
                const b = statusBadge(d.status)
                return (
                  <div
                    key={d.id}
                    className={`voice-ds-item ${sel === d.id ? 'active' : ''}`}
                    onClick={() => setSel(d.id)}
                  >
                    <div className="voice-ds-main">
                      <strong>{d.name}</strong>
                      <span className={b.cls}>{b.text}</span>
                    </div>
                    <div className="muted" style={{ fontSize: 12 }}>
                      音色：{d.speaker} · {d.sample_rate} Hz · {d.clip_count} 段
                      {d.total_seconds > 0 ? ` · ${d.total_seconds.toFixed(0)}s` : ''}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </aside>

        {/* 右：可滚动详情区 */}
        <section className="voice-detail">
          <div className="note" style={{ marginBottom: 16 }}>
            <strong>使用前请确认：</strong>
            <ul>
              <li>本地训练需设置 <code>RVC_DIR</code> 指向 RVC 安装目录（或用 <code>start.sh</code> 自动 clone）；远程训练在主机配置里填“远程 RVC 目录”。</li>
              <li>建议准备 <code>5~30 分钟</code> 干净的目标人声（单人、无伴奏/降噪后），可切成多段上传。</li>
              <li>RVC 训练以 CUDA 为主，本地 Mac (MPS/CPU) 可跑但较慢；大数据集请优先用远程 GPU。</li>
            </ul>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            {!selected ? (
              <div className="empty">从左侧选择一个声音数据集，上传音频并开始训练。</div>
            ) : (
              <DatasetPanel
                dataset={selected}
                clips={clips}
                backends={backends}
                fileRef={fileRef}
                onChanged={() => { load(); api.listClips(selected.id).then(setClips).catch(() => { }) }}
                onDeleted={() => { setSel(null); load() }}
              />
            )}
          </div>

          <div style={{ marginBottom: 16 }}>
            <div className="card-title">训练任务</div>
            {jobs.length === 0 ? (
              <div className="card"><div className="empty sm">还没有训练任务。</div></div>
            ) : (
              <div className="table-card">
                <table>
                  <thead><tr><th>名称</th><th>状态</th><th>进度</th><th>轮次</th><th></th></tr></thead>
                  <tbody>
                    {jobs.map((j) => {
                      const b = statusBadge(j.status)
                      return (
                        <tr key={j.id}>
                          <td>{j.name}</td>
                          <td><span className={b.cls}>{b.text}</span></td>
                          <td style={{ minWidth: 140 }}>
                            <div className="progress"><div className="bar" style={{ width: `${j.progress * 100}%` }} /></div>
                          </td>
                          <td className="nowrap">{j.total_step ? `${j.current_step}/${j.total_step}` : '—'}</td>
                          <td>
                            <div className="row-actions">
                              {j.status === 'running' && (
                                <button className="btn sm ghost" onClick={() => api.stopVoiceJob(j.id).then(load)}>停止</button>
                              )}
                              <button className="btn sm danger" onClick={async () => {
                                if (!confirm(`删除任务「${j.name}」？将一并删除日志与产出模型。`)) return
                                await api.deleteVoiceJob(j.id); load()
                              }}>删除</button>
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div>
            <div className="card-title">声音模型库</div>
            {models.length === 0 ? (
              <div className="card"><div className="empty sm">训练成功后，产出的声音模型会显示在这里。</div></div>
            ) : (
              <div className="table-card">
                <table>
                  <thead><tr><th>名称</th><th>音色</th><th>轮次</th><th>采样率</th><th>索引</th><th></th></tr></thead>
                  <tbody>
                    {models.map((m) => (
                      <tr key={m.id}>
                        <td>{m.name}</td>
                        <td className="muted">{m.speaker || '—'}</td>
                        <td>{m.epoch || '—'}</td>
                        <td className="muted nowrap">{m.sample_rate}</td>
                        <td>{m.has_index ? <span className="badge green">有</span> : <span className="muted">无</span>}</td>
                        <td>
                          <div className="row-actions">
                            <a className="btn sm ghost" href={api.voiceModelDownloadUrl(m.id)}>下载</a>
                            <button className="btn sm danger" onClick={async () => {
                              if (!confirm('删除该声音模型？')) return
                              await api.deleteVoiceModel(m.id); load()
                            }}>删除</button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>
      </div>

      {creating && (
        <CreateModal
          onClose={() => setCreating(false)}
          onCreated={(d) => { setCreating(false); load(); setSel(d.id) }}
          err={err}
          setErr={setErr}
        />
      )}
    </div>
  )
}

function DatasetPanel({
  dataset, clips, backends, fileRef, onChanged, onDeleted,
}: {
  dataset: VoiceDataset
  clips: AudioClip[]
  backends: { name: string; label: string }[]
  fileRef: React.RefObject<HTMLInputElement>
  onChanged: () => void
  onDeleted: () => void
}) {
  const [uploading, setUploading] = useState(false)
  const [training, setTraining] = useState(false)
  const [backend, setBackend] = useState('local_rvc')
  const [epochs, setEpochs] = useState(100)
  const [f0method, setF0method] = useState('rmvpe')
  const [msg, setMsg] = useState('')

  useEffect(() => { if (backends[0]) setBackend(backends[0].name) }, [backends])

  const upload = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setUploading(true); setMsg('')
    try { await api.uploadClips(dataset.id, files); onChanged() }
    catch (e: any) { setMsg(e.message) }
    finally { setUploading(false); if (fileRef.current) fileRef.current.value = '' }
  }

  const train = async () => {
    setTraining(true); setMsg('')
    try {
      const job = await api.createVoiceJob({
        name: `${dataset.speaker || dataset.name}-训练`,
        dataset_id: dataset.id,
        backend,
        params: { total_epoch: epochs, f0_method: f0method, sample_rate: dataset.sample_rate },
      })
      await api.startVoiceJob(job.id)
      setMsg('训练已启动，进度见下方“训练任务”。')
      onChanged()
    } catch (e: any) { setMsg(`启动失败：${e.message}`) }
    finally { setTraining(false) }
  }

  return (
    <div>
      <div className="toolbar" style={{ marginBottom: 8 }}>
        <strong style={{ fontSize: 15 }}>{dataset.name}</strong>
        <span className="muted" style={{ marginLeft: 8, fontSize: 12 }}>
          音色 {dataset.speaker} · {dataset.sample_rate} Hz
        </span>
        <span className="spacer" />
        <button className="btn sm danger" onClick={async () => {
          if (!confirm(`删除数据集「${dataset.name}」及其所有音频？`)) return
          await api.deleteVoiceDataset(dataset.id); onDeleted()
        }}>删除数据集</button>
      </div>

      <div className="voice-upload-bar">
        <input ref={fileRef} type="file" accept="audio/*" multiple style={{ display: 'none' }}
          onChange={(e) => upload(e.target.files)} />
        <button className="btn sm" disabled={uploading} onClick={() => fileRef.current?.click()}>
          {uploading ? '上传中…' : '+ 上传音频'}
        </button>
        <span className="muted" style={{ fontSize: 12 }}>已上传 {clips.length} 段</span>
      </div>

      <div className="voice-clip-list">
        {clips.length === 0 ? (
          <div className="empty sm">还没有音频，点击“上传音频”添加目标人声片段。</div>
        ) : clips.map((c) => (
          <div key={c.filename} className="voice-clip">
            <audio controls preload="none" src={c.audio_url} style={{ height: 32 }} />
            <span className="voice-clip-name" title={c.filename}>{c.filename}</span>
            <span className="muted" style={{ fontSize: 12 }}>
              {c.seconds > 0 ? `${c.seconds}s · ` : ''}{formatBytes(c.size_bytes)}
            </span>
            <button className="btn sm danger" onClick={async () => {
              await api.deleteClip(dataset.id, c.filename); onChanged()
            }}>删除</button>
          </div>
        ))}
      </div>

      <div className="voice-train-box">
        <div className="card-title">开始训练</div>
        <div className="row">
          <div className="field">
            <label>训练后端</label>
            <Select value={backend} onChange={(v) => setBackend(String(v))}
              options={backends.map((b) => ({ value: b.name, label: b.label }))} />
          </div>
          <div className="field">
            <label>训练轮数 (epoch) <span className="help-icon" title="总训练轮数；小数据集 100~200 轮通常足够，过多会过拟合">ⓘ</span></label>
            <input className="input" type="number" value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />
          </div>
          <div className="field">
            <label>基频算法 (f0) <span className="help-icon" title="提取音高的方法，歌声训练推荐 rmvpe，最准且较快">ⓘ</span></label>
            <Select value={f0method} onChange={(v) => setF0method(String(v))} options={F0_OPTIONS} />
          </div>
        </div>
        <div className="toolbar" style={{ marginTop: 8 }}>
          {msg && <span className="muted" style={{ fontSize: 12 }}>{msg}</span>}
          <span className="spacer" />
          <button className="btn" disabled={training || clips.length === 0} onClick={train}>
            {training ? '启动中…' : '启动训练'}
          </button>
        </div>
        {clips.length === 0 && <p className="muted" style={{ fontSize: 12 }}>请先上传音频再开始训练。</p>}
      </div>
    </div>
  )
}

function CreateModal({
  onClose, onCreated, err, setErr,
}: {
  onClose: () => void
  onCreated: (d: VoiceDataset) => void
  err: string
  setErr: (s: string) => void
}) {
  const [form, setForm] = useState({ name: '', speaker: '', sample_rate: 40000 })

  const create = async () => {
    setErr('')
    if (!form.name || !form.speaker) { setErr('数据集名称和音色名必填'); return }
    try {
      const d = await api.createVoiceDataset(form)
      onCreated(d)
    } catch (e: any) { setErr(e.message) }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal modal-form" onClick={(e) => e.stopPropagation()}>
        <div className="modal-form-body">
          <div className="toolbar" style={{ marginBottom: 16 }}>
            <strong style={{ fontSize: 16 }}>新建声音数据集</strong>
            <span className="spacer" />
            <button className="btn sm ghost" onClick={onClose}>关闭</button>
          </div>
          <div className="row">
            <div className="field">
              <label>数据集名称</label>
              <input className="input" value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="如：歌手A 干声" />
            </div>
            <div className="field">
              <label>音色名 (speaker) <span className="help-icon" title="目标人声的标识，将作为模型名">ⓘ</span></label>
              <input className="input" value={form.speaker}
                onChange={(e) => setForm({ ...form, speaker: e.target.value })} placeholder="如：singerA" />
            </div>
          </div>
          <div className="field">
            <label>采样率 <span className="help-icon" title="训练目标采样率，需与上传音频一致；40000 Hz 为通用推荐">ⓘ</span></label>
            <Select value={form.sample_rate} onChange={(v) => setForm({ ...form, sample_rate: Number(v) })}
              options={SR_OPTIONS} />
          </div>
          {err && <p className="badge red">{err}</p>}
          <div className="toolbar" style={{ marginTop: 8 }}>
            <span className="spacer" />
            <button className="btn ghost" onClick={onClose}>取消</button>
            <button className="btn" onClick={create}>创建</button>
          </div>
        </div>
      </div>
    </div>
  )
}
