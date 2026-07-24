import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, BaseModel, Dataset } from '../../api/client'
import { statusBadge } from '../../components/helpers'
import Select from '../../components/Select/Select'
import PageHeader from '../../components/PageHeader/PageHeader'
import './Datasets.css'

const STYLE_LABEL: Record<string, string> = { anime: '动漫', realistic: '写实' }

export default function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [creating, setCreating] = useState(false)
  const [importing, setImporting] = useState(false)
  const [baseModels, setBaseModels] = useState<BaseModel[]>([])
  const [form, setForm] = useState({ name: '', concept: '', repeat: 10, trigger_word: '', base_model: '' })
  const [importForm, setImportForm] = useState({ name: '', concept: '', repeat: 10, trigger_word: '', base_model: '' })
  const [archive, setArchive] = useState<File | null>(null)
  const [err, setErr] = useState('')
  const nav = useNavigate()

  const load = () => api.listDatasets().then(setDatasets).catch((e) => setErr(e.message))
  useEffect(() => {
    load()
    api.baseModels().then((r) => {
      setBaseModels(r.models)
      const def = r.models.find((m) => m.is_default) || r.models[0]
      if (def) {
        setForm((f) => ({ ...f, base_model: f.base_model || def.filename }))
        setImportForm((f) => ({ ...f, base_model: f.base_model || def.filename }))
      }
    }).catch(() => { })
  }, [])

  const baseLabel = (filename: string) => {
    const m = baseModels.find((b) => b.filename === filename)
    if (!m) return filename || '默认'
    return `${m.label}（${m.is_sdxl ? 'SDXL' : 'SD1.5'} · ${STYLE_LABEL[m.style] || m.style}）`
  }

  const selectedBase = baseModels.find((b) => b.filename === form.base_model)
  const selectedImportBase = baseModels.find((b) => b.filename === importForm.base_model)

  const openCreate = () => {
    setErr('')
    setCreating(true)
  }

  const openImport = () => {
    setErr('')
    setArchive(null)
    setImporting(true)
  }

  const create = async () => {
    setErr('')
    if (!form.name || !form.concept) { setErr('名称和概念名必填'); return }
    try {
      const d = await api.createDataset(form)
      setCreating(false)
      setForm({ name: '', concept: '', repeat: 10, trigger_word: '', base_model: form.base_model })
      nav(`/datasets/${d.id}`)
    } catch (e: any) { setErr(e.message) }
  }

  const importZip = async () => {
    setErr('')
    if (!importForm.name || !importForm.concept) { setErr('名称和概念名必填'); return }
    if (!archive) { setErr('请选择已标注数据集压缩包（.zip）'); return }
    try {
      const r = await api.importDataset({ ...importForm, archive })
      setImporting(false)
      setImportForm({ name: '', concept: '', repeat: 10, trigger_word: '', base_model: importForm.base_model })
      setArchive(null)
      alert(r.detail)
      nav(`/datasets/${r.dataset.id}`)
    } catch (e: any) { setErr(e.message) }
  }

  const remove = async (id: number) => {
    if (!confirm('确认删除该数据集及其所有图片？')) return
    await api.deleteDataset(id)
    load()
  }

  return (
    <div className="page">
      <PageHeader
        title="数据集"
        actions={
          <>
            <button className="btn ghost" onClick={openImport}>导入已标注包</button>
            <button className="btn" onClick={openCreate}>+ 新建数据集</button>
          </>
        }
      />
      <div className="page-body">

        <div className="table-card">
          {datasets.length === 0 ? (
            <div className="empty">还没有数据集，点击右上角新建。</div>
          ) : (
            <table>
              <thead>
                <tr><th>名称</th><th>概念名</th><th>触发词</th><th>底模</th><th>图片数</th><th>状态</th><th></th></tr>
              </thead>
              <tbody>
                {datasets.map((d) => {
                  const b = statusBadge(d.status)
                  return (
                    <tr key={d.id}>
                      <td><Link className="linkish" to={`/datasets/${d.id}`}>{d.name}</Link></td>
                      <td>{d.repeat}_{d.concept}</td>
                      <td>{d.trigger_word || '—'}</td>
                      <td className="muted">{baseLabel(d.base_model)}</td>
                      <td>{d.image_count}</td>
                      <td><span className={b.cls}>{b.text}</span></td>
                      <td><button className="btn sm danger" onClick={() => remove(d.id)}>删除</button></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>

        {creating && (
          <div className="modal-backdrop" onClick={() => setCreating(false)}>
            <div className="modal modal-form" onClick={(e) => e.stopPropagation()}>
              <div className="modal-form-body">
                <div className="toolbar" style={{ marginBottom: 16 }}>
                  <strong style={{ fontSize: 16 }}>新建数据集</strong>
                  <span className="spacer" />
                  <button className="btn sm ghost" onClick={() => setCreating(false)}>关闭</button>
                </div>
                <div className="row">
                  <div className="field">
                    <label>名称</label>
                    <input className="input" value={form.name}
                      onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="如：角色小樱" />
                  </div>
                  <div className="field">
                    <label>概念名 (concept)</label>
                    <input className="input" value={form.concept}
                      onChange={(e) => setForm({ ...form, concept: e.target.value })} placeholder="如：mychar" />
                  </div>
                </div>
                <div className="row">
                  <div className="field">
                    <label>Repeat 次数</label>
                    <input className="input" type="number" value={form.repeat}
                      onChange={(e) => setForm({ ...form, repeat: Number(e.target.value) })} />
                  </div>
                  <div className="field">
                    <label>触发词 (trigger word)</label>
                    <input className="input" value={form.trigger_word}
                      onChange={(e) => setForm({ ...form, trigger_word: e.target.value })} placeholder="通常与概念名相同" />
                  </div>
                </div>
                <div className="field">
                  <label>底模 (base model)</label>
                  <Select
                    value={form.base_model}
                    onChange={(v) => setForm({ ...form, base_model: String(v) })}
                    placeholder="选择训练底模"
                    options={baseModels.map((m) => ({
                      value: m.filename,
                      label: `${m.label}（${m.is_sdxl ? 'SDXL' : 'SD1.5'} · ${STYLE_LABEL[m.style] || m.style}）`,
                    }))}
                  />
                  {selectedBase && (
                    <p className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      {selectedBase.style === 'realistic'
                        ? '写实风格：打标将使用 BLIP 生成自然语言描述'
                        : '动漫风格：打标将使用 WD14 生成 booru 标签'}
                      ；训练将使用 {selectedBase.is_sdxl ? 'SDXL' : 'SD1.5'} 脚本。该数据集训练时默认用此底模。
                    </p>
                  )}
                </div>
                {err && <p className="badge red">{err}</p>}
                <div className="toolbar" style={{ marginTop: 8 }}>
                  <span className="spacer" />
                  <button className="btn ghost" onClick={() => setCreating(false)}>取消</button>
                  <button className="btn" onClick={create}>创建</button>
                </div>
              </div>
            </div>
          </div>
        )}

        {importing && (
          <div className="modal-backdrop" onClick={() => setImporting(false)}>
            <div className="modal modal-form" onClick={(e) => e.stopPropagation()}>
              <div className="modal-form-body">
                <div className="toolbar" style={{ marginBottom: 16 }}>
                  <strong style={{ fontSize: 16 }}>导入已标注数据集</strong>
                  <span className="spacer" />
                  <button className="btn sm ghost" onClick={() => setImporting(false)}>关闭</button>
                </div>
                <div className="note">
                  <strong>压缩包要求：</strong>
                  <ul>
                    <li>目前支持 <code>.zip</code>。</li>
                    <li>图片文件与同名 <code>.txt</code> 标注可放在任意层级目录，例如 <code>1.png</code> + <code>1.txt</code>。</li>
                    <li>导入后会自动创建数据集、生成缩略图，并保留压缩包中的已有标注。</li>
                  </ul>
                </div>
                <div className="row">
                  <div className="field">
                    <label>名称</label>
                    <input className="input" value={importForm.name}
                      onChange={(e) => setImportForm({ ...importForm, name: e.target.value })} placeholder="如：角色小樱（已标注）" />
                  </div>
                  <div className="field">
                    <label>概念名 (concept)</label>
                    <input className="input" value={importForm.concept}
                      onChange={(e) => setImportForm({ ...importForm, concept: e.target.value })} placeholder="如：mychar" />
                  </div>
                </div>
                <div className="row">
                  <div className="field">
                    <label>Repeat 次数</label>
                    <input className="input" type="number" value={importForm.repeat}
                      onChange={(e) => setImportForm({ ...importForm, repeat: Number(e.target.value) })} />
                  </div>
                  <div className="field">
                    <label>触发词 (trigger word)</label>
                    <input className="input" value={importForm.trigger_word}
                      onChange={(e) => setImportForm({ ...importForm, trigger_word: e.target.value })} placeholder="通常与概念名相同" />
                  </div>
                </div>
                <div className="field">
                  <label>底模 (base model)</label>
                  <Select
                    value={importForm.base_model}
                    onChange={(v) => setImportForm({ ...importForm, base_model: String(v) })}
                    placeholder="选择训练底模"
                    options={baseModels.map((m) => ({
                      value: m.filename,
                      label: `${m.label}（${m.is_sdxl ? 'SDXL' : 'SD1.5'} · ${STYLE_LABEL[m.style] || m.style}）`,
                    }))}
                  />
                  {selectedImportBase && (
                    <p className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                      导入后该数据集训练时默认使用该底模；当前选择为
                      {' '}{selectedImportBase.is_sdxl ? 'SDXL' : 'SD1.5'}
                      {' · '}{STYLE_LABEL[selectedImportBase.style] || selectedImportBase.style}。
                    </p>
                  )}
                </div>
                <div className="field">
                  <label>已标注压缩包</label>
                  <input
                    className="input"
                    type="file"
                    accept=".zip,application/zip"
                    onChange={(e) => setArchive(e.target.files?.[0] || null)}
                  />
                  <p className="muted" style={{ fontSize: 12, marginTop: 6 }}>
                    {archive ? `已选择：${archive.name}` : '请选择包含图片和同名 .txt 标注的 zip 包'}
                  </p>
                </div>
                {err && <p className="badge red">{err}</p>}
                <div className="toolbar" style={{ marginTop: 8 }}>
                  <span className="spacer" />
                  <button className="btn ghost" onClick={() => setImporting(false)}>取消</button>
                  <button className="btn" onClick={importZip}>导入</button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
