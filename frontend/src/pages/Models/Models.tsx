import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, LoraModel } from '../../api/client'
import { formatBytes } from '../../components/helpers'
import PageHeader from '../../components/PageHeader/PageHeader'
import './Models.css'

export default function Models() {
  const [models, setModels] = useState<LoraModel[]>([])
  const [selected, setSelected] = useState<Set<number>>(new Set())

  const load = () =>
    api
      .listModels()
      .then((rows) => {
        setModels(rows)
        setSelected(new Set())
      })
      .catch(() => { })
  useEffect(() => { load() }, [])

  const toggle = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const toggleAll = () => {
    setSelected((prev) =>
      prev.size === models.length ? new Set() : new Set(models.map((m) => m.id))
    )
  }

  const remove = async (id: number) => {
    if (!confirm('确认删除该模型文件？output 中的对应文件也会被删除。')) return
    await api.deleteModel(id)
    load()
  }

  const removeSelected = async () => {
    const ids = Array.from(selected)
    if (ids.length === 0) return
    if (!confirm(`确认删除选中的 ${ids.length} 个模型？output 中的对应文件也会被删除。`))
      return
    await api.bulkDeleteModels(ids)
    load()
  }

  const allChecked = models.length > 0 && selected.size === models.length

  return (
    <div className="page">
      <PageHeader
        title="模型库"
        subtitle="所有训练产出的 LoRA 权重"
        actions={selected.size > 0 ? (
          <button className="btn danger" onClick={removeSelected}>
            删除选中 ({selected.size})
          </button>
        ) : undefined}
      />
      <div className="page-body">
        <div className="table-card">
          {models.length === 0 ? (
            <div className="empty">还没有产出的模型。完成一次训练后会出现在这里。</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th className="cell-check">
                    <input
                      type="checkbox"
                      checked={allChecked}
                      onChange={toggleAll}
                      aria-label="全选"
                    />
                  </th>
                  <th>文件</th>
                  <th>来源任务</th>
                  <th>底模</th>
                  <th>Epoch</th>
                  <th>大小</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id} className={selected.has(m.id) ? 'row-selected' : ''}>
                    <td className="cell-check">
                      <input
                        type="checkbox"
                        checked={selected.has(m.id)}
                        onChange={() => toggle(m.id)}
                        aria-label={`选择 ${m.name}`}
                      />
                    </td>
                    <td>{m.name}</td>
                    <td><Link className="linkish" to={`/jobs/${m.job_id}`}>#{m.job_id}</Link></td>
                    <td className="muted">{m.base_model ? m.base_model.replace(/\.safetensors$/, '') : '—'}</td>
                    <td>{m.epoch}</td>
                    <td>{formatBytes(m.file_size)}</td>
                    <td>
                      <a className="btn sm" href={api.modelDownloadUrl(m.id)}>下载</a>{' '}
                      <button className="btn sm danger" onClick={() => remove(m.id)}>删除</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}
