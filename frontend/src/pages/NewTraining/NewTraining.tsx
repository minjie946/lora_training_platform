import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api, BaseModel, Dataset } from '../../api/client'
import Select from '../../components/Select/Select'
import './NewTraining.css'

// 各训练参数的作用说明，hover label 上的 ⓘ 显示
const PARAM_HELP: Record<string, string> = {
  resolution: '训练分辨率。显存越大可越高；24G 建议 768，48G 可用 1024。越高越清晰但更慢更吃显存。',
  network_dim: 'LoRA 的秩(rank)，决定模型容量。越大能学到的细节越多但文件更大、易过拟合；人物常用 8–32。',
  network_alpha: '缩放系数，配合 rank 控制学习强度。常设为 rank 的一半或相等。',
  max_train_epochs: '训练轮数。所有图片完整训练一遍为 1 轮。轮数过多会过拟合。',
  train_batch_size: '每步同时训练的图片数。越大越稳但更吃显存；Mac 上通常设 1。',
  save_every_n_epochs: '每隔几轮保存一个权重，便于挑选效果最好的中间结果。',
  unet_lr: 'UNet(画面主体)的学习率。过大易崩、过小学不动；SDXL 常用 1e-4 量级。',
  text_encoder_lr: '文本编码器的学习率，影响对提示词的响应。通常比 UNet 略小。',
  seed: '随机种子，固定后可复现同一结果。',
  optimizer_type: '优化器算法。Mac MPS 不支持 8bit 优化器，固定用 AdamW。',
  lr_scheduler: '学习率调度策略，控制学习率随训练的变化曲线。',
  mixed_precision: '混合精度。Mac MPS 不支持 fp16/bf16，固定为 no(全精度)。',
  resource_tier: '训练时占用本机资源的档位。\n\n· 低占用 → 限制 MPS 最多用约一半统一内存、单 CPU 线程，训练更慢但电脑更流畅，适合边训练边用机器\n· 均衡（默认）→ MPS 上限约 80%，兼顾速度与可用性\n· 全速 → 不限制内存、放开线程，训练最快但机器会明显吃紧',
}

export default function NewTraining() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [backends, setBackends] = useState<{ name: string; label: string }[]>([])
  const [baseModels, setBaseModels] = useState<BaseModel[]>([])
  const [defaults, setDefaults] = useState<Record<string, any>>({})
  const [datasetId, setDatasetId] = useState<number | null>(null)
  const [name, setName] = useState('')
  const [backend, setBackend] = useState('local_mps')
  const [baseModel, setBaseModel] = useState('')
  const [resourceTier, setResourceTier] = useState('balanced')
  const [params, setParams] = useState<Record<string, any>>({})
  const [err, setErr] = useState('')
  const [sp] = useSearchParams()
  const nav = useNavigate()
  const { id: editIdParam } = useParams()
  const editId = editIdParam ? Number(editIdParam) : null
  const isEdit = editId != null

  useEffect(() => {
    api.listDatasets().then((ds) => {
      setDatasets(ds)
      if (isEdit) return // 编辑模式下由下方 loadJob 决定数据集
      const pre = Number(sp.get('dataset'))
      const initId = pre && ds.some((d) => d.id === pre) ? pre : (ds.length ? ds[0].id : null)
      setDatasetId(initId)
      const initDs = ds.find((d) => d.id === initId)
      if (initDs) setBaseModel(initDs.base_model || '')
    })
    api.backends().then((b) => { setBackends(b); if (b[0] && !isEdit) setBackend(b[0].name) })
    api.defaults().then((d) => { setDefaults(d); if (!isEdit) setParams(d) })
    api.baseModels().then((r) => setBaseModels(r.models)).catch(() => { })
  }, [])

  // 编辑模式：拉取任务并回填表单
  useEffect(() => {
    if (!isEdit) return
    api.getJob(editId!).then((job) => {
      setName(job.name)
      setDatasetId(job.dataset_id)
      setBackend(job.backend)
      setBaseModel(job.base_model || '')
      const p = { ...(job.params || {}) }
      if (p.resource_tier) { setResourceTier(String(p.resource_tier)); delete p.resource_tier }
      setParams(p)
    }).catch(() => setErr('加载任务失败'))
  }, [editId])

  const selected = datasets.find((d) => d.id === datasetId)

  // 数据集选中后，底模默认跟随数据集设定（可在下方覆盖）。
  // 编辑模式的首次回填不覆盖已保存的底模，只有用户主动切换数据集才跟随。
  const followBase = (id: number | null) => {
    const ds = datasets.find((d) => d.id === id)
    setBaseModel(ds?.base_model || '')
  }

  const totalSteps = useMemo(() => {
    if (!selected) return 0
    const epochs = Number(params.max_train_epochs || 0)
    const batch = Math.max(1, Number(params.train_batch_size || 1))
    return Math.floor((selected.image_count * selected.repeat * epochs) / batch)
  }, [selected, params])

  const setP = (k: string, v: any) => setParams((p) => ({ ...p, [k]: v }))

  const submit = async () => {
    setErr('')
    if (!datasetId) { setErr('请选择数据集'); return }
    if (!name) { setErr('请填写任务名称'); return }
    // 去掉留空的数字字段，交给后端用默认值
    const cleanParams = Object.fromEntries(
      Object.entries(params).filter(([, v]) => v !== '' && v !== null && v !== undefined)
    )
    cleanParams.resource_tier = resourceTier
    try {
      const body = { name, dataset_id: datasetId, backend, base_model: baseModel || undefined, params: cleanParams }
      const job = isEdit ? await api.updateJob(editId!, body) : await api.createJob(body)
      nav(`/jobs/${job.id}`)
    } catch (e: any) { setErr(e.message) }
  }

  const helpIcon = (key: string) =>
    PARAM_HELP[key] ? (
      <span className="help-icon tip" data-tip={PARAM_HELP[key]} tabIndex={0}>ⓘ</span>
    ) : null

  const numField = (key: string, label: string, step = 'any') => (
    <div className="field">
      <label>{label} {helpIcon(key)}</label>
      <input className="input" type="number" step={step}
        value={params[key] ?? ''}
        onChange={(e) => {
          const v = e.target.value
          setP(key, v === '' ? '' : Number(v))
        }} />
    </div>
  )

  return (
    <div>
      <div className="toolbar">
        <h1 className="page-title">{isEdit ? '编辑训练任务' : '新建训练'}</h1>
        <span className="spacer" />
        <Link className="btn ghost sm" to={isEdit ? `/jobs/${editId}` : '/jobs'}>返回</Link>
      </div>
      <p className="page-sub">
        {isEdit
          ? '修改任务配置后保存；若改动了训练参数，任务会重置为待启动，并自动清理该任务此前的日志、检查点与产出模型。'
          : '默认参数对齐 Mac M4 Pro 指南；底模默认跟随所选数据集，可在下方覆盖。鼠标悬停参数旁的 ⓘ 查看说明。'}
      </p>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="row">
          <div className="field">
            <label>任务名称</label>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)}
              placeholder="如：小樱-第一轮" />
          </div>
          <div className="field">
            <label>数据集</label>
            <Select
              value={datasetId ?? ''}
              onChange={(v) => { const id = Number(v); setDatasetId(id); followBase(id) }}
              placeholder="选择数据集"
              options={datasets.map((d) => ({ value: d.id, label: `${d.name}（${d.image_count} 张）` }))}
            />
          </div>
          <div className="field">
            <label>训练后端</label>
            <Select
              value={backend}
              onChange={(v) => setBackend(String(v))}
              options={backends.map((b) => ({ value: b.name, label: b.label }))}
            />
          </div>
        </div>
        <div className="row">
          <div className="field">
            <label>底模 (base model) <span className="help-icon tip" tabIndex={0} data-tip="LoRA 训练所基于的大模型。默认跟随数据集设定，可在此覆盖。SDXL/SD1.5 会自动选择对应训练脚本。">ⓘ</span></label>
            <Select
              value={baseModel}
              onChange={(v) => setBaseModel(String(v))}
              placeholder="跟随数据集 / 选择底模"
              options={baseModels.map((m) => ({
                value: m.filename,
                label: `${m.label}（${m.is_sdxl ? 'SDXL' : 'SD1.5'} · ${m.style === 'realistic' ? '写实' : '动漫'}）`,
              }))}
            />
            {selected && selected.base_model && baseModel === selected.base_model && (
              <p className="muted" style={{ fontSize: 12, marginTop: 6 }}>默认使用该数据集设定的底模</p>
            )}
          </div>
          <div className="field">
            <label>资源占用 {helpIcon('resource_tier')}</label>
            <Select
              value={resourceTier}
              onChange={(v) => setResourceTier(String(v))}
              options={[
                { value: 'low', label: '低占用（电脑更流畅）' },
                { value: 'balanced', label: '均衡（默认）' },
                { value: 'full', label: '全速（训练最快）' },
              ]}
            />
            <p className="muted" style={{ fontSize: 12, marginTop: 6 }}>
              {resourceTier === 'low' && '限制显存约 50% + 单线程，训练变慢但可边训边用机器'}
              {resourceTier === 'balanced' && '显存上限约 80%，兼顾速度与可用性'}
              {resourceTier === 'full' && '不限显存、放开线程，机器会明显吃紧'}
            </p>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <strong>训练参数</strong>
        <div className="row" style={{ marginTop: 12 }}>
          <div className="field">
            <label>分辨率 (48G→1024 / 24G→768) {helpIcon('resolution')}</label>
            <Select
              value={params.resolution ?? 768}
              onChange={(v) => setP('resolution', Number(v))}
              options={[{ value: 768, label: '768' }, { value: 1024, label: '1024' }]}
            />
          </div>
          {numField('network_dim', 'Rank (network_dim)')}
          {numField('network_alpha', 'Alpha (network_alpha)')}
        </div>
        <div className="row">
          {numField('max_train_epochs', '训练轮数 (epochs)')}
          {numField('train_batch_size', 'Batch Size')}
          {numField('save_every_n_epochs', '每 N 轮保存')}
        </div>
        <div className="row">
          {numField('unet_lr', 'UNet 学习率')}
          {numField('text_encoder_lr', 'TextEncoder 学习率')}
          {numField('seed', '随机种子')}
        </div>
        <div className="row">
          <div className="field">
            <label>优化器 {helpIcon('optimizer_type')}</label>
            <input className="input" value={params.optimizer_type ?? 'AdamW'} disabled />
          </div>
          <div className="field">
            <label>学习率调度 {helpIcon('lr_scheduler')}</label>
            <Select
              value={params.lr_scheduler ?? 'cosine_with_restarts'}
              onChange={(v) => setP('lr_scheduler', String(v))}
              options={[
                { value: 'cosine_with_restarts', label: 'cosine_with_restarts' },
                { value: 'cosine', label: 'cosine' },
              ]}
            />
          </div>
          <div className="field">
            <label>混合精度 {helpIcon('mixed_precision')}</label>
            <input className="input" value={params.mixed_precision ?? ''} disabled />
          </div>
        </div>
        <p className="muted">
          预计总步数：<strong style={{ color: 'var(--accent-2)' }}>{totalSteps}</strong>
          {'  '}（图片 × repeat × epochs ÷ batch；建议 1200–2000 步，避免过拟合）
        </p>
      </div>

      {err && <p className="badge red">{err}</p>}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button className="btn" onClick={submit}>{isEdit ? '保存修改' : '创建任务'}</button>
        {isEdit && (
          <button className="btn ghost" onClick={() => nav(`/jobs/${editId}`)}>取消</button>
        )}
      </div>
    </div>
  )
}
