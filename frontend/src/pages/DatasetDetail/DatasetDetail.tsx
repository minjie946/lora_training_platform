import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  Tag, Upload, Wand2, Edit3, Eye, Trash2, X, Plus, Save, Pencil, HelpCircle, ShieldCheck,
  CheckSquare, Square, ListChecks, FolderInput, Loader2,
} from 'lucide-react'
import { api, BaseModel, Dataset, ImageItem } from '../../api/client'
import Select from '../../components/Select/Select'
import './DatasetDetail.css'

export default function DatasetDetail() {
  const { id } = useParams()
  const dsId = Number(id)
  const [ds, setDs] = useState<Dataset | null>(null)
  const [images, setImages] = useState<ImageItem[]>([])
  const [baseModels, setBaseModels] = useState<BaseModel[]>([])
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState(false)
  const [uploadPct, setUploadPct] = useState<number | null>(null)
  const [captioning, setCaptioning] = useState(false)
  const [checkingQuality, setCheckingQuality] = useState(false)
  const [editing, setEditing] = useState<ImageItem | null>(null)
  const [preview, setPreview] = useState<ImageItem | null>(null)
  const [capMethod, setCapMethod] = useState('wd14') // auto | wd14 | florence2 | blip，默认 WD14（写实/动漫均可，带置信度）
  const [wd14Model, setWd14Model] = useState('swinv2-v3') // swinv2-v3（快）| eva02-large-v3（更准更慢）
  const [threshold, setThreshold] = useState(0.35)
  const [excludeBodyFace, setExcludeBodyFace] = useState(false)
  const [excludeTags, setExcludeTags] = useState('')
  const [editingBase, setEditingBase] = useState(false)
  // 图片筛选：按 WD14 置信度（high/mid/low）与质量检测结果（ok/warn/bad）过滤
  const [confFilter, setConfFilter] = useState<'all' | 'high' | 'mid' | 'low'>('all')
  const [qualFilter, setQualFilter] = useState<'all' | 'ok' | 'warn' | 'bad'>('all')
  // 批量删除：选择模式 + 已选文件名集合
  const [selectMode, setSelectMode] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [bulkDeleting, setBulkDeleting] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const load = () => {
    api.getDataset(dsId).then(setDs).catch(() => { })
    return api.listImages(dsId).then(setImages).catch(() => { })
  }

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  // 轮询后台打标状态，直到结束（done/failed/idle），期间保持“打标中”并刷新图片。
  const startPolling = () => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const s = await api.captionStatus(dsId)
        if (s.caption_status === 'running') {
          setCaptioning(true)
        } else {
          setCaptioning(false)
          stopPolling()
          setMsg(s.detail || '')
          load()
        }
      } catch { /* 网络抖动时下次轮询再试 */ }
    }, 2000)
  }

  useEffect(() => {
    load()
    // 进入页面时读取后台打标状态：若仍在打标则恢复“打标中”并继续轮询。
    api.captionStatus(dsId).then((s) => {
      if (s.caption_status === 'running') {
        setCaptioning(true)
        startPolling()
      } else if (s.detail) {
        setMsg(s.detail)
      }
    }).catch(() => { })
    return stopPolling
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dsId])
  useEffect(() => { api.baseModels().then((r) => setBaseModels(r.models)).catch(() => { }) }, [])

  const upload = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setBusy(true); setMsg(''); setUploadPct(0)
    try {
      await api.uploadImages(dsId, files, (pct) => setUploadPct(pct))
      setMsg(`上传了 ${files.length} 张图片`)
      load()
    } catch (e: any) { setMsg(e.message) } finally { setBusy(false); setUploadPct(null) }
  }

  // 从筛选产出的 single/ 目录一键导入
  const [importOpen, setImportOpen] = useState(false)
  const [importSources, setImportSources] = useState<{ name: string; source_dir: string; image_count: number }[] | null>(null)
  const [importing, setImporting] = useState('')  // 正在导入的 source_dir
  const openImportPicker = () => {
    setImportOpen(true); setImportSources(null)
    api.importSources().then(setImportSources).catch(() => setImportSources([]))
  }
  const doImport = async (source_dir: string) => {
    setImporting(source_dir); setMsg('')
    try {
      const imgs = await api.importFromDir(dsId, source_dir)
      setImages(imgs)
      setDs((d) => (d ? { ...d, image_count: imgs.length } : d))
      setImportOpen(false)
      setMsg(`已从 ${source_dir} 导入图片`)
    } catch (e: any) { setMsg(e.message) } finally { setImporting('') }
  }

  const saveCaption = async (filename: string, caption: string) => {
    await api.updateCaption(dsId, filename, caption)
    setImages((prev) => prev.map((im) => (im.filename === filename ? { ...im, caption } : im)))
    setEditing((p) => (p && p.filename === filename ? { ...p, caption } : p))
  }

  const autoCaption = async () => {
    setCaptioning(true); setMsg('')
    try {
      // 后台异步打标：立即返回，随后轮询状态（刷新页面/退出重进都能恢复）。
      await api.autoCaption(dsId, {
        method: capMethod,
        threshold,
        exclude_body_face: excludeBodyFace,
        exclude_tags: excludeTags.split(',').map((s) => s.trim()).filter(Boolean),
        wd14_model: wd14Model,
      })
      startPolling()
    } catch (e: any) { setCaptioning(false); setMsg(e.message) }
  }

  const removeImage = async (filename: string) => {
    if (!confirm('确认删除该图片？')) return
    await api.deleteImage(dsId, filename)
    load()
  }

  // 退出选择模式并清空选择
  const exitSelect = () => { setSelectMode(false); setSelected(new Set()) }

  const toggleSelect = (filename: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(filename)) next.delete(filename); else next.add(filename)
      return next
    })
  }

  // 全选/取消全选（仅针对当前筛选后可见的图片）
  const toggleSelectAll = (visible: ImageItem[]) => {
    setSelected((prev) => {
      const allSelected = visible.length > 0 && visible.every((im) => prev.has(im.filename))
      if (allSelected) return new Set()
      return new Set(visible.map((im) => im.filename))
    })
  }

  const bulkDelete = async () => {
    const names = Array.from(selected)
    if (names.length === 0) return
    if (!confirm(`确认删除选中的 ${names.length} 张图片？此操作不可恢复。`)) return
    setBulkDeleting(true); setMsg('')
    try {
      const r = await api.bulkDeleteImages(dsId, names)
      setMsg(`已删除 ${r.deleted} 张图片`)
      exitSelect()
      await load()
    } catch (e: any) { setMsg(e.message) } finally { setBulkDeleting(false) }
  }

  const runQualityCheck = async () => {
    if (images.length === 0) return
    setCheckingQuality(true); setMsg('')
    try {
      const r = await api.checkQuality(dsId)
      const flagged = r.warn + r.bad
      setMsg(flagged > 0
        ? `质量检测完成：共 ${r.total} 张，${flagged} 张建议核对（${r.bad} 张问题较大）`
        : `质量检测完成：共 ${r.total} 张，均无明显问题`)
      await load()
    } catch (e: any) { setMsg(e.message) } finally { setCheckingQuality(false) }
  }

  const saveBaseModel = async (filename: string) => {
    setEditingBase(false)
    if (!ds || filename === ds.base_model) return
    setMsg('')
    try {
      const updated = await api.updateDataset(dsId, { base_model: filename })
      setDs(updated)
      setMsg('底模已更新')
    } catch (e: any) { setMsg(e.message) }
  }

  if (!ds) return <p className="muted page-body">加载中…</p>

  const baseInfo = baseModels.find((b) => b.filename === ds.base_model)
  const baseText = baseInfo
    ? `${baseInfo.label}（${baseInfo.is_sdxl ? 'SDXL' : 'SD1.5'} · ${baseInfo.style === 'realistic' ? '写实' : '动漫'}）`
    : (ds.base_model || '默认底模')
  const isRealistic = baseInfo?.style === 'realistic'
  const autoCaptioner = isRealistic ? 'Florence-2（自然语言描述）' : 'WD14（booru 标签）'
  const effectiveMethod =
    capMethod === 'wd14' ? 'WD14（booru 标签）'
      : capMethod === 'florence2' ? 'Florence-2（自然语言描述）'
        : capMethod === 'blip' ? 'BLIP（自然语言描述）'
          : `自动（跟随底模：${autoCaptioner}）`
  const showThreshold = capMethod === 'wd14' || (capMethod === 'auto' && !isRealistic)
  // WD14 模型选择器仅在实际会走 WD14 时显示。
  const showWd14Model = showThreshold

  // 依据 WD14 置信度与质量检测结果过滤图片。
  const hasConfData = images.some((im) => imageConfidenceLevel(im) !== null)
  const hasQualData = images.some((im) => im.quality != null)
  const filteredImages = images.filter((im) => {
    if (confFilter !== 'all' && imageConfidenceLevel(im) !== confFilter) return false
    if (qualFilter !== 'all' && (im.quality?.level ?? null) !== qualFilter) return false
    return true
  })
  const filtering = confFilter !== 'all' || qualFilter !== 'all'

  return (
    <div className="detail-page">
      {/* 固定头部：标题 + 元信息 + 操作 */}
      <div className="detail-header">
        <div className="detail-header-info">
          <h1 className="detail-title">{ds.name}</h1>
          <span className="detail-divider" />
          <span className="muted detail-meta">
            {ds.repeat}_{ds.concept} · 触发词 <span className="detail-strong">{ds.trigger_word || '未设置'}</span>
            {' · 底模 '}
            {editingBase ? (
              <span className="base-edit-inline">
                <Select
                  value={ds.base_model}
                  onChange={(v) => saveBaseModel(String(v))}
                  placeholder="选择底模"
                  options={baseModels.map((m) => ({
                    value: m.filename,
                    label: `${m.label}（${m.is_sdxl ? 'SDXL' : 'SD1.5'} · ${m.style === 'realistic' ? '写实' : '动漫'}）`,
                  }))}
                />
                <button className="base-edit-cancel" onClick={() => setEditingBase(false)}>取消</button>
              </span>
            ) : (
              <>
                <span className="detail-strong">{baseText}</span>
                <button className="base-edit-btn" title="修改底模" onClick={() => setEditingBase(true)}>
                  <Pencil size={13} />
                </button>
              </>
            )}
          </span>
        </div>
        <div className="detail-header-actions">
          <Link className="btn ghost sm" to="/datasets">返回</Link>
          <Link className="btn sm" to={`/jobs/new?dataset=${ds.id}`}>用此数据集训练</Link>
        </div>
      </div>

      {/* 独立滚动的内容区：头部固定，仅此区域滚动 */}
      <div className="detail-scroll">
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="toolbar">
            <button className="btn ghost" disabled={busy || captioning} onClick={() => fileRef.current?.click()}>
              <Upload size={16} />
              {uploadPct !== null ? `上传中… ${uploadPct}%` : '上传图片'}
            </button>
            <button className="btn ghost" disabled={busy || captioning} onClick={openImportPicker}>
              <FolderInput size={16} className="icon-accent" />从筛选目录导入
            </button>
            <button className="btn ghost" disabled={busy || captioning || images.length === 0} onClick={autoCaption}>
              {captioning ? (<><span className="spinner" />打标中…</>) : (<><Wand2 size={16} className="icon-accent" />自动打标 / 注入触发词</>)}
            </button>
            <button className="btn ghost tip" data-tip={'检测图片是否适合作为训练素材：分辨率、清晰度（模糊）、曝光、宽高比，以及是否有清晰正脸/多人脸。\n结果仅作提醒，不影响训练。'} disabled={busy || checkingQuality || images.length === 0} onClick={runQualityCheck}>
              {checkingQuality ? (<><span className="spinner" />检测中…</>) : (<><ShieldCheck size={16} className="icon-accent" />检测质量</>)}
            </button>
            <input ref={fileRef} type="file" accept="image/*,.heic,.heif" multiple hidden
              onChange={(e) => upload(e.target.files)} />
            {images.length > 0 && (
              selectMode ? (
                <button className="btn ghost" onClick={exitSelect}>
                  <X size={16} />取消选择
                </button>
              ) : (
                <button className="btn ghost" disabled={busy || captioning} onClick={() => setSelectMode(true)}>
                  <ListChecks size={16} className="icon-accent" />批量删除
                </button>
              )
            )}
            <span className="spacer" />
            {msg && <span className="muted">{msg}</span>}
          </div>

          {uploadPct !== null && (
            <div className="progress" style={{ marginTop: 10 }}>
              <div className="bar" style={{ width: `${uploadPct}%` }} />
            </div>
          )}

          {/* 打标参数：内嵌子面板 */}
          <div className="cap-controls">
            <div className="cap-field cap-field-model">
              <label className="cap-label">打标模型</label>
              <Select
                value={capMethod}
                onChange={(v) => setCapMethod(String(v))}
                options={[
                  { value: 'auto', label: '自动', desc: `跟随底模：${autoCaptioner}` },
                  { value: 'wd14', label: 'WD14 标签', desc: '写实/动漫均可，适合标身材等属性' },
                  { value: 'florence2', label: 'Florence-2', desc: '自然语言描述，写实推荐' },
                  { value: 'blip', label: 'BLIP', desc: '自然语言描述' },
                ]}
              />
            </div>
            {showWd14Model && (
              <div className="cap-field cap-field-model">
                <label className="cap-label">
                  WD14 模型
                  <span className="cap-help tip" tabIndex={0}
                    data-tip={'选择 WD14 tagger 模型：\n\n· SwinV2 v3 → 默认，速度快、精度好，日常够用\n· EVA02-large v3 → 精度最高，但模型大、在本机为 CPU 推理，明显更慢\n\n首次切换需下载对应模型权重。'}>
                    <HelpCircle size={13} />
                  </span>
                </label>
                <Select
                  value={wd14Model}
                  onChange={(v) => setWd14Model(String(v))}
                  options={[
                    { value: 'swinv2-v3', label: 'SwinV2 v3', desc: '快，默认' },
                    { value: 'eva02-large-v3', label: 'EVA02-large v3', desc: '更准，更慢' },
                  ]}
                />
              </div>
            )}
            {showThreshold && (
              <div className="cap-field cap-field-strength">
                <div className="cap-label-row">
                  <label className="cap-label">
                    打标强度
                    <span className="cap-help tip" tabIndex={0}
                      data-tip={'预设一键切换 WD14 阈值（模型给每个标签打 0~1 的分，只有 ≥ 阈值的标签才写入）：\n\n· 宽松 0.25 → 标签最多，能标出身材、姿势等细粒度属性，但噪声/错标也多\n· 均衡 0.35 → 通用默认，兼顾数量与准确\n· 严格 0.50 → 只留高置信标签，最干净但可能漏标\n\n选完预设仍可用下方滑块微调。'}>
                      <HelpCircle size={13} />
                    </span>
                  </label>
                  <span className="cap-value">{threshold.toFixed(2)}</span>
                </div>
                <div className="preset-row">
                  {([
                    ['宽松', 0.25], ['均衡', 0.35], ['严格', 0.5],
                  ] as const).map(([label, val]) => (
                    <button
                      key={label}
                      type="button"
                      className={`preset-chip${Math.abs(threshold - val) < 0.001 ? ' active' : ''}`}
                      onClick={() => setThreshold(val)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <input
                  type="range" min={0.1} max={0.7} step={0.05} value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  style={{ width: '100%', marginTop: 8 }}
                />
              </div>
            )}
            {showThreshold && (
              <label className="cap-check">
                <input type="checkbox" checked={excludeBodyFace}
                  onChange={(e) => setExcludeBodyFace(e.target.checked)} />
                固化人物特征 <span className="muted" style={{ fontSize: 12 }}>(去掉身材/脸型标签)</span>
              </label>
            )}
            {showThreshold && (
              <div className="cap-field cap-field-exclude">
                <label className="cap-label">额外排除标签 <span className="muted">(逗号分隔, 可选)</span></label>
                <input className="input" value={excludeTags}
                  placeholder="如: tattoo, glasses"
                  onChange={(e) => setExcludeTags(e.target.value)} />
              </div>
            )}
          </div>

          <p className="muted" style={{ marginTop: 12, lineHeight: 1.6 }}>
            建议 15–40 张，多角度多表情多服装；核心外观特征不写入标签，触发词会被放到标签首位。
            本次将使用 <strong>{effectiveMethod}</strong>。
            {showThreshold && '　阈值越低标签越多（含身材等细粒度属性），但噪声也越多；标不出想要的属性可调低到 0.2 左右。'}
            {showThreshold && excludeBodyFace && '　已勾选“固化人物特征”：身材/脸型标签会被从标注中移除，从而烘焙进触发词，让出图时该人物的身材脸型稳定还原（推荐用于人物 LoRA）。'}
            {captioning && <span style={{ color: 'var(--accent-2)' }}>　首次运行需下载打标模型，请耐心等待…</span>}
          </p>
        </div>

        {images.length === 0 ? (
          <div className="card"><div className="empty">还没有图片，点击上方上传。</div></div>
        ) : (
          <>
            {selectMode && (
              <div className="select-bar">
                <button className="filter-chip" onClick={() => toggleSelectAll(filteredImages)}>
                  {filteredImages.length > 0 && filteredImages.every((im) => selected.has(im.filename))
                    ? (<><CheckSquare size={14} />取消全选</>)
                    : (<><Square size={14} />全选{filtering ? '（当前筛选）' : ''}</>)}
                </button>
                <span className="muted filter-count">已选 {selected.size} 张</span>
                <span className="spacer" />
                <button className="btn danger sm" disabled={selected.size === 0 || bulkDeleting} onClick={bulkDelete}>
                  {bulkDeleting ? (<><span className="spinner" />删除中…</>) : (<><Trash2 size={15} />删除选中（{selected.size}）</>)}
                </button>
              </div>
            )}
            {(hasConfData || hasQualData) && (
              <div className="filter-bar">
                {hasConfData && (
                  <div className="filter-group">
                    <span className="filter-label">置信度</span>
                    {([
                      ['all', '全部'], ['high', '可信'], ['mid', '留意'], ['low', '需核对'],
                    ] as const).map(([v, label]) => (
                      <button
                        key={v}
                        className={`filter-chip${confFilter === v ? ' active' : ''}${v !== 'all' ? ` conf-${v}` : ''}`}
                        onClick={() => setConfFilter(v)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                )}
                {hasQualData && (
                  <div className="filter-group">
                    <span className="filter-label">质量</span>
                    {([
                      ['all', '全部'], ['ok', '无问题'], ['warn', '待核对'], ['bad', '不建议'],
                    ] as const).map(([v, label]) => (
                      <button
                        key={v}
                        className={`filter-chip${qualFilter === v ? ' active' : ''}${v !== 'all' ? ` qual-${v}` : ''}`}
                        onClick={() => setQualFilter(v)}
                      >
                        {label}
                      </button>
                    ))}
                  </div>
                )}
                <span className="spacer" />
                <span className="muted filter-count">
                  {filtering ? `筛选出 ${filteredImages.length} / ${images.length} 张` : `共 ${images.length} 张`}
                </span>
                {filtering && (
                  <button className="filter-chip" onClick={() => { setConfFilter('all'); setQualFilter('all') }}>
                    清除筛选
                  </button>
                )}
              </div>
            )}
            <div className="img-grid-wrap">
              {captioning && (
                <div className="grid-overlay">
                  <span className="spinner lg" />
                  <span>正在打标…</span>
                </div>
              )}
              {filteredImages.length === 0 ? (
                <div className="card"><div className="empty">没有符合筛选条件的图片。</div></div>
              ) : (
                <div className="img-grid">
                  {filteredImages.map((im) => {
                    const tagCount = im.caption.split(',').map((t) => t.trim()).filter(Boolean).length
                    const conf = imageConfidenceLevel(im)
                    const checked = selected.has(im.filename)
                    return (
                      <div
                        className="img-cell"
                        key={im.filename}
                        data-confidence={conf ?? undefined}
                        data-selected={selectMode && checked ? '' : undefined}
                      >
                        <div className="img-media">
                          <img src={im.thumbnail_url} alt={im.filename} className="img-thumb"
                            onClick={() => (selectMode ? toggleSelect(im.filename) : setEditing(im))} />
                          {selectMode ? (
                            <button
                              className={`select-check${checked ? ' checked' : ''}`}
                              title={checked ? '取消选择' : '选择'}
                              onClick={() => toggleSelect(im.filename)}
                            >
                              {checked ? <CheckSquare size={18} /> : <Square size={18} />}
                            </button>
                          ) : (
                            /* hover 浮出操作层 */
                            <div className="img-overlay">
                              <button className="round-btn primary" title="编辑标签" onClick={() => setEditing(im)}>
                                <Edit3 size={18} />
                              </button>
                              <button className="round-btn" title="预览原图" onClick={() => setPreview(im)}>
                                <Eye size={18} />
                              </button>
                              <button className="round-btn danger" title="删除图片" onClick={() => removeImage(im.filename)}>
                                <Trash2 size={18} />
                              </button>
                            </div>
                          )}
                          {im.quality && im.quality.level !== 'ok' && (
                            <span
                              className={`quality-badge quality-${im.quality.level}`}
                              title={(im.quality.level === 'bad' ? '不建议用于训练：' : '建议核对：')
                                + (im.quality.issues.map((i) => i.label).join('、') || '存在质量问题')}
                            >
                              {im.quality.level === 'bad' ? '不建议' : '待核对'}
                            </span>
                          )}
                          {conf && !selectMode && (
                            <span className={`conf-badge conf-${conf}`}
                              title={conf === 'low' ? '存在低置信度标签，建议核对'
                                : conf === 'mid' ? '含中等置信度标签，可留意' : '标签置信度较高'}>
                              {conf === 'low' ? '需核对' : conf === 'mid' ? '留意' : '可信'}
                            </span>
                          )}
                          <span className="tag-badge" title="标签数量">
                            <Tag size={12} />
                            {tagCount}
                          </span>
                        </div>
                        <div className="cap">
                          <div className="cap-name" title={im.filename}>{im.filename}</div>
                          <div className="cap-tags" title="点击编辑标签"
                            onClick={() => (selectMode ? toggleSelect(im.filename) : setEditing(im))}>
                            {im.caption || <span className="muted">未打标</span>}
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </>
        )}
      </div>
      {/* /detail-scroll */}

      {editing && (
        <TagEditor
          key={editing.filename}
          item={editing}
          onClose={() => setEditing(null)}
          onSave={(caption) => saveCaption(editing.filename, caption)}
        />
      )}

      {preview && (
        <div className="modal-backdrop" onClick={() => setPreview(null)}>
          <div className="preview-lightbox" onClick={(e) => e.stopPropagation()}>
            <button className="round-btn preview-close" title="关闭" onClick={() => setPreview(null)}>
              <X size={18} />
            </button>
            <img src={preview.image_url} alt={preview.filename} />
          </div>
        </div>
      )}

      {importOpen && (
        <div className="modal-backdrop" onClick={() => importing || setImportOpen(false)}>
          <div className="import-modal" onClick={(e) => e.stopPropagation()}>
            <div className="import-head">
              <span className="import-title"><FolderInput size={16} className="icon-accent" /> 从筛选目录导入</span>
              <span className="spacer" />
              <button className="round-btn" title="关闭" disabled={!!importing} onClick={() => setImportOpen(false)}>
                <X size={16} />
              </button>
            </div>
            <p className="muted import-hint">选择「图片筛选」产出的单人可训练目录（single/），一键把图片导入本数据集。导入后可在此打标。</p>
            <div className="import-list">
              {importSources === null ? (
                <div className="import-empty"><span className="spinner" /> 加载中…</div>
              ) : importSources.length === 0 ? (
                <div className="import-empty muted">暂无可导入目录。请先到「图片管理 → 图片筛选」跑一次筛选，产出 single/ 后再来。</div>
              ) : (
                importSources.map((s) => (
                  <button
                    key={s.source_dir}
                    className="import-item"
                    disabled={!!importing}
                    onClick={() => doImport(s.source_dir)}
                  >
                    <FolderInput size={16} className="icon-accent" />
                    <span className="import-name">{s.source_dir}</span>
                    <span className="badge">{s.image_count} 张</span>
                    {importing === s.source_dir && <Loader2 size={15} className="spin" />}
                  </button>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function confidenceColor(c: number): string {
  if (c >= 0.75) return '#a3e635' // 高置信度：可信
  if (c >= 0.45) return '#eab308' // 中：留意
  return '#f97316'                // 低：重点核对
}

// 卡片按 WD14 最低置信度分级：low(<45%) 需重点核对，mid(45~75%) 留意，
// high(>=75%) 可信；无分数（BLIP/手动）返回 null，不着色。
function imageConfidenceLevel(im: ImageItem): 'high' | 'mid' | 'low' | null {
  const scores = (im.tag_scores || []).map((s) => s.confidence)
  if (scores.length === 0) return null
  const min = Math.min(...scores)
  if (min >= 0.75) return 'high'
  if (min >= 0.45) return 'mid'
  return 'low'
}

// 全屏侧滑标签编辑：左大图 + 右侧标签胶囊 / 快速添加 / 批量文本。
// 融合 WD14 置信度：胶囊按分数着色边框，触发词（首位）高亮。
function TagEditor({
  item,
  onClose,
  onSave,
}: {
  item: ImageItem
  onClose: () => void
  onSave: (caption: string) => Promise<void>
}) {
  const initial = item.caption.split(',').map((t) => t.trim()).filter(Boolean)
  const [tags, setTags] = useState<string[]>(initial)
  const [newTag, setNewTag] = useState('')
  const [saving, setSaving] = useState(false)

  const scoreMap = new Map((item.tag_scores || []).map((s) => [s.tag.trim().toLowerCase(), s.confidence]))

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  const removeTag = (t: string) => setTags(tags.filter((x) => x !== t))
  const addTag = (e: React.FormEvent) => {
    e.preventDefault()
    const v = newTag.trim()
    if (v && !tags.includes(v)) { setTags([...tags, v]); setNewTag('') }
  }
  const save = async () => {
    setSaving(true)
    try { await onSave(tags.join(', ')); onClose() } finally { setSaving(false) }
  }

  const lowCount = tags.filter((t) => {
    const c = scoreMap.get(t.toLowerCase())
    return c != null && c < 0.45
  }).length
  // 是否有可展示的置信度分数（仅 WD14 打标会写入 .wdtags.json）。
  const hasScores = scoreMap.size > 0

  return (
    <div className="tag-editor">
      {/* 左：大图 */}
      <div className="tag-editor-view">
        <div className="tag-editor-topbar">
          <button className="round-btn" title="关闭" onClick={onClose}><X size={18} /></button>
          <span className="tag-editor-filename">{item.filename}</span>
        </div>
        <div className="tag-editor-img">
          <img src={item.image_url} alt={item.filename} />
        </div>
      </div>

      {/* 右：标签编辑 */}
      <div className="tag-editor-side">
        <div className="tag-editor-head">
          <h2 className="tag-editor-h">
            <Tag size={18} className="icon-accent" />
            标签编辑
            <span className="tag-count">{tags.length}</span>
          </h2>
          <button className="btn sm" disabled={saving} onClick={save}>
            {saving ? (<><span className="spinner" />保存中…</>) : (<><Save size={15} />保存</>)}
          </button>
        </div>

        <div className="tag-editor-body">
          {/* 快速添加 */}
          <div className="tag-section">
            <label className="tag-section-label">添加标签</label>
            <form onSubmit={addTag} className="tag-add-form">
              <input
                className="input"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                placeholder="输入标签名并回车…"
              />
              <button type="submit" className="tag-add-btn" disabled={!newTag.trim()} title="添加">
                <Plus size={18} />
              </button>
            </form>
          </div>

          {/* 当前标签胶囊 */}
          <div className="tag-section">
            <div className="tag-section-head">
              <label className="tag-section-label">当前标签</label>
              <button className="tag-clear" onClick={() => setTags([])}>清空全部</button>
            </div>
            {lowCount > 0 && (
              <p className="tag-low-hint">{lowCount} 个低置信度标签，建议核对</p>
            )}
            <div className="tag-chips">
              {tags.map((tag, idx) => {
                const c = scoreMap.get(tag.toLowerCase())
                const isTrigger = idx === 0
                const border = c != null ? confidenceColor(c) : undefined
                return (
                  <div
                    key={`${tag}-${idx}`}
                    className={`tag-chip ${isTrigger ? 'trigger' : ''}`}
                    style={border && !isTrigger ? { borderColor: border, color: border } : undefined}
                    title={c != null ? `WD14 置信度 ${(c * 100).toFixed(0)}%` : (isTrigger ? '触发词（置于首位）' : '手动添加，无打标分数')}
                  >
                    <span>{tag}</span>
                    {c != null && !isTrigger && <b className="tag-chip-score">{(c * 100).toFixed(0)}%</b>}
                    <button className="tag-chip-x" onClick={() => removeTag(tag)}><X size={13} /></button>
                  </div>
                )
              })}
              {tags.length === 0 && <span className="muted" style={{ fontSize: 13 }}>暂无标签</span>}
            </div>
          </div>

          {/* 批量文本编辑 */}
          <div className="tag-section tag-section-bulk">
            <label className="tag-section-label tag-bulk-label">
              <span>批量文本编辑</span>
              <span className="muted" style={{ fontSize: 11 }}>逗号分隔</span>
            </label>
            <textarea
              className="tag-bulk-text"
              value={tags.join(', ')}
              onChange={(e) => setTags(e.target.value.split(',').map((t) => t.trim()).filter(Boolean))}
            />
          </div>

          {hasScores ? (
            <p className="muted" style={{ fontSize: 11, lineHeight: 1.6 }}>
              置信度颜色：<span style={{ color: '#a3e635' }}>绿≥75%</span>、
              <span style={{ color: '#eab308' }}>黄45–75%</span>、
              <span style={{ color: '#f97316' }}>橙&lt;45%</span>。首位为触发词（高亮）。
            </p>
          ) : (
            <p className="muted" style={{ fontSize: 11, lineHeight: 1.6 }}>
              该图暂无 WD14 置信度分数：当前标注来自 BLIP 自然语言描述或手动编辑，
              仅 <strong>WD14</strong> 打标会记录逐标签置信度。改用「WD14」重新打标即可看到分数着色。首位为触发词（高亮）。
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
