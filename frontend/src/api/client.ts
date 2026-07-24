// Backend API client.
export interface Dataset {
  id: number
  name: string
  concept: string
  repeat: number
  trigger_word: string
  base_model: string
  image_count: number
  status: string
  caption_status?: string // idle | running | done | failed
  caption_detail?: string
  created_at: string
}

export interface DatasetImportResult {
  dataset: Dataset
  imported: number
  captioned: number
  detail: string
}

export interface BaseModel {
  filename: string
  label: string
  is_sdxl: boolean
  style: string // "anime" | "realistic"
  size_bytes: number
  is_default: boolean
}

export interface TagScore {
  tag: string
  confidence: number
}

export interface ImageItem {
  filename: string
  caption: string
  thumbnail_url: string
  image_url: string
  tag_scores?: TagScore[] | null
  quality?: ImageQuality | null
}

export interface QualityIssue {
  code: string
  label: string
  severity: string // "warn" | "bad"
}

export interface ImageQuality {
  level: string // "ok" | "warn" | "bad"
  issues: QualityIssue[]
}

export interface QualityCheckResult {
  total: number
  ok: number
  warn: number
  bad: number
}

export interface PreflightItem {
  name: string
  ok: boolean
  detail: string
}
export interface PreflightResult {
  ok: boolean
  items: PreflightItem[]
}

export interface ResourceStats {
  platform: string
  cpu_percent?: number
  cpu_count?: number
  mem_total?: number
  mem_used?: number
  mem_percent?: number
  gpu?: {
    available?: boolean
    utilization?: number
    used_bytes?: number
    cores?: number
  }
}

export interface Job {
  id: number
  name: string
  dataset_id: number
  base_model: string
  backend: string
  params: Record<string, any>
  status: string
  progress: number
  current_step: number
  total_step: number
  latest_loss: number | null
  error: string | null
  created_at: string
  queued_at?: string | null
  finished_at: string | null
  has_checkpoint?: boolean
}

export interface LoraModel {
  id: number
  job_id: number
  name: string
  epoch: number
  base_model: string
  file_size: number
  created_at: string
}

export interface RemoteHost {
  id: number
  name: string
  host: string
  port: number
  username: string
  auth_type: string // "key" | "password"
  has_password: boolean
  private_key_path: string
  workdir: string
  kohya_dir: string
  python_cmd: string
  base_models_dir: string
  rvc_dir: string
  created_at: string
}

// ---- Voice / SVC (RVC) ----
export interface VoiceDataset {
  id: number
  name: string
  speaker: string
  sample_rate: number
  clip_count: number
  total_seconds: number
  status: string
  created_at: string
}

export interface AudioClip {
  filename: string
  seconds: number
  size_bytes: number
  audio_url: string
}

export interface VoiceJob {
  id: number
  name: string
  dataset_id: number
  backend: string
  params: Record<string, any>
  status: string
  progress: number
  current_step: number
  total_step: number
  error: string | null
  created_at: string
  finished_at: string | null
}

export interface VoiceModel {
  id: number
  job_id: number
  name: string
  speaker: string
  epoch: number
  sample_rate: number
  has_index: boolean
  file_size: number
  created_at: string
}

// ---- Image tools (微博图片管理) ----
export interface ImageTask {
  id: number
  kind: string // "pull" | "filter"
  target: string
  out_dir: string
  params: Record<string, any>
  status: string // running | paused | done | failed | stopped
  detail: string
  created_at: string
  finished_at: string | null
  // Download progress for pull tasks (0..1 fraction + raw counts).
  progress: number
  done: number
  total: number
}

export interface ImageDirEntry {
  name: string
  image_count: number
  categories: Record<string, number>
}

export interface ImageCookie {
  present: boolean
  length: number
  preview: string
  updated_at: string | null
  looks_valid: boolean
}

export interface ImagePreviewItem {
  pid: string
  thumb_url: string
  full_url: string
}

export interface ImagePreviewResult {
  out_dir_name: string
  uid: string
  album_id: string | null
  pids: ImagePreviewItem[]
}

export interface ImagePullOpts {
  uid?: string
  album?: string
  workers?: number
  start?: number
  end?: number | null
}

export interface ImageFilterOpts {
  directory: string
  recursive?: boolean
  dry_run?: boolean
  min_face?: number
  text_blocks?: number
  text_area?: number
  no_text_filter?: boolean
  no_animal_filter?: boolean
  no_quality_filter?: boolean
}

export interface ImageSettings {
  out_dir: string
  default_out_dir: string
  is_default: boolean
  exists: boolean
}

export interface ImageBrowseResult {
  path: string
  parent: string | null
  dirs: string[]
}

async function http<T>(url: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const j = await res.json()
      detail = j.detail || detail
    } catch { }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  // system
  preflight: () => http<PreflightResult>('/api/system/preflight'),
  resources: () => http<ResourceStats>('/api/system/resources'),
  baseModels: () =>
    http<{ default: string; models: BaseModel[] }>('/api/system/base-models'),

  // datasets
  listDatasets: () => http<Dataset[]>('/api/datasets'),
  createDataset: (body: Partial<Dataset>) =>
    http<Dataset>('/api/datasets', { method: 'POST', body: JSON.stringify(body) }),
  importDataset: async (
    body: Partial<Dataset> & { archive: File },
  ) => {
    const fd = new FormData()
    fd.append('name', body.name || '')
    fd.append('concept', body.concept || '')
    fd.append('repeat', String(body.repeat ?? 10))
    fd.append('trigger_word', body.trigger_word || '')
    fd.append('base_model', body.base_model || '')
    fd.append('archive', body.archive)
    const res = await fetch('/api/datasets/import', { method: 'POST', body: fd })
    if (!res.ok) throw new Error((await res.json()).detail || '导入失败')
    return res.json() as Promise<DatasetImportResult>
  },
  getDataset: (id: number) => http<Dataset>(`/api/datasets/${id}`),
  updateDataset: (id: number, body: Partial<Dataset>) =>
    http<Dataset>(`/api/datasets/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  deleteDataset: (id: number) =>
    http<{ ok: boolean }>(`/api/datasets/${id}`, { method: 'DELETE' }),
  listImages: (id: number) => http<ImageItem[]>(`/api/datasets/${id}/images`),
  importSources: () =>
    http<{ name: string; source_dir: string; image_count: number }[]>('/api/datasets/import-sources/list'),
  importFromDir: (id: number, source_dir: string) =>
    http<ImageItem[]>(`/api/datasets/${id}/import-from-dir`, {
      method: 'POST', body: JSON.stringify({ source_dir }),
    }),
  uploadImages: async (id: number, files: FileList, onProgress?: (pct: number) => void) => {
    const fd = new FormData()
    Array.from(files).forEach((f) => fd.append('files', f))
    // Use XHR so we can report real upload progress (fetch can't).
    return new Promise<ImageItem[]>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `/api/datasets/${id}/images`)
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText) as ImageItem[]) }
          catch { reject(new Error('响应解析失败')) }
        } else {
          let detail = '上传失败'
          try { detail = JSON.parse(xhr.responseText).detail || detail } catch { }
          reject(new Error(detail))
        }
      }
      xhr.onerror = () => reject(new Error('网络错误，上传失败'))
      xhr.send(fd)
    })
  },
  deleteImage: (id: number, filename: string) =>
    http(`/api/datasets/${id}/images/${encodeURIComponent(filename)}`, { method: 'DELETE' }),
  bulkDeleteImages: (id: number, filenames: string[]) =>
    http<{ ok: boolean; deleted: number }>(`/api/datasets/${id}/images/bulk-delete`, {
      method: 'POST',
      body: JSON.stringify({ filenames }),
    }),
  checkQuality: (id: number) =>
    http<QualityCheckResult>(`/api/datasets/${id}/quality-check`, { method: 'POST' }),
  updateCaption: (id: number, filename: string, caption: string) =>
    http(`/api/datasets/${id}/captions`, {
      method: 'PUT',
      body: JSON.stringify({ filename, caption }),
    }),
  autoCaption: (id: number, opts?: { threshold?: number; inject_trigger?: boolean; method?: string; exclude_body_face?: boolean; exclude_tags?: string[] }) =>
    http<{ ok: boolean; caption_status: string; detail: string }>(
      `/api/datasets/${id}/auto-caption`,
      {
        method: 'POST',
        body: JSON.stringify({
          threshold: opts?.threshold ?? 0.35,
          inject_trigger: opts?.inject_trigger ?? true,
          method: opts?.method ?? 'auto',
          exclude_body_face: opts?.exclude_body_face ?? false,
          exclude_tags: opts?.exclude_tags ?? [],
        }),
      },
    ),
  captionStatus: (id: number) =>
    http<{ dataset_id: number; caption_status: string; detail: string; status: string }>(
      `/api/datasets/${id}/caption-status`,
    ),

  // jobs
  listJobs: () => http<Job[]>('/api/jobs'),
  getJob: (id: number) => http<Job>(`/api/jobs/${id}`),
  createJob: (body: any) =>
    http<Job>('/api/jobs', { method: 'POST', body: JSON.stringify(body) }),
  updateJob: (id: number, body: any) =>
    http<Job>(`/api/jobs/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  startJob: (id: number) => http<Job>(`/api/jobs/${id}/start`, { method: 'POST' }),
  dequeueJob: (id: number) => http<Job>(`/api/jobs/${id}/dequeue`, { method: 'POST' }),
  stopJob: (id: number) => http<Job>(`/api/jobs/${id}/stop`, { method: 'POST' }),
  pauseJob: (id: number) => http<Job>(`/api/jobs/${id}/pause`, { method: 'POST' }),
  resumeJob: (id: number) => http<Job>(`/api/jobs/${id}/resume`, { method: 'POST' }),
  cloneJob: (id: number) => http<Job>(`/api/jobs/${id}/clone`, { method: 'POST' }),
  deleteJob: (id: number) => http<{ ok: boolean }>(`/api/jobs/${id}`, { method: 'DELETE' }),
  jobLog: (id: number, tail = 200) => http<{ log: string }>(`/api/jobs/${id}/log?tail=${tail}`),
  backends: () => http<{ name: string; label: string }[]>('/api/jobs/backends'),
  defaults: () => http<Record<string, any>>('/api/jobs/defaults'),

  // models
  listModels: (jobId?: number) =>
    http<LoraModel[]>(`/api/models${jobId ? `?job_id=${jobId}` : ''}`),
  deleteModel: (id: number) => http(`/api/models/${id}`, { method: 'DELETE' }),
  bulkDeleteModels: (ids: number[]) =>
    http<{ ok: boolean; deleted: number }>(`/api/models/bulk-delete`, {
      method: 'POST',
      body: JSON.stringify({ ids }),
    }),
  modelDownloadUrl: (id: number) => `/api/models/${id}/download`,

  // remote hosts (cloud GPU)
  listRemotes: () => http<RemoteHost[]>('/api/remotes'),
  createRemote: (body: any) =>
    http<RemoteHost>('/api/remotes', { method: 'POST', body: JSON.stringify(body) }),
  updateRemote: (id: number, body: any) =>
    http<RemoteHost>(`/api/remotes/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  deleteRemote: (id: number) =>
    http<{ ok: boolean }>(`/api/remotes/${id}`, { method: 'DELETE' }),
  testRemote: (id: number) =>
    http<{ ok: boolean; detail: string }>(`/api/remotes/${id}/test`, { method: 'POST' }),

  // ---- Voice / SVC (RVC) ----
  voiceBackends: () => http<{ name: string; label: string }[]>('/api/voice/backends'),
  voiceDefaults: () => http<Record<string, any>>('/api/voice/defaults'),

  listVoiceDatasets: () => http<VoiceDataset[]>('/api/voice/datasets'),
  createVoiceDataset: (body: Partial<VoiceDataset>) =>
    http<VoiceDataset>('/api/voice/datasets', { method: 'POST', body: JSON.stringify(body) }),
  getVoiceDataset: (id: number) => http<VoiceDataset>(`/api/voice/datasets/${id}`),
  deleteVoiceDataset: (id: number) =>
    http<{ ok: boolean }>(`/api/voice/datasets/${id}`, { method: 'DELETE' }),
  listClips: (id: number) => http<AudioClip[]>(`/api/voice/datasets/${id}/clips`),
  uploadClips: async (id: number, files: FileList) => {
    const fd = new FormData()
    Array.from(files).forEach((f) => fd.append('files', f))
    const res = await fetch(`/api/voice/datasets/${id}/clips`, { method: 'POST', body: fd })
    if (!res.ok) throw new Error((await res.json()).detail || '上传失败')
    return res.json() as Promise<AudioClip[]>
  },
  deleteClip: (id: number, filename: string) =>
    http(`/api/voice/datasets/${id}/clips/${encodeURIComponent(filename)}`, { method: 'DELETE' }),

  listVoiceJobs: () => http<VoiceJob[]>('/api/voice/jobs'),
  getVoiceJob: (id: number) => http<VoiceJob>(`/api/voice/jobs/${id}`),
  createVoiceJob: (body: any) =>
    http<VoiceJob>('/api/voice/jobs', { method: 'POST', body: JSON.stringify(body) }),
  startVoiceJob: (id: number) =>
    http<VoiceJob>(`/api/voice/jobs/${id}/start`, { method: 'POST' }),
  stopVoiceJob: (id: number) =>
    http<VoiceJob>(`/api/voice/jobs/${id}/stop`, { method: 'POST' }),
  deleteVoiceJob: (id: number) =>
    http<{ ok: boolean }>(`/api/voice/jobs/${id}`, { method: 'DELETE' }),
  voiceJobLog: (id: number, tail = 200) =>
    http<{ log: string }>(`/api/voice/jobs/${id}/log?tail=${tail}`),

  listVoiceModels: (jobId?: number) =>
    http<VoiceModel[]>(`/api/voice/models${jobId ? `?job_id=${jobId}` : ''}`),
  deleteVoiceModel: (id: number) => http(`/api/voice/models/${id}`, { method: 'DELETE' }),
  voiceModelDownloadUrl: (id: number) => `/api/voice/models/${id}/download`,

  // image tools (微博图片管理)
  imageConfig: () => http<{ cookie_present: boolean; cookie_path: string; out_dir: string }>('/api/images/config'),
  imageSettings: () => http<ImageSettings>('/api/images/settings'),
  setImageSettings: (out_dir: string | null) =>
    http<ImageSettings>('/api/images/settings', { method: 'PUT', body: JSON.stringify({ out_dir }) }),
  browseImageDir: (path = '') =>
    http<ImageBrowseResult>(`/api/images/browse?path=${encodeURIComponent(path)}`),
  pickImageDir: (initial = '') =>
    http<{ path: string | null }>(`/api/images/pick-dir?initial=${encodeURIComponent(initial)}`, { method: 'POST' }),
  imageCookie: (platform: 'weibo' | 'xhs' = 'weibo') =>
    http<ImageCookie>(`/api/images/cookie?platform=${platform}`),
  imageCookieRaw: (platform: 'weibo' | 'xhs' = 'weibo') =>
    http<{ cookie: string }>(`/api/images/cookie/raw?platform=${platform}`),
  setImageCookie: (cookie: string, platform: 'weibo' | 'xhs' = 'weibo') =>
    http<ImageCookie>(`/api/images/cookie?platform=${platform}`, { method: 'PUT', body: JSON.stringify({ cookie }) }),
  previewImages: (opts: { uid?: string; album?: string; start?: number; end?: number | null }) =>
    http<ImagePreviewResult>('/api/images/preview', { method: 'POST', body: JSON.stringify(opts) }),
  // Live log of the in-flight (synchronous) preview fetch, polled while waiting.
  previewLog: (platform: 'weibo' | 'xhs' = 'weibo', tail = 400) =>
    http<{ log: string }>(`/api/images/preview-log?platform=${platform}&tail=${tail}`),
  pullSelected: (opts: { pids: string[]; out_dir_name: string; workers?: number }) =>
    http<ImageTask>('/api/images/pull-selected', { method: 'POST', body: JSON.stringify(opts) }),
  // 小红书（XHS）博主主页全量
  xhsPreview: (opts: { user: string; max_notes?: number | null; headed?: boolean }) =>
    http<ImagePreviewResult>('/api/images/xhs/preview', { method: 'POST', body: JSON.stringify(opts) }),
  xhsPull: (opts: { user: string; workers?: number; max_notes?: number | null; headed?: boolean }) =>
    http<ImageTask>('/api/images/xhs/pull', { method: 'POST', body: JSON.stringify(opts) }),
  xhsPullSelected: (opts: { ids: string[]; user: string; out_dir_name: string; workers?: number }) =>
    http<ImageTask>('/api/images/xhs/pull-selected', { method: 'POST', body: JSON.stringify(opts) }),
  imageProxyUrl: (url: string) => `/api/images/proxy?url=${encodeURIComponent(url)}`,
  imageDirs: () => http<ImageDirEntry[]>('/api/images/dirs'),
  imageTasks: (kind?: 'pull' | 'filter') =>
    http<ImageTask[]>(`/api/images/tasks${kind ? `?kind=${kind}` : ''}`),
  imageTask: (id: number) => http<ImageTask>(`/api/images/tasks/${id}`),
  imageTaskLog: (id: number, tail = 400) =>
    http<{ log: string }>(`/api/images/tasks/${id}/log?tail=${tail}`),
  stopImageTask: (id: number) =>
    http<{ ok: boolean }>(`/api/images/tasks/${id}/stop`, { method: 'POST' }),
  pauseImageTask: (id: number) =>
    http<ImageTask>(`/api/images/tasks/${id}/pause`, { method: 'POST' }),
  resumeImageTask: (id: number) =>
    http<ImageTask>(`/api/images/tasks/${id}/resume`, { method: 'POST' }),
  discardImageTask: (id: number) =>
    http<{ ok: boolean }>(`/api/images/tasks/${id}/discard`, { method: 'POST' }),
  pullImages: (opts: ImagePullOpts) =>
    http<ImageTask>('/api/images/pull', { method: 'POST', body: JSON.stringify(opts) }),
  filterImages: (opts: ImageFilterOpts) =>
    http<ImageTask>('/api/images/filter', { method: 'POST', body: JSON.stringify(opts) }),
}
