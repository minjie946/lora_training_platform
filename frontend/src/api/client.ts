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
}
