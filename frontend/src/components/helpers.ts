import { Job } from '../api/client'

export function statusBadge(status: string): { cls: string; text: string } {
  switch (status) {
    case 'running':
      return { cls: 'badge blue', text: '训练中' }
    case 'succeeded':
      return { cls: 'badge green', text: '成功' }
    case 'failed':
      return { cls: 'badge red', text: '失败' }
    case 'stopped':
      return { cls: 'badge amber', text: '已停止' }
    case 'paused':
      return { cls: 'badge amber', text: '已暂停' }
    case 'pending':
      return { cls: 'badge gray', text: '待启动' }
    case 'captioned':
      return { cls: 'badge green', text: '已打标' }
    case 'ready':
      return { cls: 'badge green', text: '就绪' }
    case 'draft':
      return { cls: 'badge gray', text: '草稿' }
    default:
      return { cls: 'badge gray', text: status }
  }
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

export function jobStepText(j: Job): string {
  if (j.total_step > 0) return `${j.current_step}/${j.total_step} 步`
  return '—'
}

export interface TrainPhase {
  key: string
  label: string
  hint: string
}

// 从训练日志推断当前所处阶段，给用户一个清晰的“现在在干嘛”提示。
// kohya 的典型流程：加载模型 → 缓存图像特征(latents) → 训练循环 → 保存权重。
export function inferPhase(status: string, log: string[]): TrainPhase {
  if (status === 'succeeded') return { key: 'done', label: '已完成', hint: '训练结束，可在下方下载产出的 LoRA 权重' }
  if (status === 'failed') return { key: 'failed', label: '失败', hint: '训练中断，请查看日志排查原因' }
  if (status === 'stopped') return { key: 'stopped', label: '已停止', hint: '训练已被手动停止' }
  if (status === 'paused') return { key: 'stopped', label: '已暂停', hint: '训练已暂停，点击“继续训练”从上次检查点恢复' }
  if (status === 'pending') return { key: 'pending', label: '待启动', hint: '点击“启动训练”开始' }

  // running：扫描最近的日志判断细分阶段
  const tail = log.slice(-40).join('\n').toLowerCase()
  if (/steps:\s*\d|epoch \d+\/\d+/.test(tail) || /avr_loss/.test(tail)) {
    return { key: 'training', label: '训练中', hint: '正在迭代训练，下方步数与 Loss 实时更新' }
  }
  if (/saving|save.*\.safetensors|model saved/.test(tail)) {
    return { key: 'saving', label: '保存权重中', hint: '正在保存当前轮次的 LoRA 权重' }
  }
  if (/caching (latents|text encoder)|cache_latents/.test(tail)) {
    return { key: 'caching', label: '缓存图像特征中', hint: '训练前的预处理：把图片编码缓存（这里的进度条不是训练进度）' }
  }
  if (/loading|load stablediffusion|prepare|building|make buckets|import network/.test(tail)) {
    return { key: 'preparing', label: '准备中', hint: '正在加载底模与数据集，尚未开始训练' }
  }
  return { key: 'preparing', label: '准备中', hint: '正在初始化训练环境' }
}

