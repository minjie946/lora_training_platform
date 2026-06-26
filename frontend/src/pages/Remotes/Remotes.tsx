import { useEffect, useState } from 'react'
import { api, RemoteHost } from '../../api/client'
import Select from '../../components/Select/Select'
import './Remotes.css'

type FormState = {
  name: string
  host: string
  port: number
  username: string
  auth_type: string
  password: string
  private_key_path: string
  workdir: string
  kohya_dir: string
  python_cmd: string
  base_models_dir: string
  rvc_dir: string
}

const EMPTY: FormState = {
  name: '', host: '', port: 22, username: 'root', auth_type: 'key',
  password: '', private_key_path: '~/.ssh/id_ed25519',
  workdir: '~/loralab', kohya_dir: '~/sd-scripts', python_cmd: 'python', base_models_dir: '',
  rvc_dir: '~/Retrieval-based-Voice-Conversion-WebUI',
}

export default function Remotes() {
  const [hosts, setHosts] = useState<RemoteHost[]>([])
  const [editing, setEditing] = useState<RemoteHost | null>(null)
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState<FormState>(EMPTY)
  const [err, setErr] = useState('')
  const [testing, setTesting] = useState<number | null>(null)
  const [testMsg, setTestMsg] = useState<Record<number, { ok: boolean; detail: string }>>({})

  const load = () => api.listRemotes().then(setHosts).catch(() => { })
  useEffect(() => { load() }, [])

  const openCreate = () => { setEditing(null); setForm(EMPTY); setErr(''); setOpen(true) }
  const openEdit = (h: RemoteHost) => {
    setEditing(h)
    setForm({
      name: h.name, host: h.host, port: h.port, username: h.username, auth_type: h.auth_type,
      password: '', private_key_path: h.private_key_path,
      workdir: h.workdir, kohya_dir: h.kohya_dir, python_cmd: h.python_cmd, base_models_dir: h.base_models_dir,
      rvc_dir: h.rvc_dir,
    })
    setErr(''); setOpen(true)
  }

  const setF = (k: keyof FormState, v: any) => setForm((f) => ({ ...f, [k]: v }))

  const save = async () => {
    setErr('')
    if (!form.name || !form.host) { setErr('名称和主机地址必填'); return }
    try {
      if (editing) await api.updateRemote(editing.id, form)
      else await api.createRemote(form)
      setOpen(false)
      load()
    } catch (e: any) { setErr(e.message) }
  }

  const remove = async (h: RemoteHost) => {
    if (!confirm(`确认删除远程主机「${h.name}」？正在使用它的训练任务将无法继续。`)) return
    await api.deleteRemote(h.id)
    load()
  }

  const test = async (h: RemoteHost) => {
    setTesting(h.id)
    try {
      const r = await api.testRemote(h.id)
      setTestMsg((m) => ({ ...m, [h.id]: r }))
    } catch (e: any) {
      setTestMsg((m) => ({ ...m, [h.id]: { ok: false, detail: e.message } }))
    } finally { setTesting(null) }
  }

  return (
    <div>
      <div className="toolbar">
        <h1 className="page-title">算力 / 远程 GPU</h1>
        <span className="spacer" />
        <button className="btn" onClick={openCreate}>+ 添加远程主机</button>
      </div>
      <p className="page-sub">
        配置可通过 SSH 访问的云端 / 自有 CUDA 主机。新建训练时即可在“训练后端”里选择它，
        平台会自动上传数据集与配置、远程运行 kohya、回传日志与权重。
      </p>

      <div className="table-card">
        {hosts.length === 0 ? (
          <div className="empty">还没有远程主机。点击右上角添加，连接你的云 GPU。</div>
        ) : (
          <table>
            <thead>
              <tr><th>名称</th><th>地址</th><th>用户</th><th>认证</th><th>远程目录</th><th>连通性</th><th></th></tr>
            </thead>
            <tbody>
              {hosts.map((h) => {
                const tm = testMsg[h.id]
                return (
                  <tr key={h.id}>
                    <td>{h.name}</td>
                    <td className="muted">{h.host}:{h.port}</td>
                    <td className="muted">{h.username}</td>
                    <td>{h.auth_type === 'password' ? '密码' : '密钥'}</td>
                    <td className="muted">{h.workdir}</td>
                    <td>
                      {testing === h.id ? <span className="spinner" /> :
                        tm ? <span className={tm.ok ? 'badge green' : 'badge red'} title={tm.detail}>
                          {tm.ok ? '正常' : '失败'}
                        </span> : <span className="muted">—</span>}
                    </td>
                    <td>
                      <div className="row-actions">
                        <button className="btn sm ghost" onClick={() => test(h)} disabled={testing === h.id}>测试</button>
                        <button className="btn sm ghost" onClick={() => openEdit(h)}>编辑</button>
                        <button className="btn sm danger" onClick={() => remove(h)}>删除</button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
        {Object.values(testMsg).some(Boolean) && (
          <div style={{ marginTop: 12 }}>
            {hosts.map((h) => testMsg[h.id] && (
              <p key={h.id} className="muted" style={{ fontSize: 12, margin: '2px 0' }}>
                <b>{h.name}</b>：{testMsg[h.id].detail}
              </p>
            ))}
          </div>
        )}
      </div>

      {open && (
        <div className="modal-backdrop" onClick={() => setOpen(false)}>
          <div className="modal modal-form" onClick={(e) => e.stopPropagation()}>
            <div className="modal-form-body">
              <div className="toolbar" style={{ marginBottom: 16 }}>
                <strong style={{ fontSize: 16 }}>{editing ? '编辑远程主机' : '添加远程主机'}</strong>
                <span className="spacer" />
                <button className="btn sm ghost" onClick={() => setOpen(false)}>关闭</button>
              </div>

              <div className="note">
                <strong>使用前请确认：</strong>
                <ul>
                  <li>远程主机已预先装好 <code>sd-scripts (kohya)</code> 与 <code>accelerate</code>，且「远程 kohya 目录」下能找到 <code>train_network.py</code>。</li>
                  <li>底模 <code>.safetensors</code> 文件需放在远程底模目录中（或随数据集一同上传，视你的环境而定）。</li>
                  <li>本机可通过 SSH 直连该主机（密钥或密码任一可用），首次添加后建议先点「测试」确认连通。</li>
                </ul>
              </div>

              <div className="row">
                <div className="field">
                  <label>名称</label>
                  <input className="input" value={form.name} onChange={(e) => setF('name', e.target.value)} placeholder="如：RunPod-4090" />
                </div>
                <div className="field">
                  <label>主机地址 (host)</label>
                  <input className="input" value={form.host} onChange={(e) => setF('host', e.target.value)} placeholder="IP 或域名" />
                </div>
              </div>
              <div className="row">
                <div className="field">
                  <label>端口</label>
                  <input className="input" type="number" value={form.port} onChange={(e) => setF('port', Number(e.target.value))} />
                </div>
                <div className="field">
                  <label>用户名</label>
                  <input className="input" value={form.username} onChange={(e) => setF('username', e.target.value)} />
                </div>
                <div className="field">
                  <label>认证方式</label>
                  <Select
                    value={form.auth_type}
                    onChange={(v) => setF('auth_type', String(v))}
                    options={[{ value: 'key', label: 'SSH 密钥' }, { value: 'password', label: '密码' }]}
                  />
                </div>
              </div>

              {form.auth_type === 'key' ? (
                <div className="field">
                  <label>私钥路径</label>
                  <input className="input" value={form.private_key_path} onChange={(e) => setF('private_key_path', e.target.value)} placeholder="~/.ssh/id_ed25519" />
                </div>
              ) : (
                <div className="field">
                  <label>密码 {editing && <span className="muted" style={{ fontSize: 12 }}>（留空表示不修改）</span>}</label>
                  <input className="input" type="password" value={form.password} onChange={(e) => setF('password', e.target.value)} placeholder="远程主机密码" />
                </div>
              )}

              <div className="row">
                <div className="field">
                  <label>远程工作目录 <span className="help-icon" title="平台在远程主机上存放数据集/配置/输出的根目录">ⓘ</span></label>
                  <input className="input" value={form.workdir} onChange={(e) => setF('workdir', e.target.value)} />
                </div>
                <div className="field">
                  <label>远程 kohya 目录 <span className="help-icon" title="远程主机上 sd-scripts(kohya) 的路径，须含 train_network.py">ⓘ</span></label>
                  <input className="input" value={form.kohya_dir} onChange={(e) => setF('kohya_dir', e.target.value)} />
                </div>
              </div>
              <div className="row">
                <div className="field">
                  <label>远程 python <span className="help-icon" title="远程用于运行 accelerate 的解释器/命令，如 python 或 conda run -n env python">ⓘ</span></label>
                  <input className="input" value={form.python_cmd} onChange={(e) => setF('python_cmd', e.target.value)} />
                </div>
                <div className="field">
                  <label>远程底模目录 <span className="help-icon" title="远程主机上底模 .safetensors 所在目录；留空则默认 工作目录/models/base">ⓘ</span></label>
                  <input className="input" value={form.base_models_dir} onChange={(e) => setF('base_models_dir', e.target.value)} placeholder="留空用默认" />
                </div>
              </div>
              <div className="row">
                <div className="field">
                  <label>远程 RVC 目录 <span className="help-icon" title="远程主机上 Retrieval-based-Voice-Conversion-WebUI 的路径，用于声音克隆 / SVC 训练">ⓘ</span></label>
                  <input className="input" value={form.rvc_dir} onChange={(e) => setF('rvc_dir', e.target.value)} placeholder="~/Retrieval-based-Voice-Conversion-WebUI" />
                </div>
              </div>

              {err && <p className="badge red">{err}</p>}
              <div className="toolbar" style={{ marginTop: 8 }}>
                <span className="spacer" />
                <button className="btn ghost" onClick={() => setOpen(false)}>取消</button>
                <button className="btn" onClick={save}>{editing ? '保存' : '添加'}</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
