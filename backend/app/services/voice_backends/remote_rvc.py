"""Remote SSH RVC backend: run the RVC training pipeline on a cloud GPU host.

Reuses the RemoteHost rows + ssh_service. Per job (background thread):
  1. build the pipeline script with remote paths + remote python
  2. SFTP the audio trainset + script to the remote workdir
  3. ssh-run the script, streaming remote stdout into the LOCAL log
  4. on success, SFTP produced .pth/.index back into the local weights dir
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from ...models import RemoteHost
from .. import ssh_service, voice_config
from .base import VoiceBackend, VoiceLaunchSpec, VoicePreflight


def _remote_join(*parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p not in ("", None)]
    head = parts[0]
    prefix = "" if head.startswith("~") else ("/" if head.startswith("/") else "")
    return prefix + "/".join(cleaned)


class _RemoteVoiceProc:
    """Popen-like handle whose RVC pipeline runs over SSH in a thread."""

    def __init__(self, backend: "RemoteRvcBackend", spec: VoiceLaunchSpec):
        self.backend = backend
        self.spec = spec
        self.pid = 0
        self._returncode: Optional[int] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def poll(self) -> Optional[int]:
        return self._returncode

    def wait(self) -> int:
        self._thread.join()
        return self._returncode if self._returncode is not None else 1

    def request_stop(self) -> None:
        self._stop.set()
        self.backend._remote_kill(self.spec)

    def _log(self, line: str) -> None:
        with self.spec.log_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")

    def _run(self) -> None:
        host = self.backend.host
        spec = self.spec
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("", encoding="utf-8")
        try:
            paths = self.backend._remote_paths(spec.job_id)
            script = voice_config.build_pipeline_script(
                rvc_dir=host.rvc_dir,
                python_cmd=host.python_cmd,
                exp_name=spec.exp_name,
                trainset_dir=paths["trainset"],
                exp_dir=paths["exp_dir"],
                weights_out_dir=paths["weights"],
                params=spec.params,
                device="cuda:0",
            )
            local_script = spec.exp_dir / "pipeline.remote.sh"
            voice_config.write_script(script, local_script)

            self._log(f"[loralab] 连接远程主机 {host.host}:{host.port} …")
            client = ssh_service.connect(self.backend._conn())
            try:
                self._log("[loralab] 上传音频训练集 …")
                ssh_service.upload_dir(client, spec.trainset_dir, paths["trainset"])
                ssh_service.upload_file(client, local_script, paths["script"])

                cmd = f"chmod +x {paths['script']} && bash {paths['script']} 2>&1"
                self._log("[loralab] 远程启动 RVC 训练流水线 …")
                code = ssh_service.run_streamed(client, cmd, self._log)

                if code == 0 and not self._stop.is_set():
                    self._log("[loralab] 训练完成，拉取声音模型权重 …")
                    pth = ssh_service.download_matching(
                        client, paths["weights"], spec.weights_out_dir, ".pth"
                    )
                    idx = ssh_service.download_matching(
                        client, paths["weights"], spec.weights_out_dir, ".index"
                    )
                    self._log(
                        f"[loralab] 已下载 {len(pth)} 个权重 / {len(idx)} 个索引："
                        f"{', '.join(pth + idx) or '无'}"
                    )
                self._returncode = code if not self._stop.is_set() else 1
            finally:
                client.close()
        except Exception as e:  # noqa: BLE001
            self._log(f"[loralab] 远程训练异常：{e}")
            self._returncode = 1


class RemoteRvcBackend(VoiceBackend):
    """One instance per configured RemoteHost row (voice/RVC variant)."""

    def __init__(self, host: RemoteHost):
        self.host = host
        self.name = f"remote_rvc_{host.id}"
        self.label = f"☁ {host.name}（远程 RVC）"

    def _conn(self) -> ssh_service.RemoteConn:
        h = self.host
        return ssh_service.RemoteConn(
            host=h.host,
            port=h.port,
            username=h.username,
            auth_type=h.auth_type,
            password=h.password,
            private_key_path=h.private_key_path,
        )

    def _remote_paths(self, job_id: int) -> dict:
        job_root = _remote_join(self.host.workdir, "voice_jobs", str(job_id))
        return {
            "job_root": job_root,
            "trainset": _remote_join(job_root, "trainset"),
            "exp_dir": _remote_join(job_root, "exp"),
            "weights": _remote_join(job_root, "weights"),
            "script": _remote_join(job_root, "pipeline.sh"),
        }

    def _remote_kill(self, spec: VoiceLaunchSpec) -> None:
        paths = self._remote_paths(spec.job_id)
        try:
            client = ssh_service.connect(self._conn(), timeout=10)
            try:
                ssh_service.run_simple(client, f"pkill -TERM -f '{paths['script']}'", timeout=10)
                ssh_service.run_simple(client, f"pkill -TERM -f '{paths['exp_dir']}'", timeout=10)
            finally:
                client.close()
        except Exception:  # noqa: BLE001
            pass

    def preflight(self) -> VoicePreflight:
        try:
            client = ssh_service.connect(self._conn())
        except Exception as e:  # noqa: BLE001
            return VoicePreflight(ok=False, detail=f"无法连接远程主机: {e}")
        try:
            train_py = _remote_join(self.host.rvc_dir, "infer/modules/train/train.py")
            code, _ = ssh_service.run_simple(client, f"test -f {train_py}")
            if code != 0:
                return VoicePreflight(
                    ok=False, detail=f"远程未找到 RVC（{self.host.rvc_dir}/infer/modules/train/train.py）"
                )
            return VoicePreflight(ok=True, detail="远程 RVC 就绪")
        finally:
            client.close()

    def start(self, spec: VoiceLaunchSpec):
        return _RemoteVoiceProc(self, spec)

    def stop(self, proc) -> None:
        if isinstance(proc, _RemoteVoiceProc):
            proc.request_stop()
