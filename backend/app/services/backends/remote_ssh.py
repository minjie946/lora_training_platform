"""Remote SSH backend: run kohya training on a cloud / remote CUDA host.

Flow per job (in a background thread so it plugs into the existing supervisor):
  1. read the locally-built config.toml and rewrite its paths to remote ones
  2. SFTP the dataset directory + rewritten config to the remote workdir
  3. ssh-run `accelerate launch ...`, streaming remote stdout into the LOCAL log
  4. on success, SFTP the produced *.safetensors back into the local output dir

The returned handle mimics subprocess.Popen (pid/poll/wait) so job_manager's
supervisor and finalizer work unchanged.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import toml

from ...models import RemoteHost
from . import base as base_mod
from .base import LaunchSpec, PreflightResult, TrainingBackend
from .. import ssh_service


def _remote_join(*parts: str) -> str:
    cleaned = [p.strip("/") for p in parts if p not in ("", None)]
    head = parts[0]
    prefix = "" if head.startswith("~") else ("/" if head.startswith("/") else "")
    return prefix + "/".join(cleaned)


class _RemoteProc:
    """Popen-like handle whose work runs over SSH in a background thread."""

    def __init__(self, backend: "RemoteSshBackend", spec: LaunchSpec):
        self.backend = backend
        self.spec = spec
        self.pid = 0  # no local pid; remote process is tracked via config marker
        self._returncode: Optional[int] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # --- Popen-ish API used by job_manager ---
    def poll(self) -> Optional[int]:
        return self._returncode

    def wait(self) -> int:
        self._thread.join()
        return self._returncode if self._returncode is not None else 1

    def request_stop(self) -> None:
        self._stop.set()
        self.backend._remote_kill(self.spec)

    # --- worker ---
    def _log(self, line: str) -> None:
        with self.spec.log_path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")

    def _run(self) -> None:
        host = self.backend.host
        spec = self.spec
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        # start a fresh log
        spec.log_path.write_text("", encoding="utf-8")
        try:
            cfg = toml.load(spec.config_path)
            local_data_dir = Path(cfg["dataset"]["train_data_dir"])
            base_model_file = Path(cfg["model"]["pretrained_model_name_or_path"]).name

            paths = self.backend._remote_paths(spec.job_id, local_data_dir.name)
            # Rewrite config paths to the remote layout.
            cfg["dataset"]["train_data_dir"] = paths["data_dir"]
            cfg["training"]["output_dir"] = paths["output_dir"]
            cfg["model"]["pretrained_model_name_or_path"] = _remote_join(
                self.backend._base_models_dir(), base_model_file
            )
            remote_cfg_local_tmp = spec.config_path.parent / "config.remote.toml"
            with remote_cfg_local_tmp.open("w", encoding="utf-8") as f:
                toml.dump(cfg, f)

            self._log(f"[loralab] 连接远程主机 {host.host}:{host.port} …")
            conn = self.backend._conn()
            client = ssh_service.connect(conn)
            try:
                self._log(f"[loralab] 上传数据集 {local_data_dir.name} …")
                ssh_service.upload_dir(client, local_data_dir, paths["data_dir"])
                ssh_service.upload_file(client, remote_cfg_local_tmp, paths["config"])

                script = "sdxl_train_network.py" if spec.is_sdxl else "train_network.py"
                remote_script = _remote_join(host.kohya_dir, script)
                cmd = (
                    f"cd {host.kohya_dir} && "
                    f"mkdir -p {paths['output_dir']} && "
                    f"{host.python_cmd} -m accelerate.commands.launch "
                    f"--num_cpu_threads_per_process 1 "
                    f"{remote_script} --config_file {paths['config']} 2>&1"
                )
                self._log(f"[loralab] 远程启动训练：{script}")
                code = ssh_service.run_streamed(client, cmd, self._log)

                if code == 0 and not self._stop.is_set():
                    self._log("[loralab] 训练完成，拉取产出权重 …")
                    names = ssh_service.download_matching(
                        client, paths["output_dir"], spec.output_dir, ".safetensors"
                    )
                    self._log(f"[loralab] 已下载 {len(names)} 个权重：{', '.join(names) or '无'}")
                self._returncode = code if not self._stop.is_set() else 1
            finally:
                client.close()
        except Exception as e:  # noqa: BLE001
            self._log(f"[loralab] 远程训练异常：{e}")
            self._returncode = 1


class RemoteSshBackend(TrainingBackend):
    """One instance per configured RemoteHost row."""

    def __init__(self, host: RemoteHost):
        self.host = host
        self.name = f"remote_{host.id}"
        self.label = f"☁ {host.name}（远程 GPU）"

    # --- helpers ---
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

    def _base_models_dir(self) -> str:
        return self.host.base_models_dir or _remote_join(self.host.workdir, "models", "base")

    def _remote_paths(self, job_id: int, data_dirname: str) -> dict:
        job_root = _remote_join(self.host.workdir, "jobs", str(job_id))
        return {
            "job_root": job_root,
            "data_dir": _remote_join(job_root, "dataset", data_dirname),
            "output_dir": _remote_join(job_root, "output"),
            "config": _remote_join(job_root, "config.toml"),
        }

    def _remote_kill(self, spec: LaunchSpec) -> None:
        paths = self._remote_paths(spec.job_id, "")
        try:
            client = ssh_service.connect(self._conn(), timeout=10)
            try:
                # The config path is unique per job and appears in both the
                # launcher and the training child's argv, so this kills the tree.
                ssh_service.run_simple(client, f"pkill -TERM -f '{paths['config']}'", timeout=10)
            finally:
                client.close()
        except Exception:  # noqa: BLE001
            pass

    # --- TrainingBackend API ---
    def preflight(self) -> PreflightResult:
        try:
            client = ssh_service.connect(self._conn())
        except Exception as e:  # noqa: BLE001
            return PreflightResult(ok=False, detail=f"无法连接远程主机: {e}")
        try:
            problems: list[str] = []
            code, _ = ssh_service.run_simple(
                client, f"test -f {_remote_join(self.host.kohya_dir, 'train_network.py')}"
            )
            if code != 0:
                problems.append(f"远程未找到 kohya（{self.host.kohya_dir}/train_network.py）")
            code, out = ssh_service.run_simple(
                client, f"{self.host.python_cmd} -c 'import accelerate' 2>&1"
            )
            if code != 0:
                problems.append("远程缺少 accelerate（请在远程环境安装 kohya 依赖）")
            if problems:
                return PreflightResult(ok=False, detail="；".join(problems))
            return PreflightResult(ok=True, detail="远程主机就绪")
        finally:
            client.close()

    def start(self, spec: LaunchSpec):
        return _RemoteProc(self, spec)

    def stop(self, proc) -> None:
        if isinstance(proc, _RemoteProc):
            proc.request_stop()
