"""SSH/SFTP helper built on paramiko for the remote (cloud GPU) backend.

Supports key-file and password auth. Provides: connection test, recursive
upload/download over SFTP, and streamed remote command execution that pipes the
remote stdout/stderr into a local callback (used to mirror training logs).
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import paramiko


@dataclass
class RemoteConn:
    host: str
    port: int
    username: str
    auth_type: str  # "key" | "password"
    password: str = ""
    private_key_path: str = ""


def _connect(conn: RemoteConn, timeout: float = 15.0) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict = {
        "hostname": conn.host,
        "port": conn.port,
        "username": conn.username,
        "timeout": timeout,
        "banner_timeout": timeout,
        "auth_timeout": timeout,
    }
    if conn.auth_type == "password":
        kwargs["password"] = conn.password
        kwargs["look_for_keys"] = False
        kwargs["allow_agent"] = False
    else:
        key_path = os.path.expanduser(conn.private_key_path) if conn.private_key_path else ""
        if key_path:
            kwargs["key_filename"] = key_path
        # otherwise fall back to agent / default keys
    client.connect(**kwargs)
    return client


def test_connection(conn: RemoteConn) -> tuple[bool, str]:
    """Return (ok, detail). Runs `nvidia-smi` to report GPU info when present."""
    try:
        client = _connect(conn)
    except Exception as e:  # noqa: BLE001
        return False, f"连接失败: {e}"
    try:
        _, stdout, _ = client.exec_command(
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo NO_GPU",
            timeout=15,
        )
        out = stdout.read().decode("utf-8", "replace").strip()
        if not out or out == "NO_GPU":
            return True, "连接成功，但未检测到 NVIDIA GPU（nvidia-smi 不可用）"
        return True, f"连接成功，GPU: {out.replace(chr(10), ' / ')}"
    except Exception as e:  # noqa: BLE001
        return True, f"连接成功，但探测 GPU 失败: {e}"
    finally:
        client.close()


def _sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    cur = "/" if remote_dir.startswith("/") else ""
    for part in parts:
        if not part:
            continue
        cur = f"{cur}{part}" if cur in ("", "/") else f"{cur}/{part}"
        probe = cur if remote_dir.startswith("/") else cur
        try:
            sftp.stat(probe)
        except IOError:
            try:
                sftp.mkdir(probe)
            except IOError:
                pass


def upload_dir(client: paramiko.SSHClient, local_dir: Path, remote_dir: str) -> None:
    """Recursively upload a local directory tree to remote_dir."""
    sftp = client.open_sftp()
    try:
        _sftp_mkdirs(sftp, remote_dir)
        for root, _dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            rdir = remote_dir if rel == "." else f"{remote_dir}/{rel.replace(os.sep, '/')}"
            _sftp_mkdirs(sftp, rdir)
            for fn in files:
                sftp.put(os.path.join(root, fn), f"{rdir}/{fn}")
    finally:
        sftp.close()


def upload_file(client: paramiko.SSHClient, local_file: Path, remote_path: str) -> None:
    sftp = client.open_sftp()
    try:
        parent = remote_path.rsplit("/", 1)[0]
        _sftp_mkdirs(sftp, parent)
        sftp.put(str(local_file), remote_path)
    finally:
        sftp.close()


def download_matching(
    client: paramiko.SSHClient, remote_dir: str, local_dir: Path, suffix: str = ".safetensors"
) -> list[str]:
    """Download files under remote_dir whose name ends with suffix. Returns names."""
    local_dir.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    fetched: list[str] = []
    try:
        try:
            entries = sftp.listdir_attr(remote_dir)
        except IOError:
            return fetched
        for ent in entries:
            if stat.S_ISDIR(ent.st_mode):
                continue
            if ent.filename.endswith(suffix):
                sftp.get(f"{remote_dir}/{ent.filename}", str(local_dir / ent.filename))
                fetched.append(ent.filename)
    finally:
        sftp.close()
    return fetched


def run_streamed(
    client: paramiko.SSHClient,
    command: str,
    on_line: Callable[[str], None],
) -> int:
    """Run a command, stream combined stdout/stderr line-by-line, return exit code."""
    transport = client.get_transport()
    chan = transport.open_session()
    chan.get_pty()  # so tqdm progress flushes line-ish output
    chan.exec_command(command)
    buf = b""
    while True:
        if chan.recv_ready():
            buf += chan.recv(4096)
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                on_line(ln.decode("utf-8", "replace"))
        elif chan.exit_status_ready() and not chan.recv_ready():
            break
    if buf:
        on_line(buf.decode("utf-8", "replace"))
    return chan.recv_exit_status()


def run_simple(client: paramiko.SSHClient, command: str, timeout: float = 30.0) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, (out + err)


def connect(conn: RemoteConn, timeout: float = 15.0) -> paramiko.SSHClient:
    return _connect(conn, timeout=timeout)
