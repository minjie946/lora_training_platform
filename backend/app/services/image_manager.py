"""Background manager for the weibo image tools (拉取 / 筛选).

Both operations are long-running (album paging with throttle; CV model inference
over many files) and their heavy deps are declared as PEP 723 inline deps in the
scripts (under backend/app/image_tools/), so we launch them via ``uv run
--script`` — which provisions an ephemeral env — rather than importing them here.

The lifecycle is persisted on an ``ImageTask`` row (status/detail/pid) so the UI
can restore progress after a reload, and a supervisor thread tails the script's
stdout into a per-task log file. A run interrupted by a backend restart is
reconciled on startup (its process is either re-attached or marked failed).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import (
    IMAGE_TOOLS_COOKIE,
    IMAGE_TOOLS_DIR,
    IMAGE_TOOLS_INSIGHTFACE_ROOT,
    IMAGE_TOOLS_OUT_DIR,
    IMAGE_TOOLS_WORKSPACE,
    IMAGE_TOOLS_XHS_COOKIE,
    IMAGE_TOOLS_YOLO_WEIGHTS,
    UV_BIN,
    WORKSPACE_DIR,
)
from ..db import get_session
from ..models import ImageTask

# Logs for image tasks live here (one file per task id), like job train logs.
LOG_DIR = WORKSPACE_DIR / "image_tasks"

# Persisted user settings (configurable output/download directory, etc.).
SETTINGS_FILE = IMAGE_TOOLS_WORKSPACE / "settings.json"

_PULL_SCRIPT = "weibo_album_downloader.py"
_XHS_SCRIPT = "xhs_user_downloader.py"
_FILTER_SCRIPT = "filter_single_person.py"

# Track supervisor threads / processes in-process so we can stop them and avoid
# launching two runs of the same kind at once (single machine, be gentle).
_running: dict[int, subprocess.Popen] = {}
_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Settings (可配置的下载/拉取目录)
# --------------------------------------------------------------------------- #
def _read_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def get_out_dir() -> Path:
    """The effective download/pull/filter root: user-configured or default."""
    cfg = _read_settings().get("out_dir")
    return Path(cfg).expanduser() if cfg else IMAGE_TOOLS_OUT_DIR


def resolve_source_dir(rel: str) -> Path:
    """Resolve a relative path (e.g. "uid_123/single") under the output root.

    Guards against path traversal outside the managed output dir. Raises
    ValueError on an empty / illegal / missing directory.
    """
    rel = (rel or "").strip().strip("/")
    if not rel:
        raise ValueError("缺少来源目录")
    out_root = get_out_dir().resolve()
    target = (out_root / rel).resolve()
    if not (str(target) == str(out_root) or str(target).startswith(str(out_root) + os.sep)):
        raise ValueError("非法的来源目录路径")
    if not target.is_dir():
        raise ValueError(f"来源目录不存在：{rel}")
    return target


def read_settings() -> dict:
    """Settings for the UI: effective out_dir + whether it's the default."""
    out = get_out_dir()
    return {
        "out_dir": str(out),
        "default_out_dir": str(IMAGE_TOOLS_OUT_DIR),
        "is_default": str(out) == str(IMAGE_TOOLS_OUT_DIR),
        "exists": out.exists(),
    }


def list_single_sources() -> list[dict]:
    """List filter output dirs that contain a single/ folder with images.

    Returns [{name, source_dir, image_count}] for the dataset import picker,
    where source_dir is the relative path (e.g. "uid_123/single").
    """
    from ..config import ALLOWED_IMAGE_EXTS

    out = get_out_dir()
    if not out.exists():
        return []
    sources: list[dict] = []
    for d in sorted(out.iterdir()):
        if not d.is_dir():
            continue
        single = d / "single"
        if not single.is_dir():
            continue
        n = sum(
            1 for p in single.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS
        )
        if n > 0:
            sources.append(
                {"name": d.name, "source_dir": f"{d.name}/single", "image_count": n}
            )
    return sources


def write_settings(*, out_dir: Optional[str]) -> dict:
    """Persist settings. An empty/None out_dir resets to the default."""
    cfg = _read_settings()
    val = (out_dir or "").strip()
    if val:
        p = Path(val).expanduser()
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"无法创建目录：{e}")
        cfg["out_dir"] = str(p)
    else:
        cfg.pop("out_dir", None)  # reset to default
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return read_settings()


def browse_dir(path: Optional[str]) -> dict:
    """List subdirectories of ``path`` for the settings folder picker.

    An empty path starts at the user's home directory. Returns the resolved
    absolute path, its parent (for an "up" button), and the immediate child
    directories (sorted, hidden ones skipped).
    """
    raw = (path or "").strip()
    base = Path(raw).expanduser() if raw else Path.home()
    try:
        base = base.resolve()
    except Exception:  # noqa: BLE001
        base = Path.home()
    if not base.is_dir():
        base = Path.home()

    dirs: list[str] = []
    try:
        for d in sorted(base.iterdir(), key=lambda x: x.name.lower()):
            if d.is_dir() and not d.name.startswith("."):
                dirs.append(d.name)
    except PermissionError:
        pass

    parent = str(base.parent)
    return {
        "path": str(base),
        "parent": parent if parent != str(base) else None,
        "dirs": dirs,
    }


def pick_dir_native(initial: Optional[str] = None) -> Optional[str]:
    """Open the machine's native folder chooser and return the picked path.

    This only works because the backend runs on the same (local) machine as the
    user. On macOS we drive Finder's ``choose folder`` via osascript. Returns the
    POSIX path, or None if the user cancelled. Raises RuntimeError on unsupported
    platforms / when no GUI is available.
    """
    import sys

    if sys.platform != "darwin":
        raise RuntimeError("原生文件选择仅支持在本机 macOS 上使用")

    start = (initial or "").strip()
    default_clause = ""
    if start and Path(start).expanduser().is_dir():
        # AppleScript wants a POSIX file reference for the default location.
        safe = str(Path(start).expanduser()).replace('"', '\\"')
        default_clause = f' default location (POSIX file "{safe}")'

    script = (
        f'POSIX path of (choose folder with prompt "选择下载 / 拉取 / 筛选目录"{default_clause})'
    )
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError("未找到 osascript，无法调起系统文件选择")
    except subprocess.TimeoutExpired:
        raise RuntimeError("选择超时")

    if proc.returncode != 0:
        # osascript returns 1 and "User canceled." when the dialog is dismissed.
        if "User canceled" in (proc.stderr or ""):
            return None
        raise RuntimeError((proc.stderr or "系统文件选择失败").strip())

    picked = (proc.stdout or "").strip().rstrip("/")
    return picked or None


def log_path(task_id: int) -> Path:
    return LOG_DIR / f"{task_id}.log"


def read_log(task_id: int, tail: int = 400) -> str:
    lp = log_path(task_id)
    if not lp.exists():
        return ""
    with lp.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-tail:])


# Preview (预览抓图片列表) is synchronous — the caller waits for the pid/image
# list — but the fetch can be slow (weibo album paging; 小红书 spins up a real
# browser and reads each note). We still tee its stdout to a per-platform log
# file so the UI can poll and show live progress instead of a bare spinner.
def preview_log_path(kind: str) -> Path:
    """Log file for a synchronous preview run (weibo | xhs)."""
    return LOG_DIR / f"preview_{kind}.log"


def read_preview_log(kind: str, tail: int = 400) -> str:
    lp = preview_log_path(kind)
    if not lp.exists():
        return ""
    with lp.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-tail:])


def _run_preview(cmd: list[str], kind: str, timeout: float) -> tuple[int, str]:
    """Run a preview command, streaming stdout to its preview log file.

    Returns (returncode, last_nonempty_line). Raises subprocess.TimeoutExpired
    (after killing the child) when the run exceeds ``timeout``. The log is
    truncated at the start of each preview so the UI shows only the current run.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Force unbuffered child stdout so lines land in the file immediately.
    env["PYTHONUNBUFFERED"] = "1"
    lp = preview_log_path(kind)
    with open(lp, "w", encoding="utf-8", buffering=1) as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(IMAGE_TOOLS_DIR),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(proc.pid)
            raise
    tail = read_preview_log(kind, tail=8).strip().splitlines()
    return code, (tail[-1] if tail else "")


# Downloader progress lines look like: "  进度 10/91  成功 10 跳过 0 失败 0",
# the header "[下载] 共 91 张,..." gives the total up-front, and "[完成] ..."
# marks the end. Parse the log tail so the UI can render a progress bar.
_PROGRESS_RE = re.compile(r"进度\s+(\d+)\s*/\s*(\d+)")
_TOTAL_RE = re.compile(r"\[下载\]\s*共\s*(\d+)\s*张")
_DONE_RE = re.compile(r"\[完成\]")


def parse_progress(task_id: int, status: str) -> dict:
    """Derive {progress, done, total} for a pull task from its log tail.

    ``progress`` is a 0..1 float; done/total are counts. Returns zeros when no
    download has started yet.
    """
    log = read_log(task_id, tail=200)
    total = 0
    done = 0
    for m in _TOTAL_RE.finditer(log):
        total = int(m.group(1))
    matches = list(_PROGRESS_RE.finditer(log))
    if matches:
        done = int(matches[-1].group(1))
        total = int(matches[-1].group(2)) or total
    # A finished (done) task or an explicit 完成 line means fully downloaded.
    if status == "done" or _DONE_RE.search(log):
        if total:
            done = total
    progress = (done / total) if total else 0.0
    return {"progress": round(progress, 4), "done": done, "total": total}


# --------------------------------------------------------------------------- #
# Cookie management (微博登录态)
# --------------------------------------------------------------------------- #
def _cookie_path(platform: str) -> Path:
    """Cookie file for a platform: weibo (default) or xhs (小红书)."""
    return IMAGE_TOOLS_XHS_COOKIE if platform == "xhs" else IMAGE_TOOLS_COOKIE


def read_cookie_info(platform: str = "weibo") -> dict:
    """Return metadata about the stored cookie (never the full value)."""
    p = _cookie_path(platform)
    if not p.exists():
        return {"present": False, "length": 0, "preview": "", "updated_at": None,
                "looks_valid": False}
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    n = len(text)
    # Masked preview: head + tail only, so the UI can confirm which cookie is set
    # without exposing the whole credential.
    preview = text if n <= 24 else f"{text[:12]}…{text[-8:]}"
    if platform == "xhs":
        # 小红书 needs a1 + web_session (webId is nice-to-have) for signing/auth.
        looks_valid = "a1=" in text and "web_session=" in text
    else:
        looks_valid = "SUB=" in text and "SUBP=" in text
    return {
        "present": bool(text),
        "length": n,
        "preview": preview,
        "updated_at": datetime.utcfromtimestamp(p.stat().st_mtime),
        "looks_valid": looks_valid,
    }


def write_cookie(cookie: str, platform: str = "weibo") -> None:
    """Persist the login cookie string for a platform to its cookie file."""
    text = (cookie or "").strip()
    if not text:
        raise ValueError("Cookie 内容不能为空")
    p = _cookie_path(platform)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def read_cookie_raw(platform: str = "weibo") -> str:
    """Return the full stored cookie text (single-user local tool, for editing)."""
    p = _cookie_path(platform)
    return p.read_text(encoding="utf-8", errors="replace").strip() if p.exists() else ""


def is_kind_running(kind: str) -> bool:
    """True if a task of this kind is currently running (in-process registry)."""
    with _lock:
        ids = list(_running.keys())
    if not ids:
        return False
    from sqlmodel import select

    with get_session() as session:
        rows = session.exec(
            select(ImageTask).where(ImageTask.id.in_(ids))
        ).all()
        return any(r.kind == kind and r.status == "running" for r in rows)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def start_pull(
    *,
    uid: str,
    album: str,
    workers: int,
    start: int,
    end: Optional[int],
) -> int:
    """Launch a weibo album download in the background. Returns the task id."""
    if is_kind_running("pull"):
        raise ValueError("已有拉取任务在进行中，请等待完成")
    if not uid and not album:
        raise ValueError("请提供用户 UID 或相册链接")
    if not IMAGE_TOOLS_COOKIE.exists():
        raise ValueError(f"缺少微博 Cookie 文件：{IMAGE_TOOLS_COOKIE}")

    # The downloader writes to <out>/uid_<uid> or <out>/album_<album_id>.
    if uid:
        target = f"uid:{uid}"
        out_dir = f"uid_{uid}"
    else:
        target = album
        out_dir = ""  # resolved from the log/dir listing after it runs

    params = {"uid": uid, "album": album, "workers": workers, "start": start, "end": end}
    cmd = [
        UV_BIN, "run", "--script", _PULL_SCRIPT,
        "--out", str(get_out_dir()),
        "--cookie", str(IMAGE_TOOLS_COOKIE),
        "--workers", str(workers),
        "--start", str(start),
    ]
    if end is not None:
        cmd += ["--end", str(end)]
    if uid:
        cmd += ["--uid", uid]
    else:
        cmd += ["--album", album]

    return _launch("pull", target, out_dir, params, cmd)


def _build_cmd_base() -> list[str]:
    return [
        UV_BIN, "run", "--script", _PULL_SCRIPT,
        "--out", str(get_out_dir()),
        "--cookie", str(IMAGE_TOOLS_COOKIE),
    ]


def _pid_urls(pid: str) -> dict:
    """Derive preview (thumbnail) + original URLs from a weibo pid.

    orj360 is a ~360px preview served by the sinaimg CDN (fast, fine for a grid);
    large is the original the downloader actually saves.
    """
    ext = "gif" if pid.startswith("8") else "jpg"
    return {
        "pid": pid,
        "thumb_url": f"https://wx1.sinaimg.cn/orj360/{pid}.{ext}",
        "full_url": f"https://wx1.sinaimg.cn/large/{pid}.{ext}",
    }


def preview_pids(
    *,
    uid: str,
    album: str,
    start: int,
    end: Optional[int],
    timeout: float = 300.0,
) -> dict:
    """Synchronously fetch the pid list (no download) for preview.

    Returns {out_dir_name, pids: [{pid, thumb_url, full_url}]}. Raises ValueError
    on bad input / cookie / fetch failure.
    """
    if not uid and not album:
        raise ValueError("请提供用户 UID 或相册链接")
    if not IMAGE_TOOLS_COOKIE.exists():
        raise ValueError(f"缺少微博 Cookie 文件：{IMAGE_TOOLS_COOKIE}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(
        "r", suffix=".json", delete=False, dir=str(LOG_DIR), encoding="utf-8"
    ) as tf:
        list_out = tf.name

    cmd = _build_cmd_base() + [
        "--list-only", "--list-out", list_out, "--start", str(start),
    ]
    if end is not None:
        cmd += ["--end", str(end)]
    if uid:
        cmd += ["--uid", uid]
    else:
        cmd += ["--album", album]

    try:
        code, last = _run_preview(cmd, "weibo", timeout)
        if code != 0:
            raise ValueError(last or "抓取图片列表失败")
        with open(list_out, "r", encoding="utf-8") as f:
            data = json.load(f)
    except subprocess.TimeoutExpired:
        raise ValueError("抓取超时，请缩小范围（设置结束张数）后重试")
    finally:
        try:
            os.unlink(list_out)
        except OSError:
            pass

    pids = data.get("pids", []) or []
    return {
        "out_dir_name": data.get("out_dir_name", ""),
        "uid": data.get("uid", ""),
        "album_id": data.get("album_id"),
        "pids": [_pid_urls(p) for p in pids],
    }


def start_pull_selected(*, pids: list[str], out_dir_name: str, workers: int) -> int:
    """Download only the given pids into out_dir_name, in the background."""
    if is_kind_running("pull"):
        raise ValueError("已有拉取任务在进行中，请等待完成")
    pids = [p.strip() for p in pids if p and p.strip()]
    if not pids:
        raise ValueError("未选择任何图片")
    if not out_dir_name:
        raise ValueError("缺少目标目录名")
    if not IMAGE_TOOLS_COOKIE.exists():
        raise ValueError(f"缺少微博 Cookie 文件：{IMAGE_TOOLS_COOKIE}")

    # The downloader always appends out_dir_name under --out itself, so recover the
    # uid/album from the dir name to pass the right selector back to it.
    if out_dir_name.startswith("uid_"):
        selector = ["--uid", out_dir_name[len("uid_"):]]
    elif out_dir_name.startswith("album_"):
        # album mode needs a URL; reconstruct a minimal one the parser accepts.
        selector = ["--album", f"https://photo.weibo.com/0/albums/detail/album_id/{out_dir_name[len('album_'):]}"]
    else:
        raise ValueError("无法识别的目标目录名")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, dir=str(LOG_DIR), encoding="utf-8"
    ) as tf:
        tf.write("\n".join(pids))
        pids_file = tf.name

    params = {"selected": len(pids), "out_dir_name": out_dir_name, "workers": workers}
    cmd = _build_cmd_base() + ["--workers", str(workers), "--pids-file", pids_file] + selector
    return _launch("pull", f"{out_dir_name} · {len(pids)}张", out_dir_name, params, cmd)


# --------------------------------------------------------------------------- #
# 小红书（XHS）：博主主页全量 —— 纯 Python，签名由 Playwright 在脚本内完成
# --------------------------------------------------------------------------- #
def _xhs_cmd_base() -> list[str]:
    return [
        UV_BIN, "run", "--script", _XHS_SCRIPT,
        "--out", str(get_out_dir()),
        "--cookie", str(IMAGE_TOOLS_XHS_COOKIE),
    ]


def preview_xhs_user(
    *, user: str, max_notes: Optional[int], headed: bool = False, timeout: float = 600.0
) -> dict:
    """Synchronously fetch a 小红书 author's image list (no download) for preview.

    Returns {out_dir_name, user_id, pids:[{pid, thumb_url, full_url}]} to match the
    weibo preview shape (pid == the per-image id the selective download expects).
    Signing spins up a headless browser, so this is slower — hence a longer timeout.
    """
    if not user.strip():
        raise ValueError("请提供小红书博主主页链接")
    if not IMAGE_TOOLS_XHS_COOKIE.exists():
        raise ValueError(f"缺少小红书 Cookie 文件：{IMAGE_TOOLS_XHS_COOKIE}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(
        "r", suffix=".json", delete=False, dir=str(LOG_DIR), encoding="utf-8"
    ) as tf:
        list_out = tf.name

    cmd = _xhs_cmd_base() + ["--user", user.strip(), "--list-only", "--list-out", list_out]
    if max_notes is not None:
        cmd += ["--max-notes", str(max_notes)]
    if headed:
        cmd += ["--headed"]

    try:
        code, last = _run_preview(cmd, "xhs", timeout)
        if code != 0:
            raise ValueError(last or "抓取小红书图片列表失败")
        with open(list_out, "r", encoding="utf-8") as f:
            data = json.load(f)
    except subprocess.TimeoutExpired:
        raise ValueError("抓取超时，请减少解析笔记数（max-notes）后重试")
    finally:
        try:
            os.unlink(list_out)
        except OSError:
            pass

    items = data.get("items", []) or []
    return {
        "out_dir_name": data.get("out_dir_name", ""),
        "uid": data.get("user_id", ""),
        "album_id": None,
        # Reuse the weibo preview item shape; the "pid" here is the xhs image id.
        "pids": [
            {"pid": it["id"], "thumb_url": it["thumb_url"], "full_url": it["full_url"]}
            for it in items
        ],
    }


def start_pull_xhs(*, user: str, workers: int, max_notes: Optional[int], headed: bool = False) -> int:
    """Download ALL images of a 小红书 author in the background."""
    if is_kind_running("pull"):
        raise ValueError("已有拉取任务在进行中，请等待完成")
    if not user.strip():
        raise ValueError("请提供小红书博主主页链接")
    if not IMAGE_TOOLS_XHS_COOKIE.exists():
        raise ValueError(f"缺少小红书 Cookie 文件：{IMAGE_TOOLS_XHS_COOKIE}")

    user_id = _xhs_user_id(user)
    out_dir = f"xhs_user_{user_id}"
    params = {"platform": "xhs", "user": user.strip(), "workers": workers,
              "max_notes": max_notes, "headed": headed}
    cmd = _xhs_cmd_base() + ["--user", user.strip(), "--workers", str(workers)]
    if max_notes is not None:
        cmd += ["--max-notes", str(max_notes)]
    if headed:
        cmd += ["--headed"]
    return _launch("pull", f"xhs:{user_id}", out_dir, params, cmd)


def start_pull_xhs_selected(*, ids: list[str], user: str, out_dir_name: str, workers: int) -> int:
    """Download only the chosen 小红书 image ids for an author, in the background."""
    if is_kind_running("pull"):
        raise ValueError("已有拉取任务在进行中，请等待完成")
    ids = [i.strip() for i in ids if i and i.strip()]
    if not ids:
        raise ValueError("未选择任何图片")
    if not user.strip():
        raise ValueError("缺少博主链接")
    if not IMAGE_TOOLS_XHS_COOKIE.exists():
        raise ValueError(f"缺少小红书 Cookie 文件：{IMAGE_TOOLS_XHS_COOKIE}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, dir=str(LOG_DIR), encoding="utf-8"
    ) as tf:
        tf.write("\n".join(ids))
        ids_file = tf.name

    params = {"platform": "xhs", "user": user.strip(), "selected": len(ids),
              "out_dir_name": out_dir_name, "workers": workers}
    cmd = _xhs_cmd_base() + [
        "--user", user.strip(), "--workers", str(workers), "--ids-file", ids_file,
    ]
    return _launch("pull", f"{out_dir_name} · {len(ids)}张", out_dir_name, params, cmd)


def _xhs_user_id(user: str) -> str:
    """Parse the 小红书 user_id from a profile URL or a bare id (for dir naming)."""
    m = re.search(r"/user/profile/([0-9a-f]+)", user)
    if m:
        return m.group(1)
    return user.strip().strip("/").split("/")[-1] or "unknown"


def start_filter(
    *,
    directory: str,
    recursive: bool,
    dry_run: bool,
    min_face: float,
    text_blocks: int,
    text_area: float,
    no_text_filter: bool,
    no_animal_filter: bool,
    no_quality_filter: bool = False,
) -> int:
    """Launch the single-person filter over a downloaded directory."""
    if is_kind_running("filter"):
        raise ValueError("已有筛选任务在进行中，请等待完成")
    out_root = get_out_dir()
    target_dir = (out_root / directory).resolve()
    # Guard against path traversal outside the managed output dir.
    if not str(target_dir).startswith(str(out_root.resolve())):
        raise ValueError("非法的目录路径")
    if not target_dir.is_dir():
        raise ValueError(f"目录不存在：{directory}")

    params = {
        "directory": directory,
        "recursive": recursive,
        "dry_run": dry_run,
        "min_face": min_face,
        "text_blocks": text_blocks,
        "text_area": text_area,
        "no_text_filter": no_text_filter,
        "no_animal_filter": no_animal_filter,
        "no_quality_filter": no_quality_filter,
    }
    cmd = [
        UV_BIN, "run", "--script", _FILTER_SCRIPT,
        str(target_dir),
        "--min-face", str(min_face),
        "--text-blocks", str(text_blocks),
        "--text-area", str(text_area),
    ]
    if recursive:
        cmd += ["--recursive"]
    if dry_run:
        cmd += ["--dry-run"]
    if no_text_filter:
        cmd += ["--no-text-filter"]
    if no_animal_filter:
        cmd += ["--no-animal-filter"]
    if no_quality_filter:
        cmd += ["--no-quality-filter"]

    # Point the script's model caches at the persistent workspace location so we
    # never re-download InsightFace (~600MB) / YOLOv8n between runs.
    IMAGE_TOOLS_INSIGHTFACE_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_TOOLS_YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    extra_env = {
        "INSIGHTFACE_ROOT": str(IMAGE_TOOLS_INSIGHTFACE_ROOT),
        "YOLO_WEIGHTS": str(IMAGE_TOOLS_YOLO_WEIGHTS),
    }
    return _launch("filter", directory, directory, params, cmd, extra_env=extra_env)


def stop_task(task_id: int) -> bool:
    """Stop a running image task (kills its process group)."""
    with _lock:
        proc = _running.get(task_id)

    with get_session() as session:
        task = session.get(ImageTask, task_id)
        if not task:
            return False
        pid = task.pid
        status = task.status

    if status != "running":
        return False

    _set_status(task_id, "stopped", "已手动停止", finished=True)
    if proc is not None:
        _terminate(proc.pid)
    elif pid:
        _terminate(pid)
    with _lock:
        _running.pop(task_id, None)
    return True


def pause_task(task_id: int) -> bool:
    """Pause a running pull task: kill the process but keep files + state.

    The task's status becomes "paused"; already-downloaded images stay on disk
    and the launch command is preserved so it can be resumed later. Because we
    set the terminal status *before* killing, the supervisor won't overwrite it.
    """
    with _lock:
        proc = _running.get(task_id)

    with get_session() as session:
        task = session.get(ImageTask, task_id)
        if not task:
            return False
        pid = task.pid
        status = task.status
        kind = task.kind

    if status != "running" or kind != "pull":
        return False

    prog = parse_progress(task_id, status)
    detail = f"已暂停（已下载 {prog['done']}/{prog['total'] or '?'} 张）"
    _set_status(task_id, "paused", detail)
    if proc is not None:
        _terminate(proc.pid)
    elif pid:
        _terminate(pid)
    with _lock:
        _running.pop(task_id, None)
    return True


def resume_task(task_id: int) -> bool:
    """Resume a paused pull task by relaunching its stored command.

    The downloader skips files that already exist, so it continues from where
    the pause left off. The log is appended to (not truncated).
    """
    if is_kind_running("pull"):
        raise ValueError("已有拉取任务在进行中，请等待完成")

    with get_session() as session:
        task = session.get(ImageTask, task_id)
        if not task:
            return False
        if task.status != "paused":
            return False
        try:
            params = json.loads(task.params or "{}")
        except Exception:  # noqa: BLE001
            params = {}

    cmd = params.get("_cmd")
    if not cmd:
        raise ValueError("该任务缺少可续跑的启动信息，无法继续")
    extra_env = params.get("_extra_env")

    _set_status(task_id, "running", "继续下载中…")
    _spawn(task_id, cmd, extra_env, append_log=True)
    return True


def discard_task(task_id: int) -> bool:
    """Stop (if needed) a pull task and delete the images it downloaded.

    Removes the task's output directory under the managed out root. Guards
    against deleting anything outside that root.
    """
    import shutil

    with _lock:
        proc = _running.get(task_id)

    with get_session() as session:
        task = session.get(ImageTask, task_id)
        if not task:
            return False
        status = task.status
        pid = task.pid
        out_dir = task.out_dir
        kind = task.kind

    if kind != "pull":
        return False

    # Kill the process first if it's still alive.
    if status == "running":
        _set_status(task_id, "stopped", "已放弃并清空", finished=True)
        if proc is not None:
            _terminate(proc.pid)
        elif pid:
            _terminate(pid)
        with _lock:
            _running.pop(task_id, None)
    else:
        _set_status(task_id, "stopped", "已放弃并清空", finished=True)

    # Delete the downloaded directory (defensive path check).
    if out_dir:
        out_root = get_out_dir().resolve()
        target = (out_root / out_dir).resolve()
        if str(target).startswith(str(out_root) + os.sep) and target.is_dir():
            try:
                shutil.rmtree(target)
            except Exception:  # noqa: BLE001
                pass
    return True


# --------------------------------------------------------------------------- #
# Launch + supervise
# --------------------------------------------------------------------------- #
def _launch(
    kind: str,
    target: str,
    out_dir: str,
    params: dict,
    cmd: list[str],
    extra_env: Optional[dict[str, str]] = None,
) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Stash the launch spec inside params so a paused task can be resumed by
    # relaunching the very same command (the downloader skips existing files,
    # so it naturally continues from where it left off).
    stored = dict(params)
    stored["_cmd"] = cmd
    if extra_env:
        stored["_extra_env"] = extra_env

    # Persist the task first so we have an id for the log file name.
    with get_session() as session:
        task = ImageTask(
            kind=kind,
            target=target,
            out_dir=out_dir,
            params=json.dumps(stored, ensure_ascii=False),
            status="running",
            detail="任务已启动…",
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        task_id = task.id

    _spawn(task_id, cmd, extra_env, append_log=False)
    return task_id


def _spawn(
    task_id: int,
    cmd: list[str],
    extra_env: Optional[dict[str, str]],
    append_log: bool,
) -> None:
    """Start the child process for a task and attach a supervisor thread.

    ``append_log`` keeps the previous log (used when resuming) instead of
    truncating it, so the user sees a continuous history across pauses.
    """
    env = os.environ.copy()
    # Force unbuffered child stdout so log lines land in the file immediately.
    # Without this, Python block-buffers stdout when redirected to a file,
    # making the frontend log appear frozen until the buffer fills or the
    # process exits.
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    lp = log_path(task_id)
    logf = open(lp, "a" if append_log else "w", encoding="utf-8", buffering=1)
    # Launch in its own session/process group so it survives a backend reload
    # and can be killed as a group. stdout/stderr go to the log file (no PIPE),
    # so a dead reader can't SIGPIPE-kill the child.
    proc = subprocess.Popen(
        cmd,
        cwd=str(IMAGE_TOOLS_DIR),
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    _update_task(task_id, pid=proc.pid)

    with _lock:
        _running[task_id] = proc
    thread = threading.Thread(target=_supervise, args=(task_id, proc), daemon=True)
    thread.start()


def _supervise(task_id: int, proc: subprocess.Popen) -> None:
    """Wait for the process to finish, then finalize status from its exit code.

    The script writes its own progress into the log file; the UI polls the log
    endpoint, so here we only need to detect completion.
    """
    try:
        code = proc.wait()
    except Exception:  # noqa: BLE001
        code = 1
    finally:
        with _lock:
            _running.pop(task_id, None)

    # A manual stop already set a terminal status; don't overwrite it.
    with get_session() as session:
        task = session.get(ImageTask, task_id)
        if task and task.status != "running":
            return

    tail = read_log(task_id, tail=8).strip().splitlines()
    last = tail[-1] if tail else ""
    if code == 0:
        _set_status(task_id, "done", last or "任务完成", finished=True)
    else:
        _set_status(task_id, "failed", last or f"任务失败（退出码 {code}）", finished=True)


def reconcile_on_startup() -> None:
    """Fix image tasks whose supervisor died with the previous backend process.

    If the child is still alive, re-attach a supervisor; otherwise mark it failed
    so the UI stops showing a running state.
    """
    from sqlmodel import select

    with get_session() as session:
        rows = session.exec(
            select(ImageTask).where(ImageTask.status == "running")
        ).all()
        running = [(t.id, t.pid) for t in rows]

    for task_id, pid in running:
        with _lock:
            if task_id in _running:
                continue
        if pid and _pid_alive(pid):
            proc = _ExternalProc(pid)
            with _lock:
                _running[task_id] = proc  # type: ignore[assignment]
            threading.Thread(
                target=_supervise, args=(task_id, proc), daemon=True
            ).start()
        else:
            _set_status(
                task_id,
                "failed",
                "任务在后端重启时中断，请重新运行",
                finished=True,
            )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _set_status(task_id: int, status: str, detail: str, finished: bool = False) -> None:
    fields: dict = {"status": status, "detail": detail}
    if finished:
        fields["finished_at"] = datetime.utcnow()
    _update_task(task_id, **fields)


def _update_task(task_id: int, **fields) -> None:
    with get_session() as session:
        task = session.get(ImageTask, task_id)
        if not task:
            return
        for k, v in fields.items():
            setattr(task, k, v)
        session.add(task)
        session.commit()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _terminate(pid: int, timeout: float = 8.0) -> None:
    """Kill a process group by PID (tasks run in their own session)."""
    if not pid:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    except Exception:  # noqa: BLE001
        pgid = pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:  # noqa: BLE001
        pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, Exception):  # noqa: BLE001
            return
        time.sleep(0.3)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


class _ExternalProc:
    """Popen-like wrapper for a re-attached process we don't own the handle to."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self) -> Optional[int]:
        return None if _pid_alive(self.pid) else 0

    def wait(self) -> int:
        while _pid_alive(self.pid):
            time.sleep(1.0)
        return 0
