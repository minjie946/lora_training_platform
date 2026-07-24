#!/usr/bin/env -S uv run --script
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "requests>=2.28.0",
# ]
# ///
"""
微博相册原图批量下载脚本。

支持两种模式:
  1. 按用户 UID 下载其"图片墙"中的全部原图(自动翻页)。
  2. 按 photo.weibo.com 的单个相册 URL 下载该相册全部原图。

登录态从当前目录下的 cookie.txt 读取(浏览器复制的 Cookie 字符串即可)。

用法示例:
  # 按 UID 全量下载
  uv run weibo_album_downloader.py --uid 1234567890

  # 按单个相册链接下载
  uv run weibo_album_downloader.py --album "https://photo.weibo.com/1234567890/albums/detail/album_id/9876543210"

  # 指定输出目录、并发数、cookie 文件
  uv run weibo_album_downloader.py --uid 1234567890 --out ./downloads --workers 8 --cookie ./cookie.txt
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time

import requests

DEFAULT_COOKIE_FILE = "cookie.txt"
DEFAULT_OUT_DIR = "weibo_photos"
DEFAULT_WORKERS = 6
REQUEST_TIMEOUT = 30
RETRY = 3
SLEEP_BETWEEN_PAGES = 0.8  # 翻页节流,避免触发风控

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def load_cookie(cookie_file):
    """从文件读取 Cookie 字符串。"""
    if not os.path.exists(cookie_file):
        sys.exit(
            f"[错误] 找不到 Cookie 文件: {cookie_file}\n"
            f"请把浏览器登录后复制的 Cookie 粘贴到该文件中(单行即可)。"
        )
    with open(cookie_file, "r", encoding="utf-8") as f:
        cookie = f.read().strip()
    if not cookie:
        sys.exit(f"[错误] Cookie 文件为空: {cookie_file}")
    return cookie


def build_session(cookie):
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": UA,
            "Cookie": cookie,
            "Referer": "https://weibo.com/",
            "Accept": "application/json, text/plain, */*",
        }
    )
    return session


def request_json(session, url, params=None, referer=None):
    """带重试的 GET 并解析 JSON。"""
    headers = {}
    if referer:
        headers["Referer"] = referer
    last_err = None
    for attempt in range(1, RETRY + 1):
        try:
            resp = session.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * attempt)
    print(f"[警告] 请求失败 {url} params={params}: {last_err}")
    return None


def candidate_urls_from_pid(pid):
    """
    由 pid 生成原图地址。
    实测(带登录态)对比各尺寸档字节数:/large/ == /original/ 始终最大,
    是微博保留的原始图;/woriginal/ 反而是被压缩的小图,不能用。
    """
    ext = "gif" if pid.startswith("8") else "jpg"
    return [f"https://wx1.sinaimg.cn/large/{pid}.{ext}"]


# --------------------------------------------------------------------------- #
# 模式一:按 UID 下载图片墙
# --------------------------------------------------------------------------- #
def fetch_by_uid(session, uid, max_needed=None):
    """
    使用 weibo.com ajax 接口分页抓取用户图片墙的全部 pid。
    接口: https://weibo.com/ajax/profile/getImageWall
    通过响应里的 since_id 翻页,直到没有更多。
    max_needed: 收集到这么多张后提前停止翻页(用于范围下载,省请求)。
    """
    api = "https://weibo.com/ajax/profile/getImageWall"
    referer = f"https://weibo.com/u/{uid}"
    pids = []
    seen = set()
    since_id = "0"
    page = 0
    while True:
        page += 1
        params = {"uid": uid, "sinceid": since_id, "has_album": "true"}
        data = request_json(session, api, params=params, referer=referer)
        if not data or data.get("ok") != 1:
            print(f"[提示] 第 {page} 页无有效数据,停止。")
            break
        block = data.get("data", {})
        items = block.get("list", []) or []
        new_count = 0
        for it in items:
            pid = it.get("pid")
            if pid and pid not in seen:
                seen.add(pid)
                pids.append(pid)
                new_count += 1
        print(f"[UID] 第 {page} 页获取 {len(items)} 张,新增 {new_count},累计 {len(pids)}")

        if max_needed is not None and len(pids) >= max_needed:
            print(f"[UID] 已收集 {len(pids)} 张,满足所需 {max_needed},停止翻页。")
            break

        # 翻页游标 since_id 为空/0 才表示到底。
        # 注意:不能用 new_count==0 作为停止条件——某页可能整页都是无 pid
        # 的项(如视频)或重复项,但后面仍有图,提前停会漏图。
        next_since = block.get("since_id")
        if not next_since or str(next_since) == "0":
            break
        if next_since == since_id:
            # 游标未推进,防御性退出,避免死循环
            print("[UID] since_id 未推进,停止。")
            break
        since_id = str(next_since)
        time.sleep(SLEEP_BETWEEN_PAGES)
    return pids


# --------------------------------------------------------------------------- #
# 模式二:按单个相册 URL 下载
# --------------------------------------------------------------------------- #
def parse_album_url(url):
    """从相册链接中解析出 uid 和 album_id。"""
    # 形如 https://photo.weibo.com/{uid}/albums/detail/album_id/{album_id}
    m_uid = re.search(r"photo\.weibo\.com/(\d+)", url)
    m_album = re.search(r"album_id/(\d+)", url)
    uid = m_uid.group(1) if m_uid else None
    album_id = m_album.group(1) if m_album else None
    return uid, album_id


def fetch_by_album(session, uid, album_id, max_needed=None):
    """
    使用 photo.weibo.com 接口分页抓取指定相册的全部 pid。
    接口: https://photo.weibo.com/photos/get_photo_wall_v2
    max_needed: 收集到这么多张后提前停止翻页(用于范围下载,省请求)。
    """
    api = "https://photo.weibo.com/photos/get_photo_wall_v2"
    referer = f"https://photo.weibo.com/{uid}/albums/detail/album_id/{album_id}"
    pids = []
    seen = set()
    page = 0
    count = 100
    while True:
        page += 1
        params = {
            "uid": uid,
            "album_id": album_id,
            "type": "3",
            "page": page,
            "count": count,
        }
        data = request_json(session, api, params=params, referer=referer)
        if not data:
            break
        photo_list = (data.get("data") or {}).get("photo_list", []) or []
        if not photo_list:
            print(f"[相册] 第 {page} 页无数据,停止。")
            break
        new_count = 0
        for p in photo_list:
            pid = p.get("pid") or p.get("photo_id")
            if pid and pid not in seen:
                seen.add(pid)
                pids.append(pid)
                new_count += 1
        print(f"[相册] 第 {page} 页获取 {len(photo_list)} 张,累计 {len(pids)}")
        if max_needed is not None and len(pids) >= max_needed:
            print(f"[相册] 已收集 {len(pids)} 张,满足所需 {max_needed},停止翻页。")
            break
        if len(photo_list) < count or new_count == 0:
            break
        time.sleep(SLEEP_BETWEEN_PAGES)
    return pids


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def download_one(session, pid, out_dir):
    urls = candidate_urls_from_pid(pid)
    ext = urls[0].rsplit(".", 1)[-1]
    filename = os.path.join(out_dir, f"{pid}.{ext}")
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return "skip"
    # 依次尝试候选地址(/woriginal/ 优先,回退 /large/)
    for url in urls:
        for attempt in range(1, RETRY + 1):
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
                if resp.status_code == 404:
                    # 该尺寸不存在,换下一个候选地址
                    break
                resp.raise_for_status()
                tmp = filename + ".part"
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, filename)
                return "ok"
            except Exception:  # noqa: BLE001
                time.sleep(1.0 * attempt)
    return "fail"


def download_all(session, pids, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    total = len(pids)
    if total == 0:
        print("[提示] 没有可下载的图片。")
        return
    print(f"[下载] 共 {total} 张,输出目录: {out_dir},并发: {workers}")
    stats = {"ok": 0, "skip": 0, "fail": 0}
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, session, pid, out_dir): pid for pid in pids}
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            stats[result] += 1
            done += 1
            if done % 10 == 0 or done == total:
                print(
                    f"  进度 {done}/{total}  "
                    f"成功 {stats['ok']} 跳过 {stats['skip']} 失败 {stats['fail']}"
                )
    print(
        f"[完成] 成功 {stats['ok']},跳过(已存在) {stats['skip']},失败 {stats['fail']}"
    )
    if stats["fail"]:
        print("[提示] 有失败项,可重新运行脚本自动续传(已下载的会跳过)。")


def apply_range(pids, start, end):
    """按 1-based 的 [start, end] 闭区间截取 pid 列表。end 为 None 表示到末尾。"""
    total = len(pids)
    lo = (start - 1) if start else 0
    hi = end if end else total
    selected = pids[lo:hi]
    print(
        f"[范围] 共抓到 {total} 张,选取第 {lo + 1} 到 {min(hi, total)} 张,"
        f"实际 {len(selected)} 张。"
    )
    return selected


def main():
    parser = argparse.ArgumentParser(description="微博相册原图批量下载")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--uid", help="用户 UID,下载其图片墙全部原图")
    group.add_argument("--album", help="单个相册 URL(photo.weibo.com)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument(
        "--cookie", default=DEFAULT_COOKIE_FILE, help="Cookie 文件路径"
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help="并发下载线程数"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="从第几张开始下载(1-based,含。默认 1)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="下载到第几张结束(1-based,含。默认到末尾)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="只抓取 pid 列表(不下载),配合 --list-out 写出 JSON,用于预览",
    )
    parser.add_argument(
        "--list-out",
        default=None,
        help="--list-only 模式下,把 {uid,album_id,out_dir_name,pids} 写到该 JSON 文件",
    )
    parser.add_argument(
        "--pids-file",
        default=None,
        help="只下载该文件里列出的 pid(每行一个),跳过抓取与范围裁剪,用于选择性下载",
    )
    args = parser.parse_args()

    # 校验范围参数
    if args.start < 1:
        sys.exit("[错误] --start 必须 >= 1")
    if args.end is not None and args.end < args.start:
        sys.exit(f"[错误] --end({args.end}) 不能小于 --start({args.start})")

    cookie = load_cookie(args.cookie)
    session = build_session(cookie)

    # 解析输出子目录名 + 抓取所需的 uid/album_id
    if args.uid:
        uid, album_id = args.uid, None
        out_dir_name = f"uid_{args.uid}"
    else:
        uid, album_id = parse_album_url(args.album)
        if not uid or not album_id:
            sys.exit(
                "[错误] 无法从相册 URL 解析出 uid / album_id,请确认链接格式,"
                "例如: https://photo.weibo.com/1234567890/albums/detail/album_id/9876543210"
            )
        out_dir_name = f"album_{album_id}"
    out_dir = os.path.join(args.out, out_dir_name)

    # 选择性下载:直接用给定的 pid 列表,跳过抓取与范围裁剪。
    if args.pids_file:
        with open(args.pids_file, "r", encoding="utf-8") as f:
            pids = [ln.strip() for ln in f if ln.strip()]
        print(f"[模式] 选择性下载: {len(pids)} 张 -> {out_dir}")
        download_all(session, pids, out_dir, args.workers)
        return

    # 只需抓到 end 张即可,提前停止翻页省请求
    max_needed = args.end

    if args.uid:
        print(f"[模式] 按 UID 抓取: {args.uid}")
        pids = fetch_by_uid(session, uid, max_needed=max_needed)
    else:
        print(f"[模式] 按相册抓取: uid={uid} album_id={album_id}")
        pids = fetch_by_album(session, uid, album_id, max_needed=max_needed)

    pids = apply_range(pids, args.start, args.end)

    # 预览模式:写出 pid 列表 JSON,不下载。
    if args.list_only:
        payload = {
            "uid": uid,
            "album_id": album_id,
            "out_dir_name": out_dir_name,
            "pids": pids,
        }
        if args.list_out:
            with open(args.list_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        print(f"[完成] 抓取到 {len(pids)} 个 pid(仅列表,未下载)。")
        return

    download_all(session, pids, out_dir, args.workers)


if __name__ == "__main__":
    main()
