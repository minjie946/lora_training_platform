#!/usr/bin/env -S uv run --script
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "playwright>=1.40.0",
#     "requests>=2.28.0",
# ]
# ///
"""小红书「博主主页全量」原图批量下载脚本（纯 Python，真实浏览器会话）。

给定一个小红书博主主页链接（带 xsec_token），抓取其**全部公开笔记**中的原图并
下载到本地。命令行契约刻意对齐 weibo_album_downloader.py，以便后端 image_manager
复用同一套「预览 → 勾选 → 下载 / 进度解析 / 断点续传」逻辑。

反爬策略：不再自己伪造签名（那样会被判 300011「账号异常」）。改为用 Playwright
注入你**完整的登录 Cookie**，直接驱动真实浏览器：
  1. 打开博主主页（保留 URL 上的 xsec_token），滚动到底，
     拦截页面自己发出的 user_posted 接口，收集全部 {note_id, xsec_token}。
  2. 逐篇打开笔记页（explore/<id>?xsec_token=...），读取页面里的
     window.__INITIAL_STATE__.note.noteDetailMap 拿到该篇的 imageList 原图。
所有请求签名（x-s / x-s-common / x-t 等）都由真实页面完成，风控视作正常用户。

首次使用需安装浏览器内核：
  uv run --script xhs_user_downloader.py --install-browser

用法：
  # 预览：抓取该博主全部笔记的图片列表，写出 JSON，不下载
  uv run xhs_user_downloader.py --user "<主页链接?xsec_token=...>" \
      --cookie ./xhs_cookie.txt --list-only --list-out out.json

  # 全量下载
  uv run xhs_user_downloader.py --user "<主页链接?xsec_token=...>" \
      --cookie ./xhs_cookie.txt --out ./downloads

  # 选择性下载（配合预览产出的 image id 列表，每行一个）
  uv run xhs_user_downloader.py --user "<主页链接?xsec_token=...>" \
      --cookie ./xhs_cookie.txt --out ./downloads --ids-file ./ids.txt
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time

import requests

REQUEST_TIMEOUT = 30
RETRY = 3
# 页面交互节流：滚动 / 逐篇打开之间的等待，尽量像真人，降低风控概率。
SCROLL_PAUSE = 1.2
NOTE_PAUSE = 0.8
MAX_SCROLL_ROUNDS = 60  # 主页滚动收集笔记的最大轮数，防止死循环

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

_STEALTH_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stealth.min.js")


# --------------------------------------------------------------------------- #
# Cookie / URL 解析
# --------------------------------------------------------------------------- #
def load_cookie(cookie_file: str) -> str:
    if not os.path.exists(cookie_file):
        sys.exit(f"[错误] 找不到 Cookie 文件: {cookie_file}")
    with open(cookie_file, "r", encoding="utf-8") as f:
        cookie = f.read().strip()
    if not cookie:
        sys.exit(f"[错误] Cookie 文件为空: {cookie_file}")
    if "web_session=" not in cookie or "a1=" not in cookie:
        print("[警告] Cookie 建议同时包含 a1 / web_session / webId，否则可能被判账号异常。")
    return cookie


def cookie_to_playwright(cookie: str) -> list:
    """把整段 Cookie 字符串拆成 Playwright add_cookies 需要的结构。"""
    items = []
    for part in cookie.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        items.append(
            {
                "name": name.strip(),
                "value": value.strip(),
                "domain": ".xiaohongshu.com",
                "path": "/",
            }
        )
    return items


def parse_user_id(user: str) -> str:
    m = re.search(r"/user/profile/([0-9a-f]+)", user)
    if m:
        return m.group(1)
    if re.fullmatch(r"[0-9a-f]{16,}", user.strip()):
        return user.strip()
    sys.exit(
        "[错误] 无法解析博主 user_id，请传主页链接，形如 "
        "https://www.xiaohongshu.com/user/profile/<uid>?xsec_token=..."
    )


def _pick_original_url(img: dict) -> str:
    """从一条 imageList 项里取**原图** URL。

    页面 state / 缩略图 CDN 给的是降档图，如：
      sns-webpic-qc.xhscdn.com/<ts>/<hash>/<imageid>!nd_dft_wlteh_webp_3
    其中最后一段 <imageid> 拼到 ci.xiaohongshu.com/<imageid> 即为原图（实测
    体积约为缩略图的 2~3 倍）。因此这里统一提取 imageid 走原图 CDN。
    """
    raw = ""
    for key in ("urlDefault", "url_default", "urlPre", "url_pre", "url"):
        v = img.get(key)
        if v:
            raw = v
            break
    if not raw:
        info = img.get("infoList") or img.get("info_list") or []
        if info and isinstance(info[-1], dict):
            raw = info[-1].get("url", "")
    return _to_original(raw)


# imageid 形如 1040g2sg322sglc337k705q6jljf8cbt47v5o02g（字母数字，长度 ~40）。
def _to_original(url: str) -> str:
    """把小红书缩略图 URL 转为原图 URL（ci.xiaohongshu.com/<imageid>）。

    带 imageView2/format/jpg 强制转标准 JPEG——ci 默认可能返回 HEIF/HEIC，
    PIL 等训练/打标库读不了，转 JPG 后通用且尺寸不变。
    """
    if not url:
        return url
    fmt = "?imageView2/format/jpg"
    if "ci.xiaohongshu.com/" in url:
        return url if "imageView2" in url else url.split("?")[0] + fmt
    # 取路径最后一段作为 imageid（去掉 !xxx 尺寸后缀）。
    tail = url.split("?")[0].rstrip("/").split("/")[-1]
    imgid = tail.split("!")[0]
    if imgid and re.fullmatch(r"[0-9a-zA-Z]{20,}", imgid):
        return f"https://ci.xiaohongshu.com/{imgid}{fmt}"
    return url


# --------------------------------------------------------------------------- #
# 抓取：真实浏览器会话
# --------------------------------------------------------------------------- #
def collect_with_browser(user_url: str, cookie: str, max_notes=None, headed=False) -> list:
    """打开博主主页收集全部笔记，再逐篇读取原图，返回图片项列表。

    每项：{id, note_id, thumb_url, full_url}，id="笔记id_序号"。

    收集分两路，互补以对抗验证码：
      1) 首屏 SSR：直接读 __INITIAL_STATE__.user.notes（不受验证码影响，最可靠）。
      2) 翻页 XHR：滚动触发 user_posted 累加后续页；若被验证码拦截则至少保住首屏。

    ``headed=True`` 会弹出真实浏览器窗口：headless 下滚动翻页常触发滑块验证码，
    页面就不再发 user_posted，只能拿到首屏。有头模式下你可以手动过一次验证码，
    之后脚本继续滚动就能翻完全部页（这是稳定拿到"全量"笔记的推荐方式）。
    """
    from playwright.sync_api import sync_playwright

    items: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        context = browser.new_context(user_agent=UA, viewport={"width": 1440, "height": 900})
        if os.path.exists(_STEALTH_JS):
            context.add_init_script(path=_STEALTH_JS)
        context.add_cookies(cookie_to_playwright(cookie))
        page = context.new_page()

        notes: dict[str, str] = {}  # note_id -> xsec_token

        # 拦截翻页接口 user_posted，累加后续页的笔记。
        def on_response(resp):
            if "/api/sns/web/v1/user_posted" not in resp.url:
                return
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001
                return
            for n in (data.get("data", {}) or {}).get("notes", []) or []:
                nid = n.get("note_id") or n.get("id")
                if nid and nid not in notes:
                    notes[nid] = n.get("xsec_token", "")

        page.on("response", on_response)
        print(f"[主页] 打开 {user_url}")
        page.goto(user_url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2.5)

        # 路 1：首屏 SSR 直接读 user.notes（关键，绕开验证码）
        for nid, token in _read_initial_notes(page):
            notes.setdefault(nid, token)
        print(f"[主页] 首屏收集到 {len(notes)} 篇笔记")

        if headed:
            print(
                "[提示] 已弹出浏览器窗口。若出现滑块/验证码请手动完成；"
                "看到笔记正常加载后，脚本会自动滚动翻页收集全部笔记，请勿关闭窗口。"
            )
            # 给用户时间过验证码再开始滚动。
            time.sleep(8)

        # 路 2：尽力滚动翻页
        last_count = len(notes)
        stagnant = 0
        for rnd in range(MAX_SCROLL_ROUNDS):
            page.mouse.wheel(0, 4000)
            time.sleep(SCROLL_PAUSE)
            # 每轮也再读一次 state（前端可能把新页并进 user.notes）
            for nid, token in _read_initial_notes(page):
                notes.setdefault(nid, token)
            cur = len(notes)
            if cur == last_count:
                stagnant += 1
                # 有头模式给更长的停滞容忍，方便用户过验证码后继续加载。
                if stagnant >= (6 if headed else 3):
                    break
            else:
                stagnant = 0
            last_count = cur
            if cur != last_count or rnd % 3 == 0:
                print(f"[主页] 第 {rnd + 1} 轮滚动，已收集 {cur} 篇笔记")
            if max_notes and cur >= max_notes:
                break

        note_list = list(notes.items())
        if max_notes:
            note_list = note_list[:max_notes]
        if not note_list:
            browser.close()
            sys.exit(
                "[错误] 未收集到任何笔记。请确认：主页链接带了 xsec_token；"
                "Cookie 为最新登录态；该博主有公开图文笔记。"
            )
        print(f"[主页] 共收集 {len(note_list)} 篇笔记，开始逐篇读取原图…")

        for idx, (note_id, token) in enumerate(note_list, 1):
            note_url = (
                f"https://www.xiaohongshu.com/explore/{note_id}"
                f"?xsec_token={token}&xsec_source=pc_user"
            )
            # 笔记详情页是 SSR，图片写在 __INITIAL_STATE__ 里（是缩略 URL），
            # _pick_original_url 会据此换算出原图 CDN 地址。
            _read_note_images(page, note_url, note_id)
            imgs = _read_state_images(page, note_id)
            for i, url in enumerate(imgs):
                items.append(
                    {
                        "id": f"{note_id}_{i}",
                        "note_id": note_id,
                        "thumb_url": url,
                        "full_url": url,
                    }
                )
            if idx % 5 == 0 or idx == len(note_list):
                print(f"[笔记] 已解析 {idx}/{len(note_list)} 篇，累计图片 {len(items)} 张")
            time.sleep(NOTE_PAUSE)

        browser.close()
    return items


def _read_initial_notes(page) -> list:
    """从 __INITIAL_STATE__.user.notes 读出 [(note_id, xsec_token), ...]。

    notes 是分组的嵌套数组（每组是一页），且是含循环引用的响应式对象，
    因此在浏览器里手动挑字段扁平化，避免 JSON.stringify 报错。
    """
    try:
        pairs = page.evaluate(
            """() => {
                try {
                    const u = window.__INITIAL_STATE__ && window.__INITIAL_STATE__.user;
                    if (!u) return [];
                    let arr = u.notes;
                    if (arr && arr._rawValue) arr = arr._rawValue;
                    if (!Array.isArray(arr)) return [];
                    const out = [];
                    arr.forEach(group => {
                        const g = Array.isArray(group) ? group : [group];
                        g.forEach(item => {
                            const nc = item.noteCard || item.note_card || item;
                            const id = item.id || nc.noteId || nc.note_id;
                            const tk = item.xsecToken || nc.xsecToken || nc.xsec_token || '';
                            if (id) out.push([id, tk]);
                        });
                    });
                    return out;
                } catch (e) { return []; }
            }"""
        )
        return [(p[0], p[1]) for p in pairs if p and p[0]]
    except Exception:  # noqa: BLE001
        return []


def _read_note_images(page, note_url: str, note_id: str) -> None:
    """打开单篇笔记页，触发其 feed 请求（由 on_response 抓取原图）。

    只负责导航 + 等待；原图优先来自被拦截的 feed 响应，读取逻辑在调用处。
    """
    for attempt in range(1, RETRY + 1):
        try:
            page.goto(note_url, wait_until="domcontentloaded", timeout=45000)
            time.sleep(1.0)  # 等 feed XHR 回来
            return
        except Exception as e:  # noqa: BLE001
            if attempt == RETRY:
                print(f"[跳过] 笔记 {note_id} 打开失败：{e}")
                return
            time.sleep(1.0 * attempt)


def _read_state_images(page, note_id: str) -> list:
    """兜底：feed 没抓到时，从 __INITIAL_STATE__.note.noteDetailMap 读 imageList。"""
    try:
        raw = page.evaluate(
            """(nid) => {
                try {
                    const s = window.__INITIAL_STATE__;
                    if (!s || !s.note) return null;
                    const map = s.note.noteDetailMap || {};
                    const entry = map[nid] || Object.values(map)[0];
                    if (!entry || !entry.note) return null;
                    const list = entry.note.imageList || [];
                    return list.map(im => ({
                        urlDefault: im.urlDefault || im.url_default || '',
                        urlPre: im.urlPre || im.url_pre || '',
                        infoList: (im.infoList || im.info_list || []).map(x => ({ url: x.url }))
                    }));
                } catch (e) { return null; }
            }""",
            note_id,
        )
        if raw:
            urls = [_pick_original_url(im) for im in raw]
            return [u for u in urls if u]
    except Exception:  # noqa: BLE001
        pass
    return []


# --------------------------------------------------------------------------- #
# 下载
# --------------------------------------------------------------------------- #
def download_one(session, item: dict, out_dir: str):
    """下载单张原图。返回 (status, reason)：status ∈ ok|skip|fail。

    失败时 reason 记录最后一次的 HTTP 状态码或异常，便于在日志里定位到底是
    限流(403/429)、原图不存在(404) 还是网络问题，而不是笼统的“失败”。
    """
    url = item["full_url"]
    ext = "png" if ".png" in url.lower() else "webp" if ".webp" in url.lower() else "jpg"
    filename = os.path.join(out_dir, f"{item['id']}.{ext}")
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        return "skip", ""
    reason = ""
    for attempt in range(1, RETRY + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            if resp.status_code == 200:
                tmp = filename + ".part"
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, filename)
                return "ok", ""
            reason = f"HTTP {resp.status_code}"
            # 404 视为原图不存在，重试无意义；403/429 多为限流，退避后再试。
            if resp.status_code == 404:
                break
            time.sleep(1.5 * attempt)
        except Exception as e:  # noqa: BLE001
            reason = type(e).__name__
            time.sleep(1.0 * attempt)
    return "fail", reason


def download_all(items: list, out_dir: str, workers: int):
    os.makedirs(out_dir, exist_ok=True)
    total = len(items)
    if total == 0:
        print("[提示] 没有可下载的图片。")
        return
    print(f"[下载] 共 {total} 张,输出目录: {out_dir},并发: {workers}")
    session = requests.Session()
    session.headers.update(
        {"User-Agent": UA, "Referer": "https://www.xiaohongshu.com/"}
    )
    stats = {"ok": 0, "skip": 0, "fail": 0}
    reasons: dict[str, int] = {}
    fail_samples: list[str] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, session, it, out_dir): it for it in items}
        for fut in concurrent.futures.as_completed(futures):
            status, reason = fut.result()
            stats[status] += 1
            if status == "fail":
                reasons[reason or "unknown"] = reasons.get(reason or "unknown", 0) + 1
                if len(fail_samples) < 3:
                    fail_samples.append(f"{reason or 'unknown'} · {futures[fut]['full_url']}")
            done += 1
            if done % 10 == 0 or done == total:
                print(
                    f"  进度 {done}/{total}  "
                    f"成功 {stats['ok']} 跳过 {stats['skip']} 失败 {stats['fail']}"
                )
    print(f"[完成] 成功 {stats['ok']},跳过(已存在) {stats['skip']},失败 {stats['fail']}")
    if reasons:
        summary = ", ".join(f"{k}×{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        print(f"[失败原因] {summary}")
        for s in fail_samples:
            print(f"[失败样本] {s}")


def main():
    parser = argparse.ArgumentParser(description="小红书博主主页全量原图下载")
    parser.add_argument("--user", help="博主主页链接（建议带 xsec_token）或 user_id")
    parser.add_argument("--out", default="xhs_photos", help="输出根目录")
    parser.add_argument("--cookie", default="xhs_cookie.txt", help="Cookie 文件路径")
    parser.add_argument("--workers", type=int, default=6, help="并发下载线程数")
    parser.add_argument("--max-notes", type=int, default=None, help="最多解析多少篇笔记")
    parser.add_argument("--list-only", action="store_true", help="仅抓图片列表不下载")
    parser.add_argument("--list-out", default=None, help="--list-only 时写出的 JSON")
    parser.add_argument("--ids-file", default=None, help="仅下载文件中列出的 image id")
    parser.add_argument(
        "--headed", action="store_true",
        help="弹出真实浏览器窗口，可手动过验证码后翻页收集全部笔记（拿全量推荐）",
    )
    parser.add_argument(
        "--install-browser", action="store_true", help="安装 playwright chromium 后退出"
    )
    args = parser.parse_args()

    if args.install_browser:
        import subprocess

        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
        return

    if not args.user:
        sys.exit("[错误] 需要 --user 指定博主主页链接或 user_id")

    cookie = load_cookie(args.cookie)
    user_id = parse_user_id(args.user)
    out_dir_name = f"xhs_user_{user_id}"
    out_dir = os.path.join(args.out, out_dir_name)

    # 若传入的是裸 id，则补一个主页 URL（但没有 xsec_token 命中率会很低）。
    user_url = args.user.strip()
    if not user_url.startswith("http"):
        user_url = f"https://www.xiaohongshu.com/user/profile/{user_id}"

    print(f"[模式] 博主全量抓取（真实浏览器会话）: {user_id}")
    items = collect_with_browser(user_url, cookie, max_notes=args.max_notes, headed=args.headed)

    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
        items = [it for it in items if it["id"] in wanted]
        print(f"[模式] 选择性下载: 命中 {len(items)} 张")

    if args.list_only:
        payload = {"user_id": user_id, "out_dir_name": out_dir_name, "items": items}
        if args.list_out:
            with open(args.list_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        print(f"[完成] 抓取到 {len(items)} 张图片(仅列表,未下载)。")
        return

    download_all(items, out_dir, args.workers)


if __name__ == "__main__":
    main()
