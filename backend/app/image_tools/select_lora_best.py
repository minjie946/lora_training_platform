#!/usr/bin/env -S uv run --script
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "opencv-python>=4.8.0",
#     "numpy>=1.24.0",
#     "insightface>=0.7.3",
#     "onnxruntime>=1.16.0",
# ]
# ///
"""
LoRA 精选脚本：从"合格单人"目录(single/)里挑出最适合训练的 Top-N 张。

前置：先跑 filter_single_person.py 得到 single/(单人·可训练)。本脚本在其上
做两件事：
  1) 质量打分：复用 InsightFace 已产出的信号(人脸 bbox / 5 点关键点 kps /
     检测置信度 det_score)+ cv2，把清晰度、人脸占比、分辨率、正脸程度、曝光、
     置信度归一化为 0~1 的连续分并加权求综合质量分（不再是通过/不通过的硬门）。
  2) 多样性去重：加载 InsightFace recognition 模块取 512 维人脸 embedding，用
     "质量 + 最远点采样(FPS)"贪心选取，避免选出 50 张几乎一样的自拍/同一姿势，
     让训练集在保持人物一致的前提下覆盖更多角度/表情/服装。

选中的图拷贝(不移动，保留 single/ 原图)到 single_best/。重跑会先清空 single_best/。

用法：
  uv run select_lora_best.py ./photos/uid_123/single --out ./photos/uid_123/single_best --count 50
"""

import argparse
import math
import os
import shutil
import sys

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

INSIGHTFACE_ROOT = os.environ.get(
    "INSIGHTFACE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_insightface"),
)


def collect_images(root):
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            out.append(p)
    return out


def load_image(image_path):
    """兼容中文路径读取；失败返回 None。"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def _largest_face(faces):
    """返回面积最大的人脸对象；无脸返回 None。"""
    best = None
    best_area = 0.0
    for fc in faces:
        x1, y1, x2, y2 = fc.bbox
        area = max(x2 - x1, 1.0) * max(y2 - y1, 1.0)
        if area > best_area:
            best_area = area
            best = fc
    return best


def quality_score(img, face):
    """把一张单人图的多维质量归一化为综合分(0~1，越高越好)。

    各维度都用连续值软加权，而非 filter 里的硬阈值门。返回 (score, detail_dict)。
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    fw, fh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    # 1) 分辨率：短边，1536 及以上给满分
    short_side = min(h, w)
    s_res = min(short_side / 1536.0, 1.0)

    # 2) 人脸占比：越大主体特征越足，25% 给满分
    face_ratio = (fw * fh) / (w * h) * 100.0
    s_face = min(face_ratio / 25.0, 1.0)

    # 3) 清晰度：人脸区域拉普拉斯方差(裁剪归一化到 256px，去除分辨率/磨皮干扰)
    fx1, fy1 = max(int(x1), 0), max(int(y1), 0)
    fx2, fy2 = min(int(x2), w), min(int(y2), h)
    lap_var = 0.0
    if fx2 > fx1 and fy2 > fy1:
        gray = cv2.cvtColor(img[fy1:fy2, fx1:fx2], cv2.COLOR_BGR2GRAY)
        gh, gw = gray.shape
        scale = 256.0 / max(gh, gw)
        if scale < 1.0:
            gray = cv2.resize(
                gray, (max(int(gw * scale), 1), max(int(gh * scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    s_sharp = min(lap_var / 300.0, 1.0)

    # 4) 曝光：整图灰度均值，越接近中性(约 128)越好，两端衰减
    mean_lum = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())
    s_expo = max(0.0, 1.0 - abs(mean_lum - 128.0) / 128.0)

    # 5) 正脸程度：yaw(鼻尖偏离眼中线/眼距) + roll(两眼连线倾角)
    s_pose = 0.6  # 无关键点时给中性分
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 5:
        le, re, nose = kps[0], kps[1], kps[2]
        eye_dx, eye_dy = (re[0] - le[0]), (re[1] - le[1])
        eye_dist = math.hypot(eye_dx, eye_dy) or 1.0
        eye_cx = (le[0] + re[0]) / 2.0
        yaw = abs(nose[0] - eye_cx) / eye_dist
        roll = abs(math.degrees(math.atan2(eye_dy, eye_dx)))
        s_yaw = max(0.0, 1.0 - yaw / 0.5)     # yaw>=0.5 视为完全侧脸
        s_roll = max(0.0, 1.0 - roll / 45.0)  # roll>=45° 视为严重歪头
        s_pose = 0.6 * s_yaw + 0.4 * s_roll

    # 6) 人脸置信度(遮挡/低质代理)：det_score 本身即 0~1
    s_det = float(getattr(face, "det_score", 1.0) or 1.0)

    # 加权综合：清晰度与正脸对训练稳定性影响最大
    score = (
        0.25 * s_sharp
        + 0.20 * s_pose
        + 0.18 * s_face
        + 0.15 * s_det
        + 0.12 * s_res
        + 0.10 * s_expo
    )
    detail = {
        "res": round(s_res, 3), "face": round(s_face, 3), "sharp": round(s_sharp, 3),
        "expo": round(s_expo, 3), "pose": round(s_pose, 3), "det": round(s_det, 3),
        "short_side": short_side, "face_ratio": round(face_ratio, 1),
        "lap_var": round(lap_var, 0),
    }
    return score, detail


def select_diverse(candidates, count, quality_weight):
    """质量 + 最远点采样(FPS)贪心选取 Top-N。

    candidates: [{path, score, emb(np.ndarray 已归一化 或 None)}]
    每步选取 combined = q_w*quality + (1-q_w)*diversity 最大的候选，其中
    diversity = 与已选集合的最小余弦距离(归一化到 0~1)。首个先选质量最高者。
    """
    if len(candidates) <= count:
        return list(candidates)

    remaining = list(candidates)
    remaining.sort(key=lambda c: c["score"], reverse=True)
    selected = [remaining.pop(0)]  # 质量最高的作为种子

    have_emb = all(c["emb"] is not None for c in candidates)
    while remaining and len(selected) < count:
        best_idx = 0
        best_combined = -1.0
        for i, c in enumerate(remaining):
            if have_emb:
                # 归一化 embedding 的余弦相似度 = 点积；距离 = 1 - sim ∈ [0,2]
                sims = [float(np.dot(c["emb"], s["emb"])) for s in selected]
                min_dist = 1.0 - max(sims)  # 与最相近已选项的距离
                diversity = max(0.0, min(min_dist / 2.0, 1.0))
            else:
                diversity = 0.0
            combined = quality_weight * c["score"] + (1.0 - quality_weight) * diversity
            if combined > best_combined:
                best_combined = combined
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected


def unique_dest(dest_dir, filename):
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{i}{ext}")
        i += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(description="从 single/ 里挑出最适合 LoRA 训练的 Top-N")
    parser.add_argument("directory", help="合格单人目录(single/)")
    parser.add_argument("--out", required=True, help="输出目录(single_best/)")
    parser.add_argument("--count", type=int, default=50, help="精选数量(默认 50)")
    parser.add_argument(
        "--quality-weight", type=float, default=0.6,
        help="质量权重 0~1(默认 0.6)；越高越偏高质量，越低越偏多样性",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="人脸检测置信度阈值(默认 0.5)")
    parser.add_argument(
        "--min-face", type=float, default=0.5,
        help="最小人脸面积占比%(默认 0.5)，小于此的脸视为误检忽略",
    )
    parser.add_argument(
        "--no-diversity", action="store_true",
        help="关闭多样性去重，纯按质量分取 Top-N(不加载 recognition 模型)",
    )
    args = parser.parse_args()

    root = args.directory
    if not os.path.isdir(root):
        sys.exit(f"[错误] 目录不存在: {root}")

    images = collect_images(root)
    if not images:
        sys.exit(f"[提示] 目录下没有图片: {root}")
    print(
        f"[扫描] 共 {len(images)} 张合格单人图，目标精选 {args.count} 张 "
        f"(质量权重={args.quality_weight}{'，纯质量' if args.no_diversity else '，含多样性去重'})"
    )

    print("[模型] 加载 InsightFace (首次会自动下载约 280MB)...")
    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    from insightface.app import FaceAnalysis

    # 需要多样性去重时额外加载 recognition 拿 embedding；否则只加载 detection。
    modules = ["detection"] if args.no_diversity else ["detection", "recognition"]
    app = FaceAnalysis(name="buffalo_l", allowed_modules=modules, root=INSIGHTFACE_ROOT)
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=args.conf)

    min_ratio = args.min_face / 100.0
    candidates = []
    for idx, path in enumerate(images, 1):
        img = load_image(path)
        if img is None:
            print(f"  [{idx}/{len(images)}] 读取失败  {os.path.basename(path)}")
            continue
        h, w = img.shape[:2]
        total_area = w * h
        faces = [
            fc for fc in app.get(img)
            if (max(fc.bbox[2] - fc.bbox[0], 1.0) * max(fc.bbox[3] - fc.bbox[1], 1.0)) / total_area >= min_ratio
        ]
        face = _largest_face(faces)
        if face is None:
            print(f"  [{idx}/{len(images)}] 无有效人脸(跳过)  {os.path.basename(path)}")
            continue
        score, detail = quality_score(img, face)
        emb = None
        if not args.no_diversity:
            e = getattr(face, "normed_embedding", None)
            if e is not None:
                emb = np.asarray(e, dtype=np.float32)
        candidates.append({"path": path, "score": score, "emb": emb, "detail": detail})
        print(
            f"  [{idx}/{len(images)}] 质量 {score:.3f} "
            f"(清晰{detail['sharp']}/正脸{detail['pose']}/占比{detail['face']}/置信{detail['det']})  "
            f"{os.path.basename(path)}"
        )

    if not candidates:
        sys.exit("[提示] 没有可评分的单人图")

    chosen = select_diverse(candidates, args.count, args.quality_weight)
    # 输出保持质量分从高到低，便于人工复核。
    chosen.sort(key=lambda c: c["score"], reverse=True)

    out_dir = args.out
    # 重跑先清空输出目录，避免历史精选残留累加。
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    for c in chosen:
        dest = unique_dest(out_dir, os.path.basename(c["path"]))
        shutil.copy2(c["path"], dest)

    avg = sum(c["score"] for c in chosen) / len(chosen)
    print(
        f"\n[完成] 已精选 {len(chosen)}/{len(candidates)} 张 → single_best/ "
        f"(平均质量分 {avg:.3f})"
    )


if __name__ == "__main__":
    main()
