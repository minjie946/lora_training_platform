#!/usr/bin/env -S uv run --script
# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "opencv-python>=4.8.0",
#     "numpy>=1.24.0",
#     "insightface>=0.7.3",
#     "onnxruntime>=1.16.0",
#     "rapidocr-onnxruntime>=1.3.0",
#     "ultralytics>=8.0.0",
# ]
# ///
"""
单人图片筛选脚本。

用 InsightFace(RetinaFace)人脸检测挑出"单人照",按类别归档:
  single/   恰好 1 张有效人脸,且非拼图、非海报、非纯动物照
  multi/    >=2 张有效人脸(合照)
  poster/   海报 / 大量文字图
  collage/  多图拼接
  animal/   纯动物照(有动物且画面中无人)
无人脸的图留在原处不动。

为什么用 InsightFace 而非轻量的 YuNet:实测明星侧脸写真、艺术滤镜、
玻璃反光背景等图,YuNet 会漏检或把背景误判成多张脸,调参无法解决;
InsightFace 对侧脸/滤镜/反光鲁棒得多(依赖 onnxruntime,比 torch 轻)。

为什么额外加 YOLO 动物检测:InsightFace 只训练人脸,但猫狗等动物特写偶尔
会被误检成人脸而错归 single/。用 YOLOv8n(约 6MB)检测画面主体,若检出
动物且无人 -> animal/,既能把纯动物照单独归档,又能拦截"动物脸当人脸"的
误检。人+动物的合照(person 存在)仍按人脸数正常分类,动物脸不干扰计数。

海报判定:RapidOCR 检测文字,文字块数或面积超阈值即判为海报。衣服上的
零星文字/水印面积很小不会误判(实测正常人像文字面积 <0.5%,海报达 14%)。
拼图判定:拼接缝像素突变 + 超细长比例。

用法示例:
  uv run filter_single_person.py ./weibo_photos/uid_1234567890 --dry-run
  uv run filter_single_person.py ./weibo_photos --recursive

首次运行会自动下载 InsightFace(约 280MB)、RapidOCR(约 10MB)与
YOLOv8n(约 6MB)。模型分别缓存在脚本同目录的 models_insightface/、
models_yolo/ 下。加 --no-animal-filter 可跳过 YOLO 动物检测。
"""

import argparse
import math
import os
import shutil
import sys

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# InsightFace 模型缓存目录。默认脚本同目录,可用环境变量 INSIGHTFACE_ROOT 覆盖
# (平台会指向 workspace 下的持久化缓存,避免污染源码树、且免重复下载 ~600MB)。
INSIGHTFACE_ROOT = os.environ.get(
    "INSIGHTFACE_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_insightface"),
)

# YOLOv8n 权重路径。默认脚本同目录,可用环境变量 YOLO_WEIGHTS 覆盖。首次运行自动下载(约 6MB)。
YOLO_WEIGHTS = os.environ.get(
    "YOLO_WEIGHTS",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models_yolo", "yolov8n.pt"
    ),
)

# COCO 数据集中的动物类别 id(cat/dog/bird/horse/sheep/cow/elephant/bear/zebra/giraffe)
COCO_ANIMAL_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
COCO_PERSON_ID = 0


def collect_images(root, recursive):
    """收集待处理图片(跳过分类输出目录)。"""
    skip_dirs = {"single", "single_lowq", "multi", "poster", "collage", "animal"}
    images = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for name in filenames:
                if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                    images.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                images.append(path)
    return sorted(images)


def load_image(image_path):
    """兼容中文路径读取图片;失败返回 None。"""
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def count_faces(app, img, min_face_ratio):
    """
    返回 (有效人脸数量, 有效人脸对象列表)。
    只统计面积占比 >= min_face_ratio 的人脸,过滤画像/盘饰/背景小脸
    (实测真人脸占比通常 >1%,而画上的小脸只有 0.01%~0.1%)。
    """
    h, w = img.shape[:2]
    total_area = w * h
    faces = app.get(img)
    valid = []
    for fc in faces:
        x1, y1, x2, y2 = fc.bbox
        fw, fh = (x2 - x1), (y2 - y1)
        if (fw * fh) / total_area >= min_face_ratio:
            valid.append(fc)
    return len(valid), valid


def assess_lora_quality(img, face, args):
    """
    判断一张"单人照"是否适合 LoRA 训练。返回 (ok, reasons)。

    只用 InsightFace 已产出的信息(bbox / 5 点关键点 kps / 检测置信度 det_score)
    加 cv2,不额外引入依赖。任一维度不达标即 ok=False,reasons 收集所有原因。
    """
    reasons = []
    h, w = img.shape[:2]

    # 1) 最小分辨率:按短边(LoRA 常用 512~1024,过小放大后细节丢失)
    short_side = min(h, w)
    if short_side < args.q_min_short_side:
        reasons.append(f"分辨率低(短边{short_side}<{args.q_min_short_side})")

    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    fw, fh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    # 2) 人脸足够大:占比过小则主体特征不足
    face_ratio = (fw * fh) / (w * h) * 100.0
    if face_ratio < args.q_min_face_area:
        reasons.append(f"人脸过小({face_ratio:.1f}%<{args.q_min_face_area}%)")

    # 3) 人脸完整:bbox 贴边说明脸被画面裁切
    m = args.q_edge_margin
    if x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m:
        reasons.append("人脸贴边(可能被裁切)")

    # 4) 清晰度:人脸区域的拉普拉斯方差。先把人脸裁剪缩放到统一尺寸(最长边
    #    256px)再计算,消除分辨率与美颜磨皮的干扰——否则大图/磨皮自拍的皮肤
    #    区域梯度天然偏小,原始方差会把清晰的正常自拍误判成模糊。
    fx1, fy1 = max(int(x1), 0), max(int(y1), 0)
    fx2, fy2 = min(int(x2), w), min(int(y2), h)
    if fx2 > fx1 and fy2 > fy1:
        crop = img[fy1:fy2, fx1:fx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gh, gw = gray.shape
        scale = 256.0 / max(gh, gw)
        if scale < 1.0:
            gray = cv2.resize(
                gray, (max(int(gw * scale), 1), max(int(gh * scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < args.q_blur_var:
            reasons.append(f"模糊(清晰度{lap_var:.0f}<{args.q_blur_var})")

    # 5) 曝光:整图灰度均值过暗/过曝
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_lum = float(gray_full.mean())
    if mean_lum < args.q_dark_mean:
        reasons.append(f"过暗(亮度{mean_lum:.0f})")
    elif mean_lum > args.q_bright_mean:
        reasons.append(f"过曝(亮度{mean_lum:.0f})")

    # 6) 正面 + 遮挡:用 5 点关键点(左眼/右眼/鼻/左嘴角/右嘴角)估姿态
    #    kps 形如 [[x,y]*5]。yaw 用鼻尖相对两眼中点的水平偏移/眼距衡量,
    #    roll 用两眼连线倾角衡量。偏移/倾角过大 -> 侧脸/歪头,不利于稳定学习。
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 5:
        le, re, nose = kps[0], kps[1], kps[2]
        eye_dx, eye_dy = (re[0] - le[0]), (re[1] - le[1])
        eye_dist = math.hypot(eye_dx, eye_dy) or 1.0
        eye_cx = (le[0] + re[0]) / 2.0
        # yaw 近似:鼻尖偏离两眼中线的比例(正面≈0,侧脸增大)
        yaw = abs(nose[0] - eye_cx) / eye_dist
        if yaw > args.q_yaw:
            reasons.append(f"侧脸(yaw{yaw:.2f}>{args.q_yaw})")
        # roll 近似:两眼连线与水平线夹角
        roll = abs(math.degrees(math.atan2(eye_dy, eye_dx)))
        if roll > args.q_roll:
            reasons.append(f"歪头(roll{roll:.0f}°>{args.q_roll}°)")

    # 7) 遮挡/低质检测:检测置信度低往往对应遮挡、糊脸、极端角度
    det = float(getattr(face, "det_score", 1.0) or 1.0)
    if det < args.q_det_score:
        reasons.append(f"人脸置信度低({det:.2f}<{args.q_det_score},可能遮挡)")

    return (len(reasons) == 0), reasons


def detect_subjects(yolo_model, img, min_ratio, conf):
    """
    用 YOLO 检测画面主体,返回 (是否有人, 是否有动物)。
    只统计边框面积占比 >= min_ratio 的目标,过滤背景里的小猫小狗/路人,
    与人脸判定的面积过滤思路一致(避免背景动物图案误判)。
    """
    h, w = img.shape[:2]
    total_area = w * h
    results = yolo_model(img, conf=conf, verbose=False)
    has_person = False
    has_animal = False
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if ((x2 - x1) * (y2 - y1)) / total_area < min_ratio:
                continue
            if cls_id == COCO_PERSON_ID:
                has_person = True
            elif cls_id in COCO_ANIMAL_IDS:
                has_animal = True
    return has_person, has_animal


def is_collage(img, seam_thresh, ar_thresh, strong_seam):
    """
    判断是否为多图拼接。
    依据:相邻行/列的整体亮度突变(拼接缝),实测拼图可达 76~146,
    正常照片仅 2~3。判定规则:
      - 强缝(>= strong_seam)且缝位于图像中部 -> 直接判拼图(如 3:4 双拼);
      - 中等缝(>= seam_thresh)还需配合超细长比例(长条多图拼接)。
    缝须在中部(0.15~0.85)以排除边缘黑边/水印造成的边缘突变。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row_prof = gray.mean(axis=1)
    col_prof = gray.mean(axis=0)
    row_diff = np.abs(np.diff(row_prof)) if h > 1 else np.array([0.0])
    col_diff = np.abs(np.diff(col_prof)) if w > 1 else np.array([0.0])
    h_seam = float(row_diff.max())
    v_seam = float(col_diff.max())
    h_pos = row_diff.argmax() / max(len(row_diff), 1)
    v_pos = col_diff.argmax() / max(len(col_diff), 1)

    def in_middle(pos):
        return 0.15 <= pos <= 0.85

    # 强缝且在中部 -> 拼图(不要求细长)
    if (h_seam >= strong_seam and in_middle(h_pos)) or (
        v_seam >= strong_seam and in_middle(v_pos)
    ):
        return True
    # 中等缝 + 超细长比例 -> 长条拼接
    ar = w / h
    elongated = ar <= ar_thresh or ar >= (1.0 / ar_thresh)
    return max(h_seam, v_seam) >= seam_thresh and elongated


def count_text(ocr_engine, img, min_text_h):
    """
    返回 (文字块数, 文字总面积占比%)。
    只统计字高占比 >= min_text_h 的文字块,忽略极小噪点。
    """
    result, _ = ocr_engine(img)
    if not result:
        return 0, 0.0
    h, w = img.shape[:2]
    total = h * w
    n = 0
    area = 0.0
    for box, _txt, _score in result:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        bh = max(ys) - min(ys)
        bw = max(xs) - min(xs)
        if bh / h * 100 < min_text_h:
            continue
        n += 1
        area += bw * bh
    return n, area / total * 100


def unique_dest(dest_dir, filename):
    """目标目录若已存在同名文件,追加序号避免覆盖。"""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{base}_{i}{ext}")
        i += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(description="筛选单人图片并移动到 single/ 文件夹")
    parser.add_argument("directory", help="待筛选的图片目录")
    parser.add_argument("--recursive", action="store_true", help="递归处理子目录")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="人脸检测置信度阈值(默认 0.5)"
    )
    parser.add_argument(
        "--min-face",
        type=float,
        default=0.5,
        help="最小人脸面积占比(百分比,默认 0.5)。小于此值的脸视为画像/背景误检忽略",
    )
    parser.add_argument(
        "--text-blocks",
        type=int,
        default=5,
        help="文字块数达到此值判为海报/文字图排除(默认 5)",
    )
    parser.add_argument(
        "--text-area",
        type=float,
        default=5.0,
        help="文字总面积占比%达到此值判为海报/文字图排除(默认 5.0)",
    )
    parser.add_argument(
        "--min-text-h",
        type=float,
        default=1.5,
        help="纳入统计的最小字高占比%(默认 1.5),过滤极小噪点",
    )
    parser.add_argument(
        "--seam",
        type=float,
        default=40.0,
        help="中等拼接缝阈值(默认 40),需配合超细长比例才判拼图",
    )
    parser.add_argument(
        "--strong-seam",
        type=float,
        default=60.0,
        help="强拼接缝阈值(默认 60),缝在图像中部即判拼图(不要求细长)",
    )
    parser.add_argument(
        "--collage-ar",
        type=float,
        default=0.6,
        help="拼图细长比阈值(默认 0.6):纵横比<=此值或>=其倒数才算细长",
    )
    parser.add_argument(
        "--no-text-filter", action="store_true", help="关闭文字/海报排除(不加载 OCR)"
    )
    parser.add_argument(
        "--no-animal-filter",
        action="store_true",
        help="关闭动物检测/归档(不加载 YOLO)",
    )
    parser.add_argument(
        "--animal-conf",
        type=float,
        default=0.4,
        help="动物/人体检测置信度阈值(默认 0.4)",
    )
    parser.add_argument(
        "--min-animal",
        type=float,
        default=3.0,
        help="最小动物/人体边框面积占比%(默认 3.0),小于此值视为背景忽略",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印结果,不移动文件")

    # ---- LoRA 训练质量筛选(在"单人"基础上再挑可训练的图)----
    q = parser.add_argument_group("LoRA 训练质量筛选")
    q.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="关闭质量筛选:单人照全部进 single/,不再细分 single_lowq/",
    )
    q.add_argument(
        "--q-min-short-side",
        type=int,
        default=768,
        help="质量:最小短边像素(默认 768),小于此判分辨率不足",
    )
    q.add_argument(
        "--q-min-face-area",
        type=float,
        default=4.0,
        help="质量:人脸最小面积占比%(默认 4.0),小于此判人脸过小",
    )
    q.add_argument(
        "--q-edge-margin",
        type=float,
        default=2.0,
        help="质量:人脸框贴边容差像素(默认 2),更近判为被裁切",
    )
    q.add_argument(
        "--q-blur-var",
        type=float,
        default=60.0,
        help="质量:人脸区域拉普拉斯方差阈值(默认 60,人脸裁剪归一化到 256px 后计算),低于此判模糊",
    )
    q.add_argument(
        "--q-dark-mean",
        type=float,
        default=40.0,
        help="质量:整图灰度均值下限(默认 40),低于此判过暗",
    )
    q.add_argument(
        "--q-bright-mean",
        type=float,
        default=220.0,
        help="质量:整图灰度均值上限(默认 220),高于此判过曝",
    )
    q.add_argument(
        "--q-yaw",
        type=float,
        default=0.35,
        help="质量:偏航(侧脸)阈值(默认 0.35,鼻尖偏离眼中线/眼距比),大于此判侧脸",
    )
    q.add_argument(
        "--q-roll",
        type=float,
        default=30.0,
        help="质量:滚转(歪头)角度阈值(默认 30°),大于此判歪头",
    )
    q.add_argument(
        "--q-det-score",
        type=float,
        default=0.55,
        help="质量:人脸检测置信度下限(默认 0.55),低于此判疑似遮挡/低质",
    )
    args = parser.parse_args()

    root = args.directory
    if not os.path.isdir(root):
        sys.exit(f"[错误] 目录不存在: {root}")

    images = collect_images(root, args.recursive)
    if not images:
        sys.exit(f"[提示] 目录下没有图片: {root}")
    print(
        f"[扫描] 共 {len(images)} 张,人脸conf={args.conf} 最小脸占比={args.min_face}% "
        f"文字排除:块数>={args.text_blocks}或面积>={args.text_area}% "
        f"{'(已关闭)' if args.no_text_filter else ''} "
        f"动物归档{'(已关闭)' if args.no_animal_filter else f':最小占比>={args.min_animal}%'}"
    )

    print("[模型] 加载 InsightFace (首次会自动下载约 280MB)...")
    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l", allowed_modules=["detection"], root=INSIGHTFACE_ROOT
    )
    app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=args.conf)

    ocr_engine = None
    if not args.no_text_filter:
        print("[模型] 加载 RapidOCR (首次会自动下载约 10MB)...")
        from rapidocr_onnxruntime import RapidOCR

        ocr_engine = RapidOCR()

    yolo_model = None
    if not args.no_animal_filter:
        print("[模型] 加载 YOLOv8n (首次会自动下载约 6MB)...")
        os.makedirs(os.path.dirname(YOLO_WEIGHTS), exist_ok=True)
        from ultralytics import YOLO

        yolo_model = YOLO(YOLO_WEIGHTS)

    # 各分类的目标子目录
    category_dirs = {
        "single": os.path.join(root, "single"),
        "single_lowq": os.path.join(root, "single_lowq"),
        "multi": os.path.join(root, "multi"),
        "poster": os.path.join(root, "poster"),
        "collage": os.path.join(root, "collage"),
        "animal": os.path.join(root, "animal"),
    }

    def move_to(category, src_path):
        """把文件移动到对应分类目录(dry-run 时只记录不动)。"""
        if args.dry_run:
            return
        dest_dir = category_dirs[category]
        os.makedirs(dest_dir, exist_ok=True)
        dest = unique_dest(dest_dir, os.path.basename(src_path))
        shutil.move(src_path, dest)

    min_ratio = args.min_face / 100.0
    min_animal_ratio = args.min_animal / 100.0
    stats = {
        "single": 0,
        "single_lowq": 0,
        "multi": 0,
        "none": 0,
        "collage": 0,
        "text": 0,
        "animal": 0,
        "error": 0,
    }
    for idx, path in enumerate(images, 1):
        img = load_image(path)
        if img is None:
            stats["error"] += 1
            print(f"  [{idx}/{len(images)}] 读取失败  {os.path.basename(path)}")
            continue

        # 优先级:海报/文字图 > 拼图 > 按人脸数分类
        # (海报底部文字条带常造成强像素突变,若先判拼图会误归 collage,
        #  故文字/海报判定优先)
        if ocr_engine is not None and _is_text_image(
            ocr_engine, img, args.min_text_h, args.text_blocks, args.text_area
        ):
            stats["text"] += 1
            tag = "海报/文字图 📝 → poster/"
            move_to("poster", path)
            print(f"  [{idx}/{len(images)}] {tag}  {os.path.basename(path)}")
            continue

        if is_collage(img, args.seam, args.collage_ar, args.strong_seam):
            stats["collage"] += 1
            tag = "拼图 ✂️ → collage/"
            move_to("collage", path)
            print(f"  [{idx}/{len(images)}] {tag}  {os.path.basename(path)}")
            continue

        # YOLO 检测画面主体:既用于挑出纯动物照,又用于拦截"动物脸误检为人脸"
        has_person, has_animal = (False, False)
        if yolo_model is not None:
            has_person, has_animal = detect_subjects(
                yolo_model, img, min_animal_ratio, args.animal_conf
            )

        n, faces = count_faces(app, img, min_ratio)

        # 有动物且 YOLO 未检出人体 -> 纯动物照。即便人脸检测误报了一张脸,
        # 只要 YOLO 确认画面主体里没有人,就归入 animal/,避免污染 single/。
        if has_animal and not has_person:
            stats["animal"] += 1
            tag = "动物 🐾 → animal/"
            move_to("animal", path)
        elif n == 0:
            stats["none"] += 1
            tag = "无人脸(不移动)"
        elif n >= 2:
            stats["multi"] += 1
            tag = f"{n} 张脸 → multi/"
            move_to("multi", path)
        else:
            # 单人照:再做一道 LoRA 训练质量筛选,达标→single/,不达标→single_lowq/
            if args.no_quality_filter:
                stats["single"] += 1
                tag = "单人 ✅ → single/"
                move_to("single", path)
            else:
                ok, reasons = assess_lora_quality(img, faces[0], args)
                if ok:
                    stats["single"] += 1
                    tag = "单人·可训练 ✅ → single/"
                    move_to("single", path)
                else:
                    stats["single_lowq"] += 1
                    tag = f"单人·质量不足 ⚠️ → single_lowq/（{'; '.join(reasons)}）"
                    move_to("single_lowq", path)
        print(f"  [{idx}/{len(images)}] {tag}  {os.path.basename(path)}")

    action = "将分类" if args.dry_run else "已分类"
    print(
        f"\n[完成] {action}:单人·可训练 {stats['single']}→single/,"
        f"单人·质量不足 {stats['single_lowq']}→single_lowq/,"
        f"多人 {stats['multi']}→multi/,海报/文字图 {stats['text']}→poster/,"
        f"拼图 {stats['collage']}→collage/,动物 {stats['animal']}→animal/,"
        f"无人脸 {stats['none']}(留原处),失败 {stats['error']}"
    )
    if args.dry_run:
        print("[提示] 这是 dry-run,未真正移动文件。去掉 --dry-run 执行实际移动。")


def _is_text_image(ocr_engine, img, min_text_h, blocks_thresh, area_thresh):
    """文字块数或文字面积达阈值即判为海报/文字图。"""
    n, area = count_text(ocr_engine, img, min_text_h)
    return n >= blocks_thresh or area >= area_thresh


if __name__ == "__main__":
    main()
