import json
import os
import pickle
import time
from scipy.spatial.transform import Rotation as R_

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoImageProcessor, AutoModel,  # DINOv2
)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from scipy.spatial import distance
from scipy.spatial import cKDTree
from Cameras.OrbbecCamera import OrbbecCamera
from transformers import pipeline, AutoModelForDepthEstimation, AutoImageProcessor
# from DiffRenderNetNT import DiffRenderNetNT
import open3d as o3d

import shutil


def copy_best_template_files(template_dir, tpl_index, output_dir, use_subfolder=True):
    """
    复制模板编号对应的所有文件，包括：
    - rgb_xxxx.png
    - mask_xxxx.png
    - pose_xxxx.npy
    - pc_xxxx.ply
    - pc_visible_pixel_xxxx.ply
    - 其他以相同编号结尾的文件（自动匹配）

    支持两种情况：
    1）template_dir 本身就是扁平模板库目录（旧结构）
    2）template_dir 是父目录，下面有 front / back 子目录（新结构）
    """
    tpl_index_str = f"{tpl_index:04d}"

    if use_subfolder:
        tpl_output_dir = os.path.join(output_dir, f"template_{tpl_index_str}")
    else:
        tpl_output_dir = output_dir
    os.makedirs(tpl_output_dir, exist_ok=True)

    copied_files = []

    def _copy_in_one_dir(one_dir):
        local_copied = []
        if not os.path.isdir(one_dir):
            return local_copied
        for fname in os.listdir(one_dir):
            if tpl_index_str in fname:
                src_path = os.path.join(one_dir, fname)
                dst_path = os.path.join(tpl_output_dir, fname)
                shutil.copy(src_path, dst_path)
                local_copied.append(dst_path)
        return local_copied

    has_rgb_here = os.path.isdir(template_dir) and any(
        f.startswith("rgb_") and f.endswith(".png")
        for f in os.listdir(template_dir)
    )
    if has_rgb_here:
        copied_files.extend(_copy_in_one_dir(template_dir))
    else:
        for sub in ["front", "back"]:
            sub_dir = os.path.join(template_dir, sub)
            copied_files.extend(_copy_in_one_dir(sub_dir))

    if copied_files:
        print(f"[INFO] 已复制以下模板文件到 {tpl_output_dir}:")
        for f in copied_files:
            print("  →", os.path.basename(f))
    else:
        print(f"[WARN] 在 {template_dir} (以及其 front/back 子目录) 中未找到编号 {tpl_index_str} 对应的文件。")

    return copied_files


def mask_iou(maskA, maskB):
    """
    计算两个二值 mask 的 IoU（交并比）
    """
    intersection = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    return 0.0 if union == 0 else intersection / union


def robust_chamfer_distance(contourA, contourB, delta=1.0):
    """
    对称且鲁棒的 Chamfer Distance
    contourA, contourB: Nx2 numpy arrays
    delta: Huber loss 截断阈值
    """
    # 计算 A 中每个点到 B 最近点的距离
    treeB = cKDTree(contourB)
    dists_A_to_B, _ = treeB.query(contourA)

    # 对距离使用 Huber 风格的鲁棒损失，降低离群点影响
    dists_A_to_B = np.where(dists_A_to_B <= delta,
                            0.5 * dists_A_to_B ** 2,
                            delta * (dists_A_to_B - 0.5 * delta))

    # 计算 B 中每个点到 A 最近点的距离
    treeA = cKDTree(contourA)
    dists_B_to_A, _ = treeA.query(contourB)
    dists_B_to_A = np.where(dists_B_to_A <= delta,
                            0.5 * dists_B_to_A ** 2,
                            delta * (dists_B_to_A - 0.5 * delta))

    # 对称 Chamfer：A->B 与 B->A 的平均损失相加
    cd = dists_A_to_B.mean() + dists_B_to_A.mean()
    return cd


def chamfer_distance(contourA, contourB):
    """
    基础 Chamfer Distance
    contourA, contourB: Nx2 numpy arrays
    """
    # A 到 B 的最近邻距离
    treeB = cKDTree(contourB)
    dists_A_to_B, _ = treeB.query(contourA)

    # B 到 A 的最近邻距离
    treeA = cKDTree(contourA)
    dists_B_to_A, _ = treeA.query(contourB)

    # 对称 Chamfer 距离
    cd = dists_A_to_B.mean() + dists_B_to_A.mean()
    return cd


def resample_contour(contour, num_points):
    """
    对轮廓进行重采样，使其具有 num_points 个均匀分布的点

    :param contour: 输入轮廓
    :param num_points: 目标点数
    :return: 重采样后的轮廓
    """
    # 将 OpenCV 轮廓格式转换为 (N,2)
    contour = contour.reshape(-1, 2)

    # 计算轮廓总长度
    arc_length = cv2.arcLength(contour, closed=True)

    # 每个采样点对应的长度间隔
    step_size = arc_length / num_points

    resampled_contour = []
    accumulated_length = 0

    for i in range(num_points):
        target_length = i * step_size

        # 沿轮廓前进，直到接近目标弧长位置
        while accumulated_length < target_length:
            dist = np.linalg.norm(contour[0] - contour[1])
            if accumulated_length + dist >= target_length:
                break
            accumulated_length += dist

        # 这里直接取 contour 中对应位置点作为近似采样点
        resampled_contour.append(contour[i % len(contour)])

    return np.array(resampled_contour)


def normalize_contour(contour):
    """
    将轮廓做中心化 + 尺度归一化
    """
    # 计算轮廓中心
    center = np.mean(contour, axis=0)

    # 平移到以中心为原点
    contour_centered = contour - center

    # 用最大半径作为缩放尺度
    scale = np.max(np.linalg.norm(contour_centered, axis=1))
    contour_normalized = contour_centered / scale
    return contour_normalized


def get_normalized_contour(image):
    """
    处理图像，提取轮廓并归一化

    :param image: 输入图像（已经是灰度图像）
    :return: 归一化后的轮廓
    """
    # 二值化
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

    # 转灰度（如果输入是三通道）
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # 提取外轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    # 将轮廓重采样为固定点数，便于后续统一比较
    num_points = 400
    contour_resampled = resample_contour(contour, num_points)

    # 归一化轮廓
    contour_normalized = normalize_contour(contour_resampled)
    return contour_normalized


def extract_depth_feature(depth_map, mask, bins=50):
    """
    提取深度直方图特征（min-max 归一化到 [0,1]）
    """
    # 取 mask 区域内的深度值
    depth_vals = depth_map[mask > 0]

    # 去掉 NaN 和无效深度
    depth_vals = depth_vals[~np.isnan(depth_vals)]
    depth_vals = depth_vals[depth_vals > 1e-6]

    if len(depth_vals) > 50:
        # 深度归一化到 [0,1]
        min_val, max_val = np.min(depth_vals), np.max(depth_vals)
        depth_vals_norm = (depth_vals - min_val) / (max_val - min_val + 1e-6)

        # 统计直方图
        hist, _ = np.histogram(depth_vals_norm, bins=bins, range=(0, 1), density=False)
        hist = hist.astype(np.float32)

        # 归一化成概率分布
        hist = hist / (np.sum(hist) + 1e-6)
    else:
        # 点数太少时返回全零特征
        hist = np.zeros(bins, dtype=np.float32)

    return hist


def depth_to_colormap(depth, mask):
    """
    输入:
    depth: (H, W) numpy 数组, 深度图
    mask: (H, W) bool 数组, 掩码

    输出:
    color_map: (H, W, 3) uint8, 伪彩色图
    """
    # 初始化归一化深度图
    norm_depth = np.zeros_like(depth, dtype=np.float32)

    # 提取掩码区域深度
    masked_depth = depth[mask]

    if len(masked_depth) > 0:
        # 将掩码区域深度做 min-max 归一化
        d_min, d_max = masked_depth.min(), masked_depth.max()
        if d_max > d_min:
            norm = (masked_depth - d_min) / (d_max - d_min)
        else:
            norm = np.zeros_like(masked_depth)

        # 填回原图
        norm_depth[mask] = norm

    # 转为 0~255 的 uint8 便于做伪彩色映射
    norm_depth_uint8 = (norm_depth * 255).astype(np.uint8)

    # 使用 OpenCV VIRIDIS 伪彩色
    color_map = cv2.applyColorMap(norm_depth_uint8, cv2.COLORMAP_VIRIDIS)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    # 掩码外区域全部置黑
    color_map[~mask] = 0
    return color_map


def run_depth_anything(image_color_bgr,
                       model_dir="./depth-anything",
                       cam_parameters=[2439.038515225739, 2439.621607590318, 959.059761340049, 612.6729280875528, 1200, 1920],
                       show=True):
    """
    使用 depth-anything 模型预测深度 + 渲染

    Args:
        image_color_bgr: (H,W,3) numpy BGR 图像
        model_dir: depth-anything 模型目录
        cam_parameters: 相机参数 [fx, fy, cx, cy, H, W]
        show: 是否显示结果

    Returns:
        depth_np: numpy float32 深度图
        rendered_img: 渲染后的伪彩色图 (RGB)
    """
    # 1. 加载 depth-anything 模型
    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir)
    pipe = pipeline(
        task="depth-estimation",
        model=model,
        image_processor=processor,
        device=0
    )

    # 2. BGR 转 RGB，再转 PIL 用于推理
    image_rgb = cv2.cvtColor(image_color_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    # 3. 执行深度估计
    out = pipe(pil_img)
    depth = out["depth"]

    # 4. 转为 numpy
    depth_np = np.array(depth).astype(np.float32)

    # 5. 显示预测深度图
    if show:
        plt.figure()
        plt.title("Depth (predicted by depth-anything)")
        plt.imshow(depth_np, cmap="plasma")
        plt.colorbar(label="Depth value")
        plt.show()

    # 6. 全图都参与渲染
    image_mask = np.ones(depth_np.shape, dtype=bool)

    # 7. 使用外部渲染网络将深度变成伪彩色（这里依赖 DiffRenderNetNT）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    render = DiffRenderNetNT(device, cam_parameters, "deg", "mm").to(device)
    rendered_img = render.depth_to_colormap(255 - depth_np, image_mask)

    # 8. 显示渲染结果
    if show:
        plt.figure()
        plt.title("Rendered Depth Colormap")
        plt.imshow(cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return depth_np, rendered_img


def normalize_outputs(masks, boxes, scores):
    """
    将 SAM3 输出统一转成 numpy 数组
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    return np.array(masks), np.array(boxes), np.array(scores)


def split_connected_masks(bin_mask):
    """
    将二值 mask 按连通域拆分成多个独立 mask
    """
    num_labels, labels = cv2.connectedComponents(bin_mask)
    masks = []
    for i in range(1, num_labels):
        masks.append((labels == i).astype(np.uint8) * 255)
    return masks


def enhance_edges(image_rgb):
    """
    对 RGB 图像做锐化 + 边缘增强，提升 SAM3 对零件边界的感知
    """
    # 转灰度提取边缘
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # 锐化卷积核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(image_rgb, -1, kernel)

    # 将锐化图与边缘图融合
    enhanced = cv2.addWeighted(sharp, 0.8, edges_rgb, 0.2, 0)
    return enhanced


def remove_small_components(mask_bool, min_area=300):
    """
    去掉 mask 里的小连通域碎片，避免零散小像素
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    cleaned = np.zeros_like(mask_u8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


class ImageCoarseSegmentor():
    def __init__(self, template_dir="template_output"):
        # 模板目录路径
        self.template_dir = template_dir

        # 自动检测是否为 front/back 双模板库结构
        front_dir = os.path.join(self.template_dir, "front")
        back_dir = os.path.join(self.template_dir, "back")
        if os.path.isdir(front_dir) and os.path.isdir(back_dir):
            self.front_template_dir = front_dir
            self.back_template_dir = back_dir
            self._load_front_back_libraries = True
            print(f"[INFO] 检测到 front/back 双模板库模式: {self.template_dir}")
        else:
            self.front_template_dir = None
            self.back_template_dir = None
            self._load_front_back_libraries = False
            print(f"[INFO] 使用单模板库模式: {self.template_dir}")

        # 加载 DINOv2 模型，用于模板特征编码
        self.load_dinov2_model()

        # 加载 SAM3 模型，用于候选分割
        print("Loading SAM3...")
        self.sam3_model = build_sam3_image_model()

        # move to cuda
        self.sam3_model.backbone.to("cuda")
        self.sam3_model.transformer.to("cuda")
        self.sam3_model.transformer.encoder.to("cuda")
        self.sam3_model.transformer.decoder.to("cuda")
        self.sam3_model.geometry_encoder.to("cuda")

        self.sam3_processor = Sam3Processor(self.sam3_model, device="cuda")

        print("SAM3 Loaded!")

        # 检查并加载模板缓存特征
        self.check_and_load_template()

        # 加载模板的位姿与相机参数信息
        self.load_pose6s_with_depth()

    # ======================= DINOv2 编码（模板匹配） =======================
    def load_dinov2_model(self):
        print("[INIT] 加载 DinoV2 模型...")

        # 优先从本地读取 DinoV2 模型
        local_model_dir = "./DinoV2/facebook/dinov2-base"
        hf_model_id = "facebook/dinov2-base"

        if os.path.exists(local_model_dir):
            model_id = local_model_dir
            print(f"[INFO] 使用本地 DinoV2: {model_id}")
        else:
            model_id = hf_model_id
            print(f"[WARN] 本地 DinoV2 不存在，改用 HuggingFace: {model_id}")

        try:
            # 加载图像预处理器与模型
            self.dinov2_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
            self.dinov2_model = AutoModel.from_pretrained(model_id)
            self.dinov2_model = self.dinov2_model.to("cuda").eval()
            print("[OK] DinoV2 模型加载成功")
        except Exception as e:
            print(f"[ERROR] DinoV2 加载失败: {e}")
            raise

        return self.dinov2_model, self.dinov2_processor, "cuda"

    def crop_resize_pad(self, image_rgb, mask=None, target_size=224):
        """
        根据 mask 裁剪目标区域，然后等比例缩放，最后 padding 到 target_size
        """
        if mask is not None:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0 or len(ys) == 0:
                # 如果 mask 为空，返回一张全黑图
                return Image.fromarray(np.zeros((target_size, target_size, 3), dtype=np.uint8))
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cropped = image_rgb[y_min:y_max + 1, x_min:x_max + 1]
        else:
            cropped = image_rgb

        h, w = cropped.shape[:2]

        # 这里缩放比例写成 /2，意味着目标区域会在最终图中更小，保留更多边界留白
        scale = target_size / max(h, w) / 2
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cropped, (new_w, new_h))

        # 中心填充到固定大小
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_off = (target_size - new_h) // 2
        x_off = (target_size - new_w) // 2
        padded[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        return padded

    def encode_image_with_dinov2(self, pil_img):
        """
        用 DinoV2 提取图像全局特征，并做 L2 归一化
        """
        inputs = self.dinov2_processor(images=pil_img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            feat = self.dinov2_model(**inputs).last_hidden_state[:, 0, :]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def get_template_contours(self, template_dir):
        """
        遍历模板库，提取每个模板的：
        - 文件名
        - 归一化轮廓
        - 裁剪图像
        - DinoV2 特征
        """
        count = 0
        features = []

        # 读取模板 RGB 与 mask 文件列表
        files = sorted([f for f in os.listdir(template_dir) if f.startswith("rgb_") and f.endswith(".png")])
        files_mask = sorted([f for f in os.listdir(template_dir) if f.startswith("mask_") and f.endswith(".png")])

        for fname, fname_mask in tqdm(zip(files, files_mask), desc="[DINOv2] 编码模板"):
            img = cv2.cvtColor(cv2.imread(os.path.join(template_dir, fname)), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(template_dir, fname_mask), 0)
            mask = mask > 0

            # 根据 mask 裁剪并规范化模板图
            cropped_image = self.crop_resize_pad(img, mask=mask)

            # 提取模板轮廓
            contour = get_normalized_contour(cropped_image)

            # 提取 DinoV2 特征
            dino_feature = self.encode_image_with_dinov2(Image.fromarray(cropped_image))
            features.append((fname, contour, cropped_image, dino_feature))

        return features


    def _ensure_side_prefix(self, feats, side):
        """
        确保双模板库模式下，模板名带有 front/ 或 back/ 前缀
        """
        new_feats = []
        for name, contour, cropped, feat in feats:
            if isinstance(name, str) and "/" not in name:
                name = f"{side}/{name}"
            new_feats.append((name, contour, cropped, feat))
        return new_feats

    def _tpl_index_from_name(self, tpl_name):
        """
        支持：
        - rgb_0001.png
        - front/rgb_0001.png
        - back/rgb_0001.png
        - 0001
        """
        if isinstance(tpl_name, str) and "/" in tpl_name:
            tpl_name = tpl_name.split("/", 1)[1]

        if isinstance(tpl_name, str) and tpl_name.startswith("rgb_"):
            try:
                return int(tpl_name.split("_")[1].split(".")[0])
            except Exception:
                pass

        try:
            return int(tpl_name)
        except Exception:
            raise ValueError(f"无法从模板名解析编号: {tpl_name}")

    def _get_template_real_path(self, tpl_name):
        """
        根据模板名返回真实文件路径
        支持单模板库和 front/back 双模板库
        """
        if tpl_name is None:
            return None

        if isinstance(tpl_name, str) and "/" in tpl_name:
            side, fname = tpl_name.split("/", 1)
            if side == "front" and self.front_template_dir is not None:
                return os.path.join(self.front_template_dir, fname)
            if side == "back" and self.back_template_dir is not None:
                return os.path.join(self.back_template_dir, fname)

        return os.path.join(self.template_dir, tpl_name)

    def _get_pose6_from_tpl_name(self, tpl_name):
        """
        根据模板名返回对应 pose6
        """
        tpl_index = self._tpl_index_from_name(tpl_name)

        if isinstance(tpl_name, str) and tpl_name.startswith("front/"):
            if hasattr(self, "front_pose6s") and tpl_index < len(self.front_pose6s):
                return self.front_pose6s[tpl_index]["pose6"]
            return None

        if isinstance(tpl_name, str) and tpl_name.startswith("back/"):
            if hasattr(self, "back_pose6s") and tpl_index < len(self.back_pose6s):
                return self.back_pose6s[tpl_index]["pose6"]
            return None

        if hasattr(self, "pose6s_feats") and tpl_index < len(self.pose6s_feats):
            return self.pose6s_feats[tpl_index]["pose6"]

        return None

    def check_and_load_template(self):
        """
        若模板库结构为：
            template_output/front
            template_output/back
        则分别加载 front/back，并构造统一的 self.template_feats。

        否则保持单模板库模式，并在 template_dir 内保存 pkl。
        """
        # ---------- 双模板库模式 ----------
        if getattr(self, "_load_front_back_libraries", False):
            print("[INFO] front/back 双模板库模式启用")

            tpl_name_front = os.path.basename(os.path.normpath(self.front_template_dir))
            cache_front = os.path.join(self.front_template_dir, f"{tpl_name_front}.pkl")

            if os.path.exists(cache_front):
                with open(cache_front, "rb") as f:
                    feats_front = pickle.load(f)
                self.front_template_feats = self._ensure_side_prefix(feats_front, "front")
                print(f"[OK] Front 模板缓存已加载: {cache_front}")
            else:
                print(f"[WARN] Front 缓存不存在，开始编码模板库: {self.front_template_dir}")
                feats_front = self.get_template_contours(self.front_template_dir)
                self.front_template_feats = self._ensure_side_prefix(feats_front, "front")
                with open(cache_front, "wb") as f:
                    pickle.dump(self.front_template_feats, f)
                print(f"[DONE] Front 模板库编码完成，已保存缓存文件: {cache_front}")

            tpl_name_back = os.path.basename(os.path.normpath(self.back_template_dir))
            cache_back = os.path.join(self.back_template_dir, f"{tpl_name_back}.pkl")

            if os.path.exists(cache_back):
                with open(cache_back, "rb") as f:
                    feats_back = pickle.load(f)
                self.back_template_feats = self._ensure_side_prefix(feats_back, "back")
                print(f"[OK] Back 模板缓存已加载: {cache_back}")
            else:
                print(f"[WARN] Back 缓存不存在，开始编码模板库: {self.back_template_dir}")
                feats_back = self.get_template_contours(self.back_template_dir)
                self.back_template_feats = self._ensure_side_prefix(feats_back, "back")
                with open(cache_back, "wb") as f:
                    pickle.dump(self.back_template_feats, f)
                print(f"[DONE] Back 模板库编码完成，已保存缓存文件: {cache_back}")

            self.template_feats = self.front_template_feats + self.back_template_feats
            print(f"[INFO] front+back 模板总数: {len(self.template_feats)}")
            return

        # ---------- 单模板库模式 ----------
        os.makedirs(self.template_dir, exist_ok=True)

        tpl_name = os.path.basename(os.path.normpath(self.template_dir))
        cache_path = os.path.join(self.template_dir, f"{tpl_name}.pkl")

        print(f"[INFO] 正在检查并加载缓存文件: {cache_path} ...")

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.template_feats = pickle.load(f)
            print(f"[OK] 模板缓存已加载: {cache_path}")
        else:
            print(f"[WARN] 缓存不存在，开始编码模板库: {self.template_dir}")
            self.template_feats = self.get_template_contours(self.template_dir)

            with open(cache_path, "wb") as f:
                pickle.dump(self.template_feats, f)
            print(f"[DONE] 模板库编码完成，已保存缓存文件: {cache_path}")

    def load_pose6s_with_depth(self):
        """
        支持：
        - 单模板库模式：template_dir/library_metadata.json
        - 双模板库模式：template_dir/front/library_metadata.json + template_dir/back/library_metadata.json
        """
        if getattr(self, "_load_front_back_libraries", False):
            print("[INIT] front/back 模板库加载 pose6 metadata")

            front_meta_path = os.path.join(self.front_template_dir, "library_metadata.json")
            if os.path.exists(front_meta_path):
                with open(front_meta_path, "r", encoding="utf-8") as f:
                    front_meta = json.load(f)
                self.front_pose6s = front_meta.get("pose6_list", [])
                print(f"[OK] Front pose6 loaded ({len(self.front_pose6s)})")
            else:
                print("[WARN] front/library_metadata.json 不存在")
                self.front_pose6s = []

            back_meta_path = os.path.join(self.back_template_dir, "library_metadata.json")
            if os.path.exists(back_meta_path):
                with open(back_meta_path, "r", encoding="utf-8") as f:
                    back_meta = json.load(f)
                self.back_pose6s = back_meta.get("pose6_list", [])
                print(f"[OK] Back pose6 loaded ({len(self.back_pose6s)})")
            else:
                print("[WARN] back/library_metadata.json 不存在")
                self.back_pose6s = []

            return

        json_path = os.path.join(self.template_dir, "library_metadata.json")
        if not os.path.exists(json_path):
            print("[WARN] library_metadata.json 不存在")
            self.pose6s_feats = []
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.pose6s_feats = data.get("pose6_list", [])

        cam_info = data.get("camera_intrinsics", {})
        self.fx = cam_info.get("fx", 0)
        self.fy = cam_info.get("fy", 0)
        self.cx = cam_info.get("cx", 0)
        self.cy = cam_info.get("cy", 0)
        self.H = cam_info.get("H", 0)
        self.W = cam_info.get("W", 0)
        self.cam_distance = data.get("camera_distance_mm", 0)

        rot_info = data.get("rotation_params", {})
        self.rot_ex = rot_info.get("ex", [])
        self.rot_ey = rot_info.get("ey", [])
        self.rot_ez = rot_info.get("ez", [])

        print(f"[INFO] 成功加载模板库，共 {len(self.pose6s_feats)} 个模板")
        print(f"      相机 fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, dist={self.cam_distance}")

    # # ======================= 第一帧：零件掩码（SAM3+模板相似度） =======================
    def find_part_mask(self, image_bgr, image_depth,
                       text_prompt="each individual metal part",
                       score_threshold=0.6,
                       mask_threshold=0.5,
                       min_area=300):
        """
        SAM3 版本的掩码查找逻辑：
        - 使用 SAM3 文本提示分割
        - 将候选 mask 按连通域拆分
        - 候选筛选逻辑：avg_depth 最小 -> 面积最大 -> score 最大
        - 再用 DinoV2 模板相似度选最优模板
        """
        # ================= 读取并预处理 RGB =================
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        enhanced_rgb = enhance_edges(image_rgb)
        img_pil = Image.fromarray(enhanced_rgb)

        # ================= SAM3 文本提示分割 =================
        state = self.sam3_processor.set_image(img_pil)
        output = self.sam3_processor.set_text_prompt(state=state, prompt=text_prompt)

        masks, boxes, scores = normalize_outputs(
            output["masks"], output["boxes"], output["scores"]
        )

        if len(scores) == 0:
            print("[WARN] SAM3 没有生成 mask")
            return None, None

        # 统一为 (N, H, W)
        masks = masks.squeeze(1)
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]

        candidate_masks = []
        candidate_infos = []

        # ================= SAM3 候选筛选 =================
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            score = float(score)

            if mask.ndim == 3:
                mask = mask[0]

            # 二值化
            bin_mask = (mask > mask_threshold).astype(np.uint8) * 255
            area = int((bin_mask > 0).sum())

            # 过滤面积过小或分数过低的候选
            if area < min_area or score < score_threshold:
                continue

            # 将候选 mask 按连通域拆分
            split_masks = split_connected_masks(bin_mask)

            for j, sm in enumerate(split_masks):
                # 去掉小碎片
                sm = remove_small_components(sm > 0, min_area=min_area)

                if np.sum(sm > 0) < min_area:
                    continue

                mask_bool = sm > 0

                # 计算平均深度，深度越小通常表示越靠近相机
                valid_depth = image_depth[mask_bool]
                valid_depth = valid_depth[valid_depth > 0]
                avg_depth = float(valid_depth.mean()) if len(valid_depth) > 5 else 9999.0

                candidate_masks.append(mask_bool)
                candidate_infos.append({
                    "candidate_id": len(candidate_masks) - 1,
                    "src_mask_id": i,
                    "split_id": j,
                    "score": score,
                    "avg_depth": avg_depth,
                    "area": int(np.sum(mask_bool > 0)),
                    "box": box
                })

        if len(candidate_masks) == 0:
            print("[WARN] SAM3 候选经过筛选后为空")
            return None, None

        # ================= 选择最佳候选 =================
        # 规则：
        # 1. 平均深度最小（更靠近相机）
        # 2. 面积更大
        # 3. 分数更高
        best_candidate_idx = min(
            range(len(candidate_masks)),
            key=lambda idx: (
                -candidate_infos[idx]["area"],
                candidate_infos[idx]["avg_depth"],
                -candidate_infos[idx]["score"]
            )
        )

        best_data = candidate_infos[best_candidate_idx]
        best_mask = candidate_masks[best_candidate_idx]

        print(f"[INFO] SAM3 best candidate = {best_candidate_idx}")
        print(f"[INFO] SAM3 best score = {best_data['score']:.3f}, "
              f"avg_depth = {best_data['avg_depth']:.3f}, "
              f"area = {best_data['area']}")

        # ================= 使用 DinoV2 进行模板匹配 =================
        best_score, best_tpl = -1.0, None

        # 只保留目标区域
        image_rgb_masked = image_rgb.copy()
        image_rgb_masked[~best_mask] = 0

        # 裁剪并规范化
        cropped_img = self.crop_resize_pad(image_rgb_masked, mask=best_mask, target_size=224)

        # 当前候选区域特征
        region_feat = self.encode_image_with_dinov2(Image.fromarray(cropped_img))

        scores = []
        for tpl_name, tpl_contour, tpl_mask, tpl_feat in self.template_feats:
            score = torch.cosine_similarity(region_feat, tpl_feat).item()
            scores.append(score)

            if score > best_score:
                best_score = score
                self.best_tpl = tpl_name
                best_tpl = tpl_name

                depth = image_depth.copy()
                depth[~best_mask] = 0.0
                valid_depths = depth[(depth > 0) & (~np.isnan(depth))]
                self.best_depth = np.mean(valid_depths) if len(valid_depths) > 0 else 0.0

        top_k_indices = np.argsort(np.array(scores))[-5:]
        print("[INFO] Top5 template idx =", top_k_indices)
        print("[INFO] Best template =", self.best_tpl)

        return (best_mask.astype(bool) if best_mask is not None else None), best_tpl



    def find_part_mask_dino(self, image_bgr, image_depth,
                            text_prompt="each individual metal part",
                            score_threshold=0.65,
                            mask_threshold=0.8,
                            min_area=8000,
                            max_area=17000):
        """
        当前版本：
        - 使用 SAM3 做候选分割
        - 第一轮：面积在 [min_area, max_area] 之间
        - 第二轮：在第一轮结果中取 score 排前2
        - 第三轮：在前2中取 avg_depth 最小
        - 保留后半段模板匹配逻辑
        - 额外返回所有候选 mask / 候选信息 / 候选框
        """

        # ================= 读取并预处理 RGB =================
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        enhanced_rgb = enhance_edges(image_rgb)
        img_pil = Image.fromarray(enhanced_rgb)

        # ================= SAM3 文本提示分割 =================
        state = self.sam3_processor.set_image(img_pil)
        output = self.sam3_processor.set_text_prompt(state=state, prompt=text_prompt)

        # 将输出统一转为 numpy
        masks, boxes, scores = normalize_outputs(
            output["masks"], output["boxes"], output["scores"]
        )

        if len(scores) == 0:
            print("[WARN] SAM3 没有生成 mask")
            return None

        # 去掉多余维度，统一为 (N,H,W)
        masks = masks.squeeze(1)
        if masks.ndim == 2:
            masks = masks[np.newaxis, :, :]

        # 用于保存所有候选结果
        candidate_masks = []
        candidate_infos = []
        candidate_boxes = []

        # ================= 遍历 SAM3 候选 =================
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            score = float(score)

            if mask.ndim == 3:
                mask = mask[0]

            # 根据 mask_threshold 二值化
            bin_mask = (mask > mask_threshold).astype(np.uint8) * 255
            area = int((bin_mask > 0).sum())

            # 第一轮：面积必须在 [min_area, max_area] 之间，且分数不低于阈值
            if area < min_area or score < score_threshold:
                continue
            if max_area is not None and area > max_area:
                continue

            # 将一个候选里可能包含的多个连通区域拆开
            split_masks = split_connected_masks(bin_mask)

            for j, sm in enumerate(split_masks):
                # 再去一次小碎片
                sm = remove_small_components(sm > 0, min_area=min_area)

                split_area = int(np.sum(sm > 0))
                if split_area < min_area:
                    continue
                if max_area is not None and split_area > max_area:
                    continue

                mask_bool = sm > 0

                # 计算候选区域平均深度，深度越小通常表示越靠近相机
                valid_depth = image_depth[mask_bool]
                valid_depth = valid_depth[valid_depth > 0]
                avg_depth = float(valid_depth.mean()) if len(valid_depth) > 5 else 9999.0

                # 记录当前候选信息
                candidate_masks.append(mask_bool)
                candidate_boxes.append(box)
                candidate_infos.append({
                    "candidate_id": len(candidate_masks) - 1,
                    "src_mask_id": i,
                    "split_id": j,
                    "score": score,
                    "avg_depth": avg_depth,
                    "area": split_area,
                    "box": box
                })

        if len(candidate_masks) == 0:
            print("[WARN] SAM3 候选经过筛选后为空")
            return None

        # ================= 三轮筛选最佳候选 =================
        # 第二轮：在第一轮结果中，按 score 从高到低取前2
        top2_candidate_idx = sorted(
            range(len(candidate_masks)),
            key=lambda idx: candidate_infos[idx]["score"],
            reverse=True
        )[:1]

        # 第三轮：在前2中选 avg_depth 最小的
        best_candidate_idx = min(
            top2_candidate_idx,
            key=lambda idx: candidate_infos[idx]["avg_depth"]
        )

        best_data = candidate_infos[best_candidate_idx]
        part_mask = candidate_masks[best_candidate_idx]
        best_box = candidate_boxes[best_candidate_idx]
        best_score = best_data["score"]

        print(f"[INFO] SAM3 top2 candidates = {top2_candidate_idx}")
        print(f"[INFO] SAM3 best candidate = {best_candidate_idx}")
        print(
            f"[INFO] SAM3 best score = {best_score:.3f}, avg_depth = {best_data['avg_depth']:.3f}, area = {best_data['area']}"
        )

        # 打印所有候选的排序信息，便于调试
        print("\n===== SAM3 Candidate Ranking =====")
        for info in candidate_infos:
            box = info["box"]
            x1, y1, x2, y2 = [int(v) for v in box]
            print(
                f"{info['candidate_id']:02d} | "
                f"src={info['src_mask_id']:02d} split={info['split_id']:02d} | "
                f"score={info['score']:.3f} | "
                f"avg_depth={info['avg_depth']:.3f} | "
                f"area={info['area']} | "
                f"box=({x1},{y1},{x2},{y2})"
            )

        # ================= 后面模板匹配逻辑保持原来不变 =================
        # 将目标区域外清零，只保留当前目标
        image_rgb_masked = image_rgb.copy()
        image_rgb_masked[~part_mask] = 0

        # 裁剪 + 缩放 + padding
        cropped_img = self.crop_resize_pad(image_rgb_masked, mask=part_mask, target_size=224)

        # 深度统一成单通道 float32
        if image_depth.ndim == 3:
            image_depth = cv2.cvtColor(image_depth, cv2.COLOR_BGR2GRAY)
        if image_depth.dtype != np.float32:
            image_depth = image_depth.astype(np.float32)

        # 将深度转伪彩色，再裁剪成与 RGB 相同格式
        depth_color_real = depth_to_colormap(image_depth, part_mask)
        cropped_depthcolor = self.crop_resize_pad(depth_color_real, mask=part_mask, target_size=224)

        # 提取场景轮廓与 DINO 特征
        contour_scene = get_normalized_contour(cropped_img)
        dino_feature_scene_rgb = self.encode_image_with_dinov2(Image.fromarray(cropped_img))
        dino_feature_scene_depth = self.encode_image_with_dinov2(Image.fromarray(cropped_depthcolor))

        # ================== 第一步：IoU 全量筛选 ==================
        scoresB = []
        for _, _, tpl_mask, _ in self.template_feats:
            scoresB.append(mask_iou(cropped_img, tpl_mask))
        scoresB = np.array(scoresB)

        # ================== 第二步：IoU Top50 ==================
        if len(scoresB) < 50:
            top_iou_idx = np.argsort(scoresB)[-len(scoresB):]
        else:
            top_iou_idx = np.argsort(scoresB)[-50:]

        print("\n===== IoU Top50 =====")
        for j in top_iou_idx:
            print(f"{j:4d} | {self.template_feats[j][0]} | IoU={scoresB[j]:.3f}")

        # ================== 第三步：Chamfer（在 IoU Top50 内筛选） ==================
        chamfer_scores = {}
        for j in top_iou_idx:
            _, tpl_contour, _, _ = self.template_feats[j]
            try:
                chamfer_scores[j] = robust_chamfer_distance(contour_scene, tpl_contour)
            except ValueError:
                chamfer_scores[j] = np.inf

        # 取 Chamfer 最小的前三个模板
        top_chamfer10_idx = sorted(chamfer_scores.keys(), key=lambda x: chamfer_scores[x])[:3]

        print("\n===== Chamfer Top10 (in IoU Top50) =====")
        for j in top_chamfer10_idx:
            print(f"{j:4d} | {self.template_feats[j][0]} | IoU={scoresB[j]:.3f} | Chamfer={chamfer_scores[j]:.6f}")

        # ================== 第四步：DINO 最终决策 ==================
        dino_scores = {}
        best_idx, best_score = None, -1
        for j in top_chamfer10_idx:
            _, _, _, tpl_dino_feature = self.template_feats[j]

            # 当前场景 RGB 与模板 RGB 的 DINO 相似度
            score_rgb = torch.cosine_similarity(dino_feature_scene_rgb, tpl_dino_feature).item()

            # 如果模板存在深度伪彩图，也计算深度相似度
            tpl_index = int(self.template_feats[j][0].split("_")[-1].split(".")[0])
            tpl_depthcolor_path = os.path.join(self.template_dir, f"depthcolor_{tpl_index:04d}.png")
            if os.path.exists(tpl_depthcolor_path):
                tpl_depthcolor_img = cv2.cvtColor(cv2.imread(tpl_depthcolor_path), cv2.COLOR_BGR2RGB)
                tpl_dino_feature_depth = self.encode_image_with_dinov2(Image.fromarray(tpl_depthcolor_img))
                score_depth = torch.cosine_similarity(dino_feature_scene_depth, tpl_dino_feature_depth).item()
            else:
                score_depth = 0.0

            # 当前总分只使用 RGB 分数，深度分数暂未融合
            score_total = 1.0 * score_rgb
            dino_scores[j] = score_total

            if score_total > best_score:
                best_score = score_total
                best_idx = j

        print("\n===== DINO Top10 (Final Decision from Chamfer Top10) =====")
        sorted_final = sorted(top_chamfer10_idx, key=lambda x: dino_scores[x], reverse=True)
        for j in sorted_final:
            print(f"{j:4d} | {self.template_feats[j][0]} | "
                  f"IoU={scoresB[j]:.3f} | Chamfer={chamfer_scores[j]:.6f} | DINO={dino_scores[j]:.3f}")

        print(f"\n✅ 最优模板: {self.template_feats[best_idx][0]} | "
              f"IoU={scoresB[best_idx]:.3f} | Chamfer={chamfer_scores[best_idx]:.6f} | DINO={best_score:.3f}")

        # 解析最佳模板信息
        tpl_name, tpl_feat, tpl_mask, tpl_dino_feature = self.template_feats[best_idx]
        self.best_tpl_name = tpl_name
        tpl_index = self._tpl_index_from_name(tpl_name)
        tpl_pose6 = self._get_pose6_from_tpl_name(tpl_name)

        # 返回：
        # 1. 最优目标 mask
        # 2. 最优模板 mask
        # 3. 对应 pose6
        # 4. 模板编号
        # 5. 全部候选 mask
        # 6. 全部候选信息
        # 7. 全部候选框
        return (
            part_mask > 0,
            tpl_mask > 0,
            tpl_pose6,
            tpl_index,
            candidate_masks,
            candidate_infos,
            candidate_boxes,
        )

    # ======================= 保存 mask 与可视化 =======================
    def save_mask_and_vis(self, frame, mask_bool, mask_dir, masked_dir, tag, frame_idx):
        """
        保存二值 mask 图和“仅保留目标区域”的可视化图
        """
        mask_path = os.path.join(mask_dir, f"mask_{tag}_{frame_idx:04d}.png")
        cv2.imwrite(mask_path, (mask_bool.astype(np.uint8) * 255))

        vis = frame.copy()
        vis[~mask_bool] = 0
        masked_path = os.path.join(masked_dir, f"masked_{tag}_{frame_idx:04d}.png")
        cv2.imwrite(masked_path, vis)

        return mask_path, masked_path

    # ======================================================================
    # =======================  新增功能块  ==========================
    # ======================================================================

    def _tpl_index_from_name(self, tpl_name):
        """
        从模板名中解析模板编号
        支持：
        - rgb_0001.png
        - 0001
        """
        if isinstance(tpl_name, str) and tpl_name.startswith("rgb_"):
            try:
                idx = int(tpl_name.split("_")[1].split(".")[0])
                return idx
            except Exception:
                pass
        try:
            return int(tpl_name)
        except Exception:
            raise ValueError(f"无法从模板名解析编号: {tpl_name}")

    def _depth_to_pointcloud(self, depth_mm, mask_bool):
        """
        将深度图（单位 mm）与场景 mask 转为点云 (N,3)
        """
        if depth_mm.dtype != np.float32 and depth_mm.dtype != np.float64:
            depth_mm = depth_mm.astype(np.float32)

        # 从内参矩阵中取 fx, fy, cx, cy
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]

        # 有效点：mask 内且深度 > 0
        m = (mask_bool.astype(bool)) & (depth_mm > 0)
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        Z = depth_mm[ys, xs]
        X = (xs.astype(np.float32) - cx) * Z / fx
        Y = (ys.astype(np.float32) - cy) * Z / fy

        # 再过滤一次 Z>0
        index_z = np.where(Z > 0)[0]
        X = X[index_z]
        Y = Y[index_z]
        Z = Z[index_z]

        return np.stack([X, Y, Z], axis=-1).astype(np.float32)

    def _save_ply_ascii(self, points_xyz, save_path):
        """
        不依赖 open3d 的简单 PLY 保存（ASCII）
        """
        pts = points_xyz.reshape(-1, 3).astype(np.float32)
        with open(save_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for x, y, z in pts:
                f.write(f"{x} {y} {z}\n")

    def apply_mask_to_depth_and_pointcloud(self, mask, depth_path, output_dir, intrinsic):
        """
        应用 mask 到深度图并生成点云，同时返回保存路径。
        """
        # === 读取深度图 ===
        # 这里假设 depth_raw 存储方式可通过 view(np.float32) 还原为 float32 深度
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError("深度图加载失败: " + depth_path)
        depth = depth_raw.view(np.float32).reshape(depth_raw.shape[0], depth_raw.shape[1])

        os.makedirs(output_dir, exist_ok=True)

        # === 保存裁剪后的深度图 ===
        masked_depth = depth.copy()
        masked_depth[~mask] = 0
        depth_save_path = os.path.join(output_dir, "masked_depth.png")
        cv2.imwrite(depth_save_path, (masked_depth * 1000).astype(np.uint16))  # 转成 mm 保存
        print(f"[INFO] 深度图保存完成: {depth_save_path}")

        # === 用 mask + 内参生成点云 ===
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        ys, xs = np.where(mask)
        points = []
        for y, x in zip(ys, xs):
            Z = depth[y, x]
            if Z <= 0:
                continue
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            points.append([X, Y, Z])

        if len(points) == 0:
            print("[WARN] 未生成有效点云。")
            return None, depth_save_path, None

        points = np.array(points)

        # 使用 open3d 保存点云
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pointcloud_path = os.path.join(output_dir, "masked_pointcloud.ply")
        o3d.io.write_point_cloud(pointcloud_path, pcd)
        print(f"[INFO] 点云保存完成: {pointcloud_path}")

        # 返回路径与数据，便于后续处理
        return {
            "masked_depth_path": depth_save_path,
            "pointcloud_path": pointcloud_path,
            "points": points,
            "masked_depth": masked_depth
        }


if __name__ == "__main__":
    # ========== 1. 参数设置 ==========
    # 模板库目录
    template_dir = "/home/sunddy/Programming/FoundationPose/templates1280*720/back"

    # 输出目录
    output_dir = "/home/sunddy/Programming/FoundationPose/output_segmented_camera"
    os.makedirs(output_dir, exist_ok=True)

    # 相机内参矩阵
    intrinsic = np.array([
        [609.99963379, 0, 641.85406494],
        [0, 610.17034912, 360.86437988],
        [0, 0, 1]
    ], dtype=np.float32)

    # 分别建立原图、候选框、候选 mask、最优 mask、最佳模板的保存目录
    raw_dir = os.path.join(output_dir, "raw")
    sam3_boxes_dir = os.path.join(output_dir, "sam3_boxes")
    candidates_dir = os.path.join(output_dir, "candidate_masks")
    best_dir = os.path.join(output_dir, "best_mask")
    matched_template_dir = os.path.join(output_dir, "matched_template")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(sam3_boxes_dir, exist_ok=True)
    os.makedirs(candidates_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(matched_template_dir, exist_ok=True)

    # ========== 2. 初始化分割器 ==========
    print("\n[STEP] 初始化分割器...")
    seg = ImageCoarseSegmentor(template_dir=template_dir)

    # ========== 3. 初始化相机 ==========
    print("\n[STEP] 正在初始化相机...")
    cam = OrbbecCamera()

    try:
        # 预热相机，避免刚启动时帧不稳定
        print("[STEP] 预热相机...")
        for _ in range(10):
            if hasattr(cam, "read"):
                cam.read()
            time.sleep(0.05)

        # ========== 4. 获取当前一帧 ==========
        print("[STEP] 采集当前帧...")

        image_color = None
        depth_image = None

        # 兼容不同版本的 OrbbecCamera 接口写法
        if hasattr(cam, "current_color"):
            cam.read()
            image_color = cam.current_color.copy() if cam.current_color is not None else None
            if hasattr(cam, "current_depth_map"):
                depth_image = cam.current_depth_map.copy()
        else:
            ret, frame = cam.read()
            if ret:
                image_color = frame.copy()
            if hasattr(cam, "current_depth_map"):
                depth_image = cam.current_depth_map.copy()

        if image_color is None or depth_image is None:
            raise RuntimeError("相机未返回有效彩色图或深度图")

        # 用时间戳作为文件名，避免覆盖
        ts = time.strftime("%Y%m%d_%H%M%S")

        color_path = os.path.join(raw_dir, f"color_{ts}.png")
        depth_path = os.path.join(raw_dir, f"depth_{ts}.png")

        # 保存原始彩色图与深度图
        cv2.imwrite(color_path, image_color)
        cv2.imwrite(depth_path, depth_image.astype(np.uint16))

        print(f"[INFO] 彩色图已保存: {color_path}")
        print(f"[INFO] 深度图已保存: {depth_path}")

        # ========== 5. 执行分割 ==========
        print("\n[STEP] 正在执行 SAM3 分割...")
        result = seg.find_part_mask_dino(
            image_bgr=image_color,
            image_depth=depth_image
        )

        if result is None:
            print("[ERROR] 未检测到有效目标区域")
            raise SystemExit(0)

        # 当前 find_part_mask_dino 返回 7 个值
        best_mask, best_tpl_mask, tpl_pose6, tpl_index, candidate_masks, candidate_infos, candidate_boxes = result

        if best_mask is None:
            print("[ERROR] 未检测到有效最优 mask")
            raise SystemExit(0)

        best_tpl_name = getattr(seg, "best_tpl_name", f"rgb_{tpl_index:04d}.png")

        print(f"[INFO] 最佳模板编号: {tpl_index}")
        print(f"[INFO] 最佳模板名称: {best_tpl_name}")
        print(f"[INFO] SAM3 候选框数量: {len(candidate_boxes)}")
        print(f"[INFO] 候选 mask 数量: {len(candidate_masks)}")

        # ========== 6. 保存所有 SAM3 候选框 ==========
        print("\n[STEP] 保存所有 SAM3 候选框 ...")

        sam3_boxes_overlay = image_color.copy()

        for i, box in enumerate(candidate_boxes):
            x1, y1, x2, y2 = box.astype(int)

            box_txt_path = os.path.join(sam3_boxes_dir, f"sam3_box_{i:03d}.txt")
            with open(box_txt_path, "w", encoding="utf-8") as f:
                f.write(f"x1={x1}\n")
                f.write(f"y1={y1}\n")
                f.write(f"x2={x2}\n")
                f.write(f"y2={y2}\n")

            cv2.rectangle(sam3_boxes_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                sam3_boxes_overlay,
                f"{i}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            print(f"[SAM3 BOX {i:03d}] x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        sam3_boxes_overlay_path = os.path.join(
            sam3_boxes_dir, f"all_sam3_boxes_overlay_{ts}.png"
        )
        cv2.imwrite(sam3_boxes_overlay_path, sam3_boxes_overlay)
        print(f"[INFO] SAM3 候选框叠加图已保存: {sam3_boxes_overlay_path}")

        # ========== 7. 保存所有候选 mask ==========
        print("\n[STEP] 保存所有候选 mask ...")

        overlay = image_color.copy()

        for i, mask in enumerate(candidate_masks):
            mask_u8 = (mask.astype(np.uint8) * 255)

            mask_path = os.path.join(candidates_dir, f"candidate_{i:03d}.png")
            cv2.imwrite(mask_path, mask_u8)

            vis = image_color.copy()
            vis[~mask] = 0
            vis_path = os.path.join(candidates_dir, f"candidate_{i:03d}_vis.png")
            cv2.imwrite(vis_path, vis)

            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                cv2.putText(
                    overlay,
                    f"{i}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            if i < len(candidate_infos):
                info = candidate_infos[i]
                print(
                    f"[CANDIDATE {i:03d}] "
                    f"score={info['score']:.3f}, "
                    f"avg_depth={info['avg_depth']:.3f}, "
                    f"area={info['area']}, "
                    f"src_mask_id={info['src_mask_id']}, "
                    f"split_id={info['split_id']}"
                )

        overlay_path = os.path.join(candidates_dir, f"all_candidates_overlay_{ts}.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"[INFO] 所有候选叠加图已保存: {overlay_path}")

        # ========== 8. 保存最优 mask ==========
        print("\n[STEP] 保存最优 mask ...")

        best_mask_path = os.path.join(best_dir, f"best_mask_{ts}.png")
        cv2.imwrite(best_mask_path, (best_mask.astype(np.uint8) * 255))

        best_vis = image_color.copy()
        best_vis[~best_mask] = 0
        best_vis_path = os.path.join(best_dir, f"best_mask_vis_{ts}.png")
        cv2.imwrite(best_vis_path, best_vis)

        best_overlay = image_color.copy()
        contours, _ = cv2.findContours(
            (best_mask.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(best_overlay, contours, -1, (0, 0, 255), 2)

        if best_mask is not None:
            ys, xs = np.where(best_mask)
            if len(xs) > 0 and len(ys) > 0:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()

                cv2.rectangle(best_overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if len(candidate_infos) > 0:
                    # 第二轮：score 排前2
                    top2_infos = sorted(
                        candidate_infos,
                        key=lambda x: x["score"],
                        reverse=True
                    )[:2]

                    # 第三轮：在前2里选 avg_depth 最小
                    best_info = min(top2_infos, key=lambda x: x["avg_depth"])

                    score_txt = (
                        f"best score={best_info['score']:.3f}, "
                        f"depth={best_info['avg_depth']:.3f}, "
                        f"area={best_info['area']}"
                    )
                else:
                    score_txt = "best"

                cv2.putText(
                    best_overlay,
                    score_txt,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

        best_overlay_path = os.path.join(best_dir, f"best_mask_overlay_{ts}.png")
        cv2.imwrite(best_overlay_path, best_overlay)

        print(f"[INFO] 最优 mask 已保存: {best_mask_path}")
        print(f"[INFO] 最优 mask 可视化已保存: {best_vis_path}")
        print(f"[INFO] 最优 mask 轮廓图已保存: {best_overlay_path}")

        # ========== 9. 显示并保存“最佳模板” ==========
        print("\n[STEP] 显示最佳模板 ...")

        if isinstance(best_tpl_name, str) and "/" in best_tpl_name:
            side, rgb_fname = best_tpl_name.split("/", 1)
            mask_fname = rgb_fname.replace("rgb_", "mask_")
            template_rgb_path = seg._get_template_real_path(best_tpl_name)
            template_mask_path = seg._get_template_real_path(f"{side}/{mask_fname}")
        else:
            template_rgb_path = seg._get_template_real_path(f"rgb_{tpl_index:04d}.png")
            template_mask_path = seg._get_template_real_path(f"mask_{tpl_index:04d}.png")

        if os.path.exists(template_rgb_path):
            template_rgb = cv2.imread(template_rgb_path)
            matched_template_rgb_save_path = os.path.join(
                matched_template_dir, f"matched_template_rgb_{tpl_index:04d}_{ts}.png"
            )
            cv2.imwrite(matched_template_rgb_save_path, template_rgb)
            print(f"[INFO] 最佳模板 RGB 已保存: {matched_template_rgb_save_path}")
        else:
            template_rgb = None
            print(f"[WARN] 未找到最佳模板 RGB: {template_rgb_path}")

        if os.path.exists(template_mask_path):
            template_mask = cv2.imread(template_mask_path, cv2.IMREAD_UNCHANGED)
            matched_template_mask_save_path = os.path.join(
                matched_template_dir, f"matched_template_mask_{tpl_index:04d}_{ts}.png"
            )
            cv2.imwrite(matched_template_mask_save_path, template_mask)
            print(f"[INFO] 最佳模板 Mask 已保存: {matched_template_mask_save_path}")
        else:
            template_mask = None
            print(f"[WARN] 未找到最佳模板 Mask: {template_mask_path}")

        # ========== 10. 显示当前最优 mask 与最佳模板对比 ==========
        if template_rgb is not None:
            # 将当前最优目标区域图和模板图统一到相同高度，便于拼接显示
            show_left = best_vis.copy()
            show_right = template_rgb.copy()

            h1, w1 = show_left.shape[:2]
            h2, w2 = show_right.shape[:2]

            target_h = max(h1, h2)
            show_left = cv2.resize(show_left, (int(w1 * target_h / h1), target_h))
            show_right = cv2.resize(show_right, (int(w2 * target_h / h2), target_h))

            compare_vis = np.hstack([show_left, show_right])

            cv2.putText(
                compare_vis,
                "Best Mask",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            cv2.putText(
                compare_vis,
                f"Matched Template: {best_tpl_name}",
                (show_left.shape[1] + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

            compare_vis_path = os.path.join(
                matched_template_dir, f"best_mask_vs_template_{tpl_index:04d}_{ts}.png"
            )
            cv2.imwrite(compare_vis_path, compare_vis)
            print(f"[INFO] 最优 mask 与模板对比图已保存: {compare_vis_path}")

            cv2.imshow("Best Mask vs Matched Template", compare_vis)
            print("[INFO] 按任意键关闭模板对比窗口")
            cv2.waitKey(0)
            cv2.destroyWindow("Best Mask vs Matched Template")

        print("\n[DONE] 当前帧处理完成。")

    finally:
        # 无论程序是否异常，都尝试关闭相机
        try:
            if cam is not None:
                if hasattr(cam, "stop"):
                    cam.stop()
                elif hasattr(cam, "release"):
                    cam.release()
        except Exception as e:
            print(f"[WARN] 相机关闭异常: {e}")