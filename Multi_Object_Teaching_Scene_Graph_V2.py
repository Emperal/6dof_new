import os

# 必须在 torch / FoundationPose / vision_pose_estimator 导入之前设置
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import csv
import cv2
import sys
import glob
import json
import math
import time
import argparse
import base64
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import requests

try:
    import torch
except Exception:
    torch = None

try:
    import warp as wp
except Exception:
    wp = None

from vision_pose_estimator import (
    VisionPoseEstimator,
    refine_single_pose_with_refiner,
    rotate_pose_y_180,
)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Global runtime patch state
# ============================================================
_CURRENT_SAM3_CFG: Optional[Dict[str, Any]] = None
_SAM3_PROCESSOR_PATCHED = False
_FIND_PART_PATCHED_CLASS_IDS = set()


# ============================================================
# GPU utilities
# ============================================================
def is_cuda_oom_error(err: Any) -> bool:
    s = str(err).lower()
    return (
        "out of memory" in s
        or "failed to allocate" in s
        or ("cuda" in s and "memory" in s)
        or "warp cuda error 2" in s
    )


def cleanup_gpu_memory(reason: str = "", verbose: bool = True) -> None:
    if verbose:
        print("=" * 80)
        print(f"[GPU CLEANUP] {reason}")
        print("=" * 80)

    try:
        gc.collect()
    except Exception:
        pass

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if wp is not None:
        try:
            wp.synchronize()
        except Exception:
            pass


def safe_set_none(obj: Any, attr: str) -> None:
    try:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                print(f"[INFO] release {type(obj).__name__}.{attr}")
            setattr(obj, attr, None)
    except Exception as e:
        print(f"[WARN] failed to release {type(obj).__name__}.{attr}: {e}")


def release_init_only_modules_from_estimator(estimator: Any, object_name: str = "") -> None:
    """
    安全版释放函数：
    只释放初始化阶段可能用到的 SAM3 / DINO / 模板粗匹配相关模块。
    不释放 estimator.est / refiner / model，避免破坏后续 FoundationPose tracking。
    """
    if estimator is None:
        return

    print("=" * 80)
    print(f"[SAFE RELEASE INIT MODULES] object={object_name}")
    print("=" * 80)

    safe_attrs = [
        "segmentor",
        "coarse_segmentor",
        "image_segmentor",
        "template_matcher",
        "matcher",
        "coarse_matcher",
        "dino_model",
        "dinov2_model",
        "dino",
        "dinov2",
        "sam3_model",
        "sam_model",
        "sam",
        "sam3",
        "processor",
        "sam3_processor",
        "image_processor",
        "text_encoder",
        "tokenizer",
    ]

    for attr in safe_attrs:
        safe_set_none(estimator, attr)

    # 只释放 FoundationPose 内部 scorer，不动 refiner / model
    try:
        fp = getattr(estimator, "est", None)
        if fp is not None:
            scorer_attrs = [
                "scorer",
                "score_predictor",
                "score_model",
                "scorer_model",
                "score_net",
                "pose_scorer",
            ]
            for attr in scorer_attrs:
                safe_set_none(fp, attr)
    except Exception as e:
        print(f"[WARN] failed to release scorer: {e}")

    cleanup_gpu_memory(f"after safe releasing init modules for {object_name}")


# ============================================================
# SAM3 runtime prompt patch
# ============================================================
def _get_cfg_value(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg else default


def _current_sam3_params() -> Dict[str, Any]:
    cfg = _CURRENT_SAM3_CFG or {}

    return {
        "text_prompt": _get_cfg_value(cfg, "sam3_text_prompt", "long strip metal part"),
        "score_threshold": float(_get_cfg_value(cfg, "sam3_score_threshold", 0.05)),
        "mask_threshold": float(_get_cfg_value(cfg, "sam3_mask_threshold", 0.25)),
        "min_area": int(_get_cfg_value(cfg, "sam3_min_area", 50)),
        "max_area": int(_get_cfg_value(cfg, "sam3_max_area", 150000)),
    }


def patch_sam3_processor_text_prompt() -> bool:
    global _SAM3_PROCESSOR_PATCHED

    if _SAM3_PROCESSOR_PATCHED:
        return True

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:
        print(f"[WARN] Cannot import Sam3Processor for runtime patch: {e}")
        return False

    if not hasattr(Sam3Processor, "set_text_prompt"):
        print("[WARN] Sam3Processor has no set_text_prompt")
        return False

    original_fn = Sam3Processor.set_text_prompt

    if getattr(original_fn, "_is_runtime_sam3_prompt_patch", False):
        _SAM3_PROCESSOR_PATCHED = True
        return True

    def wrapped_set_text_prompt(self, *args, **kwargs):
        params = _current_sam3_params()
        prompt = params["text_prompt"]

        if "prompt" in kwargs:
            kwargs["prompt"] = prompt
        elif len(args) >= 2:
            args = list(args)
            args[1] = prompt
            args = tuple(args)

        print("=" * 80)
        print("[SAM3 PROCESSOR PATCH]")
        print(f"prompt = {prompt}")
        print("=" * 80)

        return original_fn(self, *args, **kwargs)

    wrapped_set_text_prompt._is_runtime_sam3_prompt_patch = True
    Sam3Processor.set_text_prompt = wrapped_set_text_prompt

    _SAM3_PROCESSOR_PATCHED = True
    print("[OK] Patched Sam3Processor.set_text_prompt")
    return True


def patch_project_find_part_mask_dino(
    project_root: str = "/home/sunddy/Programming/FoundationPose",
) -> int:
    """
    尝试给工程中 find_part_mask_dino 加运行时参数。
    如果原函数不支持这些参数，会自动回退原始调用，不中断程序。
    """
    patched_count = 0
    project_root = os.path.abspath(project_root)

    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue

        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue

        try:
            module_file_abs = os.path.abspath(module_file)
        except Exception:
            continue

        if not module_file_abs.startswith(project_root):
            continue

        try:
            module_dict = vars(module)
        except Exception:
            continue

        for _, obj in list(module_dict.items()):
            if not isinstance(obj, type):
                continue

            cls = obj
            cls_id = id(cls)

            if cls_id in _FIND_PART_PATCHED_CLASS_IDS:
                continue

            if not hasattr(cls, "find_part_mask_dino"):
                continue

            original_fn = getattr(cls, "find_part_mask_dino")

            if getattr(original_fn, "_is_runtime_sam3_text_patch", False):
                _FIND_PART_PATCHED_CLASS_IDS.add(cls_id)
                continue

            def make_wrapper(original_method):
                def wrapped_find_part_mask_dino(self, image_bgr, image_depth, *args, **kwargs):
                    params = _current_sam3_params()
                    cfg = _CURRENT_SAM3_CFG or {}
                    object_name = cfg.get("name", "unknown_object")

                    call_kwargs = dict(kwargs)
                    call_kwargs["text_prompt"] = params["text_prompt"]
                    call_kwargs["score_threshold"] = params["score_threshold"]
                    call_kwargs["mask_threshold"] = params["mask_threshold"]
                    call_kwargs["min_area"] = params["min_area"]
                    call_kwargs["max_area"] = params["max_area"]
                    call_kwargs["image_bgr"] = image_bgr
                    call_kwargs["image_depth"] = image_depth

                    print("=" * 80)
                    print("[SAM3 TEXT PATCH]")
                    print(f"object = {object_name}")
                    print(f"prompt = {params['text_prompt']}")
                    print(
                        f"score_threshold = {params['score_threshold']}, "
                        f"mask_threshold = {params['mask_threshold']}, "
                        f"min_area = {params['min_area']}, "
                        f"max_area = {params['max_area']}"
                    )
                    print("=" * 80)

                    try:
                        return original_method(self, *args, **call_kwargs)
                    except TypeError as e:
                        print(f"[WARN] find_part_mask_dino 不接受文本参数，回退原始调用: {e}")
                        return original_method(self, image_bgr, image_depth, *args)

                wrapped_find_part_mask_dino._is_runtime_sam3_text_patch = True
                return wrapped_find_part_mask_dino

            setattr(cls, "find_part_mask_dino", make_wrapper(original_fn))
            _FIND_PART_PATCHED_CLASS_IDS.add(cls_id)
            patched_count += 1
            print(f"[OK] Patched project class: {module_name}.{cls.__name__}.find_part_mask_dino")

    if patched_count == 0:
        print("[WARN] No project find_part_mask_dino class patched.")

    return patched_count


def patch_estimator_instance_find_part_mask_dino(estimator: Any) -> int:
    patched = 0
    candidates = [estimator]

    try:
        candidates.extend(list(vars(estimator).values()))
    except Exception:
        pass

    for obj in candidates:
        if obj is None:
            continue

        if not hasattr(obj, "find_part_mask_dino"):
            continue

        method = getattr(obj, "find_part_mask_dino")

        if getattr(method, "_is_runtime_sam3_text_patch", False):
            continue

        original_bound_method = method

        def wrapped_bound_find_part_mask_dino(image_bgr, image_depth, *args, **kwargs):
            params = _current_sam3_params()
            cfg = _CURRENT_SAM3_CFG or {}
            object_name = cfg.get("name", "unknown_object")

            call_kwargs = dict(kwargs)
            call_kwargs["text_prompt"] = params["text_prompt"]
            call_kwargs["score_threshold"] = params["score_threshold"]
            call_kwargs["mask_threshold"] = params["mask_threshold"]
            call_kwargs["min_area"] = params["min_area"]
            call_kwargs["max_area"] = params["max_area"]
            call_kwargs["image_bgr"] = image_bgr
            call_kwargs["image_depth"] = image_depth

            print("=" * 80)
            print("[SAM3 TEXT INSTANCE PATCH]")
            print(f"object = {object_name}")
            print(f"prompt = {params['text_prompt']}")
            print("=" * 80)

            try:
                return original_bound_method(*args, **call_kwargs)
            except TypeError as e:
                print(f"[WARN] instance find_part_mask_dino 不接受文本参数，回退原始调用: {e}")
                return original_bound_method(image_bgr, image_depth, *args)

        wrapped_bound_find_part_mask_dino._is_runtime_sam3_text_patch = True
        setattr(obj, "find_part_mask_dino", wrapped_bound_find_part_mask_dino)
        patched += 1

    if patched > 0:
        print(f"[OK] Patched {patched} estimator instance find_part_mask_dino method(s)")

    return patched


# ============================================================
# Geometry utilities
# ============================================================
def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64)


def ndarray_to_list(x: Optional[np.ndarray]) -> Optional[List[Any]]:
    if x is None:
        return None
    return np.asarray(x).tolist()


def mask_to_bbox_xyxy(mask: Optional[np.ndarray]) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_center(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[float, float]]:
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_to_mask(
    frame_shape: Tuple[int, int, int],
    bbox_xyxy: List[int],
    shrink_px: int = 0,
) -> np.ndarray:
    H, W = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

    x1 = max(0, min(W - 1, x1 + shrink_px))
    y1 = max(0, min(H - 1, y1 + shrink_px))
    x2 = max(0, min(W - 1, x2 - shrink_px))
    y2 = max(0, min(H - 1, y2 - shrink_px))

    mask = np.zeros((H, W), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255

    return mask


def clip_bbox_xyxy(
    bbox_xyxy: List[int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def expand_bbox_xyxy(
    bbox_xyxy: List[int],
    width: int,
    height: int,
    expand_px: int = 0,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    return clip_bbox_xyxy(
        [x1 - int(expand_px), y1 - int(expand_px), x2 + int(expand_px), y2 + int(expand_px)],
        width=width,
        height=height,
    )


def crop_image_by_bbox(image: Optional[np.ndarray], bbox_xyxy: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    if image is None:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    return image[y1:y2, x1:x2].copy()


def shift_camera_matrix_for_crop(K: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, _, _ = bbox_xyxy
    K_crop = np.asarray(K, dtype=np.float64).copy()
    K_crop[0, 2] -= float(x1)
    K_crop[1, 2] -= float(y1)
    return K_crop


def paste_mask_into_full_frame(
    mask_crop: np.ndarray,
    full_shape: Tuple[int, int],
    bbox_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    full_h, full_w = full_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    mask_full = np.zeros((full_h, full_w), dtype=np.uint8)

    mask_u8 = np.asarray(mask_crop, dtype=np.uint8)
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    if mask_u8.size == 0:
        return mask_full
    if mask_u8.max() <= 1:
        mask_u8 = mask_u8 * 255

    roi_h = max(0, y2 - y1)
    roi_w = max(0, x2 - x1)
    if roi_h <= 0 or roi_w <= 0:
        return mask_full

    if mask_u8.shape[0] != roi_h or mask_u8.shape[1] != roi_w:
        mask_u8 = cv2.resize(mask_u8, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    mask_full[y1:y2, x1:x2] = mask_u8
    return mask_full


def bbox_iou_xyxy(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = [int(v) for v in a]
    bx1, by1, bx2, by2 = [int(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def bbox_center_distance_px(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
) -> float:
    ca = bbox_center(a)
    cb = bbox_center(b)
    if ca is None or cb is None:
        return float('inf')
    return float(math.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def fallback_pose_from_bbox_depth_m(
    bbox_xyxy: List[int],
    depth_uint16: Optional[np.ndarray],
    K: np.ndarray,
    fallback_z_mm: float = 500.0,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)

    z_mm = float(fallback_z_mm)

    if depth_uint16 is not None:
        H, W = depth_uint16.shape[:2]
        x1c = max(0, min(W - 1, x1))
        x2c = max(0, min(W, x2))
        y1c = max(0, min(H - 1, y1))
        y2c = max(0, min(H, y2))

        crop = depth_uint16[y1c:y2c, x1c:x2c]
        valid = crop[(crop > 50) & (crop < 5000)]
        if valid.size > 20:
            z_mm = float(np.median(valid))

    z_m = z_mm / 1000.0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_m = (u - cx) * z_m / fx
    y_m = (v - cy) * z_m / fy

    return make_transform(
        np.eye(3, dtype=np.float64),
        np.array([x_m, y_m, z_m], dtype=np.float64),
    )


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    q = np.empty(4, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))

        if i == 0:
            s = math.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s

        elif i == 1:
            s = math.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s

        else:
            s = math.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    q /= np.linalg.norm(q) + 1e-12
    return q


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)

    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为 XYZ 欧拉角（单位：度）。
    返回 [roll_x_deg, pitch_y_deg, yaw_z_deg]。
    """
    R = np.asarray(R, dtype=np.float64)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    return np.degrees(np.array([x, y, z], dtype=np.float64))


def pose_mm_to_serializable(cTo_mm: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    将 4x4 位姿矩阵拆成便于 JSON / CSV 保存的字段。
    包含：平移(mm)、四元数(xyzw)、欧拉角(deg)、旋转矩阵。
    """
    result: Dict[str, Any] = {
        "cTo_mm": None,
        "x_mm": None,
        "y_mm": None,
        "z_mm": None,
        "quat_xyzw": None,
        "qx": None,
        "qy": None,
        "qz": None,
        "qw": None,
        "euler_xyz_deg": None,
        "roll_deg": None,
        "pitch_deg": None,
        "yaw_deg": None,
        "R_3x3": None,
    }

    if cTo_mm is None:
        return result

    T = np.asarray(cTo_mm, dtype=np.float64).copy()
    R = T[:3, :3]
    t = T[:3, 3]

    q = matrix_to_quat_xyzw(R)
    euler_deg = matrix_to_euler_xyz_deg(R)

    result.update({
        "cTo_mm": ndarray_to_list(T),
        "x_mm": float(t[0]),
        "y_mm": float(t[1]),
        "z_mm": float(t[2]),
        "quat_xyzw": ndarray_to_list(q),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "qz": float(q[2]),
        "qw": float(q[3]),
        "euler_xyz_deg": ndarray_to_list(euler_deg),
        "roll_deg": float(euler_deg[0]),
        "pitch_deg": float(euler_deg[1]),
        "yaw_deg": float(euler_deg[2]),
        "R_3x3": ndarray_to_list(R),
    })

    return result


def quat_slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))

    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)

    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
        q /= np.linalg.norm(q) + 1e-12
        return q

    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha

    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / (sin_theta_0 + 1e-12)
    s1 = sin_theta / (sin_theta_0 + 1e-12)

    q = s0 * q0 + s1 * q1
    q /= np.linalg.norm(q) + 1e-12
    return q


class TranslationKalmanFilter3D:
    """
    3D 平移卡尔曼滤波器。
    状态: [x, y, z, vx, vy, vz]
    单位: mm / frame
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_var: float = 80.0,
        measure_var: float = 80.0,
    ):
        self.dt = float(dt)
        self.process_var = float(process_var)
        self.measure_var = float(measure_var)

        self.initialized = False

        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1000.0

        self.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        self.Q = np.eye(6, dtype=np.float64) * self.process_var
        self.R = np.eye(3, dtype=np.float64) * self.measure_var
        self.I = np.eye(6, dtype=np.float64)

    def initialize(self, t_mm: np.ndarray):
        t_mm = np.asarray(t_mm, dtype=np.float64).reshape(3)
        self.x[:3, 0] = t_mm
        self.x[3:, 0] = 0.0
        self.P = np.eye(6, dtype=np.float64) * 1000.0
        self.initialized = True

    def predict(self) -> np.ndarray:
        if not self.initialized:
            return np.zeros(3, dtype=np.float64)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3, 0].copy()

    def update(self, t_meas_mm: np.ndarray) -> np.ndarray:
        z = np.asarray(t_meas_mm, dtype=np.float64).reshape(3, 1)

        if not self.initialized:
            self.initialize(z[:, 0])
            return self.x[:3, 0].copy()

        self.predict()

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

        return self.x[:3, 0].copy()


def project_bbox_from_estimator_pose(
    estimator: VisionPoseEstimator,
    pose_m: np.ndarray,
    image_shape: Tuple[int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """
    根据 estimator.to_origin 和 estimator.bbox 投影 3D bbox 到 2D bbox。
    pose_m 单位：米。
    """
    try:
        if pose_m is None:
            return None

        H, W = image_shape[:2]

        if not hasattr(estimator, "to_origin") or not hasattr(estimator, "bbox"):
            return None

        center_pose = pose_m @ np.linalg.inv(estimator.to_origin)

        xyz_min = estimator.bbox[0]
        xyz_max = estimator.bbox[1]

        x0, y0, z0 = xyz_min
        x1, y1, z1 = xyz_max

        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float64)

        corners_h = np.concatenate(
            [corners, np.ones((corners.shape[0], 1), dtype=np.float64)],
            axis=1,
        )

        pts_cam = (center_pose @ corners_h.T).T[:, :3]
        z = pts_cam[:, 2]

        if np.any(z <= 1e-6):
            return None

        K = estimator.MANUAL_K.astype(np.float64)

        u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]

        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)):
            return None

        x_min = int(np.floor(np.min(u)))
        y_min = int(np.floor(np.min(v)))
        x_max = int(np.ceil(np.max(u)))
        y_max = int(np.ceil(np.max(v)))

        if x_max < -W or x_min > 2 * W or y_max < -H or y_min > 2 * H:
            return None

        x_min = max(0, min(W - 1, x_min))
        y_min = max(0, min(H - 1, y_min))
        x_max = max(0, min(W - 1, x_max))
        y_max = max(0, min(H - 1, y_max))

        if x_max <= x_min or y_max <= y_min:
            return None

        return x_min, y_min, x_max, y_max

    except Exception as e:
        print(f"[WARN] project bbox failed: {e}")
        return None


def foundationpose_refiner_track_once_no_modify(
    estimator: VisionPoseEstimator,
    frame_bgr: np.ndarray,
    depth_uint16: np.ndarray,
    last_track_pose_m: np.ndarray,
    refine_iter: int = 2,
    use_y180: bool = True,
    track_roi_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    不修改 vision_pose_estimator.py，直接复用 refine_single_pose_with_refiner。
    如果传入 track_roi_bbox_xyxy，则只在上一帧附近的 ROI 内做 refiner，
    但位姿仍保持在完整相机坐标系下；投影仍使用完整图像内参。
    """
    try:
        if frame_bgr is None or depth_uint16 is None or last_track_pose_m is None:
            return None

        if not hasattr(estimator, "est") or estimator.est is None:
            raise RuntimeError("estimator.est is None, FoundationPose refiner not available")

        full_shape = frame_bgr.shape
        K_refine = estimator.MANUAL_K.astype(np.float64)
        rgb_bgr_for_refine = frame_bgr
        depth_for_refine = depth_uint16
        used_roi = None

        if track_roi_bbox_xyxy is not None:
            H, W = frame_bgr.shape[:2]
            used_roi = clip_bbox_xyxy(list(track_roi_bbox_xyxy), width=W, height=H)
            x1, y1, x2, y2 = used_roi
            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                rgb_bgr_for_refine = crop_image_by_bbox(frame_bgr, used_roi)
                depth_for_refine = crop_image_by_bbox(depth_uint16, used_roi)
                K_refine = shift_camera_matrix_for_crop(estimator.MANUAL_K, used_roi)
            else:
                used_roi = None

        rgb_input = rgb_bgr_for_refine[..., ::-1].copy()

        depth_m = depth_for_refine.astype(np.float32) / 1000.0
        depth_m[depth_m < 0.001] = 0.0

        t0 = time.time()

        refined_pose = refine_single_pose_with_refiner(
            estimator=estimator.est,
            rgb=rgb_input,
            depth_m=depth_m,
            K=K_refine,
            init_pose_orig=last_track_pose_m.astype(np.float32),
            iteration=int(refine_iter),
        )

        if refined_pose is None or refined_pose.shape != (4, 4):
            return None

        if use_y180:
            final_pose = rotate_pose_y_180(refined_pose)
        else:
            final_pose = refined_pose

        cTo_mm = final_pose.copy()
        cTo_mm[:3, 3] *= 1000.0

        bbox_xyxy = project_bbox_from_estimator_pose(
            estimator=estimator,
            pose_m=final_pose,
            image_shape=full_shape,
        )

        roi_note = f" roi={used_roi}" if used_roi is not None else ""
        print(f"[TIMING] FoundationPose refiner tracking: {time.time() - t0:.4f} s{roi_note}")

        cleanup_gpu_memory("after one FoundationPose refiner tracking", verbose=False)

        return {
            "track_pose_m": refined_pose,
            "final_pose_m": final_pose,
            "cTo_mm": cTo_mm,
            "bbox_xyxy": bbox_xyxy,
            "track_roi_bbox_xyxy": used_roi,
        }

    except Exception as e:
        print(f"[WARN] foundationpose_refiner_track_once_no_modify failed: {e}")
        print(traceback.format_exc())
        cleanup_gpu_memory("FoundationPose refiner exception", verbose=is_cuda_oom_error(e))

        if is_cuda_oom_error(e):
            raise

        return None


# ============================================================
# Recorded RGB-D utilities
# ============================================================
def load_record_meta(record_root: str) -> Dict[str, Any]:
    meta_path = os.path.join(record_root, "record_meta.json")
    if not os.path.exists(meta_path):
        print(f"[WARN] record_meta.json not found: {meta_path}")
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_recorded_rgbd_frames(
    record_root: str,
    color_dir_name: str = "color_frames",
    depth_dir_name: str = "depth_aligned_frames",
) -> Tuple[List[str], List[str]]:
    color_dir = os.path.join(record_root, color_dir_name)
    depth_dir = os.path.join(record_root, depth_dir_name)

    if not os.path.isdir(color_dir):
        raise FileNotFoundError(f"Cannot find color frame dir: {color_dir}")

    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Cannot find depth frame dir: {depth_dir}")

    color_paths = sorted(
        glob.glob(os.path.join(color_dir, "*.png"))
        + glob.glob(os.path.join(color_dir, "*.jpg"))
        + glob.glob(os.path.join(color_dir, "*.jpeg"))
    )

    depth_paths = sorted(
        glob.glob(os.path.join(depth_dir, "*.tiff"))
        + glob.glob(os.path.join(depth_dir, "*.tif"))
        + glob.glob(os.path.join(depth_dir, "*.png"))
    )

    if len(color_paths) == 0:
        raise FileNotFoundError(f"No color frames found in: {color_dir}")

    if len(depth_paths) == 0:
        raise FileNotFoundError(f"No depth frames found in: {depth_dir}")

    n = min(len(color_paths), len(depth_paths))

    if len(color_paths) != len(depth_paths):
        print(
            f"[WARN] color/depth frame count mismatch: "
            f"color={len(color_paths)}, depth={len(depth_paths)}, use first {n}"
        )

    color_paths = color_paths[:n]
    depth_paths = depth_paths[:n]

    print("=" * 80)
    print("[INFO] Loaded recorded RGB-D sequence")
    print(f"[INFO] record_root: {record_root}")
    print(f"[INFO] color_dir: {color_dir}")
    print(f"[INFO] depth_dir: {depth_dir}")
    print(f"[INFO] paired frame count: {n}")
    print(f"[INFO] first color: {color_paths[0]}")
    print(f"[INFO] first depth: {depth_paths[0]}")
    print("=" * 80)

    return color_paths, depth_paths


# ============================================================
# Hand tracking
# ============================================================
@dataclass
class HandFrameResult:
    success: bool
    handedness: Optional[str]
    score: float
    landmarks_px: Optional[np.ndarray]
    landmarks_world: Optional[np.ndarray]
    cTh: Optional[np.ndarray]
    reproj_error_px: Optional[float]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    wrist_px: Optional[Tuple[float, float]] = None
    pinch_distance_px: Optional[float] = None


def empty_hand_result(handedness: Optional[str] = None) -> HandFrameResult:
    return HandFrameResult(
        success=False,
        handedness=handedness,
        score=0.0,
        landmarks_px=None,
        landmarks_world=None,
        cTh=None,
        reproj_error_px=None,
        bbox_xyxy=None,
        wrist_px=None,
        pinch_distance_px=None,
    )


@dataclass
class MultiHandFrameResult:
    left_hand: HandFrameResult
    right_hand: HandFrameResult

    @property
    def any_hand_ok(self) -> bool:
        return bool(self.left_hand.success or self.right_hand.success)


class MediaPipeHandPoseTracker:
    def __init__(
        self,
        model_path: str,
        K: np.ndarray,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.K = K.astype(np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max(1, int(num_hands)),
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp_ms = -1

    def _bbox_from_landmarks(
        self,
        pts: np.ndarray,
        W: int,
        H: int,
        pad: int = 12,
    ) -> Tuple[int, int, int, int]:
        x1 = max(int(np.min(pts[:, 0])) - pad, 0)
        y1 = max(int(np.min(pts[:, 1])) - pad, 0)
        x2 = min(int(np.max(pts[:, 0])) + pad, W - 1)
        y2 = min(int(np.max(pts[:, 1])) + pad, H - 1)
        return x1, y1, x2, y2

    def _solve_hand_pnp(
        self,
        image_points_px: np.ndarray,
        world_points: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        object_points = world_points.astype(np.float64)
        image_points = image_points_px.astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.K,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not ok:
            return None, None

        proj, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.K,
            self.dist_coeffs,
        )

        proj = proj.reshape(-1, 2)
        reproj = float(np.linalg.norm(proj - image_points, axis=1).mean())

        T = make_transform(
            rodrigues_to_matrix(rvec),
            tvec.reshape(3),
        )

        return T, reproj

    def _single_from_index(
        self,
        result: Any,
        idx: int,
        W: int,
        H: int,
    ) -> HandFrameResult:
        lm_img = result.hand_landmarks[idx]
        lm_world = result.hand_world_landmarks[idx]
        handed_list = result.handedness[idx] if idx < len(result.handedness) else []

        handedness = None
        score = 0.0
        if len(handed_list) > 0:
            handedness = handed_list[0].category_name
            score = float(handed_list[0].score)

        pts_px = np.array(
            [[p.x * W, p.y * H] for p in lm_img],
            dtype=np.float64,
        )

        pts_world = np.array(
            [[p.x, p.y, p.z] for p in lm_world],
            dtype=np.float64,
        )

        bbox = self._bbox_from_landmarks(pts_px, W, H)
        cTh, reproj = self._solve_hand_pnp(pts_px, pts_world)

        wrist_px = tuple(pts_px[0].tolist()) if len(pts_px) > 0 else None
        pinch_distance_px = None
        if len(pts_px) > 8:
            pinch_distance_px = float(np.linalg.norm(pts_px[4] - pts_px[8]))

        return HandFrameResult(
            success=cTh is not None,
            handedness=handedness,
            score=float(score),
            landmarks_px=pts_px,
            landmarks_world=pts_world,
            cTh=cTh,
            reproj_error_px=reproj,
            bbox_xyxy=bbox,
            wrist_px=wrist_px,
            pinch_distance_px=pinch_distance_px,
        )

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> MultiHandFrameResult:
        H, W = frame_bgr.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        )

        timestamp_ms = int(timestamp_ms)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        left_hand = empty_hand_result("Left")
        right_hand = empty_hand_result("Right")

        if result is None or len(result.hand_landmarks) == 0:
            return MultiHandFrameResult(left_hand=left_hand, right_hand=right_hand)

        best_for_label: Dict[str, HandFrameResult] = {}

        for idx in range(len(result.hand_landmarks)):
            det = self._single_from_index(result, idx, W, H)
            label = det.handedness or "Unknown"
            prev = best_for_label.get(label)
            if prev is None or det.score > prev.score:
                best_for_label[label] = det

        if "Left" in best_for_label:
            left_hand = best_for_label["Left"]
        if "Right" in best_for_label:
            right_hand = best_for_label["Right"]

        return MultiHandFrameResult(left_hand=left_hand, right_hand=right_hand)


# ============================================================
# Object tracking result
# ============================================================
@dataclass
class ObjectTrackResult:
    success: bool
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    cTo_mm: Optional[np.ndarray]
    tracker_score: float
    center_xy: Optional[Tuple[float, float]]
    tpl_index: Optional[int]
    name: str
    init_source: str
    track_source: str
    error: Optional[str] = None


class FoundationPoseRefinerObjectTracker:
    def __init__(
        self,
        name: str,
        estimator: VisionPoseEstimator,
        init_track_pose_m: np.ndarray,
        init_cTo_mm: np.ndarray,
        init_mask: np.ndarray,
        tpl_index: Optional[int] = None,
        init_source: str = "foundationpose_init",
        track_refine_iter: int = 1,
        track_every_n_frames: int = 5,
        use_y180: bool = True,
        strict: bool = False,
        kalman_process_var: float = 80.0,
        kalman_measure_var: float = 80.0,
        rotation_smoothing_alpha: float = 0.90,
        max_kalman_predict_frames: int = 30,
        use_track_roi: bool = True,
        track_roi_expand_px: int = 50,
        reject_bad_measurement: bool = True,
        max_translation_jump_mm: float = 250.0,
        min_valid_z_mm: float = 100.0,
        max_valid_z_mm: float = 1200.0,
        max_abs_xy_mm: float = 800.0,
        max_bbox_center_jump_px: float = 180.0,
        min_bbox_iou: float = 0.0,
    ):
        self.name = name
        self.estimator = estimator

        self.last_track_pose_m = np.asarray(init_track_pose_m, dtype=np.float64).copy()
        self.last_cTo_mm = np.asarray(init_cTo_mm, dtype=np.float64).copy()

        self.tpl_index = tpl_index
        self.init_source = init_source
        self.track_refine_iter = int(track_refine_iter)
        self.track_every_n_frames = int(max(1, track_every_n_frames))
        self.use_y180 = bool(use_y180)
        self.strict = bool(strict)

        self.last_bbox = mask_to_bbox_xyxy(init_mask)
        self.last_center = bbox_center(self.last_bbox)

        self.refiner_disabled = False
        self.oom_count = 0
        self.frame_counter = 0

        self.translation_kf = TranslationKalmanFilter3D(
            dt=1.0,
            process_var=float(kalman_process_var),
            measure_var=float(kalman_measure_var),
        )
        self.translation_kf.initialize(self.last_cTo_mm[:3, 3])

        self.rotation_smoothing_alpha = float(np.clip(rotation_smoothing_alpha, 0.0, 1.0))
        self.max_kalman_predict_frames = int(max(1, max_kalman_predict_frames))
        self.predict_count_since_measurement = 0

        self.use_track_roi = bool(use_track_roi)
        self.track_roi_expand_px = int(max(0, track_roi_expand_px))
        self.reject_bad_measurement = bool(reject_bad_measurement)
        self.max_translation_jump_mm = float(max_translation_jump_mm)
        self.min_valid_z_mm = float(min_valid_z_mm)
        self.max_valid_z_mm = float(max_valid_z_mm)
        self.max_abs_xy_mm = float(max_abs_xy_mm)
        self.max_bbox_center_jump_px = float(max_bbox_center_jump_px)
        self.min_bbox_iou = float(max(0.0, min_bbox_iou))

    def _cTo_mm_to_final_pose_m(self, cTo_mm: np.ndarray) -> np.ndarray:
        final_pose_m = np.asarray(cTo_mm, dtype=np.float64).copy()
        final_pose_m[:3, 3] /= 1000.0
        return final_pose_m

    def _final_pose_m_to_track_pose_m(self, final_pose_m: np.ndarray) -> np.ndarray:
        if self.use_y180:
            return rotate_pose_y_180(final_pose_m).astype(np.float64)
        return final_pose_m.astype(np.float64)

    def _smooth_rotation(self, measured_cTo_mm: np.ndarray) -> np.ndarray:
        measured = np.asarray(measured_cTo_mm, dtype=np.float64).copy()

        try:
            q_prev = matrix_to_quat_xyzw(self.last_cTo_mm[:3, :3])
            q_meas = matrix_to_quat_xyzw(measured[:3, :3])
            q_smooth = quat_slerp_xyzw(q_prev, q_meas, self.rotation_smoothing_alpha)
            measured[:3, :3] = quat_xyzw_to_matrix(q_smooth)
        except Exception as e:
            print(f"[WARN] {self.name}: rotation smoothing failed: {e}")

        return measured

    def _make_track_roi(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if not self.use_track_roi or self.last_bbox is None:
            return None
        H, W = frame_bgr.shape[:2]
        return expand_bbox_xyxy(
            list(self.last_bbox),
            width=W,
            height=H,
            expand_px=self.track_roi_expand_px,
        )

    def _validate_measurement(self, ret: Dict[str, Any], frame_bgr: np.ndarray) -> Tuple[bool, str]:
        if not self.reject_bad_measurement:
            return True, "guard_disabled"

        cTo_mm = ret.get("cTo_mm", None)
        if cTo_mm is None:
            return False, "cTo_mm_none"

        cTo_mm = np.asarray(cTo_mm, dtype=np.float64)
        if cTo_mm.shape != (4, 4) or not np.all(np.isfinite(cTo_mm)):
            return False, "cTo_mm_invalid_or_nan"

        t = cTo_mm[:3, 3]
        if not np.all(np.isfinite(t)):
            return False, "translation_nan"

        if abs(float(t[0])) > self.max_abs_xy_mm or abs(float(t[1])) > self.max_abs_xy_mm:
            return False, f"xy_out_of_range:{t.tolist()}"

        if float(t[2]) < self.min_valid_z_mm or float(t[2]) > self.max_valid_z_mm:
            return False, f"z_out_of_range:{float(t[2]):.2f}"

        prev_t = self.last_cTo_mm[:3, 3]
        jump_mm = float(np.linalg.norm(t - prev_t))
        if jump_mm > self.max_translation_jump_mm:
            return False, f"translation_jump:{jump_mm:.2f}mm"

        final_pose_m = cTo_mm.copy()
        final_pose_m[:3, 3] /= 1000.0
        bbox = project_bbox_from_estimator_pose(
            estimator=self.estimator,
            pose_m=final_pose_m,
            image_shape=frame_bgr.shape,
        )
        if bbox is None:
            bbox = ret.get("bbox_xyxy", None)

        if bbox is None:
            return False, "projected_bbox_none"

        if self.last_bbox is not None:
            center_jump = bbox_center_distance_px(self.last_bbox, bbox)
            if center_jump > self.max_bbox_center_jump_px:
                return False, f"bbox_center_jump:{center_jump:.2f}px"

            if self.min_bbox_iou > 0.0:
                iou = bbox_iou_xyxy(self.last_bbox, bbox)
                if iou < self.min_bbox_iou:
                    return False, f"bbox_iou_too_low:{iou:.3f}"

        return True, "ok"

    def _accept_measurement(self, ret: Dict[str, Any], frame_bgr: np.ndarray) -> ObjectTrackResult:
        measured_cTo_mm = np.asarray(ret["cTo_mm"], dtype=np.float64).copy()

        cTo_smooth = self._smooth_rotation(measured_cTo_mm)

        t_filtered = self.translation_kf.update(cTo_smooth[:3, 3])
        cTo_smooth[:3, 3] = t_filtered

        self.last_cTo_mm = cTo_smooth.copy()

        final_pose_m = self._cTo_mm_to_final_pose_m(self.last_cTo_mm)
        self.last_track_pose_m = self._final_pose_m_to_track_pose_m(final_pose_m)

        bbox = project_bbox_from_estimator_pose(
            estimator=self.estimator,
            pose_m=final_pose_m,
            image_shape=frame_bgr.shape,
        )

        if bbox is None:
            bbox = ret.get("bbox_xyxy", None)

        if bbox is None:
            bbox = self.last_bbox

        self.last_bbox = bbox
        self.last_center = bbox_center(bbox)
        self.predict_count_since_measurement = 0

        return ObjectTrackResult(
            success=True,
            bbox_xyxy=self.last_bbox,
            cTo_mm=self.last_cTo_mm.copy(),
            tracker_score=1.0,
            center_xy=self.last_center,
            tpl_index=self.tpl_index,
            name=self.name,
            init_source=self.init_source,
            track_source="foundationpose_refiner_kalman_update",
            error=None,
        )

    def _kalman_predict_result(self, frame_bgr: np.ndarray, source: str) -> ObjectTrackResult:
        if not self.translation_kf.initialized:
            return ObjectTrackResult(
                success=True,
                bbox_xyxy=self.last_bbox,
                cTo_mm=self.last_cTo_mm.copy(),
                tracker_score=0.45,
                center_xy=self.last_center,
                tpl_index=self.tpl_index,
                name=self.name,
                init_source=self.init_source,
                track_source=source + "_keep_last_no_kf",
                error=None,
            )

        if self.predict_count_since_measurement >= self.max_kalman_predict_frames:
            return ObjectTrackResult(
                success=True,
                bbox_xyxy=self.last_bbox,
                cTo_mm=self.last_cTo_mm.copy(),
                tracker_score=0.40,
                center_xy=self.last_center,
                tpl_index=self.tpl_index,
                name=self.name,
                init_source=self.init_source,
                track_source=source + "_max_predict_keep_last",
                error=None,
            )

        pred_t = self.translation_kf.predict()

        pred_cTo = self.last_cTo_mm.copy()
        pred_cTo[:3, 3] = pred_t

        self.last_cTo_mm = pred_cTo.copy()

        final_pose_m = self._cTo_mm_to_final_pose_m(self.last_cTo_mm)
        self.last_track_pose_m = self._final_pose_m_to_track_pose_m(final_pose_m)

        bbox = project_bbox_from_estimator_pose(
            estimator=self.estimator,
            pose_m=final_pose_m,
            image_shape=frame_bgr.shape,
        )

        if bbox is not None:
            self.last_bbox = bbox
            self.last_center = bbox_center(bbox)

        self.predict_count_since_measurement += 1

        return ObjectTrackResult(
            success=True,
            bbox_xyxy=self.last_bbox,
            cTo_mm=self.last_cTo_mm.copy(),
            tracker_score=0.65,
            center_xy=self.last_center,
            tpl_index=self.tpl_index,
            name=self.name,
            init_source=self.init_source,
            track_source=source,
            error=None,
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        depth_uint16: np.ndarray,
    ) -> ObjectTrackResult:
        self.frame_counter += 1

        if self.refiner_disabled:
            return ObjectTrackResult(
                success=False,
                bbox_xyxy=self.last_bbox,
                cTo_mm=self.last_cTo_mm.copy(),
                tracker_score=0.0,
                center_xy=self.last_center,
                tpl_index=self.tpl_index,
                name=self.name,
                init_source=self.init_source,
                track_source="refiner_disabled_after_oom_keep_last",
                error="FoundationPose refiner disabled after CUDA OOM",
            )

        if self.track_every_n_frames > 1 and ((self.frame_counter - 1) % self.track_every_n_frames != 0):
            return self._kalman_predict_result(
                frame_bgr=frame_bgr,
                source=f"kalman_predict_skip_refiner_every_{self.track_every_n_frames}",
            )

        try:
            track_roi = self._make_track_roi(frame_bgr)
            ret = foundationpose_refiner_track_once_no_modify(
                estimator=self.estimator,
                frame_bgr=frame_bgr,
                depth_uint16=depth_uint16,
                last_track_pose_m=self.last_track_pose_m,
                refine_iter=self.track_refine_iter,
                use_y180=self.use_y180,
                track_roi_bbox_xyxy=track_roi,
            )

            if ret is None:
                raise RuntimeError("foundationpose refiner track returned None")

            ok, reason = self._validate_measurement(ret, frame_bgr)
            if not ok:
                print(f"[TRACK GUARD] {self.name}: reject measurement, reason={reason}")
                pred = self._kalman_predict_result(
                    frame_bgr=frame_bgr,
                    source=f"kalman_predict_after_reject_bad_measurement_{reason}",
                )
                pred.success = False
                pred.tracker_score = 0.30
                pred.error = reason
                return pred

            return self._accept_measurement(ret, frame_bgr)

        except Exception as e:
            err = str(e)
            print(f"[WARN] {self.name}: FoundationPose tracking failed: {err}")

            if is_cuda_oom_error(e):
                self.oom_count += 1
                cleanup_gpu_memory(f"{self.name} tracking OOM")
                self.refiner_disabled = True
                print(f"[WARN] {self.name}: refiner disabled after CUDA OOM")

            if self.strict:
                raise

            pred = self._kalman_predict_result(
                frame_bgr=frame_bgr,
                source="kalman_predict_after_refiner_failed",
            )
            pred.success = False
            pred.tracker_score = 0.35
            pred.error = err
            return pred


# ============================================================
# Multi-object scene manager
# ============================================================
class MultiObjectSceneManager:
    def __init__(
        self,
        object_configs: List[Dict[str, Any]],
        K: np.ndarray,
        output_root: str,
        refine_iter: int = 5,
        track_refine_iter: int = 1,
        track_every_n_frames: int = 5,
        use_y180: bool = True,
        strict_foundationpose_track: bool = False,
        kalman_process_var: float = 80.0,
        kalman_measure_var: float = 80.0,
        rotation_smoothing_alpha: float = 0.90,
        max_kalman_predict_frames: int = 30,
        use_track_roi: bool = True,
        track_roi_expand_px: int = 50,
        reject_bad_measurement: bool = True,
        max_translation_jump_mm: float = 250.0,
        min_valid_z_mm: float = 100.0,
        max_valid_z_mm: float = 1200.0,
        max_abs_xy_mm: float = 800.0,
        max_bbox_center_jump_px: float = 180.0,
        min_bbox_iou: float = 0.0,
    ):
        self.object_configs = object_configs
        self.K = K.astype(np.float64)
        self.output_root = output_root

        self.refine_iter = int(refine_iter)
        self.track_refine_iter = int(track_refine_iter)
        self.track_every_n_frames = int(max(1, track_every_n_frames))
        self.use_y180 = bool(use_y180)
        self.strict_foundationpose_track = bool(strict_foundationpose_track)

        self.kalman_process_var = float(kalman_process_var)
        self.kalman_measure_var = float(kalman_measure_var)
        self.rotation_smoothing_alpha = float(rotation_smoothing_alpha)
        self.max_kalman_predict_frames = int(max(1, max_kalman_predict_frames))

        self.use_track_roi = bool(use_track_roi)
        self.track_roi_expand_px = int(max(0, track_roi_expand_px))
        self.reject_bad_measurement = bool(reject_bad_measurement)
        self.max_translation_jump_mm = float(max_translation_jump_mm)
        self.min_valid_z_mm = float(min_valid_z_mm)
        self.max_valid_z_mm = float(max_valid_z_mm)
        self.max_abs_xy_mm = float(max_abs_xy_mm)
        self.max_bbox_center_jump_px = float(max_bbox_center_jump_px)
        self.min_bbox_iou = float(max(0.0, min_bbox_iou))

        self.trackers: Dict[str, FoundationPoseRefinerObjectTracker] = {}
        self.init_results: Dict[str, Dict[str, Any]] = {}
        self.init_errors: Dict[str, str] = {}

        os.makedirs(self.output_root, exist_ok=True)

    def initialize_from_first_frame(
        self,
        init_color_path: str,
        init_depth_path: str,
    ):
        self.init_results.clear()
        self.trackers.clear()
        self.init_errors.clear()

        init_color = cv2.imread(init_color_path)
        init_depth = cv2.imread(init_depth_path, cv2.IMREAD_UNCHANGED)

        if init_color is None:
            raise FileNotFoundError(f"Cannot read init_color: {init_color_path}")

        if init_depth is None:
            raise FileNotFoundError(f"Cannot read init_depth: {init_depth_path}")

        for cfg in self.object_configs:
            name = cfg["name"]

            try:
                ret = self._initialize_one_object(cfg, init_color, init_depth)
                self._register_init_result(name, ret)

            except Exception as e:
                self.init_errors[name] = str(e)
                print(f"[WARN] {name}: init failed: {e}")
                traceback.print_exc()
                cleanup_gpu_memory(f"{name} init failed")

        if len(self.trackers) == 0:
            raise RuntimeError(f"No object initialized. init_errors={self.init_errors}")

    def _initialize_one_object(
        self,
        cfg: Dict[str, Any],
        init_color: np.ndarray,
        init_depth: np.ndarray,
    ) -> Dict[str, Any]:
        global _CURRENT_SAM3_CFG

        name = cfg["name"]
        _CURRENT_SAM3_CFG = cfg

        patch_sam3_processor_text_prompt()
        patch_project_find_part_mask_dino(
            project_root="/home/robot4/Programming/FoundationPose"
        )

        color_for_est = init_color
        depth_for_est = init_depth
        manual_k_for_init = self.K
        roi_bbox_xyxy = None

        if "init_bbox_xyxy" in cfg and cfg["init_bbox_xyxy"] is not None:
            H, W = init_color.shape[:2]
            roi_expand_px = int(cfg.get("init_roi_expand_px", 24))
            roi_bbox_xyxy = expand_bbox_xyxy(
                cfg["init_bbox_xyxy"],
                width=W,
                height=H,
                expand_px=roi_expand_px,
            )
            color_for_est = crop_image_by_bbox(init_color, roi_bbox_xyxy)
            depth_for_est = crop_image_by_bbox(init_depth, roi_bbox_xyxy)
            manual_k_for_init = shift_camera_matrix_for_crop(self.K, roi_bbox_xyxy)

        estimator = VisionPoseEstimator(
            mesh_file=cfg["mesh_file"],
            template_dir=cfg["template_dir"],
            save_root=os.path.join(self.output_root, name),
            manual_k=manual_k_for_init,
        )

        patch_project_find_part_mask_dino(
            project_root="/home/robot4/Programming/FoundationPose"
        )
        patch_estimator_instance_find_part_mask_dino(estimator)

        print("=" * 80)
        print(f"[INIT OBJECT] {name}")
        print(f"[PROMPT] {cfg.get('sam3_text_prompt', 'long strip metal part')}")
        if roi_bbox_xyxy is not None:
            print(f"[INIT ROI HARD CONSTRAINT] {name} roi_bbox_xyxy={list(roi_bbox_xyxy)}")
        print("=" * 80)

        ret = estimator.estimate_once(
            frame_bgr=color_for_est,
            depth_uint16=depth_for_est,
            refine_iter=self.refine_iter,
            use_y180=self.use_y180,
        )

        # 关键：ROI 初始化时临时改过相机内参；初始化结束后必须恢复完整图像 K，
        # 否则后续在完整帧上 refiner tracking / 3D bbox 投影会使用错误的 cx/cy，导致乱飞。
        try:
            estimator.MANUAL_K = self.K.astype(np.float64).copy()
        except Exception as e:
            print(f"[WARN] {name}: failed to restore full-frame K after ROI init: {e}")

        release_init_only_modules_from_estimator(estimator, name)

        if ret is not None and roi_bbox_xyxy is not None:
            ret_mask = ret.get("mask", None)
            if ret_mask is not None:
                ret["mask"] = paste_mask_into_full_frame(
                    ret_mask,
                    full_shape=init_color.shape[:2],
                    bbox_xyxy=roi_bbox_xyxy,
                ).astype(bool)
            ret["init_roi_xyxy"] = list(roi_bbox_xyxy)
            ret["init_source"] = ret.get("init_source", "foundationpose_sam3_init") + "_roi"

        if ret is None:
            if "init_bbox_xyxy" in cfg:
                print(f"[WARN] {name}: estimate_once failed, use bbox fallback pose.")
                bbox = cfg["init_bbox_xyxy"]

                mask = bbox_to_mask(
                    frame_shape=init_color.shape,
                    bbox_xyxy=bbox,
                    shrink_px=int(cfg.get("bbox_shrink_px", 0)),
                )

                track_pose_m = fallback_pose_from_bbox_depth_m(
                    bbox_xyxy=bbox,
                    depth_uint16=init_depth,
                    K=self.K,
                    fallback_z_mm=float(cfg.get("fallback_z_mm", 500.0)),
                )

                cTo_mm = track_pose_m.copy()
                cTo_mm[:3, 3] *= 1000.0

                ret = {
                    "cTo": cTo_mm,
                    "refined_pose": track_pose_m,
                    "mask": mask.astype(bool),
                    "tpl_index": -1,
                    "init_source": "bbox_fallback_pose",
                    "init_roi_xyxy": list(roi_bbox_xyxy) if roi_bbox_xyxy is not None else None,
                }
            else:
                raise RuntimeError("estimate_once returned None")

        if "mask" not in ret or ret["mask"] is None:
            raise RuntimeError("estimate_once did not return mask")

        if "cTo" not in ret or ret["cTo"] is None:
            raise RuntimeError("estimate_once did not return cTo")

        if "refined_pose" not in ret or ret["refined_pose"] is None:
            cTo_mm = np.asarray(ret["cTo"], dtype=np.float64)
            track_pose_m = cTo_mm.copy()
            track_pose_m[:3, 3] /= 1000.0
            ret["refined_pose"] = track_pose_m

        ret["estimator"] = estimator
        ret["cfg"] = cfg
        ret["init_source"] = ret.get("init_source", "foundationpose_sam3_init")
        ret["tpl_index"] = int(ret.get("tpl_index", -1))

        _CURRENT_SAM3_CFG = None

        return ret

    def _register_init_result(
        self,
        name: str,
        ret: Dict[str, Any],
    ):
        cTo_mm = np.asarray(ret["cTo"], dtype=np.float64)
        track_pose_m = np.asarray(ret["refined_pose"], dtype=np.float64)

        mask = ret["mask"].astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255

        tpl_index = int(ret.get("tpl_index", -1))
        init_source = str(ret.get("init_source", "unknown"))
        estimator = ret["estimator"]

        self.init_results[name] = {
            "name": name,
            "cTo_mm": cTo_mm,
            "track_pose_m": track_pose_m,
            "mask": mask,
            "tpl_index": tpl_index,
            "init_source": init_source,
            "init_roi_xyxy": ret.get("init_roi_xyxy", None),
        }

        self.trackers[name] = FoundationPoseRefinerObjectTracker(
            name=name,
            estimator=estimator,
            init_track_pose_m=track_pose_m,
            init_cTo_mm=cTo_mm,
            init_mask=mask,
            tpl_index=tpl_index,
            init_source=init_source,
            track_refine_iter=self.track_refine_iter,
            track_every_n_frames=self.track_every_n_frames,
            use_y180=self.use_y180,
            strict=self.strict_foundationpose_track,
            kalman_process_var=self.kalman_process_var,
            kalman_measure_var=self.kalman_measure_var,
            rotation_smoothing_alpha=self.rotation_smoothing_alpha,
            max_kalman_predict_frames=self.max_kalman_predict_frames,
            use_track_roi=bool(ret.get("cfg", {}).get("use_track_roi", self.use_track_roi)),
            track_roi_expand_px=int(ret.get("cfg", {}).get("track_roi_expand_px", self.track_roi_expand_px)),
            reject_bad_measurement=bool(ret.get("cfg", {}).get("reject_bad_measurement", self.reject_bad_measurement)),
            max_translation_jump_mm=float(ret.get("cfg", {}).get("max_translation_jump_mm", self.max_translation_jump_mm)),
            min_valid_z_mm=float(ret.get("cfg", {}).get("min_valid_z_mm", self.min_valid_z_mm)),
            max_valid_z_mm=float(ret.get("cfg", {}).get("max_valid_z_mm", self.max_valid_z_mm)),
            max_abs_xy_mm=float(ret.get("cfg", {}).get("max_abs_xy_mm", self.max_abs_xy_mm)),
            max_bbox_center_jump_px=float(ret.get("cfg", {}).get("max_bbox_center_jump_px", self.max_bbox_center_jump_px)),
            min_bbox_iou=float(ret.get("cfg", {}).get("min_bbox_iou", self.min_bbox_iou)),
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        depth_uint16: np.ndarray,
    ) -> Dict[str, ObjectTrackResult]:
        results: Dict[str, ObjectTrackResult] = {}

        for name, tracker in self.trackers.items():
            results[name] = tracker.update(frame_bgr, depth_uint16)

        return results

    def compute_relative_poses_mm(
        self,
        object_results: Dict[str, ObjectTrackResult],
    ) -> Dict[str, Any]:
        relative: Dict[str, Any] = {}
        names = list(object_results.keys())

        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue

                src = names[i]
                dst = names[j]

                src_res = object_results[src]
                dst_res = object_results[dst]

                if src_res.cTo_mm is None or dst_res.cTo_mm is None:
                    continue

                relative[f"{src}_to_{dst}_mm"] = ndarray_to_list(
                    invert_transform(dst_res.cTo_mm) @ src_res.cTo_mm
                )

        return relative


# ============================================================
# Recorder
# ============================================================
class AtomicPhase(str, Enum):
    IDLE = "IDLE"
    RIGHT_APPROACH = "RIGHT_APPROACH"
    RIGHT_CONTACT = "RIGHT_CONTACT"
    RIGHT_GRASP = "RIGHT_GRASP"
    RIGHT_LIFT = "RIGHT_LIFT"
    LEFT_APPROACH_SUPPORT = "LEFT_APPROACH_SUPPORT"
    DUAL_HAND_STABILIZE = "DUAL_HAND_STABILIZE"
    LEFT_TAKEOVER = "LEFT_TAKEOVER"
    LEFT_TRANSFER = "LEFT_TRANSFER"
    RELEASE = "RELEASE"


@dataclass
class AtomicActionSegment:
    phase: str
    start_frame: int
    end_frame: int
    object_name: Optional[str]
    primary_actor: Optional[str]
    secondary_actor: Optional[str]
    description: str


@dataclass
class MultiObjectFrameInfo:
    frame_idx: int
    timestamp_ms: int
    active_object_name: Optional[str]
    active_contact_score: float
    active_object_motion_px: float
    active_grasp_by_motion: bool
    dominant_hand: Optional[str]
    atomic_phase: str
    left_contact_object: Optional[str]
    right_contact_object: Optional[str]
    hand_results: Dict[str, Any]
    object_results: Dict[str, Any]
    relative_poses_mm: Dict[str, Any]
    qwen_state_applied: bool = False
    qwen_grasp_state: Optional[Dict[str, Any]] = None


class MultiObjectTeachRecorder:
    def __init__(
        self,
        contact_distance_px: float = 50.0,
        motion_threshold_px: float = 8.0,
        contact_threshold: float = 0.15,
        ref_update_contact_max: float = 0.05,
        approach_distance_px: float = 90.0,
        lift_threshold_px: float = 18.0,
        transfer_threshold_px: float = 25.0,
        qwen_grasp_state_by_frame: Optional[Dict[int, Dict[str, Any]]] = None,
        qwen_grasp_state_confidence_threshold: float = 0.35,
    ):
        self.contact_distance_px = float(contact_distance_px)
        self.motion_threshold_px = float(motion_threshold_px)
        self.contact_threshold = float(contact_threshold)
        self.ref_update_contact_max = float(ref_update_contact_max)
        self.approach_distance_px = float(approach_distance_px)
        self.lift_threshold_px = float(lift_threshold_px)
        self.transfer_threshold_px = float(transfer_threshold_px)
        self.qwen_grasp_state_by_frame = qwen_grasp_state_by_frame or {}
        self.qwen_grasp_state_confidence_threshold = float(qwen_grasp_state_confidence_threshold)

        self.frames: List[MultiObjectFrameInfo] = []
        self.object_center_refs: Dict[str, np.ndarray] = {}
        self.prev_object_centers: Dict[str, np.ndarray] = {}
        self.object_velocities_px: Dict[str, float] = {}

        self.current_phase: str = AtomicPhase.IDLE.value
        self.current_phase_start: Optional[int] = None
        self.current_phase_object: Optional[str] = None
        self.current_primary_actor: Optional[str] = None
        self.current_secondary_actor: Optional[str] = None
        self.atomic_actions: List[AtomicActionSegment] = []

    def _bbox_distance(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        dx = max(0, max(ax1, bx1) - min(ax2, bx2))
        dy = max(0, max(ay1, by1) - min(ay2, by2))

        if dx == 0 and dy == 0:
            return 0.0

        return float(math.hypot(dx, dy))

    def _serialize_hand(self, hand: HandFrameResult) -> Dict[str, Any]:
        return {
            "success": bool(hand.success),
            "handedness": hand.handedness,
            "score": float(hand.score),
            "bbox_xyxy": hand.bbox_xyxy,
            "wrist_px": list(hand.wrist_px) if hand.wrist_px is not None else None,
            "reproj_error_px": hand.reproj_error_px,
            "pinch_distance_px": hand.pinch_distance_px,
        }

    def _hand_object_features(
        self,
        hand_res: HandFrameResult,
        obj_name: str,
        obj_res: ObjectTrackResult,
    ) -> Dict[str, Any]:
        contact_score = 0.0
        proximity_score = 0.0
        object_motion_px = 0.0
        object_velocity_px = 0.0
        vertical_lift_px = 0.0

        if (
            hand_res.success
            and obj_res.success
            and hand_res.bbox_xyxy is not None
            and obj_res.bbox_xyxy is not None
        ):
            dist = self._bbox_distance(hand_res.bbox_xyxy, obj_res.bbox_xyxy)

            if dist < self.contact_distance_px:
                contact_score = max(
                    0.0,
                    1.0 - dist / max(1.0, self.contact_distance_px),
                )

            if dist < self.approach_distance_px:
                proximity_score = max(
                    0.0,
                    1.0 - dist / max(1.0, self.approach_distance_px),
                )

        if obj_res.success and obj_res.center_xy is not None:
            center_np = np.array(obj_res.center_xy, dtype=np.float64)

            if obj_name not in self.object_center_refs:
                self.object_center_refs[obj_name] = center_np.copy()

            elif contact_score <= self.ref_update_contact_max:
                self.object_center_refs[obj_name] = (
                    0.8 * self.object_center_refs[obj_name] + 0.2 * center_np
                )

            object_motion_px = float(
                np.linalg.norm(center_np - self.object_center_refs[obj_name])
            )
            vertical_lift_px = float(self.object_center_refs[obj_name][1] - center_np[1])

            if obj_name in self.prev_object_centers:
                raw_vel = float(
                    np.linalg.norm(center_np - self.prev_object_centers[obj_name])
                )
            else:
                raw_vel = 0.0

            object_velocity_px = (
                0.6 * self.object_velocities_px.get(obj_name, 0.0)
                + 0.4 * raw_vel
            )

            self.object_velocities_px[obj_name] = object_velocity_px
            self.prev_object_centers[obj_name] = center_np.copy()

        pinch_like = False
        if hand_res.pinch_distance_px is not None:
            pinch_like = bool(hand_res.pinch_distance_px < 45.0)

        grasp_by_motion = (
            contact_score > self.contact_threshold
            and max(object_motion_px, object_velocity_px) > self.motion_threshold_px
        )

        active_score = (
            0.45 * contact_score
            + 0.15 * proximity_score
            + 0.25 * min(1.0, object_motion_px / max(1.0, self.motion_threshold_px * 2.0))
            + 0.15 * min(1.0, object_velocity_px / max(1.0, self.motion_threshold_px * 2.0))
        )

        return {
            "contact_score": float(contact_score),
            "proximity_score": float(proximity_score),
            "object_motion_px": float(object_motion_px),
            "object_velocity_px": float(object_velocity_px),
            "vertical_lift_px": float(vertical_lift_px),
            "grasp_by_motion": bool(grasp_by_motion),
            "pinch_like": bool(pinch_like),
            "active_score": float(active_score),
        }

    def _phase_info(
        self,
        phase: str,
        active_object_name: Optional[str],
        dominant_hand: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], str]:
        if phase == AtomicPhase.RIGHT_APPROACH.value:
            return "right_hand", None, "右手接近目标物体"
        if phase == AtomicPhase.RIGHT_CONTACT.value:
            return "right_hand", None, "右手接触目标物体"
        if phase == AtomicPhase.RIGHT_GRASP.value:
            return "right_hand", None, "右手形成稳定抓取"
        if phase == AtomicPhase.RIGHT_LIFT.value:
            return "right_hand", None, "右手抬起目标物体"
        if phase == AtomicPhase.LEFT_APPROACH_SUPPORT.value:
            return "left_hand", "right_hand", "左手介入并准备辅助/接管"
        if phase == AtomicPhase.DUAL_HAND_STABILIZE.value:
            return dominant_hand or "both_hands", "both_hands", "双手共同稳定目标物体"
        if phase == AtomicPhase.LEFT_TAKEOVER.value:
            return "left_hand", "right_hand", "左手接管目标物体"
        if phase == AtomicPhase.LEFT_TRANSFER.value:
            return "left_hand", None, "左手携带目标物体移动"
        if phase == AtomicPhase.RELEASE.value:
            return dominant_hand, None, "释放或脱离目标物体"
        return None, None, f"空闲/观察 {active_object_name}" if active_object_name else "空闲/观察"

    def _apply_qwen_grasp_state_override(
        self,
        frame_idx: int,
        phase: str,
        dominant_hand: Optional[str],
        active_object_name: Optional[str],
        left_contact_object: Optional[str],
        right_contact_object: Optional[str],
        primary_actor: Optional[str],
        secondary_actor: Optional[str],
        desc: str,
        object_infos: Dict[str, Any],
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], bool, Optional[Dict[str, Any]]]:
        qwen_state = self.qwen_grasp_state_by_frame.get(int(frame_idx))
        if not qwen_state:
            return phase, dominant_hand, active_object_name, left_contact_object, right_contact_object, primary_actor, secondary_actor, False, None

        try:
            conf = float(qwen_state.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf < self.qwen_grasp_state_confidence_threshold:
            return phase, dominant_hand, active_object_name, left_contact_object, right_contact_object, primary_actor, secondary_actor, False, qwen_state

        q_obj = qwen_state.get("object_name", None)
        if q_obj is not None and str(q_obj) in object_infos:
            active_object_name = str(q_obj)

        q_hand = qwen_normalize_hand_name(qwen_state.get("grasp_hand", None))
        if q_hand in {"right_hand", "left_hand", "both_hands"}:
            dominant_hand = q_hand

        phase = qwen_map_grasp_state_to_phase(qwen_state, fallback_phase=phase)
        if active_object_name is not None:
            if q_hand == "right_hand":
                right_contact_object = active_object_name
            elif q_hand == "left_hand":
                left_contact_object = active_object_name
            elif q_hand == "both_hands":
                left_contact_object = active_object_name
                right_contact_object = active_object_name

        primary_actor, secondary_actor, auto_desc = self._phase_info(phase, active_object_name, dominant_hand)
        if q_hand == "both_hands":
            primary_actor = "both_hands"
            secondary_actor = None
        elif q_hand in {"right_hand", "left_hand"}:
            primary_actor = q_hand

        if qwen_state.get("state_description"):
            desc = str(qwen_state.get("state_description"))
        else:
            desc = auto_desc

        return phase, dominant_hand, active_object_name, left_contact_object, right_contact_object, primary_actor, secondary_actor, True, qwen_state

    def _infer_atomic_phase(
        self,
        active_object_name: Optional[str],
        active_info: Dict[str, Any],
        prev_phase: str,
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str], str]:
        if not active_object_name or not active_info:
            if prev_phase != AtomicPhase.IDLE.value:
                primary, secondary, desc = self._phase_info(AtomicPhase.RELEASE.value, None, None)
                return AtomicPhase.RELEASE.value, None, None, None, primary, desc
            primary, secondary, desc = self._phase_info(AtomicPhase.IDLE.value, None, None)
            return AtomicPhase.IDLE.value, None, None, None, primary, desc

        hand_interaction = active_info.get("hand_interaction", {})
        right = hand_interaction.get("right_hand", {})
        left = hand_interaction.get("left_hand", {})

        right_contact = float(right.get("contact_score", 0.0)) > self.contact_threshold
        left_contact = float(left.get("contact_score", 0.0)) > self.contact_threshold
        right_near = float(right.get("proximity_score", 0.0)) > 0.30
        left_near = float(left.get("proximity_score", 0.0)) > 0.30

        object_motion = float(active_info.get("object_motion_px", 0.0))
        object_velocity = float(active_info.get("object_velocity_px", 0.0))
        vertical_lift = float(active_info.get("vertical_lift_px", 0.0))
        motion_big = max(object_motion, object_velocity) > self.motion_threshold_px
        transfer_big = max(object_motion, object_velocity) > self.transfer_threshold_px
        lifted = vertical_lift > self.lift_threshold_px

        right_contact_object = active_object_name if right_contact else None
        left_contact_object = active_object_name if left_contact else None

        dominant_hand = None
        if float(right.get("active_score", 0.0)) >= float(left.get("active_score", 0.0)) and (right_contact or right_near):
            dominant_hand = "right_hand"
        elif left_contact or left_near:
            dominant_hand = "left_hand"

        if right_contact and left_contact:
            phase = AtomicPhase.DUAL_HAND_STABILIZE.value
        elif right_contact and not left_contact:
            if motion_big and lifted:
                phase = AtomicPhase.RIGHT_LIFT.value
            elif motion_big or bool(right.get("pinch_like", False)):
                phase = AtomicPhase.RIGHT_GRASP.value
            else:
                phase = AtomicPhase.RIGHT_CONTACT.value
        elif left_contact and not right_contact:
            if prev_phase in {
                AtomicPhase.DUAL_HAND_STABILIZE.value,
                AtomicPhase.LEFT_APPROACH_SUPPORT.value,
                AtomicPhase.RIGHT_LIFT.value,
                AtomicPhase.RIGHT_GRASP.value,
                AtomicPhase.RIGHT_CONTACT.value,
            }:
                phase = AtomicPhase.LEFT_TRANSFER.value if transfer_big else AtomicPhase.LEFT_TAKEOVER.value
            else:
                phase = AtomicPhase.LEFT_TRANSFER.value if motion_big else AtomicPhase.LEFT_TAKEOVER.value
        else:
            if prev_phase in {
                AtomicPhase.RIGHT_CONTACT.value,
                AtomicPhase.RIGHT_GRASP.value,
                AtomicPhase.RIGHT_LIFT.value,
                AtomicPhase.DUAL_HAND_STABILIZE.value,
                AtomicPhase.LEFT_TAKEOVER.value,
                AtomicPhase.LEFT_TRANSFER.value,
            }:
                phase = AtomicPhase.RELEASE.value
            elif right_near and not left_near:
                phase = AtomicPhase.RIGHT_APPROACH.value
            elif left_near and prev_phase in {
                AtomicPhase.RIGHT_APPROACH.value,
                AtomicPhase.RIGHT_CONTACT.value,
                AtomicPhase.RIGHT_GRASP.value,
                AtomicPhase.RIGHT_LIFT.value,
            }:
                phase = AtomicPhase.LEFT_APPROACH_SUPPORT.value
            else:
                phase = AtomicPhase.IDLE.value

        primary, secondary, desc = self._phase_info(phase, active_object_name, dominant_hand)
        return phase, dominant_hand, left_contact_object, right_contact_object, primary, desc

    def _maybe_close_segment(
        self,
        frame_idx: int,
        new_phase: str,
        new_object_name: Optional[str],
        new_primary_actor: Optional[str],
        new_secondary_actor: Optional[str],
        new_description: str,
    ) -> None:
        if self.current_phase_start is None:
            self.current_phase = new_phase
            self.current_phase_start = frame_idx
            self.current_phase_object = new_object_name
            self.current_primary_actor = new_primary_actor
            self.current_secondary_actor = new_secondary_actor
            return

        if (
            new_phase == self.current_phase
            and new_object_name == self.current_phase_object
            and new_primary_actor == self.current_primary_actor
            and new_secondary_actor == self.current_secondary_actor
        ):
            return

        self.atomic_actions.append(
            AtomicActionSegment(
                phase=self.current_phase,
                start_frame=int(self.current_phase_start),
                end_frame=max(int(self.current_phase_start), int(frame_idx) - 1),
                object_name=self.current_phase_object,
                primary_actor=self.current_primary_actor,
                secondary_actor=self.current_secondary_actor,
                description=self._phase_info(self.current_phase, self.current_phase_object, None)[2],
            )
        )

        self.current_phase = new_phase
        self.current_phase_start = frame_idx
        self.current_phase_object = new_object_name
        self.current_primary_actor = new_primary_actor
        self.current_secondary_actor = new_secondary_actor

    def update(
        self,
        frame_idx: int,
        timestamp_ms: int,
        hand_res: MultiHandFrameResult,
        object_results: Dict[str, ObjectTrackResult],
        relative_poses_mm: Dict[str, Any],
    ) -> None:
        object_infos: Dict[str, Any] = {}
        active_object_name = None
        best_active_score = -1.0
        best_dominant_hand: Optional[str] = None
        left_contact_object = None
        right_contact_object = None

        for obj_name, obj_res in object_results.items():
            left_feats = self._hand_object_features(hand_res.left_hand, obj_name, obj_res)
            right_feats = self._hand_object_features(hand_res.right_hand, obj_name, obj_res)
            pose_info = pose_mm_to_serializable(obj_res.cTo_mm)

            object_motion_px = max(left_feats["object_motion_px"], right_feats["object_motion_px"])
            object_velocity_px = max(left_feats["object_velocity_px"], right_feats["object_velocity_px"])
            vertical_lift_px = max(left_feats["vertical_lift_px"], right_feats["vertical_lift_px"])
            combined_contact = max(left_feats["contact_score"], right_feats["contact_score"])
            combined_active = max(left_feats["active_score"], right_feats["active_score"])
            dominant_hand = None
            if right_feats["active_score"] >= left_feats["active_score"] and (right_feats["contact_score"] > 0 or right_feats["proximity_score"] > 0):
                dominant_hand = "right_hand"
            elif left_feats["contact_score"] > 0 or left_feats["proximity_score"] > 0:
                dominant_hand = "left_hand"

            object_infos[obj_name] = {
                "success": obj_res.success,
                "bbox_xyxy": obj_res.bbox_xyxy,
                "center_xy": obj_res.center_xy,
                "tpl_index": obj_res.tpl_index,
                "init_source": obj_res.init_source,
                "track_source": obj_res.track_source,
                **pose_info,
                "contact_score": combined_contact,
                "object_motion_px": object_motion_px,
                "object_velocity_px": object_velocity_px,
                "vertical_lift_px": vertical_lift_px,
                "grasp_by_motion": bool(left_feats["grasp_by_motion"] or right_feats["grasp_by_motion"]),
                "dominant_hand": dominant_hand,
                "hand_interaction": {
                    "left_hand": left_feats,
                    "right_hand": right_feats,
                },
                "error": obj_res.error,
            }

            if left_feats["contact_score"] > self.contact_threshold and left_contact_object is None:
                left_contact_object = obj_name
            if right_feats["contact_score"] > self.contact_threshold and right_contact_object is None:
                right_contact_object = obj_name

            if combined_active > best_active_score and (
                object_infos[obj_name]["grasp_by_motion"]
                or combined_contact > self.contact_threshold
                or left_feats["proximity_score"] > 0.30
                or right_feats["proximity_score"] > 0.30
            ):
                best_active_score = combined_active
                active_object_name = obj_name
                best_dominant_hand = dominant_hand

        active_info = object_infos.get(active_object_name, {}) if active_object_name else {}
        phase, dominant_hand, inferred_left_contact, inferred_right_contact, primary_actor, desc = self._infer_atomic_phase(
            active_object_name=active_object_name,
            active_info=active_info,
            prev_phase=self.current_phase,
        )
        if dominant_hand is None:
            dominant_hand = best_dominant_hand
        if inferred_left_contact is not None:
            left_contact_object = inferred_left_contact
        if inferred_right_contact is not None:
            right_contact_object = inferred_right_contact

        qwen_state_applied = False
        qwen_grasp_state = None
        phase, dominant_hand, active_object_name, left_contact_object, right_contact_object, primary_actor, secondary_actor, qwen_state_applied, qwen_grasp_state = self._apply_qwen_grasp_state_override(
            frame_idx=frame_idx,
            phase=phase,
            dominant_hand=dominant_hand,
            active_object_name=active_object_name,
            left_contact_object=left_contact_object,
            right_contact_object=right_contact_object,
            primary_actor=primary_actor,
            secondary_actor=None,
            desc=desc,
            object_infos=object_infos,
        )
        active_info = object_infos.get(active_object_name, {}) if active_object_name else {}

        secondary_actor = secondary_actor
        if phase in {AtomicPhase.DUAL_HAND_STABILIZE.value, AtomicPhase.LEFT_APPROACH_SUPPORT.value, AtomicPhase.LEFT_TAKEOVER.value}:
            secondary_actor = "right_hand" if primary_actor == "left_hand" else "left_hand"
            if phase == AtomicPhase.DUAL_HAND_STABILIZE.value:
                primary_actor = "both_hands"
                secondary_actor = None

        self._maybe_close_segment(
            frame_idx=frame_idx,
            new_phase=phase,
            new_object_name=active_object_name,
            new_primary_actor=primary_actor,
            new_secondary_actor=secondary_actor,
            new_description=desc,
        )

        active_contact = float(active_info.get("contact_score", 0.0)) if active_info else 0.0
        active_motion = float(active_info.get("object_motion_px", 0.0)) if active_info else 0.0
        active_grasp = bool(active_info.get("grasp_by_motion", False)) if active_info else False
        if qwen_state_applied and qwen_grasp_state is not None:
            active_grasp = bool(qwen_grasp_state.get("is_grasping", False))

        frame_info = MultiObjectFrameInfo(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            active_object_name=active_object_name,
            active_contact_score=active_contact,
            active_object_motion_px=active_motion,
            active_grasp_by_motion=active_grasp,
            dominant_hand=dominant_hand,
            atomic_phase=phase,
            left_contact_object=left_contact_object,
            right_contact_object=right_contact_object,
            hand_results={
                "left_hand": self._serialize_hand(hand_res.left_hand),
                "right_hand": self._serialize_hand(hand_res.right_hand),
            },
            object_results=object_infos,
            relative_poses_mm=relative_poses_mm,
            qwen_state_applied=bool(qwen_state_applied),
            qwen_grasp_state=qwen_grasp_state,
        )

        self.frames.append(frame_info)

    def finalize_actions(self) -> None:
        if self.current_phase_start is None or len(self.frames) == 0:
            return

        last_frame_idx = int(self.frames[-1].frame_idx)
        if len(self.atomic_actions) > 0:
            last = self.atomic_actions[-1]
            if last.start_frame == self.current_phase_start and last.end_frame == last_frame_idx:
                return

        self.atomic_actions.append(
            AtomicActionSegment(
                phase=self.current_phase,
                start_frame=int(self.current_phase_start),
                end_frame=last_frame_idx,
                object_name=self.current_phase_object,
                primary_actor=self.current_primary_actor,
                secondary_actor=self.current_secondary_actor,
                description=self._phase_info(self.current_phase, self.current_phase_object, None)[2],
            )
        )
        self.current_phase_start = None

    def export_flat_object_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for frame in self.frames:
            left_hand = frame.hand_results.get("left_hand", {})
            right_hand = frame.hand_results.get("right_hand", {})

            for object_name, info in frame.object_results.items():
                pose_matrix = info.get("cTo_mm")
                rot = info.get("R_3x3")
                bbox_xyxy = info.get("bbox_xyxy")
                hand_interaction = info.get("hand_interaction", {})
                left_int = hand_interaction.get("left_hand", {})
                right_int = hand_interaction.get("right_hand", {})

                row = {
                    "frame_idx": frame.frame_idx,
                    "timestamp_ms": frame.timestamp_ms,
                    "object_name": object_name,
                    "x_mm": info.get("x_mm"),
                    "y_mm": info.get("y_mm"),
                    "z_mm": info.get("z_mm"),
                    "qx": info.get("qx"),
                    "qy": info.get("qy"),
                    "qz": info.get("qz"),
                    "qw": info.get("qw"),
                    "roll_deg": info.get("roll_deg"),
                    "pitch_deg": info.get("pitch_deg"),
                    "yaw_deg": info.get("yaw_deg"),
                    "track_source": info.get("track_source"),
                    "success": info.get("success"),
                    "bbox_xyxy": json.dumps(bbox_xyxy, ensure_ascii=False) if bbox_xyxy is not None else "",
                    "contact_score": info.get("contact_score"),
                    "grasp_by_motion": info.get("grasp_by_motion"),
                    "object_motion_px": info.get("object_motion_px"),
                    "object_velocity_px": info.get("object_velocity_px"),
                    "vertical_lift_px": info.get("vertical_lift_px"),
                    "dominant_hand": info.get("dominant_hand"),
                    "atomic_phase": frame.atomic_phase,
                    "left_contact_object": frame.left_contact_object,
                    "right_contact_object": frame.right_contact_object,
                    "left_hand_ok": left_hand.get("success"),
                    "right_hand_ok": right_hand.get("success"),
                    "left_hand_bbox": json.dumps(left_hand.get("bbox_xyxy"), ensure_ascii=False) if left_hand.get("bbox_xyxy") is not None else "",
                    "right_hand_bbox": json.dumps(right_hand.get("bbox_xyxy"), ensure_ascii=False) if right_hand.get("bbox_xyxy") is not None else "",
                    "left_contact_score": left_int.get("contact_score"),
                    "right_contact_score": right_int.get("contact_score"),
                    "left_proximity_score": left_int.get("proximity_score"),
                    "right_proximity_score": right_int.get("proximity_score"),
                    "left_grasp_by_motion": left_int.get("grasp_by_motion"),
                    "right_grasp_by_motion": right_int.get("grasp_by_motion"),
                    "qwen_state_applied": frame.qwen_state_applied,
                    "qwen_object_name": (frame.qwen_grasp_state or {}).get("object_name") if frame.qwen_grasp_state else None,
                    "qwen_grasp_hand": (frame.qwen_grasp_state or {}).get("grasp_hand") if frame.qwen_grasp_state else None,
                    "qwen_is_grasping": (frame.qwen_grasp_state or {}).get("is_grasping") if frame.qwen_grasp_state else None,
                    "qwen_grasp_phase": (frame.qwen_grasp_state or {}).get("grasp_phase") if frame.qwen_grasp_state else None,
                    "qwen_confidence": (frame.qwen_grasp_state or {}).get("confidence") if frame.qwen_grasp_state else None,
                    "qwen_state_description": (frame.qwen_grasp_state or {}).get("state_description") if frame.qwen_grasp_state else None,
                    "tpl_index": info.get("tpl_index"),
                    "init_source": info.get("init_source"),
                    "error": info.get("error"),
                    "pose_matrix_cTo_mm": json.dumps(pose_matrix, ensure_ascii=False) if pose_matrix is not None else "",
                    "rotation_matrix_3x3": json.dumps(rot, ensure_ascii=False) if rot is not None else "",
                }

                row["is_active_object"] = (frame.active_object_name == object_name)
                row["active_object_name"] = frame.active_object_name
                row["active_contact_score"] = frame.active_contact_score
                row["active_object_motion_px"] = frame.active_object_motion_px
                row["active_grasp_by_motion"] = frame.active_grasp_by_motion

                rows.append(row)

        return rows

    def export(self) -> Dict[str, Any]:
        self.finalize_actions()
        return {
            "frames": [asdict(f) for f in self.frames],
            "object_center_refs_px": {
                k: v.tolist() for k, v in self.object_center_refs.items()
            },
            "atomic_actions": [asdict(a) for a in self.atomic_actions],
        }


# ============================================================
# Visualization
# ============================================================
def project_points_to_image(
    pts_cam: np.ndarray,
    K: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    debug = {
        "valid_input": bool(pts_cam.ndim == 2 and pts_cam.shape[1] == 3),
        "z_min": None,
        "z_max": None,
        "negative_or_zero_z": False,
        "finite": False,
    }

    if pts_cam.ndim != 2 or pts_cam.shape[1] != 3:
        return None, debug

    z = pts_cam[:, 2]
    debug["z_min"] = float(np.min(z)) if z.size > 0 else None
    debug["z_max"] = float(np.max(z)) if z.size > 0 else None

    if np.any(z <= 1e-6):
        debug["negative_or_zero_z"] = True
        return None, debug

    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]

    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)):
        return None, debug

    debug["finite"] = True
    return np.stack([u, v], axis=1), debug


def bbox_points_visible_status(corners_2d: Optional[np.ndarray], image_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    H, W = image_shape[:2]
    status = {
        "any_inside": False,
        "all_left": False,
        "all_right": False,
        "all_above": False,
        "all_below": False,
    }
    if corners_2d is None or len(corners_2d) == 0:
        return status

    xs = corners_2d[:, 0]
    ys = corners_2d[:, 1]
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    status["any_inside"] = bool(np.any(inside))
    status["all_left"] = bool(np.all(xs < 0))
    status["all_right"] = bool(np.all(xs >= W))
    status["all_above"] = bool(np.all(ys < 0))
    status["all_below"] = bool(np.all(ys >= H))
    return status


def project_3d_bbox_corners_from_cTo_mm(
    estimator: VisionPoseEstimator,
    cTo_mm: Optional[np.ndarray],
    image_shape: Optional[Tuple[int, int, int]] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    根据 estimator.to_origin 和 estimator.bbox，将当前 6D 位姿对应的 3D 包围盒八个角点投影到图像平面。
    返回 (shape=(8,2) 的像素点, debug_info)。
    """
    debug: Dict[str, Any] = {
        "has_pose": cTo_mm is not None,
        "has_to_origin": hasattr(estimator, "to_origin"),
        "has_bbox": hasattr(estimator, "bbox"),
        "projection_ok": False,
        "reason": None,
        "pose_t_mm": None,
        "z_min": None,
        "z_max": None,
        "visible": None,
    }
    try:
        if cTo_mm is None:
            debug["reason"] = "cTo_mm_is_none"
            return None, debug

        if not hasattr(estimator, "to_origin") or not hasattr(estimator, "bbox"):
            debug["reason"] = "missing_to_origin_or_bbox"
            return None, debug

        pose_mm = np.asarray(cTo_mm, dtype=np.float64).copy()
        debug["pose_t_mm"] = pose_mm[:3, 3].tolist()

        pose_m = pose_mm.copy()
        pose_m[:3, 3] /= 1000.0

        center_pose = pose_m @ np.linalg.inv(estimator.to_origin)

        xyz_min = np.asarray(estimator.bbox[0], dtype=np.float64)
        xyz_max = np.asarray(estimator.bbox[1], dtype=np.float64)
        x0, y0, z0 = xyz_min
        x1, y1, z1 = xyz_max

        corners = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float64)

        corners_h = np.concatenate(
            [corners, np.ones((corners.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        pts_cam = (center_pose @ corners_h.T).T[:, :3]
        corners_2d, proj_debug = project_points_to_image(pts_cam, estimator.MANUAL_K.astype(np.float64))
        debug.update(proj_debug)

        if corners_2d is None:
            debug["reason"] = "projection_failed"
            return None, debug

        debug["projection_ok"] = True
        debug["reason"] = "ok"
        if image_shape is not None:
            debug["visible"] = bbox_points_visible_status(corners_2d, image_shape)
        return corners_2d, debug
    except Exception as e:
        debug["reason"] = f"exception:{e}"
        print(f"[WARN] project_3d_bbox_corners_from_cTo_mm failed: {e}")
        return None, debug


def project_pose_axes_from_cTo_mm(
    cTo_mm: Optional[np.ndarray],
    K: np.ndarray,
    axis_length_mm: float = 40.0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    将物体坐标系的三个轴投影到图像平面。
    返回 shape=(4,2) 的像素点: [origin, x_end, y_end, z_end]
    颜色约定在 draw_pose_axes 中定义: X红 Y绿 Z蓝。
    """
    debug: Dict[str, Any] = {
        "has_pose": cTo_mm is not None,
        "projection_ok": False,
        "reason": None,
        "axis_length_mm": float(axis_length_mm),
    }
    try:
        if cTo_mm is None:
            debug["reason"] = "cTo_mm_is_none"
            return None, debug

        T_mm = np.asarray(cTo_mm, dtype=np.float64).copy()
        T_m = T_mm.copy()
        T_m[:3, 3] /= 1000.0

        L = float(axis_length_mm) / 1000.0
        axes_obj = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [L,   0.0, 0.0, 1.0],
            [0.0, L,   0.0, 1.0],
            [0.0, 0.0, L,   1.0],
        ], dtype=np.float64)

        pts_cam = (T_m @ axes_obj.T).T[:, :3]
        pts_2d, proj_debug = project_points_to_image(pts_cam, np.asarray(K, dtype=np.float64))
        debug.update(proj_debug)
        if pts_2d is None:
            debug["reason"] = "projection_failed"
            return None, debug

        debug["projection_ok"] = True
        debug["reason"] = "ok"
        return pts_2d, debug
    except Exception as e:
        debug["reason"] = f"exception:{e}"
        print(f"[WARN] project_pose_axes_from_cTo_mm failed: {e}")
        return None, debug


def draw_pose_axes(
    img: np.ndarray,
    axis_pts_2d: Optional[np.ndarray],
    thickness: int = 2,
) -> np.ndarray:
    if axis_pts_2d is None or not isinstance(axis_pts_2d, np.ndarray) or axis_pts_2d.shape != (4, 2):
        return img

    pts = np.round(axis_pts_2d).astype(int)
    origin = tuple(pts[0])
    x_end = tuple(pts[1])
    y_end = tuple(pts[2])
    z_end = tuple(pts[3])

    cv2.line(img, origin, x_end, (0, 0, 255), thickness, lineType=cv2.LINE_AA)
    cv2.line(img, origin, y_end, (0, 255, 0), thickness, lineType=cv2.LINE_AA)
    cv2.line(img, origin, z_end, (255, 0, 0), thickness, lineType=cv2.LINE_AA)

    cv2.circle(img, origin, 3, (255, 255, 255), -1)
    cv2.putText(img, "X", (x_end[0] + 4, x_end[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "Y", (y_end[0] + 4, y_end[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(img, "Z", (z_end[0] + 4, z_end[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img


def draw_3d_bbox(
    img: np.ndarray,
    corners_2d: Optional[np.ndarray],
    color_bgr: Tuple[int, int, int],
    thickness: int = 2,
    draw_vertices: bool = True,
) -> np.ndarray:
    if corners_2d is None or len(corners_2d) != 8:
        return img

    pts = np.round(corners_2d).astype(int)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), color_bgr, thickness, lineType=cv2.LINE_AA)

    if draw_vertices:
        for p in pts:
            cv2.circle(img, tuple(p), 3, color_bgr, -1)

    # 高亮前表面的一条边，方便看朝向
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), (255, 255, 255), max(1, thickness), lineType=cv2.LINE_AA)
    return img


def draw_label_on_point(
    img: np.ndarray,
    pt_xy: Optional[Tuple[float, float]],
    label: str,
    color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    if pt_xy is None:
        return img

    x = int(round(pt_xy[0]))
    y = int(round(pt_xy[1]))
    cv2.putText(
        img,
        label,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color_bgr,
        2,
    )
    return img


def draw_hand_landmarks(
    img: np.ndarray,
    pts_px: Optional[np.ndarray],
    color_bgr: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    if pts_px is None:
        return img

    for p in pts_px.astype(int):
        cv2.circle(img, tuple(p), 2, color_bgr, -1)

    return img


def draw_hand_result(
    img: np.ndarray,
    hand: HandFrameResult,
    label_prefix: str,
    color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    if hand is None or not hand.success:
        return img

    draw_hand_landmarks(img, hand.landmarks_px, color_bgr=color_bgr)

    if hand.wrist_px is not None:
        wrist = (int(hand.wrist_px[0]), int(hand.wrist_px[1]))
        cv2.circle(img, wrist, 5, color_bgr, -1)
        cv2.putText(
            img,
            f"{label_prefix}:{hand.score:.2f}",
            (wrist[0] + 6, max(20, wrist[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_bgr,
            2,
        )

    return img


def overlay_status(
    img: np.ndarray,
    frame_idx: int,
    frame_info: MultiObjectFrameInfo,
) -> np.ndarray:
    left_hand = frame_info.hand_results.get("left_hand", {})
    right_hand = frame_info.hand_results.get("right_hand", {})

    lines = [
        f"frame: {frame_idx}",
        f"active_object: {frame_info.active_object_name}",
        f"atomic_phase: {frame_info.atomic_phase}",
        f"dominant_hand: {frame_info.dominant_hand}",
        f"left_contact_object: {frame_info.left_contact_object}",
        f"right_contact_object: {frame_info.right_contact_object}",
        f"contact_score: {frame_info.active_contact_score:.3f}",
        f"object_motion_px: {frame_info.active_object_motion_px:.2f}",
        f"grasp_by_motion: {frame_info.active_grasp_by_motion}",
        f"qwen_applied: {frame_info.qwen_state_applied}",
        f"left_ok: {left_hand.get('success')} reproj: {left_hand.get('reproj_error_px')}",
        f"right_ok: {right_hand.get('success')} reproj: {right_hand.get('reproj_error_px')}",
    ]

    y = 30

    for line in lines:
        cv2.putText(
            img,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
        )
        y += 24

    return img



# ============================================================
# Ollama + Qwen frame-level grasp-state preprocessing
# ============================================================
@dataclass
class QwenGraspSampledFrame:
    sample_index: int
    frame_index: int
    timestamp_sec: float
    image_path: str


class OllamaQwenError(RuntimeError):
    pass


def qwen_ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def qwen_strip_think(text: str) -> str:
    """去掉 Qwen 思考模型可能输出的 <think>...</think> 内容。"""
    if not text:
        return ""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


def qwen_video_info(video_path: str) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开 Qwen 分析视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0.0 or frame_count <= 0:
        raise RuntimeError(f"Qwen 视频元信息异常: fps={fps}, frame_count={frame_count}")

    return fps, frame_count, width, height


def qwen_build_frame_indices(frame_count: int, stride: int = 1, max_frames: int = 0) -> List[int]:
    stride = max(1, int(stride))
    indices = list(range(0, int(frame_count), stride))
    if len(indices) == 0 or indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    if int(max_frames) > 0:
        indices = indices[: int(max_frames)]
    return sorted(set(int(i) for i in indices if 0 <= int(i) < frame_count))


def qwen_extract_selected_frames(
    video_path: str,
    output_dir: Path,
    fps: float,
    frame_indices: List[int],
    jpeg_quality: int = 88,
) -> List[QwenGraspSampledFrame]:
    qwen_ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开 Qwen 分析视频: {video_path}")

    samples: List[QwenGraspSampledFrame] = []
    for sample_index, frame_index in enumerate(frame_indices):
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        if not ok:
            cap.release()
            raise RuntimeError(f"Qwen 抽帧无法跳转到帧 {frame_index}")

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Qwen 抽帧读取失败: {frame_index}")

        image_path = output_dir / f"sample_{sample_index:05d}_frame_{frame_index:06d}.jpg"
        saved = cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not saved:
            cap.release()
            raise RuntimeError(f"Qwen 抽帧保存失败: {image_path}")

        samples.append(
            QwenGraspSampledFrame(
                sample_index=int(sample_index),
                frame_index=int(frame_index),
                timestamp_sec=float(frame_index / max(1e-6, fps)),
                image_path=str(image_path),
            )
        )

    cap.release()
    return samples


def qwen_create_labeled_sheet(
    samples: List[QwenGraspSampledFrame],
    output_path: Path,
    thumb_w: int = 260,
    cols: int = 4,
) -> None:
    imgs = []
    for s in samples:
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = float(thumb_w) / max(1, w)
        thumb_h = int(round(h * scale))
        img = cv2.resize(img, (thumb_w, thumb_h))
        label = f"frame={s.frame_index}  t={s.timestamp_sec:.2f}s"
        cv2.rectangle(img, (0, 0), (thumb_w, 28), (0, 0, 0), -1)
        cv2.putText(img, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        imgs.append(img)

    if not imgs:
        raise RuntimeError("Qwen sheet 没有可用图像")

    cols = max(1, min(int(cols), len(imgs)))
    rows = int(math.ceil(len(imgs) / cols))
    cell_h = max(im.shape[0] for im in imgs)
    cell_w = max(im.shape[1] for im in imgs)
    canvas = 255 * np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, im in enumerate(imgs):
        r = i // cols
        c = i % cols
        y = r * cell_h
        x = c * cell_w
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im

    cv2.imwrite(str(output_path), canvas)


def qwen_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def qwen_ollama_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_s: int = 600,
    temperature: float = 0.0,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # 新版 Ollama/Qwen thinking 模型可识别；旧版若忽略也不会影响 /no_think 提示词。
        "think": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    resp = requests.post(url, json=payload, timeout=int(timeout_s))
    if resp.status_code != 200:
        raise OllamaQwenError(f"Ollama 请求失败: {resp.status_code} {resp.text[:500]}")

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    content = qwen_strip_think(str(content))
    if not content:
        raise OllamaQwenError("Ollama 返回为空")
    return content


def qwen_extract_json_block(text: str) -> Dict[str, Any]:
    text = qwen_strip_think(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("Qwen 输出中没有可解析 JSON")


def qwen_normalize_hand_name(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"right", "right hand", "right_hand", "右手"}:
        return "right_hand"
    if s in {"left", "left hand", "left_hand", "左手"}:
        return "left_hand"
    if s in {"both", "both hands", "both_hands", "双手"}:
        return "both_hands"
    if s in {"none", "no", "null", "无", "unknown", ""}:
        return None
    return s


def qwen_normalize_object_name(x: Any, object_names: List[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s in object_names:
        return s
    low = s.lower()
    for name in object_names:
        if low == name.lower():
            return name
    if low in {"none", "null", "unknown", "无", ""}:
        return None
    return s


def qwen_build_grasp_state_prompt(
    samples: List[QwenGraspSampledFrame],
    object_names: List[str],
) -> str:
    meta = [
        {
            "frame_idx": s.frame_index,
            "timestamp_sec": round(s.timestamp_sec, 3),
        }
        for s in samples
    ]
    return (
        "/no_think\n"
        "你是视频状态标注器。请观察图片拼图中的每个小图，判断每一帧的抓取状态。\n"
        "每个小图左上角已经标注 frame 编号和时间。\n"
        "只输出 JSON，不要输出 Markdown，不要输出解释，不要输出思考过程。\n"
        "对象名称只能优先从 object_names 中选择；无法确定时填 null。\n"
        "grasp_hand 只能是 right_hand、left_hand、both_hands 或 null。\n"
        "is_grasping 必须是 true 或 false。\n"
        "grasp_phase 只能是 idle、approach、contact、grasp、lift、transfer、release、unknown。\n"
        "JSON 格式固定为：\n"
        "{\n"
        "  \"frames\": [\n"
        "    {\n"
        "      \"frame_idx\": 0,\n"
        "      \"object_name\": \"part_A\",\n"
        "      \"grasp_hand\": \"right_hand\",\n"
        "      \"is_grasping\": false,\n"
        "      \"grasp_phase\": \"approach\",\n"
        "      \"confidence\": 0.75,\n"
        "      \"state_description\": \"右手靠近 part_A，但尚未稳定抓取\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        f"object_names={json.dumps(object_names, ensure_ascii=False)}\n"
        f"需要标注的帧：{json.dumps(meta, ensure_ascii=False)}"
    )


def qwen_map_grasp_state_to_phase(state: Dict[str, Any], fallback_phase: str) -> str:
    hand = qwen_normalize_hand_name(state.get("grasp_hand"))
    phase = str(state.get("grasp_phase", "unknown")).strip().lower()
    is_grasping = bool(state.get("is_grasping", False))

    if phase == "idle":
        return AtomicPhase.IDLE.value
    if phase == "release":
        return AtomicPhase.RELEASE.value

    if hand == "right_hand":
        if phase == "approach":
            return AtomicPhase.RIGHT_APPROACH.value
        if phase == "contact":
            return AtomicPhase.RIGHT_CONTACT.value
        if phase in {"grasp", "hold"} or is_grasping:
            return AtomicPhase.RIGHT_GRASP.value
        if phase == "lift":
            return AtomicPhase.RIGHT_LIFT.value
        if phase == "transfer":
            return AtomicPhase.RIGHT_LIFT.value

    if hand == "left_hand":
        if phase == "approach":
            return AtomicPhase.LEFT_APPROACH_SUPPORT.value
        if phase == "contact":
            return AtomicPhase.LEFT_TAKEOVER.value
        if phase in {"grasp", "hold"} or is_grasping:
            return AtomicPhase.LEFT_TAKEOVER.value
        if phase in {"lift", "transfer"}:
            return AtomicPhase.LEFT_TRANSFER.value

    if hand == "both_hands":
        if phase in {"contact", "grasp", "lift", "transfer"} or is_grasping:
            return AtomicPhase.DUAL_HAND_STABILIZE.value

    return fallback_phase


def qwen_interpolate_states(
    sparse_states: Dict[int, Dict[str, Any]],
    frame_count: int,
) -> Dict[int, Dict[str, Any]]:
    if not sparse_states:
        return {}
    keys = sorted(k for k in sparse_states.keys() if 0 <= int(k) < frame_count)
    if not keys:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    ptr = 0
    current = sparse_states[keys[0]]
    for f in range(frame_count):
        while ptr + 1 < len(keys) and f >= keys[ptr + 1]:
            ptr += 1
            current = sparse_states[keys[ptr]]
        s = dict(current)
        s["frame_idx"] = int(f)
        s["source_frame_idx"] = int(keys[ptr])
        out[int(f)] = s
    return out


def run_qwen_video_grasp_state_preprocess(
    video_path: str,
    output_dir: str,
    object_names: List[str],
    model: str = "qwen3.5:35B",
    base_url: str = "http://127.0.0.1:11434",
    stride: int = 1,
    chunk_size: int = 8,
    timeout_s: int = 600,
    jpeg_quality: int = 88,
    max_frames: int = 0,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    out_dir = Path(output_dir).resolve()
    frames_dir = out_dir / "sampled_frames"
    sheets_dir = out_dir / "sheets"
    qwen_ensure_dir(out_dir)
    qwen_ensure_dir(frames_dir)
    qwen_ensure_dir(sheets_dir)

    fps, frame_count, width, height = qwen_video_info(video_path)
    frame_indices = qwen_build_frame_indices(frame_count, stride=stride, max_frames=max_frames)
    samples = qwen_extract_selected_frames(video_path, frames_dir, fps, frame_indices, jpeg_quality=jpeg_quality)

    print("=" * 80)
    print("[QWEN GRASP STATE] 开始视频预处理：逐帧抓取状态识别")
    print(f"video       : {video_path}")
    print(f"model       : {model}")
    print(f"output_dir  : {out_dir}")
    print(f"frame_count : {frame_count}, fps={fps:.3f}, resolution={width}x{height}")
    print(f"stride      : {stride}, sampled={len(samples)}, chunk_size={chunk_size}")
    print(f"objects     : {object_names}")
    print("=" * 80)

    sparse_states: Dict[int, Dict[str, Any]] = {}
    raw_chunks: List[Dict[str, Any]] = []
    chunk_size = max(1, int(chunk_size))

    for chunk_id, start in enumerate(range(0, len(samples), chunk_size)):
        chunk = samples[start:start + chunk_size]
        sheet_path = sheets_dir / f"qwen_chunk_{chunk_id:04d}.jpg"
        qwen_create_labeled_sheet(chunk, sheet_path, thumb_w=260, cols=min(4, len(chunk)))

        prompt = qwen_build_grasp_state_prompt(chunk, object_names)
        image_b64 = qwen_image_to_base64(str(sheet_path))
        print(f"[QWEN GRASP STATE] chunk={chunk_id:04d}, frames={chunk[0].frame_index}-{chunk[-1].frame_index}")
        raw = qwen_ollama_chat(
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
            timeout_s=timeout_s,
            temperature=temperature,
        )
        parsed = qwen_extract_json_block(raw)
        raw_chunks.append({
            "chunk_id": chunk_id,
            "sheet_path": str(sheet_path),
            "frames": [asdict(s) for s in chunk],
            "raw": raw,
            "parsed": parsed,
        })

        for item in parsed.get("frames", []):
            try:
                fidx = int(item.get("frame_idx"))
            except Exception:
                continue
            state = {
                "frame_idx": fidx,
                "object_name": qwen_normalize_object_name(item.get("object_name"), object_names),
                "grasp_hand": qwen_normalize_hand_name(item.get("grasp_hand")),
                "is_grasping": bool(item.get("is_grasping", False)),
                "grasp_phase": str(item.get("grasp_phase", "unknown")).strip().lower(),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "state_description": str(item.get("state_description", "")),
                "qwen_source": "qwen_video_preprocess",
            }
            sparse_states[fidx] = state

    dense_states = qwen_interpolate_states(sparse_states, frame_count=frame_count)

    payload = {
        "enabled": True,
        "video": str(Path(video_path).resolve()),
        "model": model,
        "fps": fps,
        "frame_count": frame_count,
        "resolution": [width, height],
        "object_names": object_names,
        "stride": int(stride),
        "chunk_size": int(chunk_size),
        "sparse_states": {str(k): v for k, v in sorted(sparse_states.items())},
        "dense_states": {str(k): v for k, v in sorted(dense_states.items())},
        "raw_chunks": raw_chunks,
    }

    with open(out_dir / "qwen_frame_grasp_states.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "qwen_frame_grasp_states.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx", "source_frame_idx", "object_name", "grasp_hand",
                "is_grasping", "grasp_phase", "confidence", "state_description", "qwen_source",
            ],
        )
        writer.writeheader()
        for k in sorted(dense_states.keys()):
            writer.writerow(dense_states[k])

    print("=" * 80)
    print("[QWEN GRASP STATE DONE] Qwen 逐帧抓取状态预处理完成")
    print(f"JSON: {out_dir / 'qwen_frame_grasp_states.json'}")
    print(f"CSV : {csv_path}")
    print("=" * 80)

    return {
        "enabled": True,
        "video": str(Path(video_path).resolve()),
        "model": model,
        "output_dir": str(out_dir),
        "json_path": str(out_dir / "qwen_frame_grasp_states.json"),
        "csv_path": str(csv_path),
        "frame_count": frame_count,
        "sampled_count": len(samples),
        "dense_states": dense_states,
    }


# ============================================================
# Main pipeline
# ============================================================
def load_object_configs(config_path: str) -> List[Dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("objects_config must be a JSON list")

    for item in data:
        for key in ["name", "mesh_file", "template_dir"]:
            if key not in item:
                raise ValueError(f"Missing key '{key}' in object config: {item}")

    return data


def run_pipeline(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    vis_dir = os.path.join(args.output_dir, "vis_foundationpose_refiner_tracking_3d_bbox")
    os.makedirs(vis_dir, exist_ok=True)

    _ = load_record_meta(args.record_root)

    color_paths, depth_paths = list_recorded_rgbd_frames(
        record_root=args.record_root,
        color_dir_name=args.color_dir_name,
        depth_dir_name=args.depth_dir_name,
    )

    if args.init_frame_index < 0 or args.init_frame_index >= len(color_paths):
        raise ValueError(
            f"Invalid init_frame_index={args.init_frame_index}, "
            f"valid range: 0 ~ {len(color_paths) - 1}"
        )

    init_color_path = color_paths[args.init_frame_index]
    init_depth_path = depth_paths[args.init_frame_index]

    init_color = cv2.imread(init_color_path)
    init_depth = cv2.imread(init_depth_path, cv2.IMREAD_UNCHANGED)

    if init_color is None:
        raise FileNotFoundError(f"Cannot read init color: {init_color_path}")

    if init_depth is None:
        raise FileNotFoundError(f"Cannot read init depth: {init_depth_path}")

    H0, W0 = init_color.shape[:2]

    K = np.array([
        [args.fx, 0.0, args.cx],
        [0.0, args.fy, args.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    print("=" * 80)
    print("[INFO] FoundationPose refiner tracking + Kalman")
    print(f"[INFO] record_root: {args.record_root}")
    print(f"[INFO] init_frame_index: {args.init_frame_index}")
    print(f"[INFO] init_color: {init_color_path}")
    print(f"[INFO] init_depth: {init_depth_path}")
    print(f"[INFO] image size: {W0}x{H0}")
    print(f"[INFO] K:\n{K}")
    print(f"[INFO] track_every_n_frames={args.track_every_n_frames}")
    print(f"[INFO] kalman_process_var={args.kalman_process_var}")
    print(f"[INFO] kalman_measure_var={args.kalman_measure_var}")
    print(f"[INFO] rotation_smoothing_alpha={args.rotation_smoothing_alpha}")
    print("=" * 80)

    object_configs = load_object_configs(args.objects_config)

    object_style = {
        cfg["name"]: {
            "display_name": cfg.get("display_name", cfg["name"]),
            "bbox_color_bgr": tuple(
                int(v)
                for v in cfg.get(
                    "bbox_color_bgr",
                    cfg.get("mask_color_bgr", [0, 180, 255])
                )
            ),
        }
        for cfg in object_configs
    }

    qwen_grasp_state_preprocess: Dict[str, Any] = {"enabled": False}
    qwen_grasp_state_by_frame: Dict[int, Dict[str, Any]] = {}
    if args.enable_qwen_grasp_state_preprocess:
        try:
            qwen_video_path = args.qwen_grasp_state_video or os.path.join(args.record_root, "color_video.mp4")
            qwen_output_dir = args.qwen_grasp_state_output_dir or os.path.join(args.output_dir, "qwen_grasp_state_preprocess")
            qwen_grasp_state_preprocess = run_qwen_video_grasp_state_preprocess(
                video_path=qwen_video_path,
                output_dir=qwen_output_dir,
                object_names=[cfg["name"] for cfg in object_configs],
                model=args.qwen_grasp_state_model,
                base_url=args.qwen_grasp_state_ollama_base_url,
                stride=args.qwen_grasp_state_stride,
                chunk_size=args.qwen_grasp_state_chunk_size,
                timeout_s=args.qwen_grasp_state_timeout_s,
                jpeg_quality=args.qwen_grasp_state_jpeg_quality,
                max_frames=args.qwen_grasp_state_max_frames,
                temperature=args.qwen_grasp_state_temperature,
            )
            qwen_grasp_state_by_frame = {
                int(k): v for k, v in qwen_grasp_state_preprocess.get("dense_states", {}).items()
            }
        except Exception as e:
            qwen_grasp_state_preprocess = {
                "enabled": True,
                "failed": True,
                "error": str(e),
            }
            print(f"[WARN] Qwen grasp-state preprocessing failed: {e}")
            print(traceback.format_exc())
            if args.strict_qwen_grasp_state_preprocess:
                raise
    else:
        print("[QWEN GRASP STATE] disabled, use rule-based grasp state.")

    scene_manager = MultiObjectSceneManager(
        object_configs=object_configs,
        K=K,
        output_root=os.path.join(args.output_dir, "vision_init_foundationpose_tracking"),
        refine_iter=args.refine_iter,
        track_refine_iter=args.track_refine_iter,
        track_every_n_frames=args.track_every_n_frames,
        use_y180=args.use_y180,
        strict_foundationpose_track=args.strict_foundationpose_track,
        kalman_process_var=args.kalman_process_var,
        kalman_measure_var=args.kalman_measure_var,
        rotation_smoothing_alpha=args.rotation_smoothing_alpha,
        max_kalman_predict_frames=args.max_kalman_predict_frames,
        use_track_roi=args.use_track_roi,
        track_roi_expand_px=args.track_roi_expand_px,
        reject_bad_measurement=args.reject_bad_measurement,
        max_translation_jump_mm=args.max_translation_jump_mm,
        min_valid_z_mm=args.min_valid_z_mm,
        max_valid_z_mm=args.max_valid_z_mm,
        max_abs_xy_mm=args.max_abs_xy_mm,
        max_bbox_center_jump_px=args.max_bbox_center_jump_px,
        min_bbox_iou=args.min_bbox_iou,
    )

    scene_manager.initialize_from_first_frame(
        init_color_path=init_color_path,
        init_depth_path=init_depth_path,
    )

    hand_tracker = MediaPipeHandPoseTracker(
        model_path=args.hand_model,
        K=K,
        num_hands=2,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    recorder = MultiObjectTeachRecorder(
        contact_distance_px=args.contact_distance_px,
        motion_threshold_px=args.motion_threshold_px,
        contact_threshold=args.contact_threshold,
        qwen_grasp_state_by_frame=qwen_grasp_state_by_frame,
        qwen_grasp_state_confidence_threshold=args.qwen_grasp_state_confidence_threshold,
    )

    writer = None
    last_valid_corners_by_name: Dict[str, np.ndarray] = {}
    last_valid_label_pt_by_name: Dict[str, Tuple[float, float]] = {}

    if args.save_video:
        video_path = os.path.join(
            args.output_dir,
            "teaching_vis_foundationpose_refiner_tracking_3d_bbox_3d_bbox.mp4",
        )

        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.output_fps),
            (W0, H0),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {video_path}")

    total_end = len(color_paths)

    last_valid_corners_by_name: Dict[str, np.ndarray] = {}
    last_valid_label_pt_by_name: Dict[str, Tuple[float, float]] = {}

    if args.max_frames > 0:
        total_end = min(total_end, args.init_frame_index + args.max_frames)

    for seq_idx in range(args.init_frame_index, total_end):
        frame_idx = seq_idx - args.init_frame_index

        color_path = color_paths[seq_idx]
        depth_path = depth_paths[seq_idx]

        frame = cv2.imread(color_path)
        depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if frame is None:
            print(f"[WARN] Cannot read color frame, skip: {color_path}")
            continue

        if depth_uint16 is None:
            print(f"[WARN] Cannot read depth frame, skip: {depth_path}")
            continue

        if frame.shape[:2] != (H0, W0):
            frame = cv2.resize(frame, (W0, H0))

        if depth_uint16.shape[:2] != (H0, W0):
            depth_uint16 = cv2.resize(
                depth_uint16,
                (W0, H0),
                interpolation=cv2.INTER_NEAREST,
            )

        timestamp_ms = int(1000.0 * frame_idx / max(1.0, float(args.output_fps)))

        hand_res = hand_tracker.detect(frame, timestamp_ms)

        object_results = scene_manager.update(frame, depth_uint16)
        relative_poses_mm = scene_manager.compute_relative_poses_mm(object_results)

        recorder.update(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            hand_res=hand_res,
            object_results=object_results,
            relative_poses_mm=relative_poses_mm,
        )

        vis = frame.copy()

        for name, obj_res in object_results.items():
            style = object_style.get(name, {})
            display_name = style.get("display_name", name)
            color_bgr = style.get("bbox_color_bgr", (0, 180, 255))

            tracker = scene_manager.trackers.get(name)
            estimator = tracker.estimator if tracker is not None else None
            corners_2d = None
            bbox3d_debug = {"projection_ok": False, "reason": "estimator_is_none", "visible": None}
            using_cached_3d_bbox = False
            if estimator is not None:
                corners_2d, bbox3d_debug = project_3d_bbox_corners_from_cTo_mm(
                    estimator=estimator,
                    cTo_mm=obj_res.cTo_mm,
                    image_shape=frame.shape,
                )

            if corners_2d is None and args.keep_last_valid_3d_bbox:
                cached_corners = last_valid_corners_by_name.get(name)
                cached_label_pt = last_valid_label_pt_by_name.get(name)
                if cached_corners is not None:
                    corners_2d = cached_corners.copy()
                    using_cached_3d_bbox = True
                    bbox3d_debug = dict(bbox3d_debug)
                    bbox3d_debug["using_cached"] = True
                    bbox3d_debug["cached_label_pt"] = cached_label_pt

            if corners_2d is not None and isinstance(corners_2d, np.ndarray) and corners_2d.shape == (8, 2):
                last_valid_corners_by_name[name] = corners_2d.copy()
                label_pt = tuple(np.mean(corners_2d.astype(np.float64), axis=0).tolist())
                last_valid_label_pt_by_name[name] = label_pt
            else:
                label_pt = last_valid_label_pt_by_name.get(name) if using_cached_3d_bbox else None

            if args.debug_3d_bbox:
                pose_t_dbg = None if obj_res.cTo_mm is None else obj_res.cTo_mm[:3, 3].tolist()
                print(
                    f"[3D_BBOX_DBG] name={name} success={obj_res.success} "
                    f"pose_t_mm={pose_t_dbg} proj_ok={bbox3d_debug.get('projection_ok')} "
                    f"reason={bbox3d_debug.get('reason')} using_cached={using_cached_3d_bbox} "
                    f"visible={bbox3d_debug.get('visible')}"
                )

            draw_3d_bbox(
                img=vis,
                corners_2d=corners_2d,
                color_bgr=color_bgr,
                thickness=int(args.bbox_thickness),
            )

            axis_pts_2d, axis_debug = project_pose_axes_from_cTo_mm(
                cTo_mm=obj_res.cTo_mm,
                K=K,
                axis_length_mm=float(args.axis_length_mm),
            )
            if args.debug_3d_bbox:
                print(
                    f"[AXIS_DBG] name={name} ok={axis_debug.get('projection_ok')} "
                    f"reason={axis_debug.get('reason')} axis_len_mm={args.axis_length_mm}"
                )
            draw_pose_axes(
                img=vis,
                axis_pts_2d=axis_pts_2d,
                thickness=max(1, int(args.axis_thickness)),
            )

            if label_pt is not None:
                label = f"{display_name}:{obj_res.track_source}"
                draw_label_on_point(
                    img=vis,
                    pt_xy=label_pt,
                    label=label,
                    color_bgr=color_bgr,
                )

        if args.draw_hand:
            draw_hand_result(
                img=vis,
                hand=hand_res.left_hand,
                label_prefix="L",
                color_bgr=(255, 255, 0),
            )
            draw_hand_result(
                img=vis,
                hand=hand_res.right_hand,
                label_prefix="R",
                color_bgr=(0, 255, 255),
            )

        if args.show_status:
            overlay_status(
                img=vis,
                frame_idx=frame_idx,
                frame_info=recorder.frames[-1],
            )

        save_path = os.path.join(vis_dir, f"{frame_idx:06d}.jpg")
        cv2.imwrite(save_path, vis)

        if writer is not None:
            writer.write(vis)

        if args.show:
            cv2.imshow("FoundationPose Refiner Tracking + Kalman", vis)
            key = cv2.waitKey(1)

            if key == 27 or key == ord("q"):
                print("[INFO] User stopped.")
                break

        print(
            f"[RUN] frame={frame_idx:06d}/{total_end - args.init_frame_index:06d} "
            f"timestamp_ms={timestamp_ms} "
            f"active={recorder.frames[-1].active_object_name} "
            f"phase={recorder.frames[-1].atomic_phase} "
            f"dominant={recorder.frames[-1].dominant_hand} "
            f"grasp={recorder.frames[-1].active_grasp_by_motion}"
        )

    if writer is not None:
        writer.release()

    if args.show:
        cv2.destroyAllWindows()

    export_details = recorder.export()

    # Qwen grasp-state preprocessing has already run before tracking.

    result = {
        "meta": {
            "record_root": args.record_root,
            "color_dir_name": args.color_dir_name,
            "depth_dir_name": args.depth_dir_name,
            "init_frame_index": args.init_frame_index,
            "init_color": init_color_path,
            "init_depth": init_depth_path,
            "objects_config": args.objects_config,
            "camera_K": K.tolist(),
            "tracking_mode": "FoundationPose refiner tracking + Kalman translation prediction + dual-hand human action parsing",
            "depth_note": "depth_aligned_frames are aligned to color frame, uint16, unit mm",
            "timestamp_note": "timestamp_ms generated from frame_idx to ensure MediaPipe monotonically increasing input",
            "strict_foundationpose_track": args.strict_foundationpose_track,
            "track_every_n_frames": args.track_every_n_frames,
            "kalman_process_var": args.kalman_process_var,
            "kalman_measure_var": args.kalman_measure_var,
            "rotation_smoothing_alpha": args.rotation_smoothing_alpha,
            "max_kalman_predict_frames": args.max_kalman_predict_frames,
            "human_hand_video_mode": True,
            "qwen_grasp_state_preprocess": {k: v for k, v in qwen_grasp_state_preprocess.items() if k != "dense_states"},
            "note": "This version parses human hand-object interaction, draws 3D projected cuboids and pose axes for objects, and removes 2D bbox visualization.",
        },
        "initialized_objects": {
            name: {
                "tpl_index": ret["tpl_index"],
                "init_source": ret["init_source"],
                **pose_mm_to_serializable(ret["cTo_mm"]),
                "track_pose_m": ndarray_to_list(ret["track_pose_m"]),
            }
            for name, ret in scene_manager.init_results.items()
        },
        "init_errors": scene_manager.init_errors,
        "details": export_details,
        "atomic_actions": export_details.get("atomic_actions", []),
        "qwen_grasp_state_preprocess": {k: v for k, v in qwen_grasp_state_preprocess.items() if k != "dense_states"},
    }

    json_path = os.path.join(
        args.output_dir,
        "multi_object_teaching_result_foundationpose_refiner_tracking.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(
        args.output_dir,
        "multi_object_teaching_result_foundationpose_refiner_tracking_flat.csv",
    )
    flat_rows = recorder.export_flat_object_rows()

    if len(flat_rows) > 0:
        fieldnames = list(flat_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(flat_rows)
    else:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "frame_idx", "timestamp_ms", "object_name",
                "x_mm", "y_mm", "z_mm",
                "qx", "qy", "qz", "qw",
                "roll_deg", "pitch_deg", "yaw_deg"
            ])

    print("=" * 80)
    print("[DONE] FoundationPose refiner tracking + Kalman pipeline finished")
    print(f"Result JSON: {json_path}")
    print(f"Result CSV: {csv_path}")
    print(f"Vis dir: {vis_dir}")

    if args.save_video:
        print(
            "Vis video: "
            f"{os.path.join(args.output_dir, 'teaching_vis_foundationpose_refiner_tracking_3d_bbox_3d_bbox.mp4')}"
        )

    print(f"Objects initialized: {list(scene_manager.init_results.keys())}")

    if scene_manager.init_errors:
        print(f"Objects failed: {scene_manager.init_errors}")

    print("=" * 80)


# ============================================================
# CLI
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-object teaching from recorded Orbbec RGB-D frames with FoundationPose refiner tracking + Kalman + dual-hand human action parsing + 3D bbox visualization"
    )

    parser.add_argument(
        "--record_root",
        type=str,
        default="/home/robot4/Programming/FoundationPose/video/5.22",
    )

    parser.add_argument(
        "--color_dir_name",
        type=str,
        default="color_frames",
    )

    parser.add_argument(
        "--depth_dir_name",
        type=str,
        default="depth_aligned_frames",
    )

    parser.add_argument(
        "--init_frame_index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--objects_config",
        type=str,
        default="/home/robot4/Programming/FoundationPose/video/5.22/objects_config_sam3_text_runtime.json",
    )

    parser.add_argument(
        "--hand_model",
        type=str,
        default="/home/robot4/Programming/FoundationPose/weights/hand_landmarker.task",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/robot4/Programming/FoundationPose/video/5.22/outputs/output_multi_teach_foundationpose_tracking",
    )

    parser.add_argument("--fx", type=float, default=365.99)
    parser.add_argument("--fy", type=float, default=366.10)
    parser.add_argument("--cx", type=float, default=320.91)
    parser.add_argument("--cy", type=float, default=240.51)

    parser.add_argument("--refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=1)

    parser.add_argument(
        "--track_every_n_frames",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--kalman_process_var",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--kalman_measure_var",
        type=float,
        default=80.0,
    )

    parser.add_argument(
        "--rotation_smoothing_alpha",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--max_kalman_predict_frames",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--use_track_roi",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--track_roi_expand_px", type=int, default=50)
    parser.add_argument(
        "--reject_bad_measurement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_translation_jump_mm", type=float, default=250.0)
    parser.add_argument("--min_valid_z_mm", type=float, default=100.0)
    parser.add_argument("--max_valid_z_mm", type=float, default=1200.0)
    parser.add_argument("--max_abs_xy_mm", type=float, default=800.0)
    parser.add_argument("--max_bbox_center_jump_px", type=float, default=180.0)
    parser.add_argument("--min_bbox_iou", type=float, default=0.0)

    parser.add_argument("--use_y180", action="store_true", default=True)

    parser.add_argument(
        "--strict_foundationpose_track",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--bbox_thickness", type=int, default=2)

    parser.add_argument(
        "--draw_hand",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--show_status",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
    )

    parser.add_argument("--min_hand_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_hand_presence_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

    parser.add_argument("--contact_distance_px", type=float, default=50.0)
    parser.add_argument("--motion_threshold_px", type=float, default=8.0)
    parser.add_argument("--contact_threshold", type=float, default=0.15)

    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--max_frames", type=int, default=0)

    parser.add_argument(
        "--debug_3d_bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--keep_last_valid_3d_bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--axis_length_mm", type=float, default=40.0)
    parser.add_argument("--axis_thickness", type=int, default=2)

    parser.add_argument(
        "--enable_qwen_grasp_state_preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在跟踪开始前调用 Ollama + Qwen 预处理视频，生成逐帧抓取状态",
    )
    parser.add_argument(
        "--strict_qwen_grasp_state_preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--qwen_grasp_state_video",
        type=str,
        default="/home/robot4/Programming/FoundationPose/video/5.22/color_video.mp4",
    )
    parser.add_argument(
        "--qwen_grasp_state_output_dir",
        type=str,
        default="",
    )
    parser.add_argument("--qwen_grasp_state_model", type=str, default="qwen3.5:35B")
    parser.add_argument("--qwen_grasp_state_ollama_base_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--qwen_grasp_state_stride", type=int, default=1)
    parser.add_argument("--qwen_grasp_state_chunk_size", type=int, default=8)
    parser.add_argument("--qwen_grasp_state_timeout_s", type=int, default=600)
    parser.add_argument("--qwen_grasp_state_jpeg_quality", type=int, default=88)
    parser.add_argument("--qwen_grasp_state_max_frames", type=int, default=0)
    parser.add_argument("--qwen_grasp_state_temperature", type=float, default=0.0)
    parser.add_argument("--qwen_grasp_state_confidence_threshold", type=float, default=0.35)

    parser.add_argument(
        "--save_video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_pipeline(args)