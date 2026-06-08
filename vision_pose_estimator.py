# =========================
# vision_pose_estimator.py
# 视觉识别与位姿估计模块
#
# 功能：
# 1. 调用粗分割模块，找到当前目标 mask 与最佳模板
# 2. 保存彩色图、深度图、mask、模板等中间结果
# 3. 基于模板位姿 + 当前深度，构造初始位姿
# 4. 调用 FoundationPose refiner 对单个位姿做优化
# 5. 保存最终位姿与可视化图
# 6. 支持作为模块被主程序导入，也支持单独运行测试
# =========================

# =========================
# 标准库 / 第三方库导入
# =========================
import os
import cv2
import time
import argparse
import numpy as np
import trimesh
import torch

# =========================
# FoundationPose 相关导入
# 说明：
# - estimater 中包含 FoundationPose、ScorePredictor、PoseRefinePredictor 等核心类
# - datareader 中包含一些深度处理/可视化工具函数
# =========================
from estimater import *
from datareader import *

# =========================
# 粗匹配与模板复制模块
# 说明：
# - ImageCoarseSegmentor：负责分割当前目标、匹配模板
# - copy_best_template_files：将最佳模板相关文件复制到输出目录
# =========================
# from ImageCoarseSegmentor_pointcloud import ImageCoarseSegmentor, copy_best_template_files
from ImageCoarseSegmentor_pointcloud_many import ImageCoarseSegmentor, copy_best_template_files


# ==========================================================
# 在图像上绘制三维坐标轴
# 说明：
# 给定位姿 pose 和内参 K，把物体局部坐标系的 xyz 三轴投影到图像上
# 常用于可视化最终位姿是否正确
# ==========================================================
def draw_axis_on_image(image: np.ndarray, K: np.ndarray, pose: np.ndarray,
                       axis_length: float = 0.05, line_width: int = 3) -> np.ndarray:
    """
    在输入图像上绘制物体局部坐标轴

    参数：
    - image: 输入 BGR 图像
    - K: 相机内参矩阵
    - pose: 4x4 物体位姿矩阵（物体坐标系 -> 相机坐标系）
    - axis_length: 坐标轴长度（通常单位与位姿平移一致）
    - line_width: 画线粗细

    返回：
    - vis: 绘制好坐标轴后的图像
    """
    # 拷贝原图，避免原地修改
    vis = image.copy()

    # 定义局部坐标系中的 4 个点：
    # 原点、x 轴终点、y 轴终点、z 轴终点
    axis_3d = np.array([
        [0.0, 0.0, 0.0],
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
    ], dtype=np.float32)

    # 从 4x4 位姿矩阵中提取旋转和平移
    R = pose[:3, :3].astype(np.float32)
    t = pose[:3, 3].astype(np.float32).reshape(3, 1)

    # OpenCV 的 projectPoints 需要 Rodrigues 旋转向量
    rvec, _ = cv2.Rodrigues(R)

    # 假设相机无畸变
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # 将 3D 坐标轴端点投影到 2D 图像平面
    pts_2d, _ = cv2.projectPoints(axis_3d, rvec, t, K.astype(np.float32), dist_coeffs)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    # 原点
    origin = tuple(pts_2d[0])

    # 绘制三轴：
    # x 轴红色，y 轴绿色，z 轴蓝色
    cv2.line(vis, origin, tuple(pts_2d[1]), (0, 0, 255), line_width)
    cv2.line(vis, origin, tuple(pts_2d[2]), (0, 255, 0), line_width)
    cv2.line(vis, origin, tuple(pts_2d[3]), (255, 0, 0), line_width)

    # 原点画一个黄色圆点
    cv2.circle(vis, origin, 5, (0, 255, 255), -1)

    return vis


# ==========================================================
# 在复制出来的模板文件列表中查找 pose_xxxx.npy
# 说明：
# 粗匹配后会把最佳模板相关文件复制到输出目录
# 这里用于找到其中对应的模板位姿文件
# ==========================================================
def find_pose_file(copied_files):
    """
    在 copied_files 中查找模板位姿文件 pose_xxxx.npy
    """
    for f in copied_files:
        name = os.path.basename(f)
        if name.startswith("pose_") and name.endswith(".npy"):
            return f
    return None


# ==========================================================
# 基于模板姿态构建初始位姿
# 说明：
# - 旋转：直接用模板位姿的旋转
# - 平移：根据当前深度图 + mask，由 FoundationPose 内部函数估计
# ==========================================================
def build_init_pose_from_template(estimator, template_pose, depth_m, mask_bool, K):
    """
    用模板位姿构造当前场景的初始位姿估计
    """
    # 初始化为单位阵
    init_pose = np.eye(4, dtype=np.float32)

    # 旋转部分直接采用模板姿态的旋转
    init_pose[:3, :3] = template_pose[:3, :3].astype(np.float32)

    # 平移部分通过当前深度图和 mask 估计
    init_pose[:3, 3] = estimator.guess_translation(
        depth=depth_m,
        mask=mask_bool,
        K=K
    ).astype(np.float32)

    return init_pose


# ==========================================================
# 使用 FoundationPose refiner 对单个位姿假设做优化
# 说明：
# 这里不是 register 全量搜索，而是：
# - 已经有一个初始位姿 init_pose_orig
# - 直接调用 refiner 进行若干次迭代优化
# 这对“模板先验较好”的场景更高效
# ==========================================================
def refine_single_pose_with_refiner(estimator, rgb, depth_m, K, init_pose_orig, iteration=4):
    """
    对单个初始位姿进行 refiner 优化

    参数：
    - estimator: FoundationPose 对象
    - rgb: RGB 图像
    - depth_m: 深度图（单位米）
    - K: 相机内参
    - init_pose_orig: 初始位姿（原始 mesh 坐标系）
    - iteration: refiner 迭代次数

    返回：
    - refined_pose_orig: 优化后的位姿（原始 mesh 坐标系）
    """
    # 如果 OpenGL/CUDA 光栅上下文还没创建，则创建
    if estimator.glctx is None:
        estimator.glctx = dr.RasterizeCudaContext()

    # 对深度图做预处理：
    # 1. 腐蚀，去掉边缘毛刺
    # 2. 双边滤波，抑制噪声同时保留边缘
    depth_m = erode_depth(depth_m, radius=2, device='cuda')
    depth_m = bilateral_filter_depth(depth_m, radius=2, device='cuda')

    # 将深度图转成 xyz map
    xyz_map = depth2xyzmap(depth_m, K)

    # 当前没有使用 normal_map
    normal_map = None

    # 获取从原始 mesh 到 centered mesh 的变换
    tf_to_centered = estimator.get_tf_to_centered_mesh().data.cpu().numpy()

    # 将初始位姿转换到 centered mesh 坐标系
    init_pose_centered = init_pose_orig @ np.linalg.inv(tf_to_centered)
    init_pose_centered = init_pose_centered.reshape(1, 4, 4).astype(np.float32)

    # 开始计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    # 调用 refiner 做位姿优化
    refined_pose_centered, vis = estimator.refiner.predict(
        mesh=estimator.mesh,
        mesh_tensors=estimator.mesh_tensors,
        rgb=rgb,
        depth=depth_m,
        K=K,
        ob_in_cams=init_pose_centered,
        normal_map=normal_map,
        xyz_map=xyz_map,
        glctx=estimator.glctx,
        mesh_diameter=estimator.diameter,
        iteration=iteration,
        get_vis=False
    )

    # 结束计时
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()

    print(f"[TIMING] Refiner耗时: {t1 - t0:.4f} s")

    # 将优化结果从 centered mesh 坐标系转换回原始 mesh 坐标系
    refined_pose_orig = (refined_pose_centered[0] @ estimator.get_tf_to_centered_mesh()).data.cpu().numpy()

    return refined_pose_orig


# ==========================================================
# 将位姿绕 y 轴额外旋转 180°
# 说明：
# 某些模板方向和实际抓取方向定义不一致时，会用这个函数做修正
# ==========================================================
def rotate_pose_y_180(pose: np.ndarray) -> np.ndarray:
    """
    对输入位姿的旋转部分绕 y 轴旋转 180°
    """
    # 绕 y 轴 180° 的旋转矩阵
    Ry_180 = np.array([
        [-1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [0.0,  0.0, -1.0]
    ], dtype=np.float64)

    # 拷贝一份原位姿
    pose_y180 = pose.copy()

    # 只更新旋转部分
    pose_y180[:3, :3] = pose[:3, :3] @ Ry_180

    return pose_y180


# ==========================================================
# 根据 mask 生成“白背景裁剪目标图”
# 说明：
# 目标区域保留原图，背景全部设为白色，再按外接框裁剪
# 主要用于后续给 Qwen/VLM 做分类输入
# ==========================================================
def make_white_bg_masked_rgb(frame_bgr: np.ndarray, mask_bool: np.ndarray, pad: int = 12) -> np.ndarray:
    """
    根据 mask 从原图中裁剪目标区域，并将背景设置为白色

    参数：
    - frame_bgr: 原始 BGR 图像
    - mask_bool: 目标 mask（bool）
    - pad: 裁剪边缘额外扩展像素

    返回：
    - cropped: 白背景裁剪图
    """
    # 输入合法性检查
    if frame_bgr is None or mask_bool is None:
        raise ValueError("输入图像或 mask 为空")

    # 找出 mask 内所有前景像素坐标
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("mask 为空，无法裁剪目标区域")

    # 创建全白背景图
    masked = np.full_like(frame_bgr, 255, dtype=np.uint8)

    # 将 mask 区域替换成原图对应区域
    masked[mask_bool] = frame_bgr[mask_bool]

    # 计算目标区域外接框，并适当加 padding
    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, frame_bgr.shape[1])
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad + 1, frame_bgr.shape[0])

    # 返回裁剪结果
    cropped = masked[y1:y2, x1:x2].copy()
    return cropped


# ==========================================================
# 对裁剪图做轻微增强
# 说明：
# 用于提升局部对比度和边缘细节，辅助后续 front/back 分类
# 这里特意保持“轻微增强”，避免过度失真
# ==========================================================
def enhance_masked_rgb_light(frame_bgr: np.ndarray) -> np.ndarray:
    """
    对白背景裁剪图做轻微增强：
    1. LAB + CLAHE 提升局部对比度
    2. 轻微锐化提升孔边缘、凸台边缘、刻字细节
    """
    if frame_bgr is None:
        raise ValueError("输入图像为空")

    # 复制原图
    img = frame_bgr.copy()

    # 转 LAB 色彩空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 对亮度通道 L 做 CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)

    # 合并回 LAB，再转回 BGR
    lab_enh = cv2.merge([l_enh, a, b])
    img_enh = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

    # 轻微锐化
    blur = cv2.GaussianBlur(img_enh, (0, 0), 1.0)
    sharpen = cv2.addWeighted(img_enh, 1.25, blur, -0.25, 0)

    return np.clip(sharpen, 0, 255).astype(np.uint8)


# ==========================================================
# 视觉位姿估计主类
# 说明：
# 该类负责：
# 1. 初始化 FoundationPose
# 2. 初始化模板粗匹配器
# 3. 保存各类中间结果
# 4. 完成一次“粗匹配 -> 初始位姿 -> refiner -> 保存可视化”的完整流程
# ==========================================================
class VisionPoseEstimator:
    def __init__(self, mesh_file, template_dir, save_root, manual_k):
        """
        初始化视觉位姿估计模块

        参数：
        - mesh_file: 目标 mesh 模型路径
        - template_dir: 模板库目录
        - save_root: 结果保存根目录
        - manual_k: 相机内参矩阵
        """
        # 保存配置
        self.mesh_file = mesh_file
        self.template_dir = template_dir
        self.save_root = save_root
        self.MANUAL_K = manual_k

        # 定义输出目录
        self.rgb_dir = os.path.join(save_root, "rgb")
        self.depth_dir = os.path.join(save_root, "depth")
        self.mask_dir = os.path.join(save_root, "mask")
        self.pose_dir = os.path.join(save_root, "ob_in_cam")
        self.vis_dir = os.path.join(save_root, "vis")
        self.tpl_dir = os.path.join(save_root, "matched_templates")

        # 创建输出目录
        for d in [self.rgb_dir, self.depth_dir, self.mask_dir, self.pose_dir, self.vis_dir, self.tpl_dir]:
            os.makedirs(d, exist_ok=True)

        print("[INIT] 初始化 FoundationPose...")

        # 加载 mesh
        mesh = trimesh.load(self.mesh_file)
        self.mesh = mesh

        # 计算 mesh 的有向包围盒
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)

        # 初始化评分器和位姿优化器
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()

        # 创建 GPU 光栅上下文
        self.glctx = dr.RasterizeCudaContext()

        # 创建 FoundationPose 主对象
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug=0,
            debug_dir="./debug",
            glctx=self.glctx
        )

        print("[INIT] 初始化模板粗匹配...")

        # 初始化粗匹配模块
        self.segmentor = ImageCoarseSegmentor(template_dir=self.template_dir)

        # 将相机内参赋给粗匹配器
        self.segmentor.intrinsic = self.MANUAL_K.copy()

        # 当前保存索引
        self.save_idx = 0

    # def estimate_once(self, frame_bgr, depth_uint16, refine_iter=5, use_y180=True):
    #     """
    #     执行一次完整的视觉位姿估计
    #
    #     参数：
    #     - frame_bgr: 当前彩色图（BGR）
    #     - depth_uint16: 当前深度图（uint16，通常单位 mm）
    #     - refine_iter: refiner 迭代次数
    #     - use_y180: 是否对最终位姿绕 y 轴补 180°
    #
    #     返回：
    #     - 包含位姿、mask、可视化、路径等信息的字典
    #     """
    #     # 当前文件编号
    #     fname = f"{self.save_idx:06d}"
    #
    #     # 总耗时起点
    #     t0 = time.time()
    #
    #     # 保存原始彩色图与深度图
    #     color_path = os.path.join(self.rgb_dir, f"{fname}.png")
    #     depth_path = os.path.join(self.depth_dir, f"{fname}.png")
    #     cv2.imwrite(color_path, frame_bgr)
    #     cv2.imwrite(depth_path, depth_uint16.astype(np.uint16))
    #
    #     # BGR 转 RGB，供 FoundationPose 使用
    #     rgb_input = frame_bgr[..., ::-1].copy()
    #
    #     # 深度图从 mm 转成 m
    #     depth_m = depth_uint16.astype(np.float32) / 1000.0
    #
    #     # 把很小的深度值清零，视作无效
    #     depth_m[depth_m < 0.001] = 0.0
    #
    #     # =========================
    #     # 第一步：粗匹配与分割
    #     # =========================
    #     t_match0 = time.time()
    #     best_mask, best_tpl_mask, tpl_pose6, tpl_index,_,_,_ = self.segmentor.find_part_mask_dino(
    #         image_bgr=frame_bgr,
    #         image_depth=depth_uint16.copy()
    #     )
    #     t_match1 = time.time()
    #     print(f"[TIMING] 粗匹配耗时: {t_match1 - t_match0:.4f} s")
    #
    #     # 没找到目标则报错
    #     if best_mask is None:
    #         raise RuntimeError("未找到有效目标")
    #
    #     # 转成 bool mask
    #     current_mask = best_mask.astype(bool)
    #
    #     # mask 太小说明分割不可靠
    #     if current_mask.sum() < 10:
    #         raise RuntimeError("mask太小")
    #
    #     # 保存 mask 图
    #     cv2.imwrite(os.path.join(self.mask_dir, f"{fname}.png"), current_mask.astype(np.uint8) * 255)
    #
    #     # =========================
    #     # 第二步：生成给 Qwen/VLM 用的裁剪图
    #     # =========================
    #     masked_rgb = make_white_bg_masked_rgb(frame_bgr, current_mask, pad=12)
    #     masked_rgb_path = os.path.join(self.rgb_dir, f"{fname}_masked_crop_white.png")
    #     cv2.imwrite(masked_rgb_path, masked_rgb)
    #
    #     # 再生成轻微增强版
    #     masked_rgb_enh = enhance_masked_rgb_light(masked_rgb)
    #     masked_rgb_enh_path = os.path.join(self.rgb_dir, f"{fname}_masked_crop_white_enh.png")
    #     cv2.imwrite(masked_rgb_enh_path, masked_rgb_enh)
    #
    #     # =========================
    #     # 第三步：复制最佳模板文件
    #     # =========================
    #     current_tpl_save_dir = os.path.join(self.tpl_dir, f"{fname}_tpl_{tpl_index:04d}")
    #     copied_files = copy_best_template_files(
    #         template_dir=self.template_dir,
    #         tpl_index=tpl_index,
    #         output_dir=current_tpl_save_dir,
    #         use_subfolder=True
    #     )
    #
    #     # 找到模板位姿文件
    #     pose_path = find_pose_file(copied_files)
    #     if pose_path is None:
    #         raise RuntimeError("没找到pose_xxxx.npy")
    #
    #     # 读取模板位姿
    #     template_pose = np.load(pose_path).astype(np.float32)
    #
    #     # =========================
    #     # 第四步：构造初始位姿
    #     # =========================
    #     init_pose = build_init_pose_from_template(
    #         estimator=self.est,
    #         template_pose=template_pose,
    #         depth_m=depth_m,
    #         mask_bool=current_mask,
    #         K=self.MANUAL_K
    #     )
    #
    #     # =========================
    #     # 第五步：调用 refiner 优化位姿
    #     # =========================
    #     refined_pose = refine_single_pose_with_refiner(
    #         estimator=self.est,
    #         rgb=rgb_input,
    #         depth_m=depth_m,
    #         K=self.MANUAL_K,
    #         init_pose_orig=init_pose,
    #         iteration=refine_iter
    #     )
    #
    #     # =========================
    #     # 第六步：是否补一个绕 y 轴 180° 的旋转
    #     # =========================
    #     if use_y180:
    #         final_pose = rotate_pose_y_180(refined_pose)
    #         cTo = final_pose
    #         cTo_mm = cTo.copy()
    #         cTo_mm[:3, 3] *= 1000.0
    #     else:
    #         final_pose = refined_pose
    #         cTo = final_pose
    #         cTo_mm = cTo.copy()
    #         cTo_mm[:3, 3] *= 1000.0
    #
    #     # 保存最终位姿矩阵
    #     np.savetxt(os.path.join(self.pose_dir, f"{fname}_final.txt"), final_pose.reshape(4, 4))
    #
    #     # =========================
    #     # 第七步：生成可视化图
    #     # =========================
    #     vis = frame_bgr.copy()
    #
    #     # 将 mask 区域叠加为绿色半透明
    #     green_overlay = np.zeros_like(vis)
    #     green_overlay[current_mask] = [0, 255, 0]
    #     vis = cv2.addWeighted(vis, 1.0, green_overlay, 0.3, 0)
    #
    #     # 画局部坐标轴
    #     vis = draw_axis_on_image(vis, self.MANUAL_K, final_pose, axis_length=0.15, line_width=4)
    #
    #     # 画 3D 包围盒
    #     center_pose = final_pose @ np.linalg.inv(self.to_origin)
    #     vis = draw_posed_3d_box(self.MANUAL_K, img=vis, ob_in_cam=center_pose, bbox=self.bbox)
    #
    #     # 叠加模板编号
    #     cv2.putText(vis, f"tpl_idx: {tpl_index}", (20, 35),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    #
    #     # 保存可视化图
    #     cv2.imwrite(os.path.join(self.vis_dir, f"{fname}.png"), vis)
    #
    #     # 打印本次总耗时
    #     total_t = time.time() - t0
    #     print(f"[TIMING] 单次视觉估计总耗时: {total_t:.4f} s")
    #
    #     # 保存编号自增
    #     self.save_idx += 1
    #
    #     # 返回本次估计结果
    #     return {
    #         "cTo": cTo_mm,
    #         "init_pose": init_pose,
    #         "refined_pose": refined_pose,
    #         "tpl_index": tpl_index,
    #         "mask": current_mask,
    #         "vis": vis,
    #         "color_path": color_path,
    #         "depth_path": depth_path,
    #         "masked_rgb": masked_rgb,
    #         "masked_rgb_path": masked_rgb_path,
    #         "masked_rgb_enh": masked_rgb_enh,
    #         "masked_rgb_enh_path": masked_rgb_enh_path,
    #     }

    def estimate_once(self, frame_bgr, depth_uint16, refine_iter=5, use_y180=True):
        """
        执行一次完整的视觉位姿估计

        参数：
        - frame_bgr: 当前彩色图（BGR）
        - depth_uint16: 当前深度图（uint16，通常单位 mm）
        - refine_iter: refiner 迭代次数
        - use_y180: 是否对最终位姿绕 y 轴补 180°

        返回：
        - 成功时：返回包含位姿、mask、可视化、路径等信息的字典
        - 失败时：返回 None
        """
        try:
            # 当前文件编号
            fname = f"{self.save_idx:06d}"

            # 总耗时起点
            t0 = time.time()

            # -------------------------
            # 0. 输入检查
            # -------------------------
            if frame_bgr is None:
                print("[视觉] 输入错误：frame_bgr 为 None")
                return None

            if depth_uint16 is None:
                print("[视觉] 输入错误：depth_uint16 为 None")
                return None

            if not isinstance(frame_bgr, np.ndarray):
                print(f"[视觉] 输入错误：frame_bgr 类型异常，type = {type(frame_bgr)}")
                return None

            if not isinstance(depth_uint16, np.ndarray):
                print(f"[视觉] 输入错误：depth_uint16 类型异常，type = {type(depth_uint16)}")
                return None

            if frame_bgr.size == 0:
                print("[视觉] 输入错误：frame_bgr 为空数组")
                return None

            if depth_uint16.size == 0:
                print("[视觉] 输入错误：depth_uint16 为空数组")
                return None

            # -------------------------
            # 1. 保存原始彩色图与深度图
            # -------------------------
            color_path = os.path.join(self.rgb_dir, f"{fname}.png")
            depth_path = os.path.join(self.depth_dir, f"{fname}.png")

            try:
                cv2.imwrite(color_path, frame_bgr)
                cv2.imwrite(depth_path, depth_uint16.astype(np.uint16))
            except Exception as e:
                print(f"[视觉] 保存原始图像失败：{e}")
                return None

            # BGR 转 RGB，供 FoundationPose 使用
            rgb_input = frame_bgr[..., ::-1].copy()

            # 深度图从 mm 转成 m
            depth_m = depth_uint16.astype(np.float32) / 1000.0

            # 把很小的深度值清零，视作无效
            depth_m[depth_m < 0.001] = 0.0

            # =========================
            # 第一步：粗匹配与分割
            # =========================
            t_match0 = time.time()

            try:
                match_ret = self.segmentor.find_part_mask_dino(
                    image_bgr=frame_bgr,
                    image_depth=depth_uint16.copy()
                )
            except Exception as e:
                print(f"[视觉] 粗匹配异常：{e}")
                return None

            t_match1 = time.time()
            print(f"[TIMING] 粗匹配耗时: {t_match1 - t_match0:.4f} s")

            if match_ret is None:
                print("[视觉] 粗匹配失败：find_part_mask_dino 返回 None，可能是 SAM3 候选为空")
                return None

            if not isinstance(match_ret, (list, tuple)):
                print(f"[视觉] 粗匹配失败：返回类型异常，type = {type(match_ret)}")
                return None

            if len(match_ret) < 4:
                print(f"[视觉] 粗匹配失败：返回内容长度不足，len = {len(match_ret)}")
                return None

            try:
                best_mask, best_tpl_mask, tpl_pose6, tpl_index, *others = match_ret
            except Exception as e:
                print(f"[视觉] 粗匹配失败：结果解包异常：{e}")
                return None

            if best_mask is None:
                print("[视觉] 粗匹配失败：best_mask 为 None")
                return None

            if tpl_index is None:
                print("[视觉] 粗匹配失败：tpl_index 为 None")
                return None

            try:
                current_mask = best_mask.astype(bool)
            except Exception as e:
                print(f"[视觉] mask 转 bool 失败：{e}")
                return None

            mask_area = int(current_mask.sum())
            if mask_area < 10:
                print(f"[视觉] 粗匹配失败：mask 太小，前景像素数 = {mask_area}")
                return None

            # 保存 mask 图
            try:
                cv2.imwrite(
                    os.path.join(self.mask_dir, f"{fname}.png"),
                    current_mask.astype(np.uint8) * 255
                )
            except Exception as e:
                print(f"[视觉] 保存 mask 图失败：{e}")
                return None

            # =========================
            # 第二步：生成给 Qwen/VLM 用的裁剪图
            # =========================
            try:
                masked_rgb = make_white_bg_masked_rgb(frame_bgr, current_mask, pad=12)
                masked_rgb_path = os.path.join(self.rgb_dir, f"{fname}_masked_crop_white.png")
                cv2.imwrite(masked_rgb_path, masked_rgb)
            except Exception as e:
                print(f"[视觉] 生成白背景裁剪图失败：{e}")
                return None

            try:
                masked_rgb_enh = enhance_masked_rgb_light(masked_rgb)
                masked_rgb_enh_path = os.path.join(self.rgb_dir, f"{fname}_masked_crop_white_enh.png")
                cv2.imwrite(masked_rgb_enh_path, masked_rgb_enh)
            except Exception as e:
                print(f"[视觉] 生成增强裁剪图失败：{e}")
                return None

            # =========================
            # 第三步：复制最佳模板文件
            # =========================
            try:
                current_tpl_save_dir = os.path.join(self.tpl_dir, f"{fname}_tpl_{int(tpl_index):04d}")
                copied_files = copy_best_template_files(
                    template_dir=self.template_dir,
                    tpl_index=tpl_index,
                    output_dir=current_tpl_save_dir,
                    use_subfolder=True
                )
            except Exception as e:
                print(f"[视觉] 复制最佳模板文件失败：{e}")
                return None

            if copied_files is None or len(copied_files) == 0:
                print("[视觉] 复制最佳模板文件失败：copied_files 为空")
                return None

            # 找到模板位姿文件
            pose_path = find_pose_file(copied_files)
            if pose_path is None:
                print("[视觉] 模板文件异常：没找到 pose_xxxx.npy")
                return None

            if not os.path.exists(pose_path):
                print(f"[视觉] 模板位姿文件不存在：{pose_path}")
                return None

            # 读取模板位姿
            try:
                template_pose = np.load(pose_path).astype(np.float32)
            except Exception as e:
                print(f"[视觉] 读取模板位姿文件失败：{e}")
                return None

            if template_pose.shape != (4, 4):
                print(f"[视觉] 模板位姿格式异常：shape = {template_pose.shape}，期望 (4, 4)")
                return None

            # =========================
            # 第四步：构造初始位姿
            # =========================
            try:
                init_pose = build_init_pose_from_template(
                    estimator=self.est,
                    template_pose=template_pose,
                    depth_m=depth_m,
                    mask_bool=current_mask,
                    K=self.MANUAL_K
                )
            except Exception as e:
                print(f"[视觉] 构造初始位姿失败：{e}")
                return None

            if init_pose is None:
                print("[视觉] 构造初始位姿失败：init_pose 为 None")
                return None

            if getattr(init_pose, "shape", None) != (4, 4):
                print(f"[视觉] 初始位姿格式异常：shape = {getattr(init_pose, 'shape', None)}")
                return None

            # =========================
            # 第五步：调用 refiner 优化位姿
            # =========================
            try:
                refined_pose = refine_single_pose_with_refiner(
                    estimator=self.est,
                    rgb=rgb_input,
                    depth_m=depth_m,
                    K=self.MANUAL_K,
                    init_pose_orig=init_pose,
                    iteration=refine_iter
                )
            except Exception as e:
                print(f"[视觉] Refiner 优化失败：{e}")
                return None

            if refined_pose is None:
                print("[视觉] Refiner 优化失败：refined_pose 为 None")
                return None

            if getattr(refined_pose, "shape", None) != (4, 4):
                print(f"[视觉] 优化后位姿格式异常：shape = {getattr(refined_pose, 'shape', None)}")
                return None

            # =========================
            # 第六步：是否补一个绕 y 轴 180° 的旋转
            # =========================
            try:
                if use_y180:
                    final_pose = rotate_pose_y_180(refined_pose)
                else:
                    final_pose = refined_pose

                cTo = final_pose
                cTo_mm = cTo.copy()
                cTo_mm[:3, 3] *= 1000.0
            except Exception as e:
                print(f"[视觉] 最终位姿处理失败：{e}")
                return None

            # 保存最终位姿矩阵
            try:
                np.savetxt(os.path.join(self.pose_dir, f"{fname}_final.txt"), final_pose.reshape(4, 4))
            except Exception as e:
                print(f"[视觉] 保存最终位姿失败：{e}")
                return None

            # =========================
            # 第七步：生成可视化图
            # =========================
            try:
                vis = frame_bgr.copy()

                # 将 mask 区域叠加为绿色半透明
                green_overlay = np.zeros_like(vis)
                green_overlay[current_mask] = [0, 255, 0]
                vis = cv2.addWeighted(vis, 1.0, green_overlay, 0.3, 0)

                # 画局部坐标轴
                vis = draw_axis_on_image(vis, self.MANUAL_K, final_pose, axis_length=0.15, line_width=4)

                # 画 3D 包围盒
                center_pose = final_pose @ np.linalg.inv(self.to_origin)
                vis = draw_posed_3d_box(self.MANUAL_K, img=vis, ob_in_cam=center_pose, bbox=self.bbox)

                # 叠加模板编号
                cv2.putText(
                    vis,
                    f"tpl_idx: {tpl_index}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv2.imwrite(os.path.join(self.vis_dir, f"{fname}.png"), vis)
            except Exception as e:
                print(f"[视觉] 生成或保存可视化图失败：{e}")
                return None

            # 打印本次总耗时
            total_t = time.time() - t0
            print(f"[TIMING] 单次视觉估计总耗时: {total_t:.4f} s")

            # 保存编号自增
            self.save_idx += 1

            # 返回本次估计结果
            return {
                "cTo": cTo_mm,
                "init_pose": init_pose,
                "refined_pose": refined_pose,
                "tpl_index": tpl_index,
                "mask": current_mask,
                "vis": vis,
                "color_path": color_path,
                "depth_path": depth_path,
                "masked_rgb": masked_rgb,
                "masked_rgb_path": masked_rgb_path,
                "masked_rgb_enh": masked_rgb_enh,
                "masked_rgb_enh_path": masked_rgb_enh_path,
            }

        except Exception as e:
            print(f"[视觉] estimate_once 总异常：{e}")
            import traceback
            print(traceback.format_exc())
            return None


# ==========================================================
# 主程序测试入口
# ==========================================================
if __name__ == "__main__":

    # mesh 模型路径
    mesh_file = "/home/ma/FoundationPose2.0/demo_data_pian/my_data0/mesh/pian_hole_m.obj"

    # 模板库路径
    template_dir = "/home/ma/FoundationPose2.0/templates1280*720/back"

    # 输出目录
    save_root = "/home/ma/FoundationPose2.0/vision_test_output"

    # 测试彩色图路径
    color_path = "/home/ma/test/color.png"

    # 测试深度图路径
    depth_path = "/home/ma/test/depth.png"

    # refiner 迭代次数
    refine_iter = 6

    # 是否启用 y 轴 180° 修正
    use_y180 = True

    # 相机内参
    fx = 609.99963379
    fy = 610.17034912
    cx = 641.85406494
    cy = 360.86437988

    # 组装相机内参矩阵
    MANUAL_K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    # =========================
    # 2. 打印当前测试配置
    # =========================
    print("=" * 60)
    print("[TEST] 当前测试配置")
    print(f"mesh_file    = {mesh_file}")
    print(f"template_dir = {template_dir}")
    print(f"save_root    = {save_root}")
    print(f"color_path   = {color_path}")
    print(f"depth_path   = {depth_path}")
    print(f"refine_iter  = {refine_iter}")
    print(f"use_y180     = {use_y180}")
    print("MANUAL_K =")
    print(MANUAL_K)
    print("=" * 60)

    # =========================
    # 3. 创建输出目录
    # =========================
    os.makedirs(save_root, exist_ok=True)

    # =========================
    # 4. 读取测试彩色图
    # =========================
    frame_bgr = cv2.imread(color_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"无法读取彩色图: {color_path}")

    # =========================
    # 5. 读取测试深度图
    # =========================
    depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_uint16 is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")

    # =========================
    # 6. 初始化视觉位姿估计模块
    # =========================
    estimator = VisionPoseEstimator(
        mesh_file=mesh_file,
        template_dir=template_dir,
        save_root=save_root,
        manual_k=MANUAL_K
    )

    # =========================
    # 7. 执行一次视觉位姿估计
    # =========================
    result = estimator.estimate_once(
        frame_bgr=frame_bgr,
        depth_uint16=depth_uint16,
        refine_iter=refine_iter,
        use_y180=use_y180
    )

    # =========================
    # 8. 打印最终结果
    # =========================
    print("=" * 60)
    print("[FINAL RESULT]")
    print(f"tpl_index            : {result['tpl_index']}")
    print(f"color_path           : {result['color_path']}")
    print(f"depth_path           : {result['depth_path']}")
    print(f"masked_rgb_path      : {result['masked_rgb_path']}")
    print(f"masked_rgb_enh_path  : {result['masked_rgb_enh_path']}")
    print("cTo(mm) =")
    print(result["cTo"])
    print("=" * 60)

    # =========================
    # 9. 显示最终可视化结果
    # =========================
    if result.get("vis", None) is not None:
        cv2.imshow("vision_pose_result", result["vis"])
        print("[INFO] 按任意键关闭结果窗口")
        cv2.waitKey(0)
        cv2.destroyAllWindows()