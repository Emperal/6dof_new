# 实现一个可微分的 3D 渲染网络，用于目标物体的 6D 姿态估计与模板生成。
# 输入：3D 模型 (.obj)，相机参数，姿态参数。
# 输出：渲染的 RGB、mask、点云、姿态矩阵、姿态库 JSON。
# 用真实图像的 mask 来优化物体的 6D 姿态；
# 批量生成模板库，为后续的匹配、识别、跟踪提供数据。
import numpy as np
import torch
import cv2
import open3d as o3d
import os
# io utils
from pytorch3d.io import load_obj
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
# datastructures
from pytorch3d.structures import Meshes
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer,
    MeshRasterizer, BlendParams,
    HardPhongShader, PointLights,
    PerspectiveCameras, TexturesVertex,
    SoftSilhouetteShader
)
from tqdm import tqdm


# ---------------------- DiffRenderNet ----------------------
class DiffRenderNetNT(nn.Module):
    def __init__(self, device, cam_parameters, rad_deg="deg", mm_m="mm", ):
        super().__init__()

        self.deg2rad = (rad_deg == "deg")
        self.mm2m = not (mm_m == "mm")

        fx, fy = cam_parameters[0], cam_parameters[1]
        cx, cy = cam_parameters[2], cam_parameters[3]
        H, W = cam_parameters[4], cam_parameters[5]
        self.create_renderers(H, W, fx, fy, cx, cy, device)
        self.device = device

        self.target_image = None
        self.mask_real = None
        self.hand_mask = None  # 新增：可选的手部/遮挡掩码

    def create_renderers(self, H, W, fx, fy, cx, cy, device):
        R_cam = torch.eye(3).unsqueeze(0).to(device)
        T_cam = torch.zeros(1, 3).to(device)
        self.cameras = PerspectiveCameras(
            device=device,
            R=R_cam, T=T_cam,
            focal_length=torch.tensor([[fx, fy]], device=device),
            principal_point=torch.tensor([[cx, cy]], device=device),
            in_ndc=False,
            image_size=torch.tensor([[H, W]], device=device)
        )
        lights = PointLights(device=device, location=[[0.0, 200.0, 200.0]])
        raster_settings_color = RasterizationSettings(image_size=(H, W), blur_radius=1e-8, faces_per_pixel=1)
        self.hardphong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings_color),
            shader=HardPhongShader(device=device, cameras=self.cameras, lights=lights,
                                   blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)))
        )
        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=RasterizationSettings(
                    image_size=(H, W),
                    blur_radius=1e-8,
                    faces_per_pixel=1,
                    bin_size=None
                )
            ),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-7, gamma=1e-7))
        )

    def load_target_mesh(self, mesh_path, init_euler=None, init_trans=None):
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        self.mesh = mesh
        if init_euler is not None:
            self.euler = nn.Parameter(init_euler.clone().detach().to(self.device))
        else:
            self.euler = nn.Parameter(torch.zeros(3, device=self.device))
        if init_trans is not None:
            self.trans = nn.Parameter(init_trans.clone().detach().to(self.device))
        else:
            self.trans = nn.Parameter(torch.zeros(3, device=self.device))

    def set_target_mesh(self, mesh, init_euler=None, init_trans=None):
        self.mesh = mesh.clone()
        if init_euler is not None:
            self.euler = nn.Parameter(init_euler.clone().detach().to(self.device))
        else:
            self.euler = nn.Parameter(torch.zeros(3, device=self.device))
        if init_trans is not None:
            self.trans = nn.Parameter(init_trans.clone().detach().to(self.device))
        else:
            self.trans = nn.Parameter(torch.zeros(3, device=self.device))

    # --------- 支持 hand_mask ---------
    def set_target_image(self, target_image, mask_real, hand_mask=None):
        """
        target_image: (H,W) 或 (H,W,3) 的 torch.Tensor
        mask_real:    (H,W) 的 torch.Tensor，值域[0,1]
        hand_mask:    (H,W) 的 torch.Tensor，值域[0,1]，1 表示手/遮挡（可选）
        """
        self.target_image = target_image
        self.mask_real = mask_real

        # hand_mask（可能为 None）
        if hand_mask is None:
            self.hand_mask = None
        else:
            if not torch.is_tensor(hand_mask):
                self.hand_mask = torch.tensor(hand_mask, dtype=torch.float32)
            self.hand_mask = hand_mask.to(mask_real.device).float()

            # 尺寸不一致时，做一次最近邻缩放
            if hand_mask.shape != mask_real.shape:
                hm_np = hand_mask.detach().cpu().numpy()
                H, W = mask_real.shape[-2], mask_real.shape[-1]
                hm_np = cv2.resize(hm_np, (W, H), interpolation=cv2.INTER_NEAREST)
                hand_mask = torch.tensor(hm_np, dtype=torch.float32, device=mask_real.device)

            # 保证在[0,1]
            self.hand_mask = hand_mask.clamp(0.0, 1.0)

    def set_init_pose(self, init_euler, init_trans):
        self.euler.data = init_euler
        self.trans.data = init_trans

    def depth_to_colormap(self, depth, mask):
        """
        输入:
        depth: (H, W) numpy 数组, 深度图
        mask: (H, W) bool 数组, 掩码
        输出:
        color_map: (H, W, 3) uint8, 伪彩色图
        """

        # 拷贝一个输出
        norm_depth = np.zeros_like(depth, dtype=np.float32)

        # 提取掩码区域深度
        masked_depth = depth[mask]

        if len(masked_depth) > 0:
            # min-max 归一化到 [0,1]
            d_min, d_max = masked_depth.min(), masked_depth.max()
            if d_max > d_min:  # 避免除零
                norm = (masked_depth - d_min) / (d_max - d_min)
            else:
                norm = np.zeros_like(masked_depth)

            # 填回原图
            norm_depth[mask] = norm

        import matplotlib.pyplot as plt

        # # depth_norm 是 [0,1] 的 float
        # plt.imshow(norm_depth, cmap="turbo")   # 也可以 'viridis', 'plasma'
        # plt.colorbar()
        # plt.show()
        # 转成 0~255
        norm_depth_uint8 = (norm_depth * 255).astype(np.uint8)

        # OpenCV 伪彩色 (COLORMAP_JET 常用)
        color_map = cv2.applyColorMap(norm_depth_uint8, cv2.COLORMAP_VIRIDIS)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)  # BGR->RGB

        # cmap = cm.get_cmap('turbo')  # 或 'viridis'
        # color_map = cmap(norm_depth)[:, :, :3]  # RGBA -> RGB
        # color_map = (color_map * 255).astype(np.uint8)

        color_map[~mask] = 0
        return color_map

    def forward(self):
        if self.deg2rad:
            R_obj = euler_angles_to_matrix(self.euler * torch.pi / 180.0, convention="XYZ")
        else:
            R_obj = euler_angles_to_matrix(self.euler, convention="XYZ")

        if self.mm2m:
            T_obj = self.trans * 0.001
        else:
            T_obj = self.trans

        verts = self.mesh.verts_packed().clone()
        verts_trans = verts @ R_obj.T + T_obj
        mesh_transformed = Meshes(
            verts=[verts_trans],
            faces=[self.mesh.faces_packed().clone()],
            textures=self.mesh.textures.clone()
        )

        image = self.hardphong_renderer(mesh_transformed)
        pred_rgb = image[0, ..., :3]

        s_image = self.silhouette_renderer(mesh_transformed)
        alpha_pred = s_image[0, ..., 3]
        self.last_rendered = image[0, ..., :3].detach().cpu().numpy()
        # cv2.imshow("rendered", (image[0, ..., 3].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.waitKey(0)

        mask_gt = self.mask_real

        # ---------- 按手部遮挡权重可见区 ----------
        if self.hand_mask is not None:
            visible_w = (1.0 - self.hand_mask).clamp(0.0, 1.0)
            vis_area = visible_w.sum()
            if vis_area.item() < 1.0:
                visible_w = torch.ones_like(mask_gt)
        else:
            visible_w = torch.ones_like(mask_gt)

        diff_sil = (alpha_pred - mask_gt) ** 2
        loss_silhouette = (diff_sil * visible_w).sum() / (visible_w.sum() + 1e-6)

        cx_p, cy_p = self.mask_centroid_torch(alpha_pred * visible_w)
        cx_r, cy_r = self.mask_centroid_torch(mask_gt * visible_w)
        loss_center = torch.sqrt((cx_p - cx_r) ** 2 + (cy_p - cy_r) ** 2)

        loss_iou = self.iou(alpha_pred, mask_gt, weight=visible_w)

        loss = (1000.0 * loss_silhouette
                + 0.5 * loss_center
                - 10.0 * loss_iou)
        # + 10.0 * loss_size)

        print(f"loss_silhouette: {loss_silhouette.item():.4f} | "
              f"loss_center: {loss_center:.4f} | "
              f"loss_iou: {loss_iou:.4f} | ")

        return loss, alpha_pred

    def iou(self, mask1, mask2, weight=None):
        if weight is None:
            # 原始全图 IoU
            intersection = mask1 * mask2
            union = torch.clamp(mask1 + mask2, 0, 1)
            intersection_area = torch.sum(intersection)
            union_area = torch.sum(union)
            return intersection_area / (union_area + 1e-6)
        else:
            # 在可见区域内计算 IoU
            m1 = (mask1 * weight).clamp(0, 1)
            m2 = (mask2 * weight).clamp(0, 1)
            intersection = (m1 * m2).sum()
            union = torch.clamp(m1 + m2, 0, 1).sum()
            return intersection / (union + 1e-6)

    def mask_centroid_torch(self, mask):
        mask_bin = mask
        total = mask_bin.sum()
        if total == 0:
            h, w = mask_bin.shape
            return torch.tensor(w // 2, device=mask.device, dtype=torch.float32), \
                torch.tensor(h // 2, device=mask.device, dtype=torch.float32)
        ys = torch.arange(mask_bin.shape[0], device=mask.device).view(-1, 1).float()
        xs = torch.arange(mask_bin.shape[1], device=mask.device).view(1, -1).float()
        cx = (mask_bin * xs).sum() / total
        cy = (mask_bin * ys).sum() / total
        return cx, cy

    def pose6_to_matrix44(self, pose6):
        pose_euler = pose6[0:3]
        pose_trans = pose6[3:6]
        if self.deg2rad:
            R_obj = euler_angles_to_matrix(torch.tensor(pose_euler, device=self.device) * torch.pi / 180.0,
                                           convention="XYZ")
        else:
            R_obj = euler_angles_to_matrix(torch.tensor(pose_euler, device=self.device), convention="XYZ")

        if self.mm2m:
            T_obj = torch.tensor(pose_trans, device=self.device) * 0.001
        else:
            T_obj = torch.tensor(pose_trans, device=self.device)

        T = np.identity(4)
        T[0:3, 0:3] = R_obj.detach().cpu().numpy()
        T[0:3, 3] = T_obj.detach().cpu().numpy()
        return T

    def matrix44_to_pose6(self, T):
        pose_euler = matrix_to_euler_angles(torch.tensor(T[0:3, 0:3], device=self.device),
                                            convention="XYZ").detach().cpu().numpy()
        pose_euler = np.rad2deg(pose_euler)
        pose_trans = T[0:3, 3]

        pose6 = pose_euler.tolist() + pose_trans.tolist()
        return pose6

    def render_hardphone_and_silhouette_images(self, pose_euler, pose_trans):
        if self.deg2rad:
            R_obj = euler_angles_to_matrix(torch.tensor(pose_euler, device=self.device) * torch.pi / 180.0,
                                           convention="XYZ")
        else:
            R_obj = euler_angles_to_matrix(torch.tensor(pose_euler, device=self.device), convention="XYZ")

        if self.mm2m:
            T_obj = torch.tensor(pose_trans, device=self.device) * 0.001
        else:
            T_obj = torch.tensor(pose_trans, device=self.device)

        verts = self.mesh.verts_packed().clone()
        verts_trans = verts @ R_obj.T + T_obj
        mesh_transformed = Meshes(
            verts=[verts_trans],
            faces=[self.mesh.faces_packed().clone()],
            textures=self.mesh.textures.clone()
        )
        hardphong_image = self.hardphong_renderer(mesh_transformed)
        silhouette_image = self.silhouette_renderer(mesh_transformed)
        return hardphong_image, silhouette_image, R_obj, T_obj

    def save_pointcloud(self, points, save_path, save_format="ply"):
        """
        保存点云
        :param points: (N, 3) numpy array
        :param save_path: 文件路径（不带后缀）
        :param save_format: "ply" 或 "npy"
        """
        if save_format == "npy":
            np.save(save_path + ".npy", points.astype(np.float32))
        elif save_format == "ply":
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
            o3d.io.write_point_cloud(save_path + ".ply", pcd)
        else:
            raise ValueError("Unsupported save_format: choose 'npy' or 'ply'")

    # def render_template_library(self, output_dir="template_output_pian2", num_pc_points=3000, pc_format="ply"):
    #     os.makedirs(output_dir, exist_ok=True)
    #     template_index = 0
    #     pose6s = []
    #
    #     fx, fy = self.cameras.focal_length[0].cpu().numpy()
    #     cx, cy = self.cameras.principal_point[0].cpu().numpy()
    #     H, W = self.cameras.image_size[0].cpu().numpy()
    #
    #     # 相机距离
    #     camera_distance = 200.0
    #     cam_t = [0, 0, camera_distance]
    #
    #     for ey in tqdm(range(-90, 90, 20), desc="EY"):
    #         for ex in range(0, 180, 20):
    #             for ez in range(0, 360, 20):
    #                 hardphong_image, silhouette_image, R_obj, T_obj = \
    #                     self.render_hardphone_and_silhouette_images([ex, ey, ez], cam_t)
    #
    #                 h_img_cv = hardphong_image.detach().squeeze(0).cpu().numpy()[:, :, 0:3]
    #                 s_img_cv = silhouette_image.detach().squeeze(0).cpu().numpy()[:, :, 3]
    #
    #                 # 保存RGB和Mask
    #                 cv2.imwrite(os.path.join(output_dir, f"rgb_{template_index:04d}.png"),
    #                             (h_img_cv * 255).astype(np.uint8))
    #                 cv2.imwrite(os.path.join(output_dir, f"mask_{template_index:04d}.png"),
    #                             (s_img_cv * 255).astype(np.uint8))
    #
    #                 # 保存姿态矩阵
    #                 pose = np.eye(4, dtype=np.float32)
    #                 pose[:3, :3] = R_obj.cpu().numpy()
    #                 pose[:3, 3] = T_obj.cpu().numpy()
    #                 np.save(os.path.join(output_dir, f"pose_{template_index:04d}.npy"), pose)
    #
    #                 fragments = self.hardphong_renderer.rasterizer(
    #                     Meshes(
    #                         verts=[self.mesh.verts_packed() @ R_obj.T + T_obj],
    #                         faces=[self.mesh.faces_packed()],
    #                         textures=self.mesh.textures
    #                     )
    #                 )
    #                 depth_map = fragments.zbuf[0, ..., 0].cpu().numpy()  # (H, W)
    #
    #                 # # 保存伪彩色深度图
    #                 # depth_color = self.depth_to_colormap(depth_map, s_img_cv > 0.5)
    #                 # cv2.imwrite(os.path.join(output_dir, f"depthcolor_{template_index:04d}.png"), depth_color)
    #                 #
    #                 # depth_vals = depth_map[s_img_cv > 0.5]
    #                 # depth_vals = depth_vals[~np.isnan(depth_vals)]
    #                 # depth_vals = depth_vals[depth_vals > 1e-6]
    #                 #
    #                 # if len(depth_vals) > 50:
    #                 #     min_val, max_val = np.min(depth_vals), np.max(depth_vals)
    #                 #
    #                 #     depth_vals_norm = (depth_vals - min_val) / (max_val - min_val + 1e-6)
    #                 #
    #                 #     hist, _ = np.histogram(depth_vals_norm, bins=50, range=(0, 1), density=True)
    #                 #     hist = hist / (np.sum(hist) + 1e-6)
    #                 #     depth_feature = hist.tolist()
    #                 # else:
    #                 #     depth_feature = [0.0] * 50
    #
    #                 # ========== 保存 pose6 + depth_feature ==========
    #                 pose6s.append({
    #                     "pose6": [ex, ey, ez, cam_t[0], cam_t[1], cam_t[2]],
    #                     # "depth_feature": depth_feature
    #                 })
    #
    #                 # 完整点云
    #                 verts = self.mesh.verts_packed()
    #                 verts_trans = verts @ R_obj.T + T_obj
    #                 mesh_transformed = Meshes(
    #                     verts=[verts_trans],
    #                     faces=[self.mesh.faces_packed()],
    #                     textures=self.mesh.textures
    #                 )
    #                 pcd = sample_points_from_meshes(mesh_transformed, num_samples=num_pc_points)
    #                 pc_np = pcd[0].cpu().numpy()
    #                 self.save_pointcloud(pc_np,
    #                                      os.path.join(output_dir, f"pc_{template_index:04d}"),
    #                                      save_format=pc_format)
    #
    #                 # 可见像素点云
    #                 ys, xs = np.nonzero(s_img_cv > 0.5)
    #                 Zs = depth_map[ys, xs]
    #                 Xs = (xs - cx) * Zs / fx
    #                 Ys = (ys - cy) * Zs / fy
    #                 pc_visible = np.stack([Xs, Ys, Zs], axis=-1)
    #                 self.save_pointcloud(
    #                     pc_visible,
    #                     os.path.join(output_dir, f"pc_visible_pixel_{template_index:04d}"),
    #                     save_format=pc_format
    #                 )
    #
    #                 template_index += 1
    #     import json
    #     with open(os.path.join(output_dir, "pose6s.json"), "w", encoding="utf-8") as f:
    #         json.dump(pose6s, f, ensure_ascii=False, indent=4)
    #
    #     print(f"[DONE] 模板渲染完成: 共输出 {template_index} 帧到 {output_dir}")
    # ===================== 生成反面 mesh =====================
    def create_backside_mesh(self):
        """
        生成反面的 mesh：翻转 Z 轴 + 翻转面片方向
        """
        mesh = self.mesh
        verts = mesh.verts_packed().clone()
        faces = mesh.faces_packed().clone()

        # 翻转 Z（镜像）
        verts[:, 2] *= -1

        # 翻转面方向
        faces = faces[:, [0, 2, 1]]

        # 创建 textures
        new_mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=mesh.textures.clone()
        )
        return new_mesh

    # ===================== 修改后的模板渲染函数 =====================
    def render_template_library(self, output_dir="template_output",
                                num_pc_points=3000, pc_format="ply",
                                front_back=True):
        """
        front_back=False：只渲染正面
        front_back=True：渲染 front + back 两套模板
        """
        if not front_back:
            # 旧逻辑：只渲染正面
            return self._render_one_side(self.mesh, output_dir,
                                         num_pc_points, pc_format)
        else:
            # 新逻辑：正反两面
            front_dir = os.path.join(output_dir, "front")
            back_dir = os.path.join(output_dir, "back")
            os.makedirs(front_dir, exist_ok=True)
            os.makedirs(back_dir, exist_ok=True)

            print("[INFO] 正在渲染 Front 模板库 …")
            self._render_one_side(self.mesh, front_dir,
                                  num_pc_points, pc_format)

            print("[INFO] 正在生成 Back 翻转模型 …")
            back_mesh = self.create_backside_mesh()

            print("[INFO] 正在渲染 Back 模板库 …")
            self._render_one_side(back_mesh, back_dir,
                                  num_pc_points, pc_format)

            print("[DONE] 正反面模板渲染完成")
            return

    # ===================== 原渲染主逻辑抽取成内部函数 =====================
    def _render_one_side(self, mesh, output_dir, num_pc_points, pc_format):
        """
        渲染正面或反面模板库
        """
        os.makedirs(output_dir, exist_ok=True)
        template_index = 0
        pose6s = []

        fx, fy = self.cameras.focal_length[0].cpu().numpy()
        cx, cy = self.cameras.principal_point[0].cpu().numpy()
        H, W = self.cameras.image_size[0].cpu().numpy()

        # 相机距离（mm）
        camera_distance_mm = 150
        cam_t = [0, 0, camera_distance_mm]

        # 使用传入 mesh 渲染（暂时覆盖 self.mesh）
        original_mesh = self.mesh
        self.mesh = mesh

        print(f"[INFO] 输出路径: {output_dir}")

        # ====== 遍历角度：完整一致性版本 ======
        ex_min, ex_max, ex_step = -20, 21,10
        ey_min, ey_max, ey_step = -20, 21,10
        ez_min, ez_max, ez_step = 0, 360, 40



        for ey in tqdm(range(ey_min, ey_max, ey_step), desc="EY"):
            for ex in range(ex_min, ex_max, ex_step):
                for ez in range(ez_min, ez_max, ez_step):
                    # ---------- 渲染 ----------
                    hardphong_image, silhouette_image, R_obj, T_obj = \
                        self.render_hardphone_and_silhouette_images([ex, ey, ez], cam_t)

                    h_img_cv = hardphong_image.detach().squeeze(0).cpu().numpy()[:, :, 0:3]
                    s_img_cv = silhouette_image.detach().squeeze(0).cpu().numpy()[:, :, 3]

                    # ---------- 保存 RGB / mask ----------

                    # ---------- 保存 RGB / mask ----------
                    rgb_u8 = (h_img_cv * 255).clip(0, 255).astype(np.uint8)
                    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(os.path.join(output_dir, f"rgb_{template_index:04d}.png"), bgr_u8)
                    cv2.imwrite(os.path.join(output_dir, f"mask_{template_index:04d}.png"),
                                (s_img_cv * 255).astype(np.uint8))
                    # cv2.imwrite(os.path.join(output_dir, f"rgb_{template_index:04d}.png"),
                    #             (h_img_cv * 255).astype(np.uint8))
                    # cv2.imwrite(os.path.join(output_dir, f"mask_{template_index:04d}.png"),
                    #             (s_img_cv * 255).astype(np.uint8))

                    # ---------- 保存姿态 4x4 ----------
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = R_obj.cpu().numpy()
                    pose[:3, 3] = T_obj.cpu().numpy()
                    np.save(os.path.join(output_dir, f"pose_{template_index:04d}.npy"), pose)

                    # ---------- 保存 pose6 (给匹配器用) ----------
                    pose6s.append({
                        "pose6": [ex, ey, ez, cam_t[0], cam_t[1], cam_t[2]],
                    })

                    # ---------- 完整点云 ----------
                    verts = self.mesh.verts_packed()
                    verts_trans = verts @ R_obj.T + T_obj
                    mesh_transformed = Meshes(
                        verts=[verts_trans],
                        faces=[self.mesh.faces_packed()],
                        textures=self.mesh.textures
                    )
                    pcd = sample_points_from_meshes(mesh_transformed, num_samples=num_pc_points)
                    pc_np = pcd[0].cpu().numpy()
                    self.save_pointcloud(pc_np,
                                         os.path.join(output_dir, f"pc_{template_index:04d}"),
                                         save_format=pc_format)

                    # ---------- 可见像素点云 ----------
                    fragments = self.hardphong_renderer.rasterizer(mesh_transformed)
                    depth_map = fragments.zbuf[0, ..., 0].cpu().numpy()

                    ys, xs = np.nonzero(s_img_cv > 0.5)
                    Zs = depth_map[ys, xs]
                    Xs = (xs - cx) * Zs / fx
                    Ys = (ys - cy) * Zs / fy
                    pc_visible = np.stack([Xs, Ys, Zs], axis=-1)

                    self.save_pointcloud(
                        pc_visible,
                        os.path.join(output_dir, f"pc_visible_pixel_{template_index:04d}"),
                        save_format=pc_format
                    )

                    template_index += 1

        # ====== 保存最终 library_metadata.json ======
        metadata = {
            "template_library": os.path.abspath(output_dir),
            "camera_intrinsics": {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "H": int(H),
                "W": int(W)
            },
            "camera_distance_mm": float(camera_distance_mm),
            "rotation_params": {
                "ex": [float(ex_min), float(ex_max), float(ex_step)],
                "ey": [float(ey_min), float(ey_max), float(ey_step)],
                "ez": [float(ez_min), float(ez_max), float(ez_step)]
            },
            "total_templates": template_index,
            "pose6_list": pose6s
        }

        import json
        with open(os.path.join(output_dir, "library_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        # ====== 恢复原 mesh ======
        self.mesh = original_mesh

        print(f"[DONE] 模板渲染完成，共输出 {template_index} 帧，metadata 已生成！")

    def render_visible_pointcloud_from_pose6(self, pose6):
        """
        根据 pose6 渲染可见点云（仅新增，不影响任何原有功能）
        pose6 = [ex, ey, ez, tx, ty, tz]
        输出：可见像素点云 (N,3)
        """
        tx, ty, tz,ex, ey, ez =pose6

        # -----------------------------
        # 1) 使用你已有的渲染函数
        # -----------------------------
        hardphong_image, silhouette_image, R_obj, T_obj = \
            self.render_hardphone_and_silhouette_images(
                [ex, ey, ez],
                [tx, ty, tz]
            )

        # mask
        s_img_cv = silhouette_image.detach().cpu().numpy()[0, ..., 3]

        # 相机内参
        fx, fy = self.cameras.focal_length[0].detach().cpu().numpy()
        cx, cy = self.cameras.principal_point[0].detach().cpu().numpy()

        # -----------------------------
        # 2) 用原来的 rasterizer 得到深度图
        # -----------------------------
        mesh_transformed = Meshes(
            verts=[self.mesh.verts_packed() @ R_obj.T + T_obj],
            faces=[self.mesh.faces_packed()],
            textures=self.mesh.textures
        )
        fragments = self.hardphong_renderer.rasterizer(mesh_transformed)
        depth_map = fragments.zbuf[0, ..., 0].detach().cpu().numpy()

        # -----------------------------
        # 3) 提取可见像素点云
        # -----------------------------
        ys, xs = np.nonzero(s_img_cv > 0.5)
        Zs = depth_map[ys, xs]

        # 过滤无效深度
        valid = (Zs > 1e-6) & np.isfinite(Zs)
        xs = xs[valid]
        ys = ys[valid]
        Zs = Zs[valid]

        # 像素坐标 → 相机坐标系点云
        Xs = (xs - cx) * Zs / fx
        Ys = (ys - cy) * Zs / fy

        return np.stack([Xs, Ys, Zs], axis=1)



if __name__ == "__main__":

    #
    # cam_parameters = [2439.0,  2439.0, 959.0,612.0, 1200, 1920]
    #
    # # obj_path = "resources/metalpart.obj"  # "resources/T/T.obj"
    # obj_path = "/home/sunddy/Programming/TrackingAnyCAD/Mesh/big.obj"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # render = DiffRenderNetNT(device, cam_parameters, "deg", "mm").to(device)
    #
    # render.load_target_mesh(obj_path)
    # render.deg2rad = True
    # # [-185.18943787 - 284.162323    497.48260498    6.34260614    1.64215797
    # #  21.51196982]
    # # pose = np.load("template_output/pose_0036.npy")
    # # pose6 = render.matrix44_to_pose6(pose)
    # hardphong_image, silhouette_image, R, T = render.render_hardphone_and_silhouette_images( [4.23,-0.16,30.7], [  31.95,27.14,327.66 ])
    # import cv2
    #
    # alpha = hardphong_image.squeeze(0)[:, :, 3].detach().cpu().numpy()  # (H, W), float [0,1]
    #
    # # 转成 0~255 uint8
    # alpha_uint8 = (alpha * 255).clip(0, 255).astype(np.uint8)
    #
    # cv2.imwrite("silhouette.png", alpha_uint8)


    # cam_parameters = [
    #     3.659997863769531250e+02, 3.661022338867187500e+02,  # fx, fy
    #     3.209124450683593750e+02, 2.405186309814453125e+02,  # cx, cy
    #     480, 640  # H, W
    # ]

    cam_parameters = [
                    365.99,366.10,320.91,240.51,480,640
                ]

    # ======= 模型路径 =======
    obj_path = "/home/robot4/Programming/FoundationPose/mesh_5.20/huan.obj"

    # ======= 输出目录 =======
    # output_dir = "/home/ma/Programming/TrackingAnyCAD/templates/template_output2026_3.13"
    output_dir = "/home/robot4/Programming/FoundationPose/templates_5.20/huan"

    # ======= 初始化渲染器 =======
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    render = DiffRenderNetNT(device, cam_parameters, rad_deg="deg", mm_m="m").to(device)
    render.load_target_mesh(obj_path)
    render.deg2rad = True  # 启用角度转弧度

    # ======= 渲染模板库 =======
    print(f"[INFO] 开始渲染模板库，输出路径: {output_dir}")
    render.render_template_library(
        output_dir=output_dir,
        num_pc_points=3000,  # 每个姿态采样的点云点数
        pc_format="ply"  # 保存格式：ply / npy
    )

    print(f"[DONE] 模板库渲染完成，输出位于 {os.path.abspath(output_dir)}")