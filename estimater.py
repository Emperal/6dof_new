# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
import sys
sys.path.append('/home/sunddy/Programming/FoundationPose/mycpp/build')

class FoundationPose:
    # 初始化 FoundationPose 核心类
    # def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer: ScorePredictor = None,
    #              refiner: PoseRefinePredictor = None, glctx=None, debug=0,
    #              debug_dir='/home/bowen/debug/novel_pose_debug/'):
    #     self.gt_pose = None
    #     self.ignore_normal_flip = True
    #     self.debug = debug
    #     self.debug_dir = debug_dir
    #     # 创建用于调试的目录
    #     os.makedirs(debug_dir, exist_ok=True)
    def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer: ScorePredictor = None,
                 refiner: PoseRefinePredictor = None, glctx=None, debug=0,
                 debug_dir='./debug'):
        self.gt_pose = None
        self.ignore_normal_flip = True
        self.debug = debug
        self.debug_dir = debug_dir

        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)


        # 重置并初始化目标物体的 3D 模型状态（中心化、下采样等）
        self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
        # 生成用于全局搜索的离散旋转网格（视点空间）
        # self.make_rotation_grid(min_n_views=40, inplane_step=120)
        self.make_rotation_grid(min_n_views=30, inplane_step=120)


        # OpenGL/CUDA 渲染上下文
        self.glctx = glctx

        # 初始化评分网络（用于评估位姿假设的好坏）
        if scorer is not None:
            self.scorer = scorer
        else:
            self.scorer = ScorePredictor()

        # 初始化优化网络（用于对粗糙位姿进行迭代微调）
        if refiner is not None:
            self.refiner = refiner
        else:
            self.refiner = PoseRefinePredictor()

        # 记录上一帧的位姿，用于连续跟踪（相对于中心化后的网格）
        self.pose_last = None  # Used for tracking; per the centered mesh

    # 处理输入的 3D 模型，提取关键特征并将其平移至原点
    def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
        # 计算模型边界框以获取中心点
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2

        # 如果提供了网格，将其平移，使模型中心对齐到坐标系原点
        if mesh is not None:
            self.mesh_ori = mesh.copy()
            mesh = mesh.copy()
            mesh.vertices = mesh.vertices - self.model_center.reshape(1, 3)

        model_pts = mesh.vertices
        # 计算物体的直径（用于确定体素大小和渲染参数）
        self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
        # 根据直径设定点云降采样的体素大小，最小为 3mm
        self.vox_size = max(self.diameter / 20.0, 0.003)
        logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')

        # 设置距离和角度的 bin（可能用于特征匹配或直方图统计）
        self.dist_bin = self.vox_size / 2
        self.angle_bin = 20  # Deg

        # 将模型转换为 Open3D 格式并进行体素下采样
        pcd = toOpen3dCloud(model_pts, normals=model_normals)
        pcd = pcd.voxel_down_sample(self.vox_size)
        self.max_xyz = np.asarray(pcd.points).max(axis=0)
        self.min_xyz = np.asarray(pcd.points).min(axis=0)

        # 将下采样后的点云和法线转为 PyTorch Tensor 并存入 GPU
        self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
        self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
        logging.info(f'self.pts:{self.pts.shape}')

        self.mesh_path = None
        self.mesh = mesh
        # 导出临时中心化网格文件，并生成供网络使用的网格张量
        if self.mesh is not None:
            self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
            self.mesh.export(self.mesh_path)
        self.mesh_tensors = make_mesh_tensors(self.mesh)

        # 记录物体的对称性变换矩阵，如果没有提供则默认为单位阵
        if symmetry_tfs is None:
            self.symmetry_tfs = torch.eye(4).float().cuda()[None]
        else:
            self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

        logging.info("reset done")

    # 获取一个 4x4 变换矩阵，用于将原始位姿转换到中心化网格的空间
    def get_tf_to_centered_mesh(self):
        tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
        tf_to_center[:3, 3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
        return tf_to_center

    # 将所有相关的数据结构和神经网络模型转移到指定的计算设备 (如 cuda:0)
    def to_device(self, s='cuda:0'):
        for k in self.__dict__:
            self.__dict__[k] = self.__dict__[k]
            # 移动张量或神经网络模块
            if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
                logging.info(f"Moving {k} to device {s}")
                self.__dict__[k] = self.__dict__[k].to(s)
        # 移动网格相关的张量
        for k in self.mesh_tensors:
            logging.info(f"Moving {k} to device {s}")
            self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
        if self.refiner is not None:
            self.refiner.model.to(s)
        if self.scorer is not None:
            self.scorer.model.to(s)
        # 重置渲染器上下文到目标设备
        if self.glctx is not None:
            self.glctx = dr.RasterizeCudaContext(s)

    # 在观察球面上生成离散的视点，加上平面内旋转，构建全局位姿假设空间
    def make_rotation_grid(self, min_n_views=40, inplane_step=60):
        # 根据正二十面体采样均匀的相机视角
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        logging.info(f'cam_in_obs:{cam_in_obs.shape}')
        rot_grid = []
        # 为每个视角添加平面内（in-plane）的旋转
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                # 转换为物体在相机坐标系下的姿态
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        logging.info(f"rot_grid:{rot_grid.shape}")
        # 结合物体的对称性对相似的位姿进行聚类去重，减少计算量
        rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
        rot_grid = np.asarray(rot_grid)
        logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
        self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
        logging.info(f"self.rot_grid: {self.rot_grid.shape}")

    # 结合预生成的旋转网格和估计的中心平移量，生成一组完整的随机位姿假设
    def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
        '''
        @scene_pts: torch tensor (N,3)
        '''
        ob_in_cams = self.rot_grid.clone()
        # 预测平移向量 (X, Y, Z)
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        # 将估计的平移应用到旋转矩阵中
        ob_in_cams[:, :3, 3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1, 3)
        return ob_in_cams

    # 根据 2D 掩码和深度图粗略猜测物体在 3D 空间中的平移 (X, Y, Z)
    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            logging.info(f'mask is all zero')
            return np.zeros((3))
        # 取掩码边界框的中心作为 2D 中心 (uc, vc)
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0

        # 获取掩码区域内的有效深度值
        valid = mask.astype(bool) & (depth >= 0.001)
        if not valid.any():
            logging.info(f"valid is empty")
            return np.zeros((3))

        # 取深度的中位数作为 Z 轴平移估计值
        zc = np.median(depth[valid])
        # 利用相机内参 K 进行反投影，得到 3D 坐标
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc

        if self.debug >= 2:
            pcd = toOpen3dCloud(center.reshape(1, 3))
            o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

        return center.reshape(3)

    # 全局注册（初始化）：输入第一帧图像，全局搜索并评估得出最佳初始位姿
    def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
        '''Copmute pose from given pts to self.pcd
        @pts: (N,3) np array, downsampled scene points
        '''
        set_seed(0)
        logging.info('Welcome')

        # 初始化渲染上下文
        if self.glctx is None:
            if glctx is None:
                self.glctx = dr.RasterizeCudaContext()
                # self.glctx = dr.RasterizeGLContext()
            else:
                self.glctx = glctx

        # 对输入的深度图进行形态学腐蚀和双边滤波处理，减少噪声
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')

        if self.debug >= 2:
            # 生成供调试的点云和掩码图
            xyz_map = depth2xyzmap(depth, K)
            valid = xyz_map[..., 2] >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
            o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply', pcd)
            cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask * 255.0).clip(0, 255))

        normal_map = None
        valid = (depth >= 0.001) & (ob_mask > 0)
        # 如果有效像素太少，直接返回猜测的平移中心，旋转置为单位阵
        if valid.sum() < 4:
            logging.info(f'valid too small, return')
            pose = np.eye(4)
            pose[:3, 3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
            return pose

        if self.debug >= 2:
            imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
            cv2.imwrite(f'{self.debug_dir}/depth.png', (depth * 1000).astype(np.uint16))
            valid = xyz_map[..., 2] >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
            o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)

        self.H, self.W = depth.shape[:2]
        self.K = K
        self.ob_id = ob_id
        self.ob_mask = ob_mask

        # 1. 生成初始的大量位姿假设
        poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
        # poses = poses[:1]
        poses = poses.data.cpu().numpy()
        logging.info(f'poses:{poses.shape}')
        # 计算初始平移，并统一赋给所有位姿假设
        center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

        poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device='cuda')

        # 计算初始假设的误差（仅做记录/调试用）
        add_errs = self.compute_add_err_to_gt_pose(poses)
        logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

        # 2. 调用细化网络 (refiner) 对初始位姿假设进行多轮迭代优化
        xyz_map = depth2xyzmap(depth, K)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_refiner_start = time.time()

        poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map,
                                          glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration,
                                          get_vis=self.debug >= 2)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_refiner_end = time.time()

        logging.info(f'[TIMING] Refiner time: {t_refiner_end - t_refiner_start:.4f} s')
        print(f'[TIMING] Refiner time: {t_refiner_end - t_refiner_start:.4f} s')

        if vis is not None:
            imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

        # 3. 调用评分网络 (scorer) 对优化后的位姿进行打分评估
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_scorer_start = time.time()

        scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K,
                                          ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map,
                                          mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter,
                                          get_vis=self.debug >= 2)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_scorer_end = time.time()

        scorer_time = t_scorer_end - t_scorer_start
        logging.info(f'[TIMING] Scorer total: {scorer_time:.4f} s')
        print(f'[TIMING] Scorer total: {scorer_time:.4f} s')

        if vis is not None:
            imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

        add_errs = self.compute_add_err_to_gt_pose(poses)
        logging.info(f"final, add_errs min:{add_errs.min()}")

        # 4. 根据打分结果降序排列，筛选出最佳位姿
        ids = torch.as_tensor(scores).argsort(descending=True)
        logging.info(f'sort ids:{ids}')
        scores = scores[ids]
        poses = poses[ids]

        logging.info(f'sorted scores:{scores}')

        # 取分数最高的结果，并转换回原始网格坐标系
        best_pose = poses[0] @ self.get_tf_to_centered_mesh()
        self.pose_last = poses[0]
        self.best_id = ids[0]

        self.poses = poses
        self.scores = scores

        return best_pose.data.cpu().numpy()

    # 计算与真实位姿(Ground Truth)之间的 ADD (Average Distance) 误差（此处为占位实现，返回 -1）
    def compute_add_err_to_gt_pose(self, poses):
        '''
        @poses: wrt. the centered mesh
        '''
        return -torch.ones(len(poses), device='cuda', dtype=torch.float)

    # 帧间跟踪：在已知上一帧位姿的情况下，对当前帧的位姿进行微调更新
    def track_one(self, rgb, depth, K, iteration, extra={}):
        # 确保位姿已通过 register() 被初始化
        if self.pose_last is None:
            logging.info("Please init pose by register first")
            raise RuntimeError
        logging.info("Welcome")

        # 对当前帧深度图进行预处理 (腐蚀与双边滤波)
        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = erode_depth(depth, radius=2, device='cuda')
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        logging.info("depth processing done")

        # 生成深度对应的 3D XYZ 点坐标图
        xyz_map = \
        depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

        # 直接使用上一帧位姿 (pose_last) 作为初始假设，送入 refiner 网络进行微调迭代
        pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K,
                                         ob_in_cams=self.pose_last.reshape(1, 4, 4).data.cpu().numpy(), normal_map=None,
                                         xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx,
                                         iteration=iteration, get_vis=self.debug >= 2)
        logging.info("pose done")
        if self.debug >= 2:
            extra['vis'] = vis

        # 更新最新位姿，并返回转换回原始未中心化的坐标系结果
        self.pose_last = pose
        return (pose @ self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)
