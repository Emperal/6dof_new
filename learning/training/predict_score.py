# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import functools
import os, sys, kornia
import time
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm

# 将上一级目录加入系统路径，以便导入上级目录中的自定义模块
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../')

# 导入项目中自定义的数据集读取、模型结构、以及一些通用工具函数
from learning.datasets.h5_dataset import *
from learning.models.score_network import *
from learning.datasets.pose_dataset import *
from Utils import *
from datareader import *


def vis_batch_data_scores(pose_data, ids, scores, pad_margin=5):
    """
    可视化一批位姿数据及其对应的打分结果。
    将渲染图 (A) 和真实图 (B) 的 RGB 及深度图拼接在一起，并在左上角绘制得分。

    :param pose_data: 包含了渲染图像(rgbAs)、真实图像(rgbBs)、深度图等的 BatchPoseData 对象
    :param ids: 要可视化的数据的索引列表，通常按得分从高到低排序
    :param scores: 对应索引位置的得分列表
    :param pad_margin: 拼接图像时中间和底部的空白间隔像素数
    :return: 拼接好的一张大图 (numpy array)
    """
    assert len(scores) == len(ids)
    canvas = []  # 存储每一行可视化结果的列表
    for id in ids:
        # 提取第 id 个渲染图 (A) 并将其从 Tensor (C,H,W) 转为 numpy array (H,W,C)
        rgbA_vis = (pose_data.rgbAs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
        # 提取第 id 个真实裁剪图 (B)
        rgbB_vis = (pose_data.rgbBs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
        H, W = rgbA_vis.shape[:2]

        # 计算当前深度图中的最大最小值，用于将深度数据归一化成可视化图像
        zmin = pose_data.depthAs[id].data.cpu().numpy().reshape(H, W).min()
        zmax = pose_data.depthAs[id].data.cpu().numpy().reshape(H, W).max()

        # 将深度图转为可显示的彩色图或者灰度图
        depthA_vis = depth_to_vis(pose_data.depthAs[id].data.cpu().numpy().reshape(H, W), zmin=zmin, zmax=zmax,
                                  inverse=False)
        depthB_vis = depth_to_vis(pose_data.depthBs[id].data.cpu().numpy().reshape(H, W), zmin=zmin, zmax=zmax,
                                  inverse=False)

        # 预留处理法向量的代码位置，目前直接pass
        if pose_data.normalAs is not None:
            pass

        # 创建纯白色的分隔条，用于隔开不同图片
        pad = np.ones((rgbA_vis.shape[0], pad_margin, 3)) * 255
        if pose_data.normalAs is not None:
            pass
        else:
            # 将: 渲染RGB、空白、渲染深度图、空白、真实RGB、空白、真实深度图，横向拼接成一行
            row = np.concatenate([rgbA_vis, pad, depthA_vis, pad, rgbB_vis, pad, depthB_vis], axis=1)

        # 将这一行图片统一缩放到高度为 100 像素
        s = 100 / row.shape[0]
        row = cv2.resize(row, fx=s, fy=s, dsize=None)

        # 在图片左上角写上它的 id 和 网络打出的 score
        row = cv_draw_text(row, text=f'id:{id}, score:{scores[id]:.3f}', uv_top_left=(10, 10), color=(0, 255, 0),
                           fontScale=0.5)
        canvas.append(row)

        # 在这一行下面添加一个横向的白色空白条
        pad = np.ones((pad_margin, row.shape[1], 3)) * 255
        canvas.append(pad)

    # 将所有行纵向拼接成一张长图并返回
    canvas = np.concatenate(canvas, axis=0).astype(np.uint8)
    return canvas


@torch.no_grad()  # 整个函数不需要计算梯度，节省显存
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, normal_map=None, mesh_diameter=None,
                         glctx=None, mesh_tensors=None, dataset: TripletH5Dataset = None, cfg=None):
    """
    核心预处理函数：批量生成用于打分网络的数据。
    它会把 3D 模型按照各种候选位姿渲染成 2D 图像，并从真实图片中裁剪出对应区域。

    :param render_size: 渲染/裁剪输出的图像分辨率，如 (128, 128)
    :param ob_in_cams: Nx4x4 的矩阵，代表 N 个候选的物体位姿
    :param mesh: 物体的 3D 网格模型
    :param rgb: 相机拍摄的完整 RGB 图像
    :param depth: 相机拍摄的完整深度图像
    :param K: 3x3 相机内参矩阵
    :param crop_ratio: 裁剪框扩大的比例系数
    """
    logging.info("Welcome make_crop_data_batch")
    H, W = depth.shape[:2]

    args = []
    method = 'box_3d'
    # 根据 3D 边界框的投影，计算从原图裁剪出目标物体的透视变换矩阵 (Nx3x3)
    tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K,
                                               crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]),
                                               method=method, mesh_diameter=mesh_diameter)
    logging.info("make tf_to_crops done")

    B = len(ob_in_cams)  # 候选位姿的总数
    poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

    bs = 512  # 渲染批大小 (Batch Size)，避免一次渲染太多爆显存
    rgb_rs = []
    depth_rs = []
    xyz_map_rs = []

    # 计算裁剪框的基准坐标系，用于基于 nvdiffrast 的高效硬件光栅化渲染
    bbox2d_crop = torch.as_tensor(
        np.array([0, 0, cfg['input_resize'][0] - 1, cfg['input_resize'][1] - 1]).reshape(2, 2), device='cuda',
        dtype=torch.float)
    bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:, None]).reshape(-1, 4)

    # 分批次执行 3D 到 2D 的渲染操作
    for b in range(0, len(ob_in_cams), bs):
        extra = {}
        # 使用 nvdiffrast 在 GPU 上快速渲染当前批次的位姿图像
        rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b + bs], context='cuda',
                                                     get_normal=cfg['use_normal'], glctx=glctx,
                                                     mesh_tensors=mesh_tensors, output_size=cfg['input_resize'],
                                                     bbox2d=bbox2d_ori[b:b + bs], use_light=True, extra=extra)
        rgb_rs.append(rgb_r)
        depth_rs.append(depth_r[..., None])
        xyz_map_rs.append(extra['xyz_map'])  # xyz_map 记录了每个像素的 3D 空间坐标

    # 将所有批次的渲染结果拼接成大张量，并调整维度为 (B, C, H, W)
    rgb_rs = torch.cat(rgb_rs, dim=0).permute(0, 3, 1, 2) * 255
    depth_rs = torch.cat(depth_rs, dim=0).permute(0, 3, 1, 2)
    xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0, 3, 1, 2)  # (B,3,H,W)
    logging.info("render done")

    # 利用前面算好的透视变换矩阵，将真实拍摄的整张 RGB 和 Depth 图像裁剪并缩放成小图，复制 B 份对齐
    rgbBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2, 0, 1)[None].expand(B, -1, -1, -1),
        tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthBs = kornia.geometry.transform.warp_perspective(
        torch.as_tensor(depth, dtype=torch.float, device='cuda')[None, None].expand(B, -1, -1, -1), tf_to_crops,
        dsize=render_size, mode='nearest', align_corners=False)

    # 如果渲染输出尺寸与要求不一致，再做一次变形对齐（A代表渲染图，B代表真实图）
    if rgb_rs.shape[-2:] != cfg['input_resize']:
        rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear',
                                                           align_corners=False)
        depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest',
                                                             align_corners=False)
    else:
        rgbAs = rgb_rs
        depthAs = depth_rs

    if xyz_map_rs.shape[-2:] != cfg['input_resize']:
        xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size,
                                                               mode='nearest', align_corners=False)
    else:
        xyz_mapAs = xyz_map_rs

    normalAs = None
    normalBs = None

    # 准备构建 Batch 数据集所需的一些参数张量
    Ks = torch.as_tensor(K, dtype=torch.float).reshape(1, 3, 3).expand(len(rgbAs), 3, 3)
    mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda') * mesh_diameter

    # 将所有成对的输入（渲染图A 与 真实图B）打包成 BatchPoseData 数据结构
    pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs,
                              normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, tf_to_crops=tf_to_crops, Ks=Ks,
                              mesh_diameters=mesh_diameters)

    # 调用 Dataset 的归一化/预处理函数，为进入神经网络做最后准备
    pose_data = dataset.transform_batch(pose_data, H_ori=H, W_ori=W, bound=1)

    logging.info("pose batch data done")

    return pose_data


class ScorePredictor:
    """
    评分网络包装类。
    用于评估给定的多个候选位姿，选出与真实图片最贴合的那个位姿。
    """

    def __init__(self, amp=True):
        self.amp = amp  # 是否启用自动混合精度(Automatic Mixed Precision)加速计算
        self.run_name = "2024-01-11-20-02-45"  # 训练好权重的保存文件夹名

        model_name = 'model_best.pth'
        code_dir = os.path.dirname(os.path.realpath(__file__))
        ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

        # 读取网络模型的配置参数文件
        self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')

        self.cfg['ckpt_dir'] = ckpt_dir
        self.cfg['enable_amp'] = True

        ########## 补充缺省配置，为了向后兼容老版本的模型
        if 'use_normal' not in self.cfg:
            self.cfg['use_normal'] = False
        if 'use_BN' not in self.cfg:
            self.cfg['use_BN'] = False
        if 'zfar' not in self.cfg:
            self.cfg['zfar'] = np.inf
        if 'c_in' not in self.cfg:
            self.cfg['c_in'] = 4
        if 'normalize_xyz' not in self.cfg:
            self.cfg['normalize_xyz'] = False
        if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
            self.cfg['crop_ratio'] = 1.2

        logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

        # 实例化数据集对象（仅用于调用其中的数据变换方法）
        self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode='test', h5_file=None, max_num_key=1)

        # 初始化打分网络模型结构并放到 GPU 上
        self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()

        # 加载预训练权重
        logging.info(f"Using pretrained model from {ckpt_dir}")
        ckpt = torch.load(ckpt_dir)
        if 'model' in ckpt:
            ckpt = ckpt['model']
        self.model.load_state_dict(ckpt)

        # 设为评估模式
        self.model.cuda().eval()
        logging.info("init done")

    @torch.inference_mode()  # 关闭梯度计算，防止显存泄漏，比 no_grad 更彻底
    def predict(self, rgb, depth, K, ob_in_cams, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None,
                glctx=None, mesh_diameter=None):
        '''
        核心推理接口：给定图像和一批候选位姿，返回每个位姿的分数。
        @rgb: 真实彩色图像 np array (H,W,3)
        @ob_in_cams: 一组需要打分的候选位姿矩阵 (N, 4, 4)
        '''
        logging.info(f"ob_in_cams:{ob_in_cams.shape}")
        ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

        logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
        if not self.cfg.use_normal:
            normal_map = None

        logging.info("making cropped data")

        # 如果没有提供显存中的网格张量，则重新构建一份
        if mesh_tensors is None:
            mesh_tensors = make_mesh_tensors(mesh)

        rgb = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)

        # 1. 数据预处理：执行前面定义的 make_crop_data_batch，获取成对的渲染图和真实裁剪图
        pose_data = make_crop_data_batch(self.cfg.input_resize, ob_in_cams, mesh, rgb, depth, K,
                                         crop_ratio=self.cfg['crop_ratio'], glctx=glctx, mesh_tensors=mesh_tensors,
                                         dataset=self.dataset, cfg=self.cfg, mesh_diameter=mesh_diameter)

        # 内部辅助函数：执行前向推理给位姿打分
        def find_best_among_pairs(pose_data: BatchPoseData):
            logging.info(f'pose_data.rgbAs.shape[0]: {pose_data.rgbAs.shape[0]}')
            ids = []
            scores = []
            bs = pose_data.rgbAs.shape[0]  # 当前批次处理的数量

            for b in range(0, pose_data.rgbAs.shape[0], bs):
                # A 组为网络预测的输入：将渲染 RGB 图与 3D 坐标系 xyz_map 在通道维度(dim=1)拼接
                A = torch.cat([pose_data.rgbAs[b:b + bs].cuda(), pose_data.xyz_mapAs[b:b + bs].cuda()], dim=1).float()
                # B 组为实际物体的特征：将裁剪出的真实 RGB 与 对应的 xyz_map 拼接
                B = torch.cat([pose_data.rgbBs[b:b + bs].cuda(), pose_data.xyz_mapBs[b:b + bs].cuda()], dim=1).float()

                # 可选：如果包含法向信息，一并拼接进去
                if pose_data.normalAs is not None:
                    A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1)
                    B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1)

                # 开启混合精度推理，将数据丢进网络
                with torch.cuda.amp.autocast(enabled=self.amp):
                    output = self.model(A, B, L=len(A))

                # 提取网络预测的分数结果
                scores_cur = output["score_logit"].float().reshape(-1)
                # 记录当前批次中得分最高的位姿索引
                ids.append(scores_cur.argmax() + b)
                scores.append(scores_cur)

            # 整合整个大批次的结果
            ids = torch.stack(ids, dim=0).reshape(-1)
            scores = torch.cat(scores, dim=0).reshape(-1)
            return ids, scores

        pose_data_iter = pose_data
        global_ids = torch.arange(len(ob_in_cams), device='cuda', dtype=torch.long)
        scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device='cuda')

        # 不断打分迭代筛选，虽然这里的实现只跑了一次就 break 了
        while 1:
            # 调用模型，给所有候选位姿打分
            ids, scores = find_best_among_pairs(pose_data_iter)

            # 这里的逻辑主要是如果只有 1 组输出了，说明打分结束，记录分数
            if len(ids) == 1:
                scores_global[global_ids] = scores + 100
                break

            # 更新剩余位姿索引（实际上这里的实现有待商榷，因为上面的 bs 就是全量长度）
            global_ids = global_ids[ids]
            pose_data_iter = pose_data.select_by_indices(global_ids)

        scores = scores_global

        logging.info(f'forward done')
        torch.cuda.empty_cache()  # 推理完毕后清空释放显存缓存

        # 2. 如果请求了可视化结果，生成对应的对比图像
        if get_vis:
            logging.info("get_vis...")
            canvas = []
            # 根据分数从大到小对所有位姿的 id 进行排序
            ids = scores.argsort(descending=True)
            canvas = vis_batch_data_scores(pose_data, ids=ids, scores=scores)
            return scores, canvas

        # 返回所有候选位姿的分数
        return scores, None