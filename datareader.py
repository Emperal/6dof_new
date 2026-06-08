# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


# 导入项目中通用工具函数和第三方依赖
from Utils import *
import json, os, sys

# BOP 挑战赛包含的标准数据集列表
BOP_LIST = ['lmo', 'tless', 'ycbv', 'hb', 'tudl', 'icbin', 'itodd']

# 从环境变量中获取 BOP 数据集根目录
BOP_DIR = os.getenv('BOP_DIR')


def get_bop_reader(video_dir, zfar=np.inf):
    """
    工厂函数：根据输入路径自动判断属于哪个 BOP 数据集，
    并返回对应的数据读取器对象。

    参数：
    - video_dir: 数据集某个视频/场景目录
    - zfar: 深度远裁剪距离，大于该值的深度会被置 0
    """
    if 'ycbv' in video_dir or 'YCB' in video_dir:
        return YcbVideoReader(video_dir, zfar=zfar)
    if 'lmo' in video_dir or 'LINEMOD-O' in video_dir:
        return LinemodOcclusionReader(video_dir, zfar=zfar)
    if 'tless' in video_dir or 'TLESS' in video_dir:
        return TlessReader(video_dir, zfar=zfar)
    if 'hb' in video_dir:
        return HomebrewedReader(video_dir, zfar=zfar)
    if 'tudl' in video_dir:
        return TudlReader(video_dir, zfar=zfar)
    if 'icbin' in video_dir:
        return IcbinReader(video_dir, zfar=zfar)
    if 'itodd' in video_dir:
        return ItoddReader(video_dir, zfar=zfar)
    else:
        raise RuntimeError("未识别的数据集路径格式")


def get_bop_video_dirs(dataset):
    """
    获取指定 BOP 数据集下的所有测试视频/场景目录。
    """
    if dataset == 'ycbv':
        video_dirs = sorted(glob.glob(f'{BOP_DIR}/ycbv/test/*'))
    elif dataset == 'lmo':
        video_dirs = sorted(glob.glob(f'{BOP_DIR}/lmo/lmo_test_bop19/test/*'))
    elif dataset == 'tless':
        video_dirs = sorted(glob.glob(f'{BOP_DIR}/tless/tless_test_primesense_bop19/test_primesense/*'))
    elif dataset == 'hb':
        video_dirs = sorted(glob.glob(f'{BOP_DIR}/hb/hb_test_primesense_bop19/test_primesense/*'))
    elif dataset == 'tudl':
        video_dirs = sorted(glob.glob(f'{BOP_DIR}/tudl/tudl_test_bop19/test/*'))
    elif dataset == 'icbin':
        video_dirs = sorted(glob.glob(f'{BOP_DIR}/icbin/icbin_test_bop19/test/*'))
    elif dataset == 'itodd':
        video_dirs = sorted(glob.glob(f'{BOP_DIR}/itodd/itodd_test_bop19/test/*'))
    else:
        raise RuntimeError("不支持的数据集名称")
    return video_dirs


class YcbineoatReader:
    """
    YCBInEOAT 数据集读取器。
    这不是标准 BOP 结构，而是特定场景/抓取任务常见的一种组织方式。
    """

    def __init__(self, video_dir, downscale=1, shorter_side=None, zfar=np.inf):
        # 数据目录
        self.video_dir = video_dir
        # 图像缩放比例
        self.downscale = downscale
        # 深度远裁剪距离
        self.zfar = zfar

        # 读取所有 RGB 图像文件路径
        self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))

        # 读取相机内参
        self.K = np.loadtxt(f'{video_dir}/cam_K.txt').reshape(3, 3)

        # 提取每一帧的文件编号字符串
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace('.png', '')
            self.id_strs.append(id_str)

        # 原始图像尺寸
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]

        # 如果指定较短边长度，则自动计算缩放比例
        if shorter_side is not None:
            self.downscale = shorter_side / min(self.H, self.W)

        # 根据缩放比例更新图像尺寸和相机内参
        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K[:2] *= self.downscale

        # 读取所有 GT 位姿文件
        self.gt_pose_files = sorted(glob.glob(f'{self.video_dir}/annotated_poses/*'))

        # 视频名 -> 物体名 的映射
        # 用于找到该视频对应的 CAD 模型
        self.videoname_to_object = {
            'bleach0': "021_bleach_cleanser",
            'bleach_hard_00_03_chaitanya': "021_bleach_cleanser",
            'cracker_box_reorient': '003_cracker_box',
            'cracker_box_yalehand0': '003_cracker_box',
            'mustard0': '006_mustard_bottle',
            'mustard_easy_00_02': '006_mustard_bottle',
            'sugar_box1': '004_sugar_box',
            'sugar_box_yalehand0': '004_sugar_box',
            'tomato_soup_can_yalehand0': '005_tomato_soup_can',
        }

    def get_video_name(self):
        # 返回当前视频目录名
        return self.video_dir.split('/')[-1]

    def __len__(self):
        # 返回帧数
        return len(self.color_files)

    def get_gt_pose(self, i):
        # 读取第 i 帧的 GT 4x4 位姿矩阵
        try:
            pose = np.loadtxt(self.gt_pose_files[i]).reshape(4, 4)
            return pose
        except:
            logging.info("GT pose not found, return None")
            return None

    def get_color(self, i):
        # 读取 RGB 图像并根据当前尺寸缩放
        color = imageio.imread(self.color_files[i])[..., :3]
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_mask(self, i):
        # 读取物体 mask
        mask = cv2.imread(self.color_files[i].replace('rgb', 'masks'), -1)

        # 若 mask 是 3 通道，取有值的那个通道
        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break

        # resize 后转成 0/1 uint8
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        return mask

    def get_depth(self, i):
        # 读取深度图，原始通常是毫米，转成米
        depth = cv2.imread(self.color_files[i].replace('rgb', 'depth'), -1) / 1e3
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        # 去除无效深度和超过 zfar 的深度
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0
        return depth

    def get_xyz_map(self, i):
        # 由深度图反投影得到 xyz_map
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_occ_mask(self, i):
        # 读取遮挡 mask，例如手部遮挡
        hand_mask_file = self.color_files[i].replace('rgb', 'masks_hand')
        occ_mask = np.zeros((self.H, self.W), dtype=bool)

        if os.path.exists(hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(hand_mask_file, -1) > 0)

        right_hand_mask_file = self.color_files[i].replace('rgb', 'masks_hand_right')
        if os.path.exists(right_hand_mask_file):
            occ_mask = occ_mask | (cv2.imread(right_hand_mask_file, -1) > 0)

        occ_mask = cv2.resize(occ_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        return occ_mask.astype(np.uint8)

    def get_gt_mesh(self):
        # 根据视频名映射出对应 YCB 物体 CAD 模型
        ob_name = self.videoname_to_object[self.get_video_name()]
        YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
        mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{ob_name}/textured_simple.obj')
        return mesh


class BopBaseReader:
    """
    标准 BOP 数据集读取基类。
    各类 BOP 数据集（LMO/TLESS/YCBV/...）都继承它。

    核心作用：
    - 统一 RGB / 深度 / mask / pose / mesh 的读取接口
    - 解析 scene_camera.json / scene_gt.json
    - 处理 resize、深度单位转换等公共逻辑
    """

    def __init__(self, base_dir, zfar=np.inf, resize=1):
        # 当前场景目录
        self.base_dir = base_dir
        # 图像缩放比例
        self.resize = resize
        # 数据集名称，子类里赋值
        self.dataset_name = None

        # 优先读取 rgb 目录；若没有，则尝试 gray 目录（如 TLESS）
        self.color_files = sorted(glob.glob(f"{self.base_dir}/rgb/*"))
        if len(self.color_files) == 0:
            self.color_files = sorted(glob.glob(f"{self.base_dir}/gray/*"))

        # 深度远裁剪距离
        self.zfar = zfar

        # 解析每一帧的相机内参
        # BOP 里有些数据集每帧 K 可能不同
        self.K_table = {}
        with open(f'{self.base_dir}/scene_camera.json', 'r') as ff:
            info = json.load(ff)

        for k in info:
            self.K_table[f'{int(k):06d}'] = np.array(info[k]['cam_K']).reshape(3, 3)
            # BOP 数据集特有的深度比例因子
            self.bop_depth_scale = info[k]['depth_scale']

        # 加载场景 GT 位姿信息
        if os.path.exists(f'{self.base_dir}/scene_gt.json'):
            with open(f'{self.base_dir}/scene_gt.json', 'r') as ff:
                self.scene_gt = json.load(ff)

            # 深拷贝，避免文件句柄、多线程或 joblib 序列化问题
            self.scene_gt = copy.deepcopy(self.scene_gt)
            assert len(self.scene_gt) == len(self.color_files)
        else:
            self.scene_gt = None

        # 生成每帧对应的 id 字符串
        self.make_id_strs()

    def make_scene_ob_ids_dict(self):
        # 根据 BOP 官方 test_targets 文件构建：
        # 每一帧有哪些目标物体 ID，且每类实例有几个
        with open(f'{BOP_DIR}/{self.dataset_name}/test_targets_bop19.json', 'r') as ff:
            self.scene_ob_ids_dict = {}
            data = json.load(ff)
            for d in data:
                if d['scene_id'] == self.get_video_id():
                    id_str = f"{d['im_id']:06d}"
                    if id_str not in self.scene_ob_ids_dict:
                        self.scene_ob_ids_dict[id_str] = []
                    self.scene_ob_ids_dict[id_str] += [d['obj_id']] * d['inst_count']

    def get_K(self, i_frame):
        # 获取第 i_frame 帧的相机内参
        # 若做过 resize，则同步调整焦距和光心
        K = self.K_table[self.id_strs[i_frame]]
        if self.resize != 1:
            K[:2, :2] *= self.resize
        return K

    def get_video_dir(self):
        # 从路径中提取当前视频/场景 ID
        video_id = int(self.base_dir.rstrip('/').split('/')[-1])
        return video_id

    def make_id_strs(self):
        # 根据图像文件名提取每帧的编号字符串，如 000001
        self.id_strs = []
        for i in range(len(self.color_files)):
            name = os.path.basename(self.color_files[i]).split('.')[0]
            self.id_strs.append(name)

    def get_instance_ids_in_image(self, i_frame: int):
        # 获取某一帧图像中出现的所有物体 ID
        ob_ids = []

        if self.scene_gt is not None:
            # 从 scene_gt.json 中读取
            name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])
            for k in self.scene_gt[str(name)]:
                ob_ids.append(k['obj_id'])

        elif self.scene_ob_ids_dict is not None:
            # 从 test_targets 构建的索引中读取
            return np.array(self.scene_ob_ids_dict[self.id_strs[i_frame]])

        else:
            # 若没有 gt 文件，则尝试从 mask_visib 文件名推断
            mask_dir = os.path.dirname(self.color_files[0]).replace('rgb', 'mask_visib')
            id_str = self.id_strs[i_frame]
            mask_files = sorted(glob.glob(f'{mask_dir}/{id_str}_*.png'))
            ob_ids = []
            for mask_file in mask_files:
                ob_id = int(os.path.basename(mask_file).split('.')[0].split('_')[1])
                ob_ids.append(ob_id)

        ob_ids = np.asarray(ob_ids)
        return ob_ids

    def get_gt_mesh_file(self, ob_id):
        # 抽象接口：由具体数据集子类实现
        raise RuntimeError("You should override this")

    def get_color(self, i):
        # 读取 RGB/灰度图
        color = imageio.imread(self.color_files[i])

        # 若是灰度图，复制成 3 通道
        if len(color.shape) == 2:
            color = np.tile(color[..., None], (1, 1, 3))

        # 若配置了 resize，则缩放
        if self.resize != 1:
            color = cv2.resize(color, fx=self.resize, fy=self.resize, dsize=None)

        return color

    def get_depth(self, i, filled=False):
        # 读取深度图
        # filled=True 时，读取补洞深度图
        if filled:
            depth_file = self.color_files[i].replace('rgb', 'depth_filled')
            depth_file = f'{os.path.dirname(depth_file)}/0{os.path.basename(depth_file)}'
            depth = cv2.imread(depth_file, -1) / 1e3
        else:
            depth_file = self.color_files[i].replace('rgb', 'depth').replace('gray', 'depth')
            # BOP 原始深度需乘深度比例，再转成米
            depth = cv2.imread(depth_file, -1) * 1e-3 * self.bop_depth_scale

        # resize 时深度图要用最近邻插值
        if self.resize != 1:
            depth = cv2.resize(depth, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)

        # 去除无效深度
        depth[depth < 0.001] = 0
        depth[depth > self.zfar] = 0
        return depth

    def get_xyz_map(self, i):
        # 用深度图和 K 反投影得到 xyz_map
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.get_K(i))
        return xyz_map

    def get_mask(self, i_frame: int, ob_id: int, type='mask_visib'):
        '''
        获取 mask
        @type:
            - mask_visib：只包含可见区域
            - mask：包含完整模型轮廓
        '''
        pos = 0
        name = int(os.path.basename(self.color_files[i_frame]).split('.')[0])

        if self.scene_gt is not None:
            # 在当前帧 scene_gt 中找到该 ob_id 对应的第几个实例
            for k in self.scene_gt[str(name)]:
                if k['obj_id'] == ob_id:
                    break
                pos += 1

            mask_file = f'{self.base_dir}/{type}/{name:06d}_{pos:06d}.png'
            if not os.path.exists(mask_file):
                logging.info(f'{mask_file} not found')
                return None
        else:
            raise RuntimeError

        # 读取 mask 并 resize
        mask = cv2.imread(mask_file, -1)
        if self.resize != 1:
            mask = cv2.resize(mask, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)

        return mask > 0

    def get_gt_mesh(self, ob_id: int):
        # 读取 CAD 模型，并把顶点单位从毫米转成米
        mesh_file = self.get_gt_mesh_file(ob_id)
        mesh = trimesh.load(mesh_file)
        mesh.vertices *= 1e-3
        return mesh

    def get_model_diameter(self, ob_id):
        # 从 models_info.json 中读取物体直径，并转成米
        dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
        info_file = f'{dir}/models_info.json'
        with open(info_file, 'r') as ff:
            info = json.load(ff)
        return info[str(ob_id)]['diameter'] / 1e3

    def get_gt_poses(self, i_frame, ob_id):
        # 获取当前帧中某个物体 ID 的所有 GT 位姿
        gt_poses = []
        name = int(self.id_strs[i_frame])
        for i_k, k in enumerate(self.scene_gt[str(name)]):
            if k['obj_id'] == ob_id:
                cur = np.eye(4)
                cur[:3, :3] = np.array(k['cam_R_m2c']).reshape(3, 3)
                cur[:3, 3] = np.array(k['cam_t_m2c']) / 1e3
                gt_poses.append(cur)
        return np.asarray(gt_poses).reshape(-1, 4, 4)

    def get_gt_pose(self, i_frame: int, ob_id, mask=None, use_my_correction=False):
        # 获取单个位姿
        # 若该帧中存在多个同类实例，可通过传入 mask 来根据 IoU 选最匹配的那个
        ob_in_cam = np.eye(4)
        best_iou = -np.inf
        best_gt_mask = None
        name = int(self.id_strs[i_frame])

        for i_k, k in enumerate(self.scene_gt[str(name)]):
            if k['obj_id'] == ob_id:
                cur = np.eye(4)
                cur[:3, :3] = np.array(k['cam_R_m2c']).reshape(3, 3)
                cur[:3, 3] = np.array(k['cam_t_m2c']) / 1e3

                # 如果传入了预测 mask，则用与 GT mask 的 IoU 选最匹配实例
                if mask is not None:
                    gt_mask = cv2.imread(f'{self.base_dir}/mask_visib/{self.id_strs[i_frame]}_{i_k:06d}.png',
                                         -1).astype(bool)
                    intersect = (gt_mask * mask).astype(bool)
                    union = (gt_mask + mask).astype(bool)
                    iou = float(intersect.sum()) / union.sum()
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_mask = gt_mask
                        ob_in_cam = cur
                else:
                    ob_in_cam = cur
                    break

        # 对某些 YCB 训练集中的已知标注问题做额外修正
        if use_my_correction:
            if 'ycb' in self.base_dir.lower() and 'train_real' in self.color_files[i_frame]:
                video_id = self.get_video_id()
                if ob_id == 1:
                    if video_id in [12, 13, 14, 17, 24]:
                        ob_in_cam = ob_in_cam @ self.symmetry_tfs[ob_id][1]

        return ob_in_cam

    def load_symmetry_tfs(self):
        # 读取所有物体的对称性信息，并生成对应的对称变换矩阵
        dir = os.path.dirname(self.get_gt_mesh_file(self.ob_ids[0]))
        info_file = f'{dir}/models_info.json'
        with open(info_file, 'r') as ff:
            info = json.load(ff)

        self.symmetry_tfs = {}
        self.symmetry_info_table = {}

        for ob_id in self.ob_ids:
            self.symmetry_info_table[ob_id] = info[str(ob_id)]
            self.symmetry_tfs[ob_id] = symmetry_tfs_from_info(info[str(ob_id)], rot_angle_discrete=5)

        self.geometry_symmetry_info_table = copy.deepcopy(self.symmetry_info_table)

    def get_video_id(self):
        # 从 base_dir 中取出当前场景编号
        return int(self.base_dir.split('/')[-1])


# =========================================================================
# 以下是各个 BOP 数据集的具体 Reader
# 主要是：
# 1. 指定 dataset_name
# 2. 指定当前数据集有哪些物体 ID
# 3. 实现 get_gt_mesh_file()
# =========================================================================

class LinemodOcclusionReader(BopBaseReader):
    def __init__(self, base_dir='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD-O/lmo_test_all/test/000002',
                 zfar=np.inf):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'lmo'
        # 取第一帧 K 作为常用 K
        self.K = list(self.K_table.values())[0]

        # LMO 数据集中的物体 ID
        self.ob_ids = [1, 5, 6, 8, 9, 10, 11, 12]

        # 物体 ID 到名字映射
        self.ob_id_to_names = {
            1: 'ape',
            2: 'benchvise',
            3: 'bowl',
            4: 'camera',
            5: 'water_pour',
            6: 'cat',
            7: 'cup',
            8: 'driller',
            9: 'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            13: 'iron',
            14: 'lamp',
            15: 'phone',
        }

        # 加载对称性变换
        self.load_symmetry_tfs()

    def get_gt_mesh_file(self, ob_id):
        # 返回 LMO 模型路径
        mesh_dir = f'{BOP_DIR}/{self.dataset_name}/models/obj_{ob_id:06d}.ply'
        return mesh_dir


class LinemodReader(LinemodOcclusionReader):
    def __init__(self, base_dir='/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD/lm_test_all/test/000001',
                 zfar=np.inf, split=None):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'lm'

        # 若指定 split，则按 split.txt 重新组织 color_files
        if split is not None:
            with open(
                    f'/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD/Linemod_preprocessed/data/{self.get_video_id():02d}/{split}.txt',
                    'r') as ff:
                lines = ff.read().splitlines()
            self.color_files = []
            for line in lines:
                id = int(line)
                self.color_files.append(f'{self.base_dir}/rgb/{id:06d}.png')
            self.make_id_strs()

        # LINEMOD 去掉 bowl 和 cup
        self.ob_ids = np.setdiff1d(np.arange(1, 16), np.array([7, 3])).tolist()
        self.load_symmetry_tfs()

    def get_gt_mesh_file(self, ob_id):
        # 逐级向上寻找 lm_models 目录
        root = self.base_dir
        while 1:
            if os.path.exists(f'{root}/lm_models'):
                mesh_dir = f'{root}/lm_models/models/obj_{ob_id:06d}.ply'
                break
            else:
                root = os.path.abspath(f'{root}/../')
        return mesh_dir

    def get_reconstructed_mesh(self, ob_id, ref_view_dir):
        # 读取重建模型
        mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
        return mesh


class YcbVideoReader(BopBaseReader):
    def __init__(self, base_dir, zfar=np.inf):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'ycbv'
        self.K = list(self.K_table.values())[0]

        self.make_id_strs()

        # YCBV 的 21 个物体
        self.ob_ids = np.arange(1, 22).astype(int).tolist()

        # 构建物体 ID <-> 名字映射
        YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
        names = sorted(os.listdir(f'{YCB_VIDEO_DIR}/models/'))
        self.ob_id_to_names = {}
        self.name_to_ob_id = {}
        for i, ob_id in enumerate(self.ob_ids):
            self.ob_id_to_names[ob_id] = names[i]
            self.name_to_ob_id[names[i]] = ob_id

        # 非 BOP 结构时读取关键帧列表
        if 'BOP' not in self.base_dir:
            with open(f'{self.base_dir}/../../keyframe.txt', 'r') as ff:
                self.keyframe_lines = ff.read().splitlines()

        self.load_symmetry_tfs()

        # 为一些具有几何对称性的物体补充更明确的对称信息
        for ob_id in self.ob_ids:
            if ob_id in [1, 4, 6, 18]:
                # 圆柱类连续对称
                self.geometry_symmetry_info_table[ob_id] = {
                    'symmetries_continuous': [
                        {'axis': [0, 0, 1], 'offset': [0, 0, 0]},
                    ],
                    'symmetries_discrete': euler_matrix(0, np.pi, 0).reshape(1, 4, 4).tolist(),
                }
            elif ob_id in [13]:
                # 另一类圆柱连续对称
                self.geometry_symmetry_info_table[ob_id] = {
                    'symmetries_continuous': [
                        {'axis': [0, 0, 1], 'offset': [0, 0, 0]},
                    ],
                }
            elif ob_id in [2, 3, 9, 21]:
                # 长方体支持多个离散翻转对称
                tfs = []
                for rz in [0, np.pi]:
                    for rx in [0, np.pi]:
                        for ry in [0, np.pi]:
                            tfs.append(euler_matrix(rx, ry, rz))
                self.geometry_symmetry_info_table[ob_id] = {
                    'symmetries_discrete': np.asarray(tfs).reshape(-1, 4, 4).tolist(),
                }
            else:
                pass

    def get_gt_mesh_file(self, ob_id):
        # 返回 YCBV 模型路径
        if 'BOP' in self.base_dir:
            mesh_file = os.path.abspath(f'{self.base_dir}/../../ycbv_models/models/obj_{ob_id:06d}.ply')
        else:
            mesh_file = f'{self.base_dir}/../../ycbv_models/models/obj_{ob_id:06d}.ply'
        return mesh_file

    def get_gt_mesh(self, ob_id: int, get_posecnn_version=False):
        # 支持两种方式读取 mesh：
        # 1) get_posecnn_version=True 时读取 textured_simple.obj
        # 2) 默认读取 BOP 的 ply 模型
        if get_posecnn_version:
            YCB_VIDEO_DIR = os.getenv('YCB_VIDEO_DIR')
            mesh = trimesh.load(f'{YCB_VIDEO_DIR}/models/{self.ob_id_to_names[ob_id]}/textured_simple.obj')
            return mesh

        mesh_file = self.get_gt_mesh_file(ob_id)
        mesh = trimesh.load(mesh_file, process=False)
        mesh.vertices *= 1e-3

        # 若存在 png 纹理，则加载纹理
        tex_file = mesh_file.replace('.ply', '.png')
        if os.path.exists(tex_file):
            from PIL import Image
            im = Image.open(tex_file)
            uv = mesh.visual.uv
            material = trimesh.visual.texture.SimpleMaterial(image=im)
            color_visuals = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
            mesh.visual = color_visuals

        return mesh

    def get_reconstructed_mesh(self, ob_id, ref_view_dir):
        # 读取重建出来的 mesh
        mesh = trimesh.load(os.path.abspath(f'{ref_view_dir}/ob_{ob_id:07d}/model/model.obj'))
        return mesh

    def get_transform_reconstructed_to_gt_model(self, ob_id):
        # 当前默认返回单位变换
        out = np.eye(4)
        return out

    def get_visible_cloud(self, ob_id):
        # 读取某个物体的可见点云
        file = os.path.abspath(f'{self.base_dir}/../../models/{self.ob_id_to_names[ob_id]}/visible_cloud.ply')
        pcd = o3d.io.read_point_cloud(file)
        return pcd

    def is_keyframe(self, i):
        # 判断第 i 帧是否为关键帧
        color_file = self.color_files[i]
        video_id = self.get_video_id()
        frame_id = int(os.path.basename(color_file).split('.')[0])
        key = f'{video_id:04d}/{frame_id:06d}'
        return (key in self.keyframe_lines)


class TlessReader(BopBaseReader):
    def __init__(self, base_dir, zfar=np.inf):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'tless'

        # TLESS 共有 30 个物体
        self.ob_ids = np.arange(1, 31).astype(int).tolist()
        self.load_symmetry_tfs()

    def get_gt_mesh_file(self, ob_id):
        # 返回 TLESS CAD 模型路径
        mesh_file = f'{self.base_dir}/../../../models_cad/obj_{ob_id:06d}.ply'
        return mesh_file

    def get_gt_mesh(self, ob_id):
        # TLESS 默认无纹理，读入后统一赋灰色纹理
        mesh = trimesh.load(self.get_gt_mesh_file(ob_id))
        mesh.vertices *= 1e-3
        mesh = trimesh_add_pure_colored_texture(mesh, color=np.ones((3)) * 200)
        return mesh


class HomebrewedReader(BopBaseReader):
    def __init__(self, base_dir, zfar=np.inf):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'hb'
        self.ob_ids = np.arange(1, 34).astype(int).tolist()
        self.load_symmetry_tfs()
        self.make_scene_ob_ids_dict()

    def get_gt_mesh_file(self, ob_id):
        # 返回 HB 模型路径
        mesh_file = f'{self.base_dir}/../../../hb_models/models/obj_{ob_id:06d}.ply'
        return mesh_file

    def get_gt_pose(self, i_frame: int, ob_id, use_my_correction=False):
        # Homebrewed 测试集通常没有公开 GT 位姿，这里给单位阵占位
        logging.info("WARN HomeBrewed doesn't have GT pose")
        return np.eye(4)


class ItoddReader(BopBaseReader):
    def __init__(self, base_dir, zfar=np.inf):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'itodd'
        self.make_id_strs()

        # ITODD 共有 28 个物体
        self.ob_ids = np.arange(1, 29).astype(int).tolist()
        self.load_symmetry_tfs()
        self.make_scene_ob_ids_dict()

    def get_gt_mesh_file(self, ob_id):
        # 返回 ITODD 模型路径
        mesh_file = f'{self.base_dir}/../../../itodd_models/models/obj_{ob_id:06d}.ply'
        return mesh_file


class IcbinReader(BopBaseReader):
    def __init__(self, base_dir, zfar=np.inf):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'icbin'
        self.ob_ids = np.arange(1, 3).astype(int).tolist()
        self.load_symmetry_tfs()

    def get_gt_mesh_file(self, ob_id):
        # 返回 ICBIN 模型路径
        mesh_file = f'{self.base_dir}/../../../icbin_models/models/obj_{ob_id:06d}.ply'
        return mesh_file


class TudlReader(BopBaseReader):
    def __init__(self, base_dir, zfar=np.inf):
        super().__init__(base_dir, zfar=zfar)
        self.dataset_name = 'tudl'
        self.ob_ids = np.arange(1, 4).astype(int).tolist()
        self.load_symmetry_tfs()

    def get_gt_mesh_file(self, ob_id):
        # 返回 TUDL 模型路径
        mesh_file = f'{self.base_dir}/../../../tudl_models/models/obj_{ob_id:06d}.ply'
        return mesh_file