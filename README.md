# 6DoF 机器人视觉抓取系统

本项目是一个面向机器人抓取/放置任务的 6D 位姿估计与执行系统，核心流程包括 RGB-D 相机采集、基于模板与点云的粗分割/粗匹配、FoundationPose 位姿精修、手眼标定坐标变换、法奥（FAIRINO）机械臂控制，以及可选的 Qwen/Ollama 视觉语言模型作业视频分析。

> ⚠️ 说明：仓库中包含多处与实验设备强绑定的绝对路径、相机内参、手眼标定矩阵、机器人 IP/接口等参数。首次部署前请务必按实际硬件和目录结构修改配置，避免机械臂误动作或文件路径错误。

## 目录

- [功能概览](#功能概览)
- [项目结构](#项目结构)
- [硬件与软件依赖](#硬件与软件依赖)
- [环境安装](#环境安装)
- [编译 C++/CUDA 扩展](#编译-ccuda-扩展)
- [数据与模型准备](#数据与模型准备)
- [快速开始](#快速开始)
- [主要脚本说明](#主要脚本说明)
- [关键配置项](#关键配置项)
- [输出文件](#输出文件)
- [标定与相机工具](#标定与相机工具)
- [Qwen 作业视频分析](#qwen-作业视频分析)
- [常见问题](#常见问题)
- [安全注意事项](#安全注意事项)

## 功能概览

- **RGB-D 图像采集**：支持 Orbbec、Mech-Eye 等相机接口。
- **粗分割与模板匹配**：通过 `ImageCoarseSegmentor_pointcloud_many.py` 在当前图像中定位目标区域，并选择最佳模板。
- **6D 位姿估计**：调用 FoundationPose 相关网络对目标物体姿态进行精修，输出物体相对相机的 4x4 位姿矩阵。
- **机器人抓取/放置**：根据手眼标定矩阵和首件示教数据，将视觉位姿转换为机械臂末端抓取位姿。
- **GUI 操作界面**：提供基于 Qt 的可视化操作入口。
- **作业视频语义分析**：支持使用本地 Ollama + Qwen 视觉模型，将机器人作业视频拆解为原子动作。
- **标定辅助工具**：包含相机内参读取、手眼标定计算/验证、AprilTag 可视化等脚本和示例数据。

## 项目结构

```text
.
├── PickPlaceSystem.py                 # 命令行抓取/放置主程序
├── PickPlaceSystem_GUI.py             # Qt GUI 版本抓取/放置系统
├── PickPlaceSystem4.0_qwen.py         # 集成 Qwen 识别/决策的抓取系统实验版
├── vision_pose_estimator.py           # 视觉识别与 FoundationPose 位姿估计封装
├── ImageCoarseSegmentor_pointcloud_many.py
│                                      # 粗分割、点云/模板匹配模块
├── estimater.py / datareader.py / Utils.py
│                                      # FoundationPose 核心推理与工具函数
├── QWEN.py                            # 本地 Ollama + Qwen 视频原子动作分析
├── qwen_remote_client.py              # OpenAI-compatible 远程视觉模型调用示例
├── Cameras/                           # RGB-D 相机驱动与采集脚本
├── Robots/                            # 机器人控制封装
├── fairino/                           # FAIRINO SDK 相关代码
├── calibration/                       # 相机内参、手眼标定和可视化辅助文件
├── learning/                          # FoundationPose score/refine 网络结构与推理脚本
├── mesh/ / mesh_5.20/                 # 示例物体纹理、材质与网格资源
├── mycpp/                             # pybind11 C++ 扩展
├── requirements.txt                   # Python 依赖
├── environment.yml                    # Conda 基础环境
├── build_all_conda.sh                 # 扩展编译脚本
└── FoundationPose_Jetson_ARM64_Install_Guide.md
                                       # Jetson/ARM64 安装记录
```

## 硬件与软件依赖

### 推荐硬件

- NVIDIA GPU 工作站或 Jetson/ARM64 设备。
- 支持深度图输出的 RGB-D 相机，例如 Orbbec 或 Mech-Eye。
- FAIRINO/法奥机械臂及其 Python SDK。
- 用于吸取/夹取的末端执行器，例如电磁铁或夹爪。

### 推荐软件

- Ubuntu/Linux 环境。
- Conda 或 Miniconda/Anaconda。
- Python 3.11（`environment.yml` 默认配置）。
- CUDA 与当前 PyTorch 版本匹配的 NVIDIA 驱动。
- CMake、Ninja、C++ 编译器、Boost、Eigen、pybind11。
- OpenCV、Open3D、PyTorch、PyRender、Trimesh 等 Python 依赖。

## 环境安装

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate 6d
python -m pip install -U pip setuptools wheel
```

如果你不想使用 `environment.yml`，也可以手动创建：

```bash
conda create -n 6d python=3.11 -y
conda activate 6d
python -m pip install -U pip setuptools wheel
```

### 2. 安装 PyTorch

请根据你的设备、CUDA 版本和平台选择合适的 PyTorch 安装命令。不要盲目覆盖已经可用的 CUDA/PyTorch 环境。

示例（仅供参考，请以实际 CUDA 版本为准）：

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Jetson/ARM64 部署请参考：

```bash
FoundationPose_Jetson_ARM64_Install_Guide.md
```

### 3. 安装 Python 依赖

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 中有部分依赖被注释，目的是避免覆盖已有的 Torch、OpenCV、Open3D 或其它大版本依赖。若你的环境缺少某个包，请按报错信息单独安装。

### 4. 安装 FoundationPose 相关 GPU 依赖

根据实际环境安装：

```bash
python -m pip install --no-build-isolation --no-cache-dir "git+https://github.com/NVlabs/nvdiffrast.git"
python -m pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git"
```

> Jetson/ARM64 或新 CUDA 版本通常需要源码编译 PyTorch3D，不能直接使用 x86_64 的旧 wheel。

## 编译 C++/CUDA 扩展

仓库提供了 `build_all_conda.sh` 用于编译扩展：

```bash
bash build_all_conda.sh
```

该脚本会：

1. 进入 `mycpp/`，使用 CMake + pybind11 编译 `mycpp` 模块。
2. 尝试进入 `bundlesdf/mycuda` 并执行 editable 安装。

> ⚠️ 当前仓库文件列表中可能未包含 `bundlesdf/mycuda` 目录。如果脚本在该步骤失败，请确认是否需要补充子模块/外部依赖，或者只手动编译 `mycpp`：

```bash
cd mycpp
rm -rf build
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
```

编译后，如果 Python 无法导入 `mycpp`，请确认 `mycpp/build` 已加入 `PYTHONPATH`，或检查 `estimater.py` 中硬编码的 `sys.path.append(...)` 是否需要改为你的本地路径。

## 数据与模型准备

运行位姿估计和抓取系统前，需要准备以下内容：

1. **目标物体网格模型**
   - 通常为 `.obj`，并配套 `.mtl` 与纹理图。
   - 示例资源位于 `mesh/`、`mesh_5.20/`。
2. **模板库**
   - `template_dir` 应包含模板 RGB、深度、mask、模板位姿等文件。
   - 主程序默认模板目录为 `templates1280*720/back`，实际使用前请改成真实路径。
3. **网络权重**
   - FoundationPose 的 scorer/refiner 权重需与 `learning/` 下推理脚本匹配。
   - 若权重不在仓库中，请按你的训练/部署目录补齐。
4. **相机内参**
   - 默认内参在 `PickPlaceSystem.py` 中以 `MANUAL_K` 形式写死。
   - 也可使用 `calibration/cam_K.txt` 或相机 SDK 读取结果替换。
5. **手眼标定矩阵**
   - 默认矩阵为 `BTC_MATRIX`，含义为 Base -> Camera。
   - 部署到新机械臂/相机后必须重新标定。
6. **首件示教数据**
   - `TEACH_CTW1` 与 `TEACH_BTE1_POSE6` 用于计算固定抓取末端相对物体的变换。
   - 更换物体、夹具或抓取点后需要重新示教。

## 快速开始

### 命令行抓取/放置主程序

```bash
python PickPlaceSystem.py \
  --mesh_file /path/to/object.obj \
  --template_dir /path/to/templates \
  --save_root /path/to/output \
  --est_refine_iter 5
```

程序启动后会打开 `preview` 与 `vis` 窗口：

- `preview`：相机实时画面。
- `vis`：最近一次识别结果可视化。
- 按 `ESC` 退出程序。

### GUI 版本

```bash
python PickPlaceSystem_GUI.py
```

GUI 版本适合调试相机预览、单次识别、机器人回到拍照位/执行抓取等交互流程。首次运行前请检查 GUI 文件中的默认模型、模板、输出目录和硬件参数。

### 单次视觉位姿估计测试

```bash
python vision_pose_estimator.py
```

该脚本底部包含测试入口，但其中默认 `mesh_file`、`template_dir`、`color_path`、`depth_path` 等为硬编码路径。请先修改为本机实际路径再运行。

## 主要脚本说明

| 文件 | 作用 |
| --- | --- |
| `PickPlaceSystem.py` | 命令行自动抓取/放置主流程，包含相机采集、视觉线程、抓放线程和 OpenCV 预览窗口。 |
| `PickPlaceSystem_GUI.py` | Qt GUI 版主程序，适合可视化调试和半自动操作。 |
| `PickPlaceSystem4.0_qwen.py` | 集成 Qwen 视觉语言模型相关逻辑的实验版抓取系统。 |
| `vision_pose_estimator.py` | 封装粗分割、模板匹配、FoundationPose refiner、坐标轴可视化和结果保存。 |
| `ImageCoarseSegmentor_pointcloud_many.py` | 多模板/点云粗匹配与 mask 处理模块。 |
| `estimater.py` | FoundationPose 主要估计器类，负责旋转网格、评分网络、refiner 调用等。 |
| `datareader.py` / `Utils.py` | 深度图处理、网格/点云、可视化、数学变换等工具函数。 |
| `Cameras/OrbbecCamera.py` | Orbbec 相机初始化、读帧和测试入口。 |
| `Cameras/mecheye_camera.py` | Mech-Eye 相机读取示例。 |
| `Robots/robot_control.py` | 机械臂控制封装。 |
| `electromagnet.py` | 电磁铁/末端执行器控制示例。 |
| `QWEN.py` | 本地 Ollama + Qwen 视觉模型分析作业视频并生成原子动作序列。 |
| `qwen_remote_client.py` | 远程 OpenAI-compatible API 图片识别调用示例。 |
| `p_V.py` | 将图片序列合成为视频的小工具。 |

## 关键配置项

### `PickPlaceSystem.py`

部署前重点检查：

- `BTC_MATRIX`：机器人基坐标系到相机坐标系的手眼矩阵。
- `MANUAL_K` / `FX` / `FY` / `CX` / `CY`：相机内参。
- `HOME_JOINT`：机械臂拍照位/初始位关节角。
- `TARGET_PLACE_POSE6`：放置目标位姿。
- `TEACH_CTW1`：首件示教时目标物体在相机系下的位姿。
- `TEACH_BTE1_POSE6`：首件示教时机器人末端在基坐标系下的位姿。
- `--mesh_file`：目标物体网格。
- `--template_dir`：模板库路径。
- `--save_root`：输出目录。
- `--est_refine_iter`：refiner 迭代次数。

### `vision_pose_estimator.py`

部署前重点检查：

- 模型路径和模板路径。
- 测试 RGB-D 图像路径。
- 相机内参 `MANUAL_K`。
- `refine_iter`：迭代次数越高通常越稳，但推理更慢。
- `use_y180`：是否对结果进行 y 轴 180° 修正，取决于模型坐标系定义。

### `estimater.py`

该文件中存在类似以下硬编码路径：

```python
sys.path.append('/home/sunddy/Programming/FoundationPose/mycpp/build')
```

请改为你的本地 `mycpp/build` 路径，或通过 `PYTHONPATH` 管理导入路径。

## 输出文件

抓取/位姿估计流程通常会在 `save_root` 下保存：

- `cam_K.txt`：当前使用的相机内参。
- 输入彩色图、深度图、mask。
- 粗分割/模板匹配中间结果。
- 最终 6D 位姿矩阵或相关文本结果。
- 绘制坐标轴/检测结果后的可视化图片。

具体文件名取决于 `vision_pose_estimator.py` 与粗分割模块中的保存逻辑。

## 标定与相机工具

`calibration/` 目录包含：

- 相机内参读取脚本。
- 手眼标定计算脚本（Python/MATLAB）。
- 手眼标定验证脚本。
- AprilTag 可视化图片和 `tag_poses.csv`。
- 示例内参文件 `cam_K.txt`。

`Cameras/` 目录包含：

- Orbbec RGB-D 读取脚本。
- Orbbec RGB-D 录制脚本。
- Mech-Eye 相机读取脚本。
- 相机工具函数。

建议部署顺序：

1. 先确认相机 SDK 可以独立读到 RGB-D 数据。
2. 读取或标定相机内参。
3. 完成手眼标定并更新 `BTC_MATRIX`。
4. 使用静态 RGB-D 图像测试 `vision_pose_estimator.py`。
5. 再联动机械臂运行抓取主程序。

## Qwen 作业视频分析

### 本地 Ollama + Qwen

启动 Ollama 服务并准备视觉模型后运行：

```bash
python QWEN.py \
  --video /path/to/color_video.mp4 \
  --output_dir /path/to/qwen_output \
  --model qwen2.5vl:7b \
  --ollama_base_url http://127.0.0.1:11434 \
  --max_frames 10 \
  --timeout_s 600
```

输出目录中会保存抽帧图片、逐帧事件分析、合并后的原子动作 JSON/TXT 等结果。

### 远程 OpenAI-compatible API

`qwen_remote_client.py` 提供了调用远程 `/v1/chat/completions` 风格视觉模型的示例。默认测试参数写在脚本底部，运行前请修改：

- `test_image`
- `remote_api_base`
- `remote_timeout`
- `remote_model_name`
- `api_key`

## 常见问题

### 1. `ModuleNotFoundError: mycpp`

请先编译 `mycpp`，并确认 `mycpp/build` 在 Python 搜索路径中：

```bash
cd mycpp
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 2. `torch.cuda.is_available()` 为 `False`

请检查：

- NVIDIA 驱动是否正常。
- CUDA 版本是否与 PyTorch wheel 匹配。
- 是否在正确的 Conda 环境中运行。
- Jetson/ARM64 是否误装了 x86_64 wheel。

### 3. OpenGL/PyRender 报错

在无显示器或远程服务器上运行时，可能需要设置离屏渲染后端：

```bash
export PYOPENGL_PLATFORM=egl
```

如果仍失败，请检查 EGL、NVIDIA 驱动和 OpenGL 运行库是否安装完整。

### 4. 相机无法打开或深度为空

请确认：

- 相机 SDK 和 udev 权限配置正确。
- 设备没有被其它程序占用。
- RGB 与 Depth 分辨率和对齐模式与代码中假设一致。
- 相机内参和深度单位是否正确。

### 5. 机械臂不动作或连接失败

请确认：

- 机器人控制柜与工控机网络互通。
- FAIRINO SDK 可独立运行。
- `Robots/` 和 `fairino/` 中的连接参数与真实控制器一致。
- 程序启动前机械臂处于安全模式，并确认急停可用。

### 6. 位姿方向反了或抓取点偏移

请检查：

- 模型坐标系与真实物体坐标系是否一致。
- 是否需要开启/关闭 `use_y180`。
- 手眼矩阵方向是否为代码预期的 Base -> Camera。
- 深度单位是米还是毫米。
- 首件示教数据是否与当前夹具和抓取点一致。

## 安全注意事项

- 第一次联动机械臂时请使用低速度、空载、远离人员的测试环境。
- 每次修改 `BTC_MATRIX`、`HOME_JOINT`、`TARGET_PLACE_POSE6` 或首件示教数据后，都应先进行仿真或单步验证。
- 请保持急停按钮可触达，并确保末端执行器不会碰撞相机、治具或工件。
- 在未确认位姿估计稳定前，不建议开启全自动连续抓取。
- 修改硬件控制代码前，建议先断开机器人伺服或进入仿真模式进行测试。

## 维护建议

- 将硬编码路径逐步迁移到 YAML/JSON 配置文件。
- 为相机、机器人和视觉推理分别增加独立的 mock/test 模式。
- 将模板库、模型权重和实验输出排除在 Git 外，使用明确的数据目录管理。
- 为关键矩阵和坐标系变换增加单元测试，减少部署时的坐标系错误。
