# Jetson / ARM64 FoundationPose 安装配置记录

适用环境：

```bash
uname -m
# aarch64

nvcc -V
# CUDA 13.0

conda 环境：
# /home/robot4/anaconda3/envs/6d
# Python 3.11
# PyTorch 2.11.0+cu130
```

---

## 1. 安装 Anaconda ARM64

下载 ARM64 版本：

```bash
Anaconda3-2025.12-2-Linux-aarch64.sh
```

安装：

```bash
cd ~/Downloads
bash Anaconda3-2025.12-2-Linux-aarch64.sh
```

安装过程中：

```text
Do you wish to initialize Anaconda3? yes
```

加载环境：

```bash
source ~/.bashrc
```

验证：

```bash
conda --version
```

---

## 2. 创建 conda 环境

推荐使用 Python 3.11：

```bash
conda create -n 6d python=3.11 -y
conda activate 6d
```

更新基础工具：

```bash
python -m pip install -U pip setuptools wheel
```

---

## 3. 检查 CUDA

```bash
nvcc -V
```

如果提示 `nvcc: command not found`，添加 CUDA 环境变量：

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

当前机器验证结果：

```text
Cuda compilation tools, release 13.0, V13.0.48
```

---

## 4. 安装 PyTorch CUDA 13

注意：本机是 `aarch64`，不要用普通 x86_64 的 cu124 源。

安装 PyTorch：

```bash
conda activate 6d

python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
PY
```

期望输出：

```text
torch: 2.11.0+cu130
cuda available: True
torch cuda: 13.0
```

---

## 5. 进入 FoundationPose 项目

```bash
cd ~/Programming/FoundationPose
conda activate 6d
```

设置环境变量：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
```

---

## 6. 安装系统编译依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  libopenblas-dev \
  libgl1 \
  libglib2.0-0
```

---

## 7. 安装 Python 基础依赖

```bash
python -m pip install -U \
  ninja \
  cmake \
  pybind11 \
  numpy \
  scipy \
  cython \
  imageio \
  imageio-ffmpeg \
  trimesh \
  joblib \
  psutil \
  pyyaml \
  tqdm \
  opencv-python \
  open3d \
  pandas \
  matplotlib \
  ruamel.yaml \
  transformations \
  kornia
```

---

## 8. 安装 NVDiffRast

```bash
python -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/NVlabs/nvdiffrast.git"
```

---

## 9. 安装 PyTorch3D

ARM64 / CUDA 13 下不要用 `py39_cu118_pyt200` wheel，需要源码编译：

```bash
python -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/facebookresearch/pytorch3d.git"
```

---

## 10. 安装 FoundationPose requirements

```bash
cd ~/Programming/FoundationPose
python -m pip install -r requirements.txt
```

如果中途已有包冲突，以当前 PyTorch CUDA 13 为准，不要降级 torch。

---

## 11. 编译 mycpp 扩展

如果运行时报：

```text
AttributeError: 'NoneType' object has no attribute 'cluster_poses'
```

说明 `mycpp` 没编译或没加载成功。

重新编译：

```bash
cd ~/Programming/FoundationPose/mycpp

rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

make -j$(nproc)
```

把生成的 `.so` 复制到项目根目录：

```bash
cp *.so ~/Programming/FoundationPose/
```

验证：

```bash
cd ~/Programming/FoundationPose

python - <<'PY'
import mycpp
print("mycpp OK")
print("has cluster_poses:", hasattr(mycpp, "cluster_poses"))
PY
```

---

## 12. 编译 FoundationPose 其它扩展

```bash
cd ~/Programming/FoundationPose
conda activate 6d

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"

bash build_all_conda.sh
```

---

## 13. 安装 Orbbec SDK

如果报错：

```text
ModuleNotFoundError: No module named 'pyorbbecsdk'
```

执行：

```bash
conda activate 6d
python -m pip install pyorbbecsdk
```

验证：

```bash
python - <<'PY'
from pyorbbecsdk import *
print("pyorbbecsdk ok")
PY
```

如果相机权限有问题，安装 udev 规则：

```bash
cd ~/Programming
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk

sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

然后重新插拔相机。

---

## 14. 安装 SAM3

如果项目需要 SAM3：

```bash
cd ~/Programming/FoundationPose
conda activate 6d

python -m pip install -U \
  transformers \
  accelerate \
  huggingface_hub \
  safetensors \
  pillow
```

安装 SAM3：

```bash
cd ~/Programming
git clone https://github.com/facebookresearch/sam3.git
cd sam3
python -m pip install -e .
```

如果项目要求本地路径在 FoundationPose 下：

```bash
cd ~/Programming/FoundationPose
git clone https://github.com/facebookresearch/sam3.git SAM3
cd SAM3
python -m pip install -e .
```

验证：

```bash
python - <<'PY'
import sam3
print("sam3 ok")
PY
```

如果需要 HuggingFace 权限：

```bash
huggingface-cli login
```

---

## 15. 修复 erode_depth 未定义

如果报错：

```text
[视觉] Refiner 优化失败：name 'erode_depth' is not defined
```

原因：

`Utils.py` 里只有在 `warp` 成功导入时才会定义 `erode_depth`：

```python
try:
  import warp as wp
  wp.init()
except:
  wp = None

if wp is not None:
  def erode_depth(...):
      ...
```

如果 `warp` 没安装成功，`erode_depth` 不会被定义。

### 方案 A：安装 warp

```bash
conda activate 6d
python -m pip install warp-lang
```

验证：

```bash
python - <<'PY'
import warp as wp
wp.init()
print("warp OK")
PY
```

### 方案 B：添加 fallback 版本

在 `~/Programming/FoundationPose/Utils.py` 文件末尾追加：

```python
def erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100):
    import numpy as np

    H, W = depth.shape
    out = np.zeros_like(depth)

    for y in range(H):
        for x in range(W):
            d = depth[y, x]
            if d < 0.001 or d >= zfar:
                continue

            bad = 0
            total = 0

            for yy in range(max(0, y-radius), min(H, y+radius+1)):
                for xx in range(max(0, x-radius), min(W, x+radius+1)):
                    total += 1
                    d2 = depth[yy, xx]
                    if d2 < 0.001 or d2 >= zfar or abs(d2 - d) > depth_diff_thres:
                        bad += 1

            if total > 0 and bad / total <= ratio_thres:
                out[y, x] = d

    return out
```

验证：

```bash
cd ~/Programming/FoundationPose

python - <<'PY'
from Utils import erode_depth
print("erode_depth OK:", erode_depth)
PY
```

---

## 16. 最终验证

```bash
cd ~/Programming/FoundationPose
conda activate 6d

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.version.cuda)

import nvdiffrast.torch as dr
print("nvdiffrast ok")

import pytorch3d
print("pytorch3d ok")

try:
    import mycpp
    print("mycpp ok:", hasattr(mycpp, "cluster_poses"))
except Exception as e:
    print("mycpp failed:", e)

try:
    from pyorbbecsdk import *
    print("pyorbbecsdk ok")
except Exception as e:
    print("pyorbbecsdk failed:", e)

try:
    from Utils import erode_depth
    print("erode_depth ok")
except Exception as e:
    print("erode_depth failed:", e)
PY
```

---

## 17. 运行项目

```bash
cd ~/Programming/FoundationPose
conda activate 6d

python Multi_Object_Teaching_Scene_Graph_V2.py
```

或者使用绝对路径：

```bash
/home/robot4/anaconda3/envs/6d/bin/python \
/home/robot4/Programming/FoundationPose/Multi_Object_Teaching_Scene_Graph_V2.py
```

---

## 18. 常见错误对照表

### 1. conda: command not found

```bash
~/anaconda3/bin/conda init
source ~/.bashrc
```

### 2. nvcc: command not found

```bash
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. No matching distribution found for torch

不要使用：

```bash
https://pypi.jetson-ai-lab.io/sbsa/cu129
```

使用：

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

### 4. No module named imageio

```bash
python -m pip install imageio imageio-ffmpeg
```

### 5. No module named pyorbbecsdk

```bash
python -m pip install pyorbbecsdk
```

### 6. mycpp is None / cluster_poses 报错

重新编译 `mycpp`：

```bash
cd ~/Programming/FoundationPose/mycpp
rm -rf build
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

make -j$(nproc)

cp *.so ~/Programming/FoundationPose/
```

### 7. erode_depth is not defined

安装 warp：

```bash
python -m pip install warp-lang
```

或者在 `Utils.py` 里补 fallback 函数。

---

## 当前最终状态

已经确认成功：

```text
torch: 2.11.0+cu130
cuda available: True
torch cuda: 13.0
nvcc: release 13.0, V13.0.48
mycpp.cluster_poses 正常
SAM / DINO / 模板匹配正常
```

剩余重点：

```text
确保 warp 或 erode_depth fallback 正常
```
