import numpy as np
import os
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# 1. 初始化管道
pipeline = Pipeline()
config = Config()

# 2. 配置彩色流 (RGB)
plist_c = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
# 这里的 1280, 800 需要确保你的相机支持该分辨率，否则请根据实际情况修改
color_profile = plist_c.get_video_stream_profile(640, 0, OBFormat.MJPG, 30)
# color_profile = plist_c.get_video_stream_profile(1280, 800, OBFormat.MJPG, 30)
config.enable_stream(color_profile)

# 3. 配置深度流 (Depth)
plist_d = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
depth_profile = plist_d.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
# depth_profile = plist_d.get_video_stream_profile(1280, 800, OBFormat.Y16, 30)
config.enable_stream(depth_profile)

# 4. 启动管道
try:
    pipeline.start(config)

    # 5. 获取相机参数
    # 必须在 start 之后调用才能获取到准确的硬件参数
    cam_param = pipeline.get_camera_param()
    rgb_intr = cam_param.rgb_intrinsic
    depth_intr = cam_param.depth_intrinsic

    # 6. 构建内参矩阵
    # RGB 内参
    K_rgb = np.array([[rgb_intr.fx, 0, rgb_intr.cx],
                      [0, rgb_intr.fy, rgb_intr.cy],
                      [0, 0, 1]])

    # Depth 内参
    K_depth = np.array([[depth_intr.fx, 0, depth_intr.cx],
                        [0, depth_intr.fy, depth_intr.cy],
                        [0, 0, 1]])

    print("RGB 内参矩阵 K_rgb =\n", K_rgb)

    # 7. 保存文件 (核心修改部分)
    # 文件名
    save_filename = "cam_K.txt"

    # 使用 numpy.savetxt 保存
    # fmt='%.18e': 使用科学计数法，保留18位小数，匹配你的附件格式
    # delimiter=' ': 使用空格作为分隔符
    np.savetxt(save_filename, K_rgb, fmt='%.18e', delimiter=' ')

    print(f"\n[成功] RGB内参已保存至当前目录下的: {save_filename}")
    print("文件内容预览:")
    with open(save_filename, 'r') as f:
        print(f.read())

finally:
    # 8. 停止管道，释放资源
    pipeline.stop()