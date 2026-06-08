import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

pipeline = Pipeline()
config   = Config()

plist_c = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
color_profile = plist_c.get_video_stream_profile(1280, 720, OBFormat.MJPG, 30)
config.enable_stream(color_profile)

plist_d = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
depth_profile = plist_d.get_video_stream_profile(1280, 720, OBFormat.Y16, 30)
config.enable_stream(depth_profile)

pipeline.start(config)

cam_param = pipeline.get_camera_param()
rgb_intr  = cam_param.rgb_intrinsic
depth_intr = cam_param.depth_intrinsic

K_rgb = np.array([[rgb_intr.fx, 0, rgb_intr.cx],
                  [0, rgb_intr.fy, rgb_intr.cy],
                  [0,           0,         1]])

K_depth = np.array([[depth_intr.fx, 0, depth_intr.cx],
                    [0, depth_intr.fy, depth_intr.cy],
                    [0,             0,           1]])

print("RGB 内参矩阵 K_rgb =\n", K_rgb)
print("Depth 内参矩阵 K_depth =\n", K_depth)

pipeline.stop()