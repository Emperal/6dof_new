import sys, os
import io
import cv2
import numpy as np

candidate_paths = [
    "/opt/Mech-Eye SDK/lib/python3",
    "/usr/local/Mech-Eye SDK/lib/python3",
    "/home/sunddy/Mech-Eye SDK/lib/python3",
    "/home/sunddy/anaconda3/envs/orbbec/lib/python3.10/site-packages"
]

for p in candidate_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

try:
    from mecheye.shared import *
    from mecheye.area_scan_3d_camera import *
    from mecheye.area_scan_3d_camera_utils import find_and_connect
except ModuleNotFoundError as e:
    raise ImportError(
        "\n❌ Could not import Mech-Eye SDK.\n"
        "Please verify the SDK path and installation.\n"
        "Detected paths:\n" + "\n".join(candidate_paths)
    ) from e


class CaptureAllData:
    """Mech-Eye 相机采集类：支持 save=True 保存或直接返回内存数据"""

    def __init__(self):
        self.camera = Camera()
        self.frame_all_2d_3d = Frame2DAnd3D()
        self.connected = False

    def connect(self):
        """自动选择第一台相机"""
        original_stdin = sys.stdin
        sys.stdin = io.StringIO("0\n")
        try:
            if not find_and_connect(self.camera):
                raise RuntimeError("[ERROR] 无法连接到 Mech-Eye 相机。")
            self.connected = True
            print("[INFO] 相机连接成功。")
        finally:
            sys.stdin = original_stdin

    def disconnect(self):
        if self.connected:
            self.camera.disconnect()
            self.connected = False
            print("[INFO] 相机已断开。")

    def capture(self, save=False, save_dir="datas", prefix=""):
        """采集彩色图、深度图，可选保存"""
        if not self.connected:
            raise RuntimeError("请先调用 connect() 连接相机。")

        show_error(self.camera.capture_2d_and_3d(self.frame_all_2d_3d))
        print("[INFO] 成功采集一帧 2D + 3D 数据。")

        frame2d = self.frame_all_2d_3d.frame_2d()
        if frame2d.color_type() == ColorTypeOf2DCamera_Color:
            color_img = frame2d.get_color_image().data().copy()
        else:
            color_img = frame2d.get_gray_scale_image().data().copy()

        frame3d = self.frame_all_2d_3d.frame_3d()
        depth_img = frame3d.get_depth_map().data().copy()

        if save:
            os.makedirs(save_dir, exist_ok=True)
            color_path = os.path.join(save_dir, f"{prefix}Color.png")
            depth_path = os.path.join(save_dir, f"{prefix}DepthMap.tiff")
            cv2.imwrite(color_path, color_img)
            cv2.imwrite(depth_path, depth_img)
            print(f"[OK] 已保存彩色图: {color_path}")
            print(f"[OK] 已保存深度图: {depth_path}")
            return color_img, depth_img, color_path, depth_path

        return color_img, depth_img, None, None


if __name__ == "__main__":
    cam = CaptureAllData()
    cam.connect()
    try:
        color_img, depth_img, color_path, depth_path = cam.capture(save=True, save_dir="datas/test")
        print(f"[INFO] 彩色图尺寸: {color_img.shape}, 深度图尺寸: {depth_img.shape}")
    finally:
        cam.disconnect()
