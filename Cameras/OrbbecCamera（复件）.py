# import os
# import time
# import cv2
# import numpy as np
# import _thread as thread
#
# from pyorbbecsdk import *
# from utils import frame_to_bgr_image
#
#
# ESC_KEY = 27
# MIN_DEPTH = 20       # mm
# MAX_DEPTH = 10000    # mm
#
#
# # =========================================================
# # 手动严格几何对齐模块
# # 用于替代硬件 D2C / AlignFilter，支持任意彩色分辨率 + 任意深度分辨率
# # =========================================================
# class ManualD2CAligner:
#     def __init__(self, pipeline: Pipeline):
#         self.pipeline = pipeline
#
#         self.color_intrin = None
#         self.depth_intrin = None
#         self.R_d2c = None
#         self.t_d2c = None
#
#         self._load_calibration()
#
#     def _to_dict(self, obj):
#         if obj is None:
#             return {}
#         if isinstance(obj, dict):
#             return obj
#
#         out = {}
#         for name in dir(obj):
#             if name.startswith("_"):
#                 continue
#             try:
#                 val = getattr(obj, name)
#                 if callable(val):
#                     continue
#                 out[name] = val
#             except Exception:
#                 pass
#         return out
#
#     def _find_first(self, d, keys, default=None):
#         for k in keys:
#             if k in d:
#                 return d[k]
#         return default
#
#     def _parse_intrinsics(self, intrin_obj):
#         d = self._to_dict(intrin_obj)
#
#         fx = self._find_first(d, ["fx", "Fx", "focal_x"])
#         fy = self._find_first(d, ["fy", "Fy", "focal_y"])
#         cx = self._find_first(d, ["cx", "Cx", "ppx", "principal_x"])
#         cy = self._find_first(d, ["cy", "Cy", "ppy", "principal_y"])
#         width = self._find_first(d, ["width", "Width", "w"])
#         height = self._find_first(d, ["height", "Height", "h"])
#
#         if fx is None or fy is None or cx is None or cy is None:
#             raise RuntimeError(f"Failed to parse intrinsics from object: {d}")
#
#         return {
#             "fx": float(fx),
#             "fy": float(fy),
#             "cx": float(cx),
#             "cy": float(cy),
#             "width": int(width) if width is not None else None,
#             "height": int(height) if height is not None else None,
#         }
#
#     def _parse_extrinsics(self, extrin_obj):
#         d = self._to_dict(extrin_obj)
#
#         rotation = self._find_first(d, ["rotation", "rot", "R"])
#         translation = self._find_first(d, ["translation", "trans", "t", "T", "transform"])
#
#         if rotation is None or translation is None:
#             raise RuntimeError(f"Failed to parse extrinsics from object: {d}")
#
#         rotation = np.array(rotation, dtype=np.float64)
#         translation = np.array(translation, dtype=np.float64)
#
#         if rotation.shape == (3, 3):
#             R = rotation
#         else:
#             R = rotation.reshape(3, 3)
#
#         t = translation.reshape(3)
#
#         return R, t
#
#     def _load_calibration(self):
#         if not hasattr(self.pipeline, "get_camera_param"):
#             raise RuntimeError("pipeline.get_camera_param() not found in your pyorbbecsdk version")
#
#         cam_param = self.pipeline.get_camera_param()
#         cp = self._to_dict(cam_param)
#
#         color_intrin_obj = self._find_first(
#             cp, ["rgb_intrinsic", "color_intrinsic", "rgbIntrinsic", "colorIntrinsic"]
#         )
#         depth_intrin_obj = self._find_first(
#             cp, ["depth_intrinsic", "depthIntrinsic"]
#         )
#         d2c_extrin_obj = self._find_first(
#             cp, ["transform", "depth_to_color_extrinsic", "depthToColorExtrinsics", "depth_to_color"]
#         )
#
#         if color_intrin_obj is None or depth_intrin_obj is None or d2c_extrin_obj is None:
#             raise RuntimeError(f"Failed to locate calibration fields in camera_param: {list(cp.keys())}")
#
#         self.color_intrin = self._parse_intrinsics(color_intrin_obj)
#         self.depth_intrin = self._parse_intrinsics(depth_intrin_obj)
#         self.R_d2c, self.t_d2c = self._parse_extrinsics(d2c_extrin_obj)
#
#         print("========== Manual D2C Calibration ==========")
#         print(f"Color intrin: {self.color_intrin}")
#         print(f"Depth intrin: {self.depth_intrin}")
#         print("R_d2c:")
#         print(self.R_d2c)
#         print("t_d2c:")
#         print(self.t_d2c)
#         print("============================================")
#
#     def align_depth_to_color(self,
#                              depth_raw_u16: np.ndarray,
#                              depth_scale: float,
#                              color_width: int,
#                              color_height: int,
#                              min_depth_mm: float = MIN_DEPTH,
#                              max_depth_mm: float = MAX_DEPTH) -> np.ndarray:
#         """
#         将原始深度图严格投影到彩色图坐标系下
#         输出尺寸 = 彩色图尺寸
#         输出内容 = 对齐后的深度图（float32）
#         """
#         z = depth_raw_u16.astype(np.float32) * float(depth_scale)
#
#         valid = (z > min_depth_mm) & (z < max_depth_mm)
#         if not np.any(valid):
#             return np.zeros((color_height, color_width), dtype=np.float32)
#
#         v, u = np.nonzero(valid)
#         z = z[v, u].astype(np.float64)
#
#         fx_d = self.depth_intrin["fx"]
#         fy_d = self.depth_intrin["fy"]
#         cx_d = self.depth_intrin["cx"]
#         cy_d = self.depth_intrin["cy"]
#
#         fx_c = self.color_intrin["fx"]
#         fy_c = self.color_intrin["fy"]
#         cx_c = self.color_intrin["cx"]
#         cy_c = self.color_intrin["cy"]
#
#         # 1) 深度像素 -> 深度相机坐标系 3D 点
#         x_d = (u.astype(np.float64) - cx_d) * z / fx_d
#         y_d = (v.astype(np.float64) - cy_d) * z / fy_d
#         pts_d = np.stack([x_d, y_d, z], axis=0)  # (3, N)
#
#         # 2) 深度相机坐标系 -> 彩色相机坐标系
#         pts_c = self.R_d2c @ pts_d + self.t_d2c.reshape(3, 1)
#         Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]
#
#         valid_z = Zc > 1e-6
#         if not np.any(valid_z):
#             return np.zeros((color_height, color_width), dtype=np.float32)
#
#         Xc = Xc[valid_z]
#         Yc = Yc[valid_z]
#         Zc = Zc[valid_z]
#
#         # 3) 彩色相机坐标系 3D 点 -> 彩色图像像素
#         u_c = np.round(fx_c * Xc / Zc + cx_c).astype(np.int32)
#         v_c = np.round(fy_c * Yc / Zc + cy_c).astype(np.int32)
#
#         inside = (u_c >= 0) & (u_c < color_width) & (v_c >= 0) & (v_c < color_height)
#         if not np.any(inside):
#             return np.zeros((color_height, color_width), dtype=np.float32)
#
#         u_c = u_c[inside]
#         v_c = v_c[inside]
#         Zc = Zc[inside].astype(np.float32)
#
#         # 4) 栅格化到彩色图平面，同像素保留最近深度
#         aligned = np.full((color_height, color_width), np.inf, dtype=np.float32)
#         flat_idx = v_c * color_width + u_c
#         aligned_flat = aligned.reshape(-1)
#         np.minimum.at(aligned_flat, flat_idx, Zc)
#
#         aligned = aligned_flat.reshape(color_height, color_width)
#         aligned[np.isinf(aligned)] = 0.0
#         return aligned
#
#     def depth_to_colormap(self,
#                           depth_mm: np.ndarray,
#                           min_depth: float = MIN_DEPTH,
#                           max_depth: float = MAX_DEPTH) -> np.ndarray:
#         """
#         将对齐后的深度图转成伪彩色图，仅用于显示
#         """
#         d = np.clip(depth_mm, min_depth, max_depth)
#         if np.count_nonzero(d) == 0:
#             return np.zeros((*d.shape, 3), dtype=np.uint8)
#
#         valid = d > 0
#         out = np.zeros_like(d, dtype=np.float32)
#
#         d_valid = d[valid]
#         d_min = d_valid.min()
#         d_max = d_valid.max()
#
#         if d_max - d_min < 1e-6:
#             out[valid] = 255
#         else:
#             out[valid] = (d_valid - d_min) / (d_max - d_min) * 255.0
#
#         out = out.astype(np.uint8)
#         vis = cv2.applyColorMap(out, cv2.COLORMAP_JET)
#         vis[~valid] = 0
#         return vis
#
#
# # =========================================================
# # 相机类
# # 将手动对齐模块融入原有相机程序
# # =========================================================
# class OrbbecCamera:
#     def __init__(self,
#                  color_width=1280,
#                  color_height=720,
#                  color_format=OBFormat.RGB,
#                  color_fps=10,
#                  depth_width=848,
#                  depth_height=480,
#                  depth_format=OBFormat.Y16,
#                  depth_fps=10):
#         print("Initializing OrbbecCamera...")
#
#         self.trigger = False
#         self.running = True
#
#         self.current_color = None                  # 当前彩色图
#         self.current_depth = None                  # 当前对齐后伪彩深度图
#         self.current_depth_map = None              # 当前对齐后原始深度图（float32）
#         self.current_raw_depth_map = None          # 当前原始深度图（深度相机分辨率下）
#         self.current_color_frame_size = None
#         self.current_depth_frame_size = None
#
#         self.color_width = color_width
#         self.color_height = color_height
#         self.color_format = color_format
#         self.color_fps = color_fps
#
#         self.depth_width = depth_width
#         self.depth_height = depth_height
#         self.depth_format = depth_format
#         self.depth_fps = depth_fps
#
#         self.pipeline = Pipeline()
#         config = Config()
#
#         # -----------------------------
#         # 配置彩色流
#         # -----------------------------
#         color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
#         try:
#             color_profile = color_profile_list.get_video_stream_profile(
#                 self.color_width, self.color_height, self.color_format, self.color_fps
#             )
#         except Exception as e:
#             print(f"[WARN] Failed to get requested color profile: {e}")
#             color_profile = None
#
#             # 回退：只要宽高匹配即可
#             for i in range(len(color_profile_list)):
#                 p = color_profile_list[i]
#                 if p.get_width() == self.color_width and p.get_height() == self.color_height:
#                     color_profile = p
#                     print("[INFO] Fallback color profile:", color_profile)
#                     break
#
#             if color_profile is None:
#                 color_profile = color_profile_list.get_default_video_stream_profile()
#                 print("[INFO] Using default color profile:", color_profile)
#
#         config.enable_stream(color_profile)
#
#         # -----------------------------
#         # 配置深度流
#         # -----------------------------
#         depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
#         try:
#             depth_profile = depth_profile_list.get_video_stream_profile(
#                 self.depth_width, self.depth_height, self.depth_format, self.depth_fps
#             )
#         except Exception as e:
#             print(f"[WARN] Failed to get requested depth profile: {e}")
#             depth_profile = None
#
#             # 回退：只要宽高匹配即可
#             for i in range(len(depth_profile_list)):
#                 p = depth_profile_list[i]
#                 if p.get_width() == self.depth_width and p.get_height() == self.depth_height:
#                     depth_profile = p
#                     print("[INFO] Fallback depth profile:", depth_profile)
#                     break
#
#             if depth_profile is None:
#                 depth_profile = depth_profile_list.get_default_video_stream_profile()
#                 print("[INFO] Using default depth profile:", depth_profile)
#
#         config.enable_stream(depth_profile)
#
#         # 明确关闭硬件 D2C
#         config.set_align_mode(OBAlignMode.DISABLE)
#
#         try:
#             self.pipeline.enable_frame_sync()
#             print("[INFO] Frame sync enabled.")
#         except Exception as e:
#             print(f"[WARN] Frame sync enable failed: {e}")
#
#         self.pipeline.start(config)
#
#         print("========== Selected Stream Profiles ==========")
#         print(
#             f"Color : {color_profile.get_width()}x{color_profile.get_height()} "
#             f"@ {color_profile.get_fps()} format={color_profile.get_format()}"
#         )
#         print(
#             f"Depth : {depth_profile.get_width()}x{depth_profile.get_height()} "
#             f"@ {depth_profile.get_fps()} format={depth_profile.get_format()}"
#         )
#         print("Align : Manual geometric D2C")
#         print("=============================================")
#
#         # 初始化手动几何对齐模块
#         self.manual_aligner = ManualD2CAligner(self.pipeline)
#
#         # 启动采集线程
#         thread.start_new_thread(self.cam_thread, ())
#
#     def cam_thread(self):
#         print("OrbbecCamera thread has been started...")
#
#         while self.running:
#             try:
#                 frames: FrameSet = self.pipeline.wait_for_frames(100)
#                 if frames is None:
#                     continue
#
#                 color_frame = frames.get_color_frame()
#                 if color_frame is None:
#                     continue
#
#                 depth_frame = frames.get_depth_frame()
#                 if depth_frame is None:
#                     continue
#
#                 if depth_frame.get_format() != OBFormat.Y16:
#                     print("depth format is not Y16")
#                     continue
#
#                 # 彩色图
#                 color_image = frame_to_bgr_image(color_frame)
#                 if color_image is None:
#                     print("failed to convert frame to image")
#                     continue
#
#                 color_h, color_w = color_image.shape[:2]
#                 self.current_color_frame_size = (color_w, color_h)
#
#                 # 原始深度图（深度相机原分辨率）
#                 depth_w = depth_frame.get_width()
#                 depth_h = depth_frame.get_height()
#                 self.current_depth_frame_size = (depth_w, depth_h)
#
#                 depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_h, depth_w))
#                 depth_scale = depth_frame.get_depth_scale()
#
#                 depth_raw_mm = depth_raw.astype(np.float32) * depth_scale
#                 depth_raw_mm = np.where(
#                     (depth_raw_mm > MIN_DEPTH) & (depth_raw_mm < MAX_DEPTH),
#                     depth_raw_mm,
#                     0
#                 )
#                 self.current_raw_depth_map = depth_raw_mm.copy()
#
#                 # 手动严格几何对齐到彩色图坐标系
#                 aligned_depth_map = self.manual_aligner.align_depth_to_color(
#                     depth_raw_u16=depth_raw,
#                     depth_scale=depth_scale,
#                     color_width=color_w,
#                     color_height=color_h,
#                     min_depth_mm=MIN_DEPTH,
#                     max_depth_mm=MAX_DEPTH
#                 )
#
#                 self.current_depth_map = aligned_depth_map.copy()
#
#                 # 生成对齐后的伪彩色深度图
#                 depth_image = self.manual_aligner.depth_to_colormap(
#                     aligned_depth_map,
#                     min_depth=MIN_DEPTH,
#                     max_depth=MAX_DEPTH
#                 )
#
#                 if self.trigger:
#                     self.current_color = color_image.copy()
#                     self.current_depth = depth_image.copy()
#                     self.trigger = False
#
#             except KeyboardInterrupt:
#                 break
#             except Exception as e:
#                 print(f"[ERROR] cam_thread: {e}")
#                 time.sleep(0.01)
#
#         cv2.destroyAllWindows()
#         self.pipeline.stop()
#         self.running = True
#
#     def read(self):
#         self.trigger = True
#         while self.trigger:
#             time.sleep(0.001)
#
#     def stop(self):
#         self.running = False
#         while self.running is False:
#             time.sleep(0.001)
#         print("camera has been safely closed!")
#
#
# # =========================================================
# # 测试 / 保存示例
# # =========================================================
# if __name__ == "__main__":
#     cam = OrbbecCamera(
#         color_width=1280,
#         color_height=720,
#         color_format=OBFormat.RGB,
#         color_fps=10,
#         depth_width=848,
#         depth_height=480,
#         depth_format=OBFormat.Y16,
#         depth_fps=10,
#     )
#
#     output_dir = "/home/ma/FoundationPose/sanpshot_depth_rgb"
#     os.makedirs(output_dir, exist_ok=True)
#
#     while True:
#         cam.read()
#
#         if cam.current_depth is not None:
#             cv2.imshow("Depth Viewer (Aligned)", cam.current_depth)
#         if cam.current_color is not None:
#             cv2.imshow("Color Viewer", cam.current_color)
#
#         key = cv2.waitKey(1)
#
#         # 按 s 保存当前帧
#         if key == ord('s'):
#             ts = time.strftime("%Y%m%d_%H%M%S")
#
#             color_path = os.path.join(output_dir, f"color_{ts}.png")
#             depth_aligned_path = os.path.join(output_dir, f"depth_aligned_{ts}.tiff")
#             depth_raw_path = os.path.join(output_dir, f"depth_raw_{ts}.tiff")
#             depth_vis_path = os.path.join(output_dir, f"depth_vis_{ts}.png")
#
#             if cam.current_color is not None:
#                 cv2.imwrite(color_path, cam.current_color)
#
#             if cam.current_depth_map is not None:
#                 cv2.imwrite(depth_aligned_path, cam.current_depth_map)
#
#             if cam.current_raw_depth_map is not None:
#                 cv2.imwrite(depth_raw_path, cam.current_raw_depth_map)
#
#             if cam.current_depth is not None:
#                 cv2.imwrite(depth_vis_path, cam.current_depth)
#
#             print(f"[INFO] 已保存:")
#             print(f"       color       -> {color_path}")
#             print(f"       depth_align -> {depth_aligned_path}")
#             print(f"       depth_raw   -> {depth_raw_path}")
#             print(f"       depth_vis   -> {depth_vis_path}")
#
#         # 按 q 或 Esc 退出
#         if key == ord('q') or key == ESC_KEY:
#             break
#
#         time.sleep(0.001)
#
#     cam.stop()

import time
import numpy as np
import cv2
import _thread as thread
from pyorbbecsdk import *
from utils import frame_to_bgr_image


ESC_KEY = 27
MIN_DEPTH = 20      # 20 mm
MAX_DEPTH = 10000   # 10000 mm


# =========================================================
# 手动严格几何对齐模块
# 不依赖硬件 D2C / AlignFilter
# 输入：原始深度图 + 标定参数
# 输出：对齐到彩色图坐标系下的深度图
# =========================================================
class ManualD2CAligner:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

        self.color_intrin = None
        self.depth_intrin = None
        self.R_d2c = None
        self.t_d2c = None

        self._load_calibration()

    def _to_dict(self, obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj

        out = {}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                val = getattr(obj, name)
                if callable(val):
                    continue
                out[name] = val
            except Exception:
                pass
        return out

    def _find_first(self, d, keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    def _parse_intrinsics(self, intrin_obj):
        d = self._to_dict(intrin_obj)

        fx = self._find_first(d, ["fx", "Fx", "focal_x"])
        fy = self._find_first(d, ["fy", "Fy", "focal_y"])
        cx = self._find_first(d, ["cx", "Cx", "ppx", "principal_x"])
        cy = self._find_first(d, ["cy", "Cy", "ppy", "principal_y"])
        width = self._find_first(d, ["width", "Width", "w"])
        height = self._find_first(d, ["height", "Height", "h"])

        if fx is None or fy is None or cx is None or cy is None:
            raise RuntimeError(f"Failed to parse intrinsics from object: {d}")

        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "width": int(width) if width is not None else None,
            "height": int(height) if height is not None else None,
        }

    def _parse_extrinsics(self, extrin_obj):
        d = self._to_dict(extrin_obj)

        rotation = self._find_first(d, ["rotation", "rot", "R"])
        translation = self._find_first(d, ["translation", "trans", "t", "T", "transform"])

        if rotation is None or translation is None:
            raise RuntimeError(f"Failed to parse extrinsics from object: {d}")

        rotation = np.array(rotation, dtype=np.float64)
        translation = np.array(translation, dtype=np.float64)

        if rotation.shape == (3, 3):
            R = rotation
        else:
            R = rotation.reshape(3, 3)

        t = translation.reshape(3)
        return R, t

    def _load_calibration(self):
        if not hasattr(self.pipeline, "get_camera_param"):
            raise RuntimeError("pipeline.get_camera_param() not found in your pyorbbecsdk version")

        cam_param = self.pipeline.get_camera_param()
        cp = self._to_dict(cam_param)

        color_intrin_obj = self._find_first(
            cp, ["rgb_intrinsic", "color_intrinsic", "rgbIntrinsic", "colorIntrinsic"]
        )
        depth_intrin_obj = self._find_first(
            cp, ["depth_intrinsic", "depthIntrinsic"]
        )
        d2c_extrin_obj = self._find_first(
            cp, ["transform", "depth_to_color_extrinsic", "depthToColorExtrinsics", "depth_to_color"]
        )

        if color_intrin_obj is None or depth_intrin_obj is None or d2c_extrin_obj is None:
            raise RuntimeError(f"Failed to locate calibration fields in camera_param: {list(cp.keys())}")

        self.color_intrin = self._parse_intrinsics(color_intrin_obj)
        self.depth_intrin = self._parse_intrinsics(depth_intrin_obj)
        self.R_d2c, self.t_d2c = self._parse_extrinsics(d2c_extrin_obj)

        print("========== Manual D2C Calibration ==========")
        print(f"Color intrin: {self.color_intrin}")
        print(f"Depth intrin: {self.depth_intrin}")
        print("R_d2c:")
        print(self.R_d2c)
        print("t_d2c:")
        print(self.t_d2c)
        print("============================================")

    def align_depth_to_color(self,
                             depth_raw_u16: np.ndarray,
                             depth_scale: float,
                             color_width: int,
                             color_height: int,
                             min_depth_mm: float = MIN_DEPTH,
                             max_depth_mm: float = MAX_DEPTH) -> np.ndarray:
        """
        输出：
            aligned_depth_mm: float32, 单位 mm, 尺寸与彩色图一致
        """
        z = depth_raw_u16.astype(np.float32) * float(depth_scale)

        valid = (z > min_depth_mm) & (z < max_depth_mm)
        if not np.any(valid):
            return np.zeros((color_height, color_width), dtype=np.float32)

        v, u = np.nonzero(valid)
        z = z[v, u].astype(np.float64)

        fx_d = self.depth_intrin["fx"]
        fy_d = self.depth_intrin["fy"]
        cx_d = self.depth_intrin["cx"]
        cy_d = self.depth_intrin["cy"]

        fx_c = self.color_intrin["fx"]
        fy_c = self.color_intrin["fy"]
        cx_c = self.color_intrin["cx"]
        cy_c = self.color_intrin["cy"]

        # 1) 深度像素 -> 深度相机坐标系 3D 点
        x_d = (u.astype(np.float64) - cx_d) * z / fx_d
        y_d = (v.astype(np.float64) - cy_d) * z / fy_d
        pts_d = np.stack([x_d, y_d, z], axis=0)  # (3, N)

        # 2) 深度相机坐标系 -> 彩色相机坐标系
        pts_c = self.R_d2c @ pts_d + self.t_d2c.reshape(3, 1)
        Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]

        valid_z = Zc > 1e-6
        if not np.any(valid_z):
            return np.zeros((color_height, color_width), dtype=np.float32)

        Xc = Xc[valid_z]
        Yc = Yc[valid_z]
        Zc = Zc[valid_z]

        # 3) 彩色相机坐标系 3D -> 彩色图像像素
        u_c = np.round(fx_c * Xc / Zc + cx_c).astype(np.int32)
        v_c = np.round(fy_c * Yc / Zc + cy_c).astype(np.int32)

        inside = (u_c >= 0) & (u_c < color_width) & (v_c >= 0) & (v_c < color_height)
        if not np.any(inside):
            return np.zeros((color_height, color_width), dtype=np.float32)

        u_c = u_c[inside]
        v_c = v_c[inside]
        Zc = Zc[inside].astype(np.float32)

        # 4) 栅格化，保留最近深度
        aligned = np.full((color_height, color_width), np.inf, dtype=np.float32)
        flat_idx = v_c * color_width + u_c
        aligned_flat = aligned.reshape(-1)
        np.minimum.at(aligned_flat, flat_idx, Zc)

        aligned = aligned_flat.reshape(color_height, color_width)
        aligned[np.isinf(aligned)] = 0.0
        return aligned

    def depth_to_colormap(self,
                          depth_mm: np.ndarray,
                          min_depth: float = MIN_DEPTH,
                          max_depth: float = MAX_DEPTH) -> np.ndarray:
        d = np.clip(depth_mm, min_depth, max_depth)

        if np.count_nonzero(d) == 0:
            return np.zeros((*d.shape, 3), dtype=np.uint8)

        valid = d > 0
        out = np.zeros_like(d, dtype=np.float32)

        d_valid = d[valid]
        d_min = d_valid.min()
        d_max = d_valid.max()

        if d_max - d_min < 1e-6:
            out[valid] = 255
        else:
            out[valid] = (d_valid - d_min) / (d_max - d_min) * 255.0

        out = out.astype(np.uint8)
        vis = cv2.applyColorMap(out, cv2.COLORMAP_JET)
        vis[~valid] = 0
        return vis


class OrbbecCamera:
    def __init__(self):
        print("Initializing OrbbecCamera...")
        self.trigger = False
        self.running = True

        # 与你原主程序保持兼容的输出成员
        self.current_color = None          # BGR 彩图
        self.current_depth = None          # 伪彩深度图（对齐后）
        self.current_depth_map = None      # 对齐后的深度图，单位 mm，float32
        self.current_raw_depth_map = None  # 原始深度图，单位 mm，float32

        config = Config()
        self.pipeline = Pipeline()

        # =====================================================
        # 目标：保持主程序原来数据形式
        # 所以彩色输出仍然走 640x480
        # 深度优先尝试 848x480，再手动对齐到 640x480
        # =====================================================

        # -------------------- COLOR --------------------
        color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None

        try:
            color_profile = color_profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        except Exception as e:
            print(f"[WARN] get color profile 640x480 RGB@30 failed: {e}")

        if color_profile is None:
            try:
                color_profile = color_profile_list.get_video_stream_profile(640, 480, OBFormat.MJPG, 30)
            except Exception as e:
                print(f"[WARN] get color profile 640x480 MJPG@30 failed: {e}")

        if color_profile is None:
            color_profile = color_profile_list.get_default_video_stream_profile()
            print("[INFO] use default color profile:", color_profile)

        config.enable_stream(color_profile)

        # -------------------- DEPTH --------------------
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = None

        # 优先 848x480 Y16
        try:
            depth_profile = depth_profile_list.get_video_stream_profile(848, 480, OBFormat.Y16, 30)
        except Exception as e:
            print(f"[WARN] get depth profile 848x480 Y16@30 failed: {e}")

        # 回退 640x480 Y16
        if depth_profile is None:
            try:
                depth_profile = depth_profile_list.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
            except Exception as e:
                print(f"[WARN] get depth profile 640x480 Y16@30 failed: {e}")

        if depth_profile is None:
            depth_profile = depth_profile_list.get_default_video_stream_profile()
            print("[INFO] use default depth profile:", depth_profile)

        config.enable_stream(depth_profile)

        # 关闭硬件 D2C
        config.set_align_mode(OBAlignMode.DISABLE)

        try:
            self.pipeline.enable_frame_sync()
            print("[INFO] Frame sync enabled.")
        except Exception as e:
            print(f"[WARN] Frame sync enable failed: {e}")

        self.pipeline.start(config)

        print("========== Selected Stream Profiles ==========")
        print(
            f"Color : {color_profile.get_width()}x{color_profile.get_height()} "
            f"@ {color_profile.get_fps()} format={color_profile.get_format()}"
        )
        print(
            f"Depth : {depth_profile.get_width()}x{depth_profile.get_height()} "
            f"@ {depth_profile.get_fps()} format={depth_profile.get_format()}"
        )
        print("Align : Manual geometric D2C")
        print("Output: Keep main-program-compatible data format")
        print("=============================================")

        # 手动对齐模块
        self.manual_aligner = ManualD2CAligner(self.pipeline)

        thread.start_new_thread(self.cam_thread, ())

    def cam_thread(self):
        print("OrbbecCamera thread has been started...")

        while self.running:
            try:
                frames: FrameSet = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue

                if depth_frame.get_format() != OBFormat.Y16:
                    print("depth format is not Y16")
                    continue

                # 转彩色图
                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    print("failed to convert frame to image")
                    continue

                color_h, color_w = color_image.shape[:2]

                # 原始深度
                depth_w = depth_frame.get_width()
                depth_h = depth_frame.get_height()
                depth_scale = depth_frame.get_depth_scale()

                depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_h, depth_w))

                # 原始深度图（mm）
                depth_raw_mm = depth_raw.astype(np.float32) * depth_scale
                depth_raw_mm = np.where(
                    (depth_raw_mm > MIN_DEPTH) & (depth_raw_mm < MAX_DEPTH),
                    depth_raw_mm,
                    0
                )
                self.current_raw_depth_map = depth_raw_mm.copy()

                # 手动严格几何对齐到彩色图分辨率
                aligned_depth_map = self.manual_aligner.align_depth_to_color(
                    depth_raw_u16=depth_raw,
                    depth_scale=depth_scale,
                    color_width=color_w,
                    color_height=color_h,
                    min_depth_mm=MIN_DEPTH,
                    max_depth_mm=MAX_DEPTH
                )

                # 这里保持主程序原来的数据语义：
                # current_depth_map = 与 current_color 同分辨率的深度图，单位 mm
                self.current_depth_map = aligned_depth_map.astype(np.float32)

                # 伪彩色深度图
                depth_image = self.manual_aligner.depth_to_colormap(
                    self.current_depth_map,
                    min_depth=MIN_DEPTH,
                    max_depth=MAX_DEPTH
                )

                if self.trigger:
                    self.current_color = color_image.copy()
                    self.current_depth = depth_image.copy()
                    self.trigger = False

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] cam_thread: {e}")
                time.sleep(0.01)

        cv2.destroyAllWindows()
        self.pipeline.stop()
        self.running = True

    def read(self):
        self.trigger = True
        while self.trigger:
            time.sleep(0.001)

    def stop(self):
        self.running = False
        while self.running is False:
            time.sleep(0.001)
        print("camera has been safely closed!")


if __name__ == "__main__":
    import os

    cam = OrbbecCamera()

    output_dir = "/home/ma/FoundationPose/sanpshot_depth_rgb"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        cam.read()

        if cam.current_depth is not None:
            cv2.imshow("Depth Viewer", cam.current_depth)
        if cam.current_color is not None:
            cv2.imshow("Color Viewer", cam.current_color)

        key = cv2.waitKey(1)

        # 按 s 保存当前帧
        if key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            depth_path = os.path.join(output_dir, f"depth_{ts}.tiff")
            color_path = os.path.join(output_dir, f"color_{ts}.png")
            depth_raw_path = os.path.join(output_dir, f"depth_raw_{ts}.tiff")
            depth_vis_path = os.path.join(output_dir, f"depth_vis_{ts}.png")

            if cam.current_depth_map is not None:
                cv2.imwrite(depth_path, cam.current_depth_map)
            if cam.current_color is not None:
                cv2.imwrite(color_path, cam.current_color)
            if cam.current_raw_depth_map is not None:
                cv2.imwrite(depth_raw_path, cam.current_raw_depth_map)
            if cam.current_depth is not None:
                cv2.imwrite(depth_vis_path, cam.current_depth)

            print(f"[INFO] 已保存 {depth_path} 和 {color_path}")

        # 按 q 或 Esc 退出
        if key == ord('q') or key == ESC_KEY:
            break

        time.sleep(0.001)

    cam.stop()