import os
import sys
import time
import json
import cv2
import numpy as np
import threading

from pyorbbecsdk import *
from utils import frame_to_bgr_image


ESC_KEY = 27
MIN_DEPTH = 20       # mm
MAX_DEPTH = 10000    # mm


# =========================================================
# 手动严格几何对齐模块
# 深度和彩色都采集 640x480，但仍然需要 D2C 对齐
# 因为彩色相机和深度相机光心不同
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
            cp,
            ["rgb_intrinsic", "color_intrinsic", "rgbIntrinsic", "colorIntrinsic"]
        )

        depth_intrin_obj = self._find_first(
            cp,
            ["depth_intrinsic", "depthIntrinsic"]
        )

        d2c_extrin_obj = self._find_first(
            cp,
            ["transform", "depth_to_color_extrinsic", "depthToColorExtrinsics", "depth_to_color"]
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

    def align_depth_to_color(
        self,
        depth_raw_u16: np.ndarray,
        depth_scale: float,
        color_width: int,
        color_height: int,
        min_depth_mm: float = MIN_DEPTH,
        max_depth_mm: float = MAX_DEPTH,
    ) -> np.ndarray:
        """
        将原始深度图投影到彩色图坐标系下。

        输出：
            aligned_depth_mm: float32
            单位：mm
            尺寸：color_height x color_width
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

        # 1. 深度像素 -> 深度相机 3D 点
        x_d = (u.astype(np.float64) - cx_d) * z / fx_d
        y_d = (v.astype(np.float64) - cy_d) * z / fy_d
        pts_d = np.stack([x_d, y_d, z], axis=0)

        # 2. 深度相机坐标系 -> 彩色相机坐标系
        pts_c = self.R_d2c @ pts_d + self.t_d2c.reshape(3, 1)
        Xc, Yc, Zc = pts_c[0], pts_c[1], pts_c[2]

        valid_z = Zc > 1e-6
        if not np.any(valid_z):
            return np.zeros((color_height, color_width), dtype=np.float32)

        Xc = Xc[valid_z]
        Yc = Yc[valid_z]
        Zc = Zc[valid_z]

        # 3. 彩色相机 3D 点 -> 彩色图像像素
        u_c = np.round(fx_c * Xc / Zc + cx_c).astype(np.int32)
        v_c = np.round(fy_c * Yc / Zc + cy_c).astype(np.int32)

        inside = (
            (u_c >= 0)
            & (u_c < color_width)
            & (v_c >= 0)
            & (v_c < color_height)
        )

        if not np.any(inside):
            return np.zeros((color_height, color_width), dtype=np.float32)

        u_c = u_c[inside]
        v_c = v_c[inside]
        Zc = Zc[inside].astype(np.float32)

        # 4. 栅格化，同一个像素保留最近深度
        aligned = np.full((color_height, color_width), np.inf, dtype=np.float32)
        flat_idx = v_c * color_width + u_c
        aligned_flat = aligned.reshape(-1)

        np.minimum.at(aligned_flat, flat_idx, Zc)

        aligned = aligned_flat.reshape(color_height, color_width)
        aligned[np.isinf(aligned)] = 0.0

        return aligned

    def depth_to_colormap(
        self,
        depth_mm: np.ndarray,
        min_depth: float = MIN_DEPTH,
        max_depth: float = MAX_DEPTH,
    ) -> np.ndarray:
        """
        深度图转伪彩色图，仅用于显示和保存预览视频。
        不要把这个当成真实深度给 FoundationPose。
        """
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


# =========================================================
# Orbbec 相机类
# 彩色 640x480
# 深度 640x480
# 手动 D2C 对齐到彩色图坐标系
# =========================================================
class OrbbecCamera:
    def __init__(
        self,
        color_width: int = 640,
        color_height: int = 480,
        color_fps: int = 30,
        depth_width: int = 640,
        depth_height: int = 480,
        depth_fps: int = 30,
    ):
        print("Initializing OrbbecCamera...")

        self.trigger = False
        self.running = True
        self.lock = threading.Lock()

        self.current_color = None             # BGR 彩图，640x480
        self.current_depth = None             # 对齐深度伪彩色图，640x480
        self.current_depth_map = None         # 对齐后的真实深度，float32，单位 mm，640x480
        self.current_raw_depth_map = None     # 原始深度，float32，单位 mm，640x480

        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps

        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps

        self.pipeline = Pipeline()
        config = Config()

        # -------------------- COLOR：严格 640x480 --------------------
        color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None

        try:
            color_profile = color_profile_list.get_video_stream_profile(
                self.color_width,
                self.color_height,
                OBFormat.RGB,
                self.color_fps,
            )
            print(f"[INFO] use color profile: {self.color_width}x{self.color_height} RGB@{self.color_fps}")
        except Exception as e:
            print(f"[WARN] get color profile {self.color_width}x{self.color_height} RGB@{self.color_fps} failed: {e}")

        if color_profile is None:
            try:
                color_profile = color_profile_list.get_video_stream_profile(
                    self.color_width,
                    self.color_height,
                    OBFormat.MJPG,
                    self.color_fps,
                )
                print(f"[INFO] use color profile: {self.color_width}x{self.color_height} MJPG@{self.color_fps}")
            except Exception as e:
                print(f"[WARN] get color profile {self.color_width}x{self.color_height} MJPG@{self.color_fps} failed: {e}")

        if color_profile is None:
            try:
                for i in range(len(color_profile_list)):
                    p = color_profile_list[i]
                    if p.get_width() == self.color_width and p.get_height() == self.color_height:
                        color_profile = p
                        print("[INFO] fallback color profile 640x480:", color_profile)
                        break
            except Exception as e:
                print(f"[WARN] iterate color profile list failed: {e}")

        if color_profile is None:
            raise RuntimeError(
                f"没有找到 {self.color_width}x{self.color_height} 的彩色流配置。"
            )

        config.enable_stream(color_profile)

        # -------------------- DEPTH：严格 640x480 --------------------
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = None

        try:
            depth_profile = depth_profile_list.get_video_stream_profile(
                self.depth_width,
                self.depth_height,
                OBFormat.Y16,
                self.depth_fps,
            )
            print(f"[INFO] use depth profile: {self.depth_width}x{self.depth_height} Y16@{self.depth_fps}")
        except Exception as e:
            print(f"[WARN] get depth profile {self.depth_width}x{self.depth_height} Y16@{self.depth_fps} failed: {e}")

        # 如果 30fps 不支持，尝试 15fps
        if depth_profile is None:
            try:
                depth_profile = depth_profile_list.get_video_stream_profile(
                    self.depth_width,
                    self.depth_height,
                    OBFormat.Y16,
                    15,
                )
                self.depth_fps = 15
                print(f"[INFO] use depth profile: {self.depth_width}x{self.depth_height} Y16@15")
            except Exception as e:
                print(f"[WARN] get depth profile {self.depth_width}x{self.depth_height} Y16@15 failed: {e}")

        # 如果仍然失败，遍历找任意 640x480 深度流
        if depth_profile is None:
            try:
                for i in range(len(depth_profile_list)):
                    p = depth_profile_list[i]
                    if p.get_width() == self.depth_width and p.get_height() == self.depth_height:
                        depth_profile = p
                        self.depth_fps = p.get_fps()
                        print("[INFO] fallback depth profile 640x480:", depth_profile)
                        break
            except Exception as e:
                print(f"[WARN] iterate depth profile list failed: {e}")

        # 不再偷偷用默认 depth profile，因为你明确要 640x480
        if depth_profile is None:
            raise RuntimeError(
                f"没有找到 {self.depth_width}x{self.depth_height} 的深度流配置。"
                "请先打印设备支持的 depth profiles，或者更换相机支持的分辨率。"
            )

        config.enable_stream(depth_profile)

        # 关闭硬件 D2C，使用手动几何对齐
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
        print("Output: color=640x480, aligned_depth=640x480")
        print("=============================================")

        self.manual_aligner = ManualD2CAligner(self.pipeline)

        self.thread = threading.Thread(target=self.cam_thread, daemon=True)
        self.thread.start()

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

                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    print("failed to convert color frame to image")
                    continue

                color_h, color_w = color_image.shape[:2]

                depth_w = depth_frame.get_width()
                depth_h = depth_frame.get_height()
                depth_scale = depth_frame.get_depth_scale()

                depth_raw = np.frombuffer(
                    depth_frame.get_data(),
                    dtype=np.uint16
                ).reshape((depth_h, depth_w))

                depth_raw_mm = depth_raw.astype(np.float32) * float(depth_scale)
                depth_raw_mm = np.where(
                    (depth_raw_mm > MIN_DEPTH) & (depth_raw_mm < MAX_DEPTH),
                    depth_raw_mm,
                    0,
                )

                aligned_depth_map = self.manual_aligner.align_depth_to_color(
                    depth_raw_u16=depth_raw,
                    depth_scale=depth_scale,
                    color_width=color_w,
                    color_height=color_h,
                    min_depth_mm=MIN_DEPTH,
                    max_depth_mm=MAX_DEPTH,
                )

                aligned_depth_map = aligned_depth_map.astype(np.float32)

                depth_vis = self.manual_aligner.depth_to_colormap(
                    aligned_depth_map,
                    min_depth=MIN_DEPTH,
                    max_depth=MAX_DEPTH,
                )

                with self.lock:
                    self.current_raw_depth_map = depth_raw_mm.copy()
                    self.current_depth_map = aligned_depth_map.copy()

                    if self.trigger:
                        self.current_color = color_image.copy()
                        self.current_depth = depth_vis.copy()
                        self.trigger = False

            except KeyboardInterrupt:
                break

            except Exception as e:
                print(f"[ERROR] cam_thread: {e}")
                time.sleep(0.01)

        try:
            self.pipeline.stop()
        except Exception as e:
            print(f"[WARN] pipeline.stop failed: {e}")

        print("[INFO] camera thread stopped.")

    def read(self):
        self.trigger = True

        while self.trigger and self.running:
            time.sleep(0.001)

        with self.lock:
            color = None if self.current_color is None else self.current_color.copy()
            depth_aligned = None if self.current_depth_map is None else self.current_depth_map.copy()
            depth_raw = None if self.current_raw_depth_map is None else self.current_raw_depth_map.copy()
            depth_vis = None if self.current_depth is None else self.current_depth.copy()

        return color, depth_aligned, depth_raw, depth_vis

    def stop(self):
        self.running = False

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        cv2.destroyAllWindows()
        print("camera has been safely closed!")


# =========================================================
# 录制主程序
# =========================================================
if __name__ == "__main__":
    output_dir = "/home/robot4/Programming/FoundationPose/video/5.22"

    color_dir = os.path.join(output_dir, "color_frames")
    depth_aligned_dir = os.path.join(output_dir, "depth_aligned_frames")
    depth_raw_dir = os.path.join(output_dir, "depth_raw_frames")
    depth_vis_dir = os.path.join(output_dir, "depth_vis_frames")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_aligned_dir, exist_ok=True)
    os.makedirs(depth_raw_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)

    color_video_path = os.path.join(output_dir, "color_video.mp4")
    depth_vis_video_path = os.path.join(output_dir, "depth_vis_video.mp4")
    meta_path = os.path.join(output_dir, "record_meta.json")

    color_width = 640
    color_height = 480
    depth_width = 640
    depth_height = 480
    stream_fps = 30

    save_color_frames = True
    save_depth_aligned_frames = True
    save_depth_raw_frames = True
    save_depth_vis_frames = True

    # 0 表示不限帧数，按 q 或 Esc 退出
    max_frames = 0

    cam = OrbbecCamera(
        color_width=color_width,
        color_height=color_height,
        color_fps=stream_fps,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_fps=stream_fps,
    )

    color_writer = None
    depth_vis_writer = None
    records = []

    frame_idx = 0
    last_time = time.time()

    print("=" * 80)
    print("[INFO] 开始录制 Orbbec RGB-D 640x480")
    print(f"[INFO] 输出目录: {output_dir}")
    print("[INFO] 按 q 或 Esc 停止录制")
    print("=" * 80)

    try:
        while True:
            color_img, depth_aligned, depth_raw, depth_vis = cam.read()

            if color_img is None or depth_aligned is None:
                print("[WARN] 当前帧 color 或 aligned depth 为空，跳过")
                continue

            if color_img.dtype != np.uint8:
                color_img = np.clip(color_img, 0, 255).astype(np.uint8)

            if len(color_img.shape) == 2:
                color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR)

            H, W = color_img.shape[:2]

            if W != 640 or H != 480:
                print(f"[WARN] 当前彩色图不是 640x480，而是 {W}x{H}")

            if depth_aligned.shape[1] != 640 or depth_aligned.shape[0] != 480:
                print(f"[WARN] 当前对齐深度不是 640x480，而是 {depth_aligned.shape[1]}x{depth_aligned.shape[0]}")

            if color_writer is None:
                color_writer = cv2.VideoWriter(
                    color_video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    stream_fps,
                    (W, H),
                )

                if not color_writer.isOpened():
                    raise RuntimeError(f"无法创建彩色视频: {color_video_path}")

            color_writer.write(color_img)

            if depth_vis is not None:
                if depth_vis.dtype != np.uint8:
                    depth_vis = np.clip(depth_vis, 0, 255).astype(np.uint8)

                if len(depth_vis.shape) == 2:
                    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                if depth_vis_writer is None:
                    depth_vis_writer = cv2.VideoWriter(
                        depth_vis_video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        stream_fps,
                        (depth_vis.shape[1], depth_vis.shape[0]),
                    )

                    if not depth_vis_writer.isOpened():
                        raise RuntimeError(f"无法创建深度可视化视频: {depth_vis_video_path}")

                depth_vis_writer.write(depth_vis)

            color_path = None
            depth_aligned_path = None
            depth_raw_path = None
            depth_vis_path = None

            if save_color_frames:
                color_path = os.path.join(color_dir, f"color_{frame_idx:06d}.png")
                cv2.imwrite(color_path, color_img)

            if save_depth_aligned_frames:
                # FoundationPose 推荐用这个：对齐到彩色图坐标系，单位 mm，uint16
                depth_aligned_u16 = np.clip(depth_aligned, 0, 65535).astype(np.uint16)
                depth_aligned_path = os.path.join(depth_aligned_dir, f"depth_{frame_idx:06d}.tiff")
                cv2.imwrite(depth_aligned_path, depth_aligned_u16)

            if save_depth_raw_frames and depth_raw is not None:
                depth_raw_u16 = np.clip(depth_raw, 0, 65535).astype(np.uint16)
                depth_raw_path = os.path.join(depth_raw_dir, f"depth_raw_{frame_idx:06d}.tiff")
                cv2.imwrite(depth_raw_path, depth_raw_u16)

            if save_depth_vis_frames and depth_vis is not None:
                depth_vis_path = os.path.join(depth_vis_dir, f"depth_vis_{frame_idx:06d}.png")
                cv2.imwrite(depth_vis_path, depth_vis)

            now = time.time()
            dt = now - last_time
            real_fps = 1.0 / dt if dt > 1e-6 else 0.0
            last_time = now

            records.append({
                "frame_idx": frame_idx,
                "timestamp_unix": now,
                "color_path": color_path,
                "depth_aligned_path": depth_aligned_path,
                "depth_raw_path": depth_raw_path,
                "depth_vis_path": depth_vis_path,
                "color_shape": list(color_img.shape),
                "depth_aligned_shape": list(depth_aligned.shape),
                "depth_raw_shape": list(depth_raw.shape) if depth_raw is not None else None,
                "depth_unit": "mm",
                "depth_aligned_saved_dtype": "uint16",
                "depth_raw_saved_dtype": "uint16",
                "real_fps_estimate": real_fps,
            })

            print(
                f"[REC] frame={frame_idx:06d} "
                f"fps={real_fps:.2f} "
                f"color={color_img.shape} "
                f"depth_aligned={depth_aligned.shape}"
            )

            cv2.imshow("Color 640x480", color_img)

            if depth_vis is not None:
                cv2.imshow("Aligned Depth 640x480", depth_vis)

            key = cv2.waitKey(1)

            if key == ord("q") or key == ESC_KEY:
                print("[INFO] 收到退出按键，停止录制")
                break

            frame_idx += 1

            if max_frames > 0 and frame_idx >= max_frames:
                print("[INFO] 达到 max_frames，停止录制")
                break

    finally:
        if color_writer is not None:
            color_writer.release()

        if depth_vis_writer is not None:
            depth_vis_writer.release()

        meta = {
            "output_dir": output_dir,
            "color_resolution": [640, 480],
            "depth_resolution": [640, 480],
            "stream_fps_target": stream_fps,
            "frame_count": len(records),
            "color_video_path": color_video_path,
            "depth_vis_video_path": depth_vis_video_path,
            "color_frames_dir": color_dir,
            "depth_aligned_frames_dir": depth_aligned_dir,
            "depth_raw_frames_dir": depth_raw_dir,
            "depth_vis_frames_dir": depth_vis_dir,
            "depth_unit": "mm",
            "records": records,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        cam.stop()

        print("=" * 80)
        print("[DONE] Orbbec RGB-D 640x480 录制完成")
        print(f"[OK] 彩色视频: {color_video_path}")
        print(f"[OK] 对齐深度帧: {depth_aligned_dir}")
        print(f"[OK] 原始深度帧: {depth_raw_dir}")
        print(f"[OK] 元数据: {meta_path}")
        print("=" * 80)