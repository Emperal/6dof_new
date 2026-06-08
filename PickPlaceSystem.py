import os
import cv2
import time
import shutil
import traceback
import argparse
import threading
import numpy as np
import math

# 相机、视觉和机器人控制模块
from Cameras.OrbbecCamera import OrbbecCamera
from vision_pose_estimator import VisionPoseEstimator
from Robots.robot_control import RobotLeft

# ===================== 配置参数 =====================
# 手眼标定矩阵 Base -> Camera
BTC_MATRIX = np.array([
    [-0.0408, -0.6495, 0.7593, 127.1348],
    [-0.9990, 0.0420, -0.0178, 307.8127],
    [-0.0204, -0.7592, -0.6505, -41.4876],
    [0, 0, 0, 1.0]
], dtype=np.float32)

# 相机内参
FX, FY, CX, CY = 609.99963379, 610.17034912, 641.85406494, 360.86437988
MANUAL_K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

# 机械臂Home位姿（关节角）
HOME_JOINT = [0.767, -163.03, -100.42, -8.67, 91.13, 43]

# 放置目标Pose6
TARGET_PLACE_POSE6 = [600, 56, -300, 177.676, -0.621, -132.258]

# 首件示教数据
TEACH_CTW1 = np.array([
    [-0.4735, 0.8793, -0.0512, -21.44],
    [0.6390, 0.3029, -0.7070, 17.53],
    [-0.6061, -0.3675, -0.7053, 511.83],
    [0, 0, 0, 1.0]
], dtype=np.float32)
TEACH_BTE1_POSE6 = [507.09, 321.40, -381.89, -177.34, 6.43, 172.85]

# -------------------- 工具函数 --------------------
def eul2rotm_zyx(rx, ry, rz):
    """欧拉角 (ZYX) 转旋转矩阵"""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def rotm2eul_zyx(R):
    """旋转矩阵 -> 欧拉角 (ZYX)"""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return rx, ry, rz


def pose6_to_matrix(pose6):
    """Pose6 (x,y,z,rx,ry,rz) -> 4x4 齐次矩阵"""
    x, y, z = pose6[:3]
    rx, ry, rz = map(math.radians, pose6[3:6])
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = eul2rotm_zyx(rx, ry, rz)
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose6(T):
    """4x4 齐次矩阵 -> Pose6"""
    rx, ry, rz = rotm2eul_zyx(T[:3, :3])
    return [
        float(T[0, 3]), float(T[1, 3]), float(T[2, 3]),
        math.degrees(rx), math.degrees(ry), math.degrees(rz)
    ]


def perform_teach_in(teach_bTe1_pose6, teach_cTw1, bTc_matrix):
    """
    根据首件示教数据计算抓取末端固定位姿 eTo_grasp
    """
    bTe_teach = pose6_to_matrix(teach_bTe1_pose6)
    cTw_teach = teach_cTw1
    eTw_fixed = np.linalg.inv(bTe_teach) @ bTc_matrix @ cTw_teach
    return eTw_fixed.astype(np.float32)


def is_cto_mm_z_axis_up(cTo_mm, cos_threshold=0.3):
    """判断相机系下z轴是否朝上"""
    z_axis = cTo_mm[:3, 2].astype(np.float32)
    norm = np.linalg.norm(z_axis)
    if norm < 1e-8:
        return False
    z_axis = z_axis / norm
    camera_up = np.array([0, -1, 0], dtype=np.float32)
    return float(np.dot(z_axis, camera_up)) > cos_threshold


# ===================== 全局信号 =====================
current_pick_result = None                 # 当前抓取目标结果
pick_lock = threading.Lock()              # 保护 current_pick_result
vision_request_event = threading.Event()  # 异步识别触发信号
stop_event = threading.Event()            # 程序退出信号

# 最近一次识别结果可视化图（给 vis 窗口显示）
latest_result_vis = None
state_lock = threading.Lock()


def set_latest_vis(vis_img):
    global latest_result_vis
    with state_lock:
        latest_result_vis = None if vis_img is None else vis_img.copy()


def get_latest_vis():
    with state_lock:
        return None if latest_result_vis is None else latest_result_vis.copy()


class PickPlaceSystem:
    """抓取放置系统主体类"""

    def __init__(self, mesh_file, template_dir, save_root):
        # 手眼矩阵和抓取位姿
        self.bTc = BTC_MATRIX
        self.eTo_grasp = perform_teach_in(TEACH_BTE1_POSE6, TEACH_CTW1, BTC_MATRIX)

        # 帧锁：防止预览线程和视觉线程同时读相机
        self.frame_lock = threading.Lock()

        # 机器人初始化
        self.robot = None
        self.Vel = 50
        try:
            self.robot = RobotLeft()
            self.robot.robot.SetSpeed(20)
            print("[INIT] 机器人连接成功")
        except Exception as e:
            print(f"[WARN] 机器人连接失败: {e}")

        # 相机和视觉模块初始化
        self.camera = OrbbecCamera()
        self.vision = VisionPoseEstimator(mesh_file, template_dir, save_root, manual_k=MANUAL_K)

    def get_current_frame(self):
        """获取当前RGB-D帧"""
        with self.frame_lock:
            self.camera.read()
            color = self.camera.current_color.copy()
            depth = self.camera.current_depth_map.copy()
        return color, depth

    def compute_pick_pose(self, cTo):
        """根据cTo计算抓取末端位姿"""
        return self.bTc @ cTo @ np.linalg.inv(self.eTo_grasp)

    def move_robot_photo_pose(self):
        """回到拍照位 / Home"""
        if self.robot is None:
            print("[WARN] 机器人未连接，无法移动到拍照位")
            return
        self.robot.robot.MoveJ(HOME_JOINT, vel=self.Vel, tool=1, user=0, blendT=-1)

    def estimate_pick_once(self, refine_iter=5):
        """单次抓取位姿识别"""
        frame, depth = self.get_current_frame()
        ret = self.vision.estimate_once(frame, depth, refine_iter, use_y180=True)

        # 保存最近一次识别可视化结果
        if ret is not None and "vis" in ret and ret["vis"] is not None:
            set_latest_vis(ret["vis"])
        else:
            set_latest_vis(None)

        if ret is None:
            return {"status": "fail", "data": None}

        if "cTo" not in ret:
            return {"status": "fail", "data": None}

        if ret["cTo"] is None:
            return {"status": "fail", "data": None}

        if not is_cto_mm_z_axis_up(ret["cTo"]):
            return {"status": "fail", "data": None}

        pick_pose6 = matrix_to_pose6(self.compute_pick_pose(ret["cTo"]))
        ret["pick_pose6"] = pick_pose6

        return {"status": "ok", "data": ret}

    def execute_pick(self, pick_pose6):
        """执行抓取动作"""
        if self.robot is None:
            print("[WARN] 机器人未连接，跳过抓取")
            return

        safe_pose = list(pick_pose6)
        safe_pose[2] += 100  # 上方安全位

        error, safe_pose_J = self.robot.robot.GetInverseKin(0, desc_pos=safe_pose, config=-1)
        safe_pose_J[5] = 43
        self.robot.robot.MoveJ(safe_pose_J, vel=self.Vel, tool=1, user=0, blendT=-1)

        error, pick_pose6_J = self.robot.robot.GetInverseKin(0, desc_pos=pick_pose6, config=-1)
        pick_pose6_J[5] = 43
        self.robot.robot.MoveJ(pick_pose6_J, vel=self.Vel, tool=1, user=0, blendT=-1)

        if hasattr(self.robot, "magnet"):
            self.robot.magnet.send("1")  # 吸取工件

        self.robot.robot.MoveJ(safe_pose_J, vel=self.Vel, tool=1, user=0, blendT=-1)

    def execute_place(self):
        """执行放置动作"""
        if self.robot is None:
            print("[WARN] 机器人未连接，跳过放置")
            return

        safe_place = list(TARGET_PLACE_POSE6)
        safe_place[2] += 30

        error, safe_place_J = self.robot.robot.GetInverseKin(0, desc_pos=safe_place, config=-1)
        safe_place_J[5] = 43
        self.robot.robot.MoveJ(safe_place_J, vel=self.Vel, tool=1, user=0, blendT=-1)

        self.robot.robot.MoveCart(TARGET_PLACE_POSE6, vel=self.Vel, tool=1, user=0, blendT=-1)

        if hasattr(self.robot, "magnet"):
            self.robot.magnet.send("2")  # 放下工件

        self.robot.robot.MoveJ(safe_place_J, vel=self.Vel, tool=1, user=0, blendT=-1)


# ===================== 线程函数 =====================
def vision_worker(system, async_refine_iter=5):
    """
    异步视觉线程：
    收到触发后，如果当前没有抓取结果，就延时1秒再识别
    """
    global current_pick_result

    print("[VISION] 异步视觉线程启动")

    while not stop_event.is_set():
        vision_request_event.wait()
        vision_request_event.clear()

        if stop_event.is_set():
            break

        print("[VISION] 收到识别请求")

        # 先读一次共享状态，不要长时间占锁
        with pick_lock:
            need_estimate = (current_pick_result is None)

        if need_estimate:
            print("[VISION] 当前没有缓存结果，1秒后开始识别")
            time.sleep(1)

            if stop_event.is_set():
                break

            ret = system.estimate_pick_once(refine_iter=async_refine_iter)

            if ret["status"] == "ok":
                with pick_lock:
                    if current_pick_result is None:
                        current_pick_result = ret["data"]
                        print("[VISION] 识别成功，已写入 current_pick_result")
            else:
                print("[VISION] 识别失败，本轮未写入结果")


def pick_place_cycle(system):
    """
    主抓取放置循环：
    首件：
    1. 到拍照位
    2. 识别第1件
    3. 抓第1件

    后续：
    4. 当前件抓起离开视野后，触发下一件识别
    5. 同时执行当前件放置
    6. 放置完成后继续下一轮
    """
    global current_pick_result

    print("[CYCLE] 抓取线程启动")

    # -------- 首件：先到拍照位 --------
    system.move_robot_photo_pose()

    while not stop_event.is_set():
        # 先尝试取走当前结果
        with pick_lock:
            if current_pick_result is not None:
                pick_data = current_pick_result
                current_pick_result = None
            else:
                pick_data = None

        # 如果当前没有识别结果，就触发视觉线程，然后在锁外等待
        if pick_data is None:
            vision_request_event.set()

            while not stop_event.is_set():
                with pick_lock:
                    if current_pick_result is not None:
                        break
                time.sleep(0.05)

            continue

        # 有结果就抓取
        system.execute_pick(pick_data["pick_pose6"])

        # 抓起离开视野后，触发后台识别下一件
        vision_request_event.set()

        # 同时执行当前件放置
        system.execute_place()


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data_pian/my_data0/mesh/pian_hole_m.obj')
    parser.add_argument('--template_dir', type=str, default=f'{code_dir}/templates1280*720/back')
    parser.add_argument('--save_root', type=str, default=f'{code_dir}/demo_data_pian/pick_place_output')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    args = parser.parse_args()

    if os.path.exists(args.save_root):
        shutil.rmtree(args.save_root)
    os.makedirs(args.save_root, exist_ok=True)
    np.savetxt(os.path.join(args.save_root, "cam_K.txt"), MANUAL_K, fmt='%.8e')

    system = None
    vis_thread = None
    cycle_thread = None

    try:
        system = PickPlaceSystem(args.mesh_file, args.template_dir, args.save_root)

        # 启动异步视觉线程
        vis_thread = threading.Thread(
            target=vision_worker,
            args=(system, args.est_refine_iter),
            daemon=True
        )
        vis_thread.start()

        # 启动抓取放置线程
        cycle_thread = threading.Thread(
            target=pick_place_cycle,
            args=(system,),
            daemon=True
        )
        cycle_thread.start()

        print("\n================ 使用说明 ================")
        print("程序启动后自动运行")
        print("按 ESC 退出")
        print("========================================\n")

        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("preview", 1280, 720)

        cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("vis", 960, 540)

        while not stop_event.is_set():
            frame, _ = system.get_current_frame()
            cv2.imshow("preview", frame)

            latest_vis = get_latest_vis()
            if latest_vis is not None:
                cv2.imshow("vis", latest_vis)
            else:
                blank = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.imshow("vis", blank)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[INFO] 检测到 ESC，准备退出")
                break

    except KeyboardInterrupt:
        print("[INFO] 检测到 Ctrl+C，准备退出")
    except Exception:
        print("[ERROR] 主程序异常：")
        print(traceback.format_exc())
    finally:
        stop_event.set()
        vision_request_event.set()

        try:
            if vis_thread is not None and vis_thread.is_alive():
                vis_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if cycle_thread is not None and cycle_thread.is_alive():
                cycle_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if system is not None and hasattr(system.camera, "release"):
                system.camera.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[INFO] 程序已退出")


if __name__ == "__main__":
    main()