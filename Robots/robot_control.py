import numpy as np
import math
import time
import serial
from fairino import Robot
from electromagnet import SocketClient




def eul2rotm_xyz(rx, ry, rz):
    """将 XYZ (roll-pitch-yaw) 欧拉角转换为旋转矩阵"""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx   # XYZ 欧拉角标准顺序


def rotm2eul_xyz(R):
    """从旋转矩阵返回 XYZ 欧拉角 (roll, pitch, yaw)"""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)

    if sy > 1e-6:
        rx = math.atan2(R[2,1], R[2,2])
        ry = math.atan2(-R[2,0], sy)
        rz = math.atan2(R[1,0], R[0,0])
    else:
        # 奇异情况
        rx = math.atan2(-R[1,2], R[1,1])
        ry = math.atan2(-R[2,0], sy)
        rz = 0

    return rx, ry, rz


# ============================================================
#                    Pose6 <-> Matrix4x4
#      pose6 = [x, y, z, rx(deg), ry(deg), rz(deg)]
# ============================================================

def pose6_to_matrix(pose6):
    """pose6 → 4x4矩阵，XYZ 欧拉角"""
    x, y, z = pose6[0], pose6[1], pose6[2]
    rx, ry, rz = map(math.radians, pose6[3:6])

    Rmat = eul2rotm_xyz(rx, ry, rz)

    T = np.eye(4)
    T[:3,:3] = Rmat
    T[:3,3] = [x, y, z]
    return T


def matrix_to_pose6(T):
    """4x4矩阵 → pose6 (XYZ 欧拉角)"""
    rx, ry, rz = rotm2eul_xyz(T[:3,:3])
    return [
        T[0,3], T[1,3], T[2,3],
        math.degrees(rx), math.degrees(ry), math.degrees(rz)
    ]


# ============================================================
#                摄像机坐标系的 pose6 转换
#            （保持你的原接口，但内部采用XYZ）
# ============================================================

def pose6_to_matrix_cam(pose6):
    """
    若相机的坐标系为：
    X 右，Y 下，Z 前（RGB/Depth常用）
    则需要加一个固定旋转补偿。
    """
    rx, ry, rz = map(math.radians, pose6[3:6])

    R_cam = eul2rotm_xyz(rx, ry, rz)


    Rmat = eul2rotm_xyz(rx, -ry, -rz)

    T = np.eye(4)
    T[:3,:3] = Rmat

    # 保持你原先的位移逻辑
    T[0,3], T[1,3], T[2,3] = -pose6[0], -pose6[1], pose6[2]

    return T

# ========== RS485 通信（手部） ==========
class RS485Driver:
    def __init__(self, port, baudrate=115200):
        self.ser = serial.Serial(port, baudrate)
    def send(self, data): self.ser.write(data)
    def close(self): self.ser.close()

def angle_to_int(value, min_angle, max_angle):
    return int((value - min_angle) / (max_angle - min_angle) * 1000)

def build_command(angles):
    cmd = [0xEB, 0x90, 0x01, 0x0F, 0x12, 0xCE, 0x05]
    for v in angles: cmd += [v & 0xFF, (v >> 8) & 0xFF]
    cmd.append(sum(cmd[2:]) & 0xFF)
    return bytes(cmd)

def send_rs485_angles(angles):
    rs = RS485Driver('/dev/ttyUSB0')
    mapped = [angle_to_int(a, mi, ma) for a,(mi,ma) in zip(
        angles, [(19,176.7)]*4 + [(-13,53.6),(90,165)])]
    rs.send(build_command(mapped))
    rs.close()

# 手部动作
OPEN_ANGLES  = [120,120,120,120,25,90]
CLOSE_ANGLES = [90,90,90,90,25,90]

def build_clear_fault_command():
    """
    构建清除故障的固定指令帧
    """
    # 按照你提供的完整帧，包含校验字节
    cmd = [0xEB, 0x90, 0x01, 0x04, 0x12, 0xEC, 0x03, 0x01, 0x07]
    return bytes(cmd)

def send_clear_fault(port='/dev/ttyUSB0'):
    """
    发送清除故障指令
    """
    rs = RS485Driver(port)
    rs.send(build_clear_fault_command())
    rs.close()
    print("👉 已发送清除故障指令")


class RobotLeft():
    def __init__(self, ip='192.168.58.2', hand=None, magnet_ip="192.168.58.4", magnet_port=2000):
        self.robot = Robot.RPC(ip)
        self.hand = hand
        self.magnet = SocketClient(magnet_ip, magnet_port)
        # self.power = PowerControl()

        # ====== 标定矩阵（示例） ======
        # self.bTc = np.array([
        #     [-0.0102,  0.6389, -0.7692, -119.6681],
        #     [ 0.0013, -0.7693, -0.6389,   61.0837],
        #     [-0.9999, -0.0075,  0.0070, -149.6871],
        #     [0, 0, 0, 1]
        # ])
        self.bTc = np.array([[0.0140 ,   0.9967 ,   0.0802 ,-430.8690],
                           [-0.0219 ,   0.0805 ,  -0.9965 , 330.7537],
                           [-0.9997  ,  0.0122  ,  0.0229 ,-231.2994],
                           [0, 0, 0, 1]])

        self.cTw = np.array( [[  0.7981 , -0.6022  , 0.0165  ,-6.8394],
                             [  0.602    ,0.796   ,-0.0635 ,-60.692 ],
                             [  0.0251  , 0.0607  , 0.9978 ,630.9057],
                             [  0.     ,  0.      , 0.       ,1.    ]])

        self.bTe = pose6_to_matrix([-441.3597412109375,-281.56732177734375,-210.9535369873047, 90.0, 45.0, 0.0])

        self.bTe_pose6 = [-441.3597412109375,-281.56732177734375,-210.9535369873047,90.0, 45.0, 0.0]

        self.pose6_home = [-556.6551513671875,-165.02503967285156,-108.84085845947266,90.0,45.0,0.0]
        self.pose6_place = [-472.5501403808594,-275.5970458984375,-81.11357116699219,90.0,45.0,0.0]
        self.bTw = self.bTc @ self.cTw
        # self.bTw_2= align_z_to_target(self.bTw_1)

        self.wTe = np.linalg.inv(self.bTw) @ self.bTe
        self.eTw = np.linalg.inv(self.wTe)

    def pose_cam_to_base_tool(self, cTw2):
        bTw2 = self.bTc @ cTw2

        bTe2 = bTw2 @ self.wTe
        pose6 = matrix_to_pose6(bTe2)
        return pose6

    def pick(self, refined_pose, tool=1, user=1, blendT=-1.0):

        pose6 = self.pose_cam_to_base_tool(refined_pose)
        print(pose6)

        self.robot.SetSpeed(5)
        pose6[1] = pose6[1] + 50
        print("go to pick pose high")
        rtn = self.robot.MoveCart(desc_pos=pose6, tool=1, user=1, blendT=blendT)
        print(f"[RightArm] MoveCart err={rtn}")

        self.robot.SetSpeed(5)
        pose6[1] = pose6[1] - 50
        # rtn = self.robot.MoveL(desc_pos=pose6, tool=tool, user=user, blendR=blendT)
        rtn = self.robot.MoveCart(desc_pos=pose6, tool=1, user=1, blendT=blendT)
        print(f"[RightArm] MoveCart err={rtn}")



        print("[RightArm] ⚡ 电磁铁通电 (吸取物体)")
        self.magnet.send("1")
        time.sleep(0.5)

        pose6[1] = pose6[1] + 50
        rtn = self.robot.MoveCart(desc_pos=pose6, tool=tool, user=user, blendT=blendT)
        print(f"[RightArm] MoveCart err={rtn}")


    def place(self , tool=1, user=1, blendT=-1.0):
        pose6 = self.pose6_place.copy()
        self.robot.SetSpeed(5)
        pose6[1] = pose6[1] + 50

        rtn = self.robot.MoveCart(desc_pos=pose6, tool=tool, user=user, blendT=blendT)
        print(f"[RightArm] MoveCart err={rtn}")

        self.robot.SetSpeed(5)
        pose6[1]=pose6[1] -48
        rtn = self.robot.MoveCart(desc_pos=pose6, tool=tool, user=user, blendT=blendT)
        print(f"[RightArm] MoveCart err={rtn}")


        print("[RightArm] ⚡ 电磁铁断电 ")
        self.magnet.send("0")
        time.sleep(1)

        self.magnet.close()
        print("电磁铁已经断开连接 ")

        rtn = self.robot.MoveCart(desc_pos=self.pose6_home, tool=tool, user=user, blendT=blendT)
        print(f"[RightArm] MoveCart err={rtn}")
    def get_current_pose(self):
        """
        读取当前末端姿态（Base→TCP），返回 pose6:
        [x, y, z, rx, ry, rz]  (mm, deg)
        """
        err, tcp = self.robot.GetActualTCPPose(0)

        if err != 0:
            print(f"[WARN] GetActualTCPPose 失败，err={err}")
            return None

        # tcp = [x, y, z, rx, ry, rz]
        pose6 = [
            tcp[0], tcp[1], tcp[2],
            tcp[3], tcp[4], tcp[5]
        ]

        print(f"[INFO] 当前 TCP pose = {pose6}")

        return pose6

    def place_with_pose(self, bTc, cTw, wTe_fixed):
        """
        根据真实姿态 cTw 计算放置姿态 bTe_place，并执行机器人移动。

        Args:
            robot: RobotLeft 实例
            bTc:   基坐标系 → 相机坐标系 (4x4)
            cTw:   相机坐标系 → 零件真实姿态 (4x4)
            wTe_fixed:  示教得到的 world→TCP 反变换 (4x4)
        """

        # 1) 计算机械臂末端放置姿态
        bTe_place = bTc @ cTw @ wTe_fixed

        print("\n========== [PLACE] 计算出的放置姿态 bTe ==========")
        print(bTe_place)

        # 2) 转成 pose6（你的控制接口需要）
        pose6 = matrix_to_pose6(bTe_place)
        print("[PLACE] pose6 =", pose6)

        # 3) 机器人执行移动
        print("[PLACE] 正在移动到放置位…")
        robot.MoveCart(desc_pos=pose6, tool=1, user=1, blendT=-1.0)

        print("[PLACE] 放置完成 √")

        return bTe_place, pose6

    def disconnet(self):
        self.robot.CloseRPC()
        self.magnet.close()


if __name__ == "__main__":

    robot = Robot.RPC('192.168.58.2')
    error, ft = robot.FT_GetForceTorqueRCS()

    if error == 0:
        Fx, Fy, Fz, Mx, My, Mz = ft
        print(f"Fx={Fx:.2f} N, Fy={Fy:.2f} N, Fz={Fz:.2f} N")
        print(f"Mx={Mx:.3f} Nm, My={My:.3f} Nm, Mz={Mz:.3f} Nm")
    else:
        print("读取力失败，error =", error)

    # # error, flange = robot.GetActualToolFlangePose(0)
    # # print(f"flange pose:{flange[0]},{flange[1]},{flange[2]},{flange[3]},{flange[4]},{flange[5]}")
    # error, tcp = robot.GetActualTCPPose(0)
    # print(f"tcp pose:{tcp[0]},{tcp[1]},{tcp[2]},{tcp[3]},{tcp[4]},{tcp[5]}")
    pose6 =[-556.6551513671875, -165.02503967285156, -108.84085845947266, 90.0, 45.0, 0.0]
    rtn = robot.MoveCart(desc_pos=pose6, tool=1, user=1, blendT=-1.0)
    print(f"[RightArm] MoveCart err={rtn}")
    error,j_deg = robot.GetActualJointPosDegree(0)
    print(f"joint pos deg:{j_deg[0]},{j_deg[1]},{j_deg[2]},{j_deg[3]},{j_deg[4]},{j_deg[5]}")
