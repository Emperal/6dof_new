import time

class RobotRight():
    def __init__(self, ip='192.168.58.2',hand=None,):
        pass

    def wait_until_reached(self, target_pose, tol_pos=0.5, tol_ang=0.5, timeout=1000):
        """
        等待机械臂到达目标位姿
        tol_pos: 位置误差阈值 (mm)
        tol_ang: 角度误差阈值 (deg)
        timeout: 超时时间 (秒)
        """
        pass

    def move_and_wait(self,refined_pose, tool=6, user=0, blendT=0.0,speed=3):
        pass

    def close_gripper(self):
        print("✊ 抓取中...")
        pass
        time.sleep(2)

    def open_gripper(self):
        # 如果需要松开，取消注释
        print("👉 松开手")
        pass

    def disconnet(self):
        robot.CloseRPC()


if __name__ == "__main__":
    robot = RobotRight()
    robot.move_and_wait()
    # robot.open_gripper()
