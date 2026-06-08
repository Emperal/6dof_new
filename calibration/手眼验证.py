from fairino import Robot
# 与机器人控制器建立连接，连接成功返回一个机器人对象
import time
robot = Robot.RPC('192.168.58.2')

# move_pose = [-607.2251586914062,-200.29595947265625,-170.2542724609375,90.67607116699219,3.4073328971862793,-0.18772107362747192]
# desc_pos5 = [764.2262573242188,44.62815475463867,-380.92474365234375,180,0,-180]#中途确认点
desc_pos5 = [0.7670850157737732,-163.0297088623047,-100.42265319824219,-8.670496940612793,91.13082885742188,43]   # 你自己改
#中途确认点
desc_pos6 = [-293.224,-149.539,-294.768,152.613,-29.006,-53.451]#最终目标点

# tool = 2
# user = 1
#
# rtn = robot.MoveCart(move_pose, tool,user, vel=20, blendT=100)
# print(f"moveCart.errcode: {rtn}")
# # vel = 100.0
# # acc = 100.0
# # ovl = 100.0

# # blendR = 0.0
# # flag = 0
# # search = 0
# robot.SetSpeed(5)577.2089233398438,-192.4515380859375,-342.2221984863281
# rtn = robot.MoveL(desc_pos=desc_pos4, tool=tool, user=user)
# robot.MoveGripper(1, 90, 90, 40, 10000, 0, 0, 0, 0, 0)  # 张开
# rtn = robot.MoveCart(desc_pos=desc_pos5, tool=1, user=0, blendT=-1.0)
# # rtn = robot.MoveCart(desc_pos=desc_pos6, tool=tool, user=user, blendT=blendT)
# robot.MoveGripper(1, 0, 90, 40, 10000, 0, 0, 0, 0, 0)  # 张开
# rtn = robot.MoveCart(desc_pos=desc_pos5, tool=tool, user=user, blendT=blendT)
# rtn = robot.MoveCart(desc_pos=desc_pos4, tool=tool, user=user, blendT=blendT)
# rtn = robot.MoveJ(joint_pos=desc_pos5,tool=1,user=0,blendT= -1.0)
# robot.MoveGripper(1, 90, 90, 10, 10000, 0, 0, 0, 0, 0)  # 张开
# print(f"MoveL errcode: {rtn}")
# error,flange = robot.GetActualToolFlangePose(0)
# print(f"flange pose:{flange[0]},{flange[1]},{flange[2]},{flange[3]},{flange[4]},{flange[5]}")
#error,flange = robot.GetActualToolFlangePose(0)
#print(f"flange pose:{flange[0]},{flange[1]},{flange[2]},{flange[3]},{flange[4]},{flange[5]}")
error,tcp = robot.GetActualTCPPose(0)
print(f"tcp pose:{tcp[0]},{tcp[1]},{tcp[2]},{tcp[3]},{tcp[4]},{tcp[5]}")
# error,[yangle, zangle] = robot.GetRobotInstallAngle()
# print(f"yangle:{yangle},zangle:{zangle}")
error,j_deg = robot.GetActualJointPosDegree(0)
print(f"joint pos deg:{j_deg[0]},{j_deg[1]},{j_deg[2]},{j_deg[3]},{j_deg[4]},{j_deg[5]}")


# desc_pos1 = [698.162109375,31.14605140686035,-365.3185119628906,176.35923767089844,1.0429496765136719,-115.0658950805664]
# error, inverseRtn = robot.GetInverseKin(0, desc_pos=desc_pos1, config=-1)
# print(f"dcs1 GetInverseKin rtn is {inverseRtn[0]}, {inverseRtn[1]}, {inverseRtn[2]}, "
#       f"{inverseRtn[3]}, {inverseRtn[4]}, {inverseRtn[5]}")


# error, inverseRtn = robot.GetInverseKinRef(0, desc_pos=desc_pos1, joint_pos_ref=j1)
# print(f"dcs1 GetInverseKinRef rtn is {inverseRtn[0]}, {inverseRtn[1]}, {inverseRtn[2]}, "
#       f"{inverseRtn[3]}, {inverseRtn[4]}, {inverseRtn[5]}")

# robot.CloseRPC()
# tcp pose:306.7895202636719,250.0909423828125,-41.39340591430664,89.61286926269531,-45.51875305175781,0.2104228287935257