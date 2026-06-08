import cv2
import numpy as np
import apriltag
import csv
import os

# ========== 参数 ==========
image_path = "/home/ma/FoundationPose/sanpshot_depth_rgb/color_20260325_135242.png"  # 输入图像路径
tag_size = 0.043  # AprilTag 边长 (米)

# 相机内参矩阵
camera_matrix = np.array(
    [[609.99963379 , 0., 641.85406494],
     [0., 610.17034912, 360.86437988],
     [0, 0, 1]], dtype=np.float32
)


# 相机畸变系数
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ========== 工具函数 ==========
def rvec_tvec_to_euler(rvec, tvec):
    """
    将旋转向量转换为欧拉角

    参数:
        rvec: 旋转向量
        tvec: 平移向量

    返回:
        euler: 欧拉角 (度)
        R: 旋转矩阵
    """
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    # 计算欧拉角
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    else:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    # 将弧度转换为角度
    euler = np.degrees([x, y, z])
    return euler, R


# ========== 检测 ==========
# 读取图像
image = cv2.imread(image_path)
# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 创建AprilTag检测器
detector = apriltag.Detector()
# 检测图像中的AprilTag
results = detector.detect(gray)

print(f"检测到 {len(results)} 个 tag")

# ========== CSV ==========
# 创建CSV文件并写入表头
csv_file = open("tag_poses.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["id", "tx_mm", "ty_mm", "tz_mm", "rx_deg", "ry_deg", "rz_deg"])

# ========== 总图可视化 ==========
# 创建用于可视化的图像副本
all_vis = image.copy()

# ========== 保存单独可视化的目录 ==========
# 创建目录用于保存每个tag的可视化结果
os.makedirs("tags_vis", exist_ok=True)

# ========== 处理每个 tag ==========
for r in results:
    tag_id = r.tag_id  # 获取tag的ID
    corners = r.corners  # 获取tag的四个角点坐标

    # 构造物理坐标 (以中心为原点, 单位: m)
    obj_pts = np.array([
        [-tag_size / 2, tag_size / 2, 0],  # 左上角
        [tag_size / 2, tag_size / 2, 0],  # 右上角
        [tag_size / 2, -tag_size / 2, 0],  # 右下角
        [-tag_size / 2, -tag_size / 2, 0]  # 左下角
    ], dtype=np.float32)
    img_pts = np.array(corners, dtype=np.float32)  # 图像坐标

    # 使用PnP算法求解位姿
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
    if not success:
        continue  # 如果求解失败则跳过

    # 将旋转向量转换为欧拉角
    euler, R = rvec_tvec_to_euler(rvec, tvec)

    # 将平移向量从米转换为毫米，并保留3位小数
    tvec_mm = (tvec.flatten() * 1000.0).round(3)

    # 将结果写入CSV文件
    writer.writerow([tag_id, *tvec_mm, *np.round(euler, 3)])

    # 坐标轴投影 - 用于可视化
    axis_len = tag_size  # 坐标轴长度
    # 定义3D坐标轴端点
    axis_3d = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]).reshape(-1, 3)
    # 将3D坐标轴投影到2D图像平面
    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int).reshape(-1, 2)

    # 计算tag中心点
    center = tuple(map(int, img_pts.mean(axis=0)))

    # ===== 画到总图 =====
    # 绘制坐标轴：X轴(红色), Y轴(绿色), Z轴(蓝色)
    cv2.line(all_vis, center, tuple(imgpts[0]), (0, 0, 255), 2)  # X轴 - 红色
    cv2.line(all_vis, center, tuple(imgpts[1]), (0, 255, 0), 2)  # Y轴 - 绿色
    cv2.line(all_vis, center, tuple(imgpts[2]), (255, 0, 0), 2)  # Z轴 - 蓝色
    # 添加tag ID标签
    cv2.putText(all_vis, f"ID:{tag_id}", (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ===== 单独可视化 =====
    single_vis = image.copy()
    # 在单独图像上绘制坐标轴
    cv2.line(single_vis, center, tuple(imgpts[0]), (0, 0, 255), 2)
    cv2.line(single_vis, center, tuple(imgpts[1]), (0, 255, 0), 2)
    cv2.line(single_vis, center, tuple(imgpts[2]), (255, 0, 0), 2)
    cv2.putText(single_vis, f"ID:{tag_id}", (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 将单个tag的可视化结果保存为单独文件
    cv2.imwrite(f"tags_vis/tag_id_{tag_id}.jpg", single_vis)

# 关闭CSV文件
csv_file.close()

# 保存包含所有tag的总可视化图
cv2.imwrite("tag_visualization.jpg", all_vis)
print("结果已保存到:")
print("  - tag_poses.csv (单位: mm, deg)")
print("  - tag_visualization.jpg (所有ID)")
print("  - tags_vis/ (每个ID单独的图)")