% =========================================================================
% 脚本: solve_pose_matrix.m
% 描述: 使用SVD方法，根据两组对应的3D点，计算从源坐标系到目标坐标系的
%       4x4位姿变换矩阵。
% =========================================================================

%% 0. 初始化工作区
clear; % 清除工作区变量
clc;   % 清除命令行窗口
close all; % 关闭所有图形窗口

%% 1. 定义已知数据点


% points_raw_homogeneous: 相机读取的码的xyz (手眼标定（眼）读出来的)


points_raw_homogeneous = [


 -52.689,-23.308,530.07,1;
 -48.628,10.447,493.327,1;
 -44.323,43.987,453.823,1;
 -102.609,-20.196,529.987,1;
 -98.323,13.58,490.769,1;
 -94.065,47.1,452.009,1;
 -151.786,-17.003,525.762,1;
 -147.9,16.67,487.896,1;
 -143.373,50.091,448.346,1;

];

% points_1_homogeneous: 机械臂拖拽到码的位置时记录的xyz (法奥sdk读取)



points_1_homogeneous = [


-501.363,-248.546,-87.776,1;
-451.955,-247.817,-90.349,1;
-403.104,-248.888,-94.765,1;
-497.318,-248.872,-37.255,1;
-447.852,-249.129,-40.278,1;
-399.078,-249.319,-45.779,1;
-493.176,-249.694,11.512,1;
-443.409,-249.515,7.629,1;
-393.759,-249.839,2.819,1;

  

];

%===========================================================
% 只需要改上面的两处就可以了   下面不用管

%上面点的格式都是x,y,z,1;
%===========================================================



points_2_homogeneous=...
    [
        656.75,12.68,340.00,1;
        730.78,-15.75,342.21,1;
        693.33,53.41,340.99,1;
        731.07,39.63,342.16,1;
        692.62,107.90,340.68,1;
        766.92,79.64,343.08,1
    ];


points_3_homogeneous=...
    [
        670.38,6.96,350.62,1;
        740.97,-19.48,325.39,1;
        706.19,49.12,344.58,1;
        741.89,35.59,331.71,1;
        706.99,102.67,352.27,1;
        776.77,75.76,325.94,1
    ];

%% 2. 提取三维坐标
% 我们只需要前三列 (x, y, z) 来计算旋转和平移。
% MATLAB 的索引是从 1 开始的。
points_raw_3d = points_raw_homogeneous(:, 1:3);
points_1_3d = points_1_homogeneous(:, 1:3);

%% 3. 计算两组点的质心 (Centroid)
% mean(A, 1) 会计算矩阵 A 每一列的平均值，结果是一个行向量。
centroid_raw = mean(points_raw_3d, 1);
centroid_1 = mean(points_1_3d, 1);

%% 4. 对两组点进行去中心化
% 将每个点减去其所在点集的质心。
% MATLAB的广播机制会自动处理维度。
centered_raw = points_raw_3d - centroid_raw;
centered_1 = points_1_3d - centroid_1;

%% 5. 计算协方差矩阵 H
% H = sum_{i=1 to n} (centered_raw_i' * centered_1_i)
% 在MATLAB中，这可以高效地通过矩阵转置和乘法完成。
% centered_raw' 是一个 3xN 矩阵, centered_1 是一个 Nx3 矩阵。
H = centered_raw' * centered_1;

%% 6. 对协方差矩阵 H 进行 SVD 分解
% [U, S, V] = svd(H) 返回 H = U*S*V'
[U, S, V] = svd(H);

%% 7. 计算旋转矩阵 R
% 最佳旋转矩阵 R = V * U'
R = V * U';

%% 特殊情况处理：反射修正 (Reflection Correction)
% 检查R的行列式。如果 det(R) 为 -1，则得到的是一个反射矩阵而非纯旋转矩阵。
% 这种情况需要修正。
if det(R) < 0
    disp('检测到反射，正在进行修正...');
    % 修正方法是翻转 V 矩阵的最后一列的符号，然后重新计算 R。
    V_corrected = V;
    V_corrected(:, 3) = V_corrected(:, 3) * -1;
    R = V_corrected * U';
end

%% 8. 计算平移向量 t
% t = centroid_1' - R * centroid_raw'
% 注意：需要将质心行向量转置为列向量进行计算。
t = centroid_1' - R * centroid_raw';

%% 9. 组合成最终的 4x4 位姿矩阵 T
% 初始化一个4x4的单位矩阵
T = eye(4);
% 将左上角的 3x3 区域替换为旋转矩阵 R
T(1:3, 1:3) = R;
% 将右上角的 3x1 区域替换为平移向量 t
T(1:3, 4) = t;

%% --- 输出结果 ---
disp('计算得到的旋转矩阵 R (3x3):');
disp(R);
disp('计算得到的平移向量 t (3x1):');
disp(t);
disp('最终求得的位姿矩阵 T (4x4, 从工件坐标系到基坐标系):');
disp(T);

%% --- 验证结果 (可选但强烈推荐) ---
% 使用计算出的矩阵T，将工件坐标系下的点转换到基坐标系，看是否与测量值匹配。
% T 是 4x4, points_raw_homogeneous' 是 4xN, 结果是 4xN, 最后转置回 Nx4。
transformed_points = (T * points_raw_homogeneous')';

disp('--- 验证 ---');
disp('原始基坐标系下的点 (points_1):');
disp(points_1_homogeneous);
disp('使用T矩阵变换工件坐标系下的点得到的结果:');
disp(transformed_points);

% 计算并显示误差
% .^2 是逐元素平方，sum(..., 2) 是按行求和，sqrt是开方
error = sqrt(sum((points_1_homogeneous - transformed_points).^2, 2));
mean_error = mean(error);

fprintf('\n每个点的欧氏距离误差:\n');
disp(error);
fprintf('平均误差: %e\n', mean_error);
figure;axis equal;
plot3(points_1_homogeneous(:,1),points_1_homogeneous(:,2),points_1_homogeneous(:,3),'bo-');hold on;
plot3(transformed_points(:,1),transformed_points(:,2),transformed_points(:,3),'r*-');hold on;

figure;axis equal;
plot3(points_1_homogeneous(:,1),points_1_homogeneous(:,2),points_1_homogeneous(:,3),'bo-');hold on;
plot3(points_raw_homogeneous(:,1),points_raw_homogeneous(:,2),points_raw_homogeneous(:,3),'r*-');hold on;
