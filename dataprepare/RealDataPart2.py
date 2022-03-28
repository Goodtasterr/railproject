import open3d as o3d
import numpy as np
import sys
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time
sys.path.append('..')

def label2color(labels):
    """
    将label 转为 open3d colors 的矩阵
    :param labels: shape=[n,]
    :return: clors: shape=[n,3] 值的范围[0,1]
    """
    list = np.asarray([
        [127,127,127],[128,0,128],[255,0,255],
        # [127,127,127],
        [0, 0, 255],[127,127,127],[0, 0, 255],
        [0, 255, 0],
        [255,0,0],
        [0,0,255],
        [127, 0, 127],
        [0, 255, 0]
    ],dtype=np.float)
    return list[labels]/255

def poly_fit_x(points):
    # degree=2 y =         c*x^2 + b*x + a
    # degree=3 y = d*x^3 + c*x^2 + b*x + a
    poly_features = PolynomialFeatures(degree=1, include_bias=False)
    lin_reg = LinearRegression()
    rail_points = points
    x_poly = rail_points[:, 0:1] #[N,1] x
    y_poly = rail_points[:, 1:2] #[N,1] y

    X_poly = poly_features.fit_transform(x_poly)
    lin_reg.fit(X_poly, y_poly)
    poly_para = [lin_reg.intercept_, lin_reg.coef_]
    print("poly params: ",poly_para)
    min_x = np.min(rail_points[:,0])
    x = np.arange(min_x, np.max(rail_points[:,0]), 0.1).reshape(-1, 1)
    n_sample = (x).shape[0]

    poly_para = poly_para
    y = poly_para[0] + poly_para[1][0][0] * x# + poly_para[1][0][1] * x ** 2  # + poly_para[1][0][2]*x**3
    h = np.max(rail_points[:,2])

    line_points_mid = np.concatenate((h * np.ones([n_sample, 1]), y, x), axis=1)
    line_points_rd = np.concatenate((h * np.ones([n_sample, 1]), y + 1.7, x), axis=1)
    line_points_ld = np.concatenate((h * np.ones([n_sample, 1]), y - 1.7, x), axis=1)
    line_points_rt = np.concatenate((h * np.ones([n_sample, 1]) + 4, y + 1.7, x), axis=1)
    line_points_lt = np.concatenate((h * np.ones([n_sample, 1]) + 4, y - 1.7, x), axis=1)
    line_points = np.concatenate((line_points_rd, line_points_ld,
                                  line_points_rt, line_points_lt,line_points_mid), axis=0)
    # print("line_points shape:", line_points.shape)
    line_points = line_points[:,(2,1,0)]
    line_lines = [[j + i * n_sample, j + 1 + i * n_sample] for i in range(4) for j in range(n_sample - 1)]
    line_lines = np.asarray(line_lines)
    # 前后两个方形框，共8条线
    line_lines_add = np.asarray([[0, n_sample], [n_sample, 3 * n_sample],
                                 [2 * n_sample, 3 * n_sample],
                                 [0 * n_sample, 2 * n_sample],
                                 [n_sample - 1, 2 * n_sample - 1],
                                 [2 * n_sample - 1, 4 * n_sample - 1],
                                 [3 * n_sample - 1, 4 * n_sample - 1],
                                 [3 * n_sample - 1, n_sample - 1]
                                 ])
    line_mid = [[i + 4*n_sample,i + 1 + 4*n_sample] for i in range(n_sample - 1)]
    line_lines = np.concatenate((line_lines, line_lines_add), axis=0)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(line_lines),
    )
    return line_set

if __name__ == '__main__':
    print(1)
    #实现功能：论文第4章实验部分。
    #1.轨道线拟合，求IoU
    #2.安全行车区域
    #3.路面分割
    file_path = "../datareal/npyfile/a3/1587957663079948.npy"
    raw_array = np.load(file_path)
    points_array = raw_array[:, (2, 1, 0)]  # turn to x-y-z
    labels_array = raw_array[:, -1]
    # step 1. 提取出轨道的点云
    rail_array = points_array[labels_array == 1]
    # step 2. 步进的距离，
    step_x = 5.0 # 步进 x 米
    step_bais = 6.0 # 从 bais 米开始
    print("max and min of rail : ",np.max(rail_array,axis=0))
    print("min and min of rail : ",np.min(rail_array,axis=0))

    pcd_box = []
    for i in range(0,10):
        part_idx = (rail_array[:,0] >= (step_bais + i*step_x)) & (rail_array[:,0] <= (step_bais + (i+1)*step_x))
        part_rail_points = rail_array[part_idx]

        part_box = poly_fit_x(part_rail_points)
        pcd_box.append(part_box)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(label2color(labels_array.astype(np.int)))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(3, center=mesh.get_center())
    pcd_box.append(pcd)
    pcd_box.append(mesh)


    o3d.visualization.draw_geometries(pcd_box)