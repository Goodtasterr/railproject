import open3d as o3d
import numpy as np
import sys
import os
import math
import copy
sys.path.append('..')

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def ckeck_npyfile():
    ##check npyfile
    file_root = "../dataunreal/npyfile1/"
    npy_paths = os.listdir(file_root)
    for i, npy_path in enumerate(npy_paths):
        npy_vec = np.load(os.path.join(file_root,npy_path))
        print("npy_vec.shape: ", npy_vec.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(npy_vec)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh.scale(3, center=mesh.get_center())
        o3d.visualization.draw_geometries([pcd,mesh])

def set_rotate_bias():
    file_root = "../dataunreal/rawpointcloud1/" #前后，左右，上下
    pc_names = os.listdir(file_root)
    rotate_paras = np.asarray([-0.3, -0.3, -0.35, -0.37, -0.42, -0.42,
                                -1.4, -1.4, -1.4, -1.4, -1.4, -1.36,
                              0, 0, 0, 0, 0, 0])

    rotate_paras2 = np.asarray([0, 0, 0, 0, 0,
                              -1.36,-1.36,-1.36,-1.36,-1.312,
                              -0.002, -0.002])
    print(rotate_paras)
    for i , pc_name in enumerate(pc_names) :
        print(i+1)
        # railwayPC = RailwayPointCloud(os.path.join(file_root,pc_name))
        # railwayPC.pc_show(name=pc_name,rotate_para=rotate_paras[i])

def label_points(points,parameter,range):
    label = np.zeros([points.shape[0]]).astype(int)
    ranged = copy.deepcopy(range)
    if parameter is not None:
        ranged[3] = ranged[3] + parameter[1]*(points[:,0]**2) + (parameter[0])*points[:,0]  # c + b*x + a*x^2
        ranged[2]  += ranged[3] - 2

    idx_x = (points[:,0] > range[0]) & (points[:,0] < range[1])
    idx_y = (points[:,1] > ranged[2]) & (points[:,1] < ranged[3])
    idx_z = (points[:,2] > range[4]) & (points[:,2] < range[5])
    idx = idx_x & idx_y & idx_z
    index = np.asarray(idx).astype(int)
    label = index + label
    return label,idx


def lable_unrealdata():
    '''
    思路：
        1.直轨道：使用矩形框；
        2.左转/右转轨道：先预设二次函数[b,a]，精确调整[上，下]，找到一根铁轨表面部分点云（曲线形状），
                        对这些点二次拟合获得准确的[b,a]并更新。
    return:
    '''
    #label params
    rail_ranges =[6, 100, .1, 0.87,  -2.74, -2.2] # 前后，左右，上下  #[-2.5,-2.73] 轨道位置
    parameter_lr = np.asarray([ 0.00114602, -0.00112116]).astype(np.float32)    # ax^2 + bx

    #label 过程
    file_root = "../dataunreal/npyfile2/"
    npy_paths = os.listdir(file_root)
    show_flag = True
    save_flag = True
    for i, npy_path in enumerate(npy_paths):
        if i>8 and i<10:
            file_path = os.path.join(file_root,npy_path)
            print(file_path)
            npy_vec = np.load(file_path)
            print("npy_vec.shape: ", npy_vec.shape)

            label,idx = label_points(npy_vec,parameter_lr,rail_ranges)

            new_points = np.concatenate((npy_vec,label[:,np.newaxis]),axis=1)
            print(new_points.shape)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(new_points[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(label2color(new_points[:,-1].astype(np.int)))
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh.scale(3, center=mesh.get_center())

            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression

            rail_points = new_points[idx]
            x_poly = rail_points[:, 0]
            x_poly = x_poly[:, np.newaxis]
            y_poly = rail_points[:, 1]
            y_poly = y_poly[:, np.newaxis]

            # degree=2 y =         c*x^2 + b*x + a
            # degree=3 y = d*x^3 + c*x^2 + b*x + a
            poly_features = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly_features.fit_transform(x_poly)
            lin_reg = LinearRegression()
            lin_reg.fit(X_poly, y_poly)
            poly_para = [lin_reg.intercept_, lin_reg.coef_]
            print('poly_para : ', poly_para)

            if show_flag:
                o3d.visualization.draw_geometries([pcd,mesh],window_name=npy_path)
            if save_flag:
                short_name = npy_path.split(".")[0]
                np.save(os.path.join(file_root, short_name), new_points)

def label2color(labels):
    """
    将label 转为 open3d colors 的矩阵
    :param labels: shape=[n,]
    :return: clors: shape=[n,3] 值的范围[0,1]
    """
    list = np.asarray([
        [127,127,127],
        [255,0,0],
        [0,0,255],
        [127, 0, 127],
        [127, 127, 0]
    ],dtype=np.float)
    return list[labels]/255

def fit_xy(degree,x,y):
    """
        拟合 y = f(x)
    :param degree: int 1或2
    :param x: [N,1]
    :param y: [N,1]
    :return: [[c],[[b],[a],...]]  c,b,a倒着排的
    """
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    lin_reg = LinearRegression()
    X_poly = poly_features.fit_transform(x)
    lin_reg.fit(X_poly, y)
    # poly_para = [lin_reg.intercept_, lin_reg.coef_]
    return [lin_reg.intercept_, lin_reg.coef_]

def gen_line_set(begin,end,poly_para,code=2):
    """

    :param begin:
    :param end:
    :param poly_para:
    :param code: code=1 显示中心线，=2 显示3d框，
    :return:
    """

    x = np.arange(begin, end, 0.1).reshape(-1, 1)
    n_sample = (x).shape[0]
    y = poly_para[0] + poly_para[1][0][0] * x + poly_para[1][0][1] * x ** 2  # + poly_para[1][0][2]*x**3
    h = -2.9
    safe_bais = 1.3

    line_points_mid = np.concatenate((h * np.ones([n_sample, 1]), y, x), axis=1)
    line_points_rd = np.concatenate((h * np.ones([n_sample, 1]), y + 1.7 + safe_bais, x), axis=1)
    line_points_ld = np.concatenate((h * np.ones([n_sample, 1]), y - 1.7 - safe_bais, x), axis=1)
    line_points_rt = np.concatenate((h * np.ones([n_sample, 1]) + 4, y + 1.7+ safe_bais, x), axis=1)
    line_points_lt = np.concatenate((h * np.ones([n_sample, 1]) + 4, y - 1.7-safe_bais, x), axis=1)
    line_points = np.concatenate((line_points_rd, line_points_ld,
                                  line_points_rt, line_points_lt,line_points_mid), axis=0)
    line_points = line_points[:,(2,1,0)]
    if code == 2:
        line_lines = [[j + i * n_sample, j + 1 + i * n_sample] for i in range(4) for j in range(n_sample - 1)]
    elif code == 1:
        line_lines = [[j + 4 * n_sample, j + 1 + 4 * n_sample] for j in range(n_sample - 1)]
    else:
        print("code illegal.")

    line_lines = np.asarray(line_lines)
    print("test: ", line_lines.shape, line_points.shape)

    # 前后两个方形框，共8条线
    line_lines_add = np.asarray([[0, n_sample], [n_sample, 3 * n_sample],
                                 [2 * n_sample, 3 * n_sample],
                                 [0 * n_sample, 2 * n_sample],
                                 [n_sample - 1, 2 * n_sample - 1],
                                 [2 * n_sample - 1, 4 * n_sample - 1],
                                 [3 * n_sample - 1, 4 * n_sample - 1],
                                 [3 * n_sample - 1, n_sample - 1]
                                 ])
    if code == 2:
        line_lines = np.concatenate((line_lines, line_lines_add), axis=0)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(line_lines),
    )

    line_color = [[1, 0, 0] for i in range(len(line_lines))]
    line_set.colors = o3d.utility.Vector3dVector(line_color)
    return line_set


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
