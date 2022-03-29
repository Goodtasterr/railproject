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
        [127,127,127],[0,0,200],[127,127,127],
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
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
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
    y = poly_para[0] + poly_para[1][0][0] * x + poly_para[1][0][1] * x ** 2  # + poly_para[1][0][2]*x**3
    h = np.max(rail_points[:,2])

    y_bias = 0.9
    z_bias = 0.17
    line_points_mid = np.concatenate((h * np.ones([n_sample, 1]), y, x), axis=1)
    line_points_rd = np.concatenate((h * np.ones([n_sample, 1]) - z_bias*2, y + y_bias, x), axis=1) #d=底部，t=顶部
    line_points_ld = np.concatenate((h * np.ones([n_sample, 1]) - z_bias*2, y - y_bias, x), axis=1)
    line_points_rt = np.concatenate((h * np.ones([n_sample, 1]) + z_bias, y + y_bias, x), axis=1)
    line_points_lt = np.concatenate((h * np.ones([n_sample, 1]) + z_bias, y - y_bias, x), axis=1)
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
    # line_lines = np.asarray(line_mid)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(line_lines),
    )
    line_color = [[1, 0, 0] for i in range(len(line_lines))]
    line_set.colors = o3d.utility.Vector3dVector(line_color)

    return line_set


def poly_fit_iou(points,labels):
    # degree=2 y =         c*x^2 + b*x + a
    # degree=3 y = d*x^3 + c*x^2 + b*x + a
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    lin_reg = LinearRegression()
    rail_points = points[labels==1]
    x_poly = rail_points[:, 0:1] #[N,1] x
    y_poly = rail_points[:, 1:2] #[N,1] y

    X_poly = poly_features.fit_transform(x_poly)
    lin_reg.fit(X_poly, y_poly)
    poly_para = [lin_reg.intercept_, lin_reg.coef_]
    # print("poly params: ",poly_para)

    y_bias = 0.93
    z_bias = 0.17

    y = poly_para[0] + poly_para[1][0][0] * points[:,0] + poly_para[1][0][1] * (points[:,0] ** 2)  # + poly_para[1][0][2]*x**3
    y_left = y - y_bias
    y_right = y + y_bias

    h_max = np.max(rail_points[:,2])+0.1
    h_min = np.min(rail_points[:,2])-0.1

    idx = (points[:,1] > y_left) & (points[:,1] < y_right) \
          & (points[:,0] > 6) & (points[:,0] < 70) & (points[:,2] > h_min) & (points[:,2] < h_max)

    pred_labels = labels[idx]
    pred_labels_rev = labels[~idx]
    # print(pred_labels.shape,pred_labels_rev.shape)
    print(pred_labels[pred_labels==0].shape,pred_labels_rev[pred_labels_rev==1].shape)
    iou_res = len(pred_labels[pred_labels==1])/(len(pred_labels[pred_labels==0]) +
                                                len(pred_labels_rev[pred_labels_rev==1])+len(pred_labels[pred_labels==1]))
    print("iou: ",iou_res)
    return iou_res

def poly_fit_iou_ours(points,labels):
    # degree=2 y =         c*x^2 + b*x + a
    # degree=3 y = d*x^3 + c*x^2 + b*x + a
    poly_features = PolynomialFeatures(degree=1, include_bias=False)
    lin_reg = LinearRegression()
    rail_points = points[labels==1]

    #超参数
    step_x = 5.0 # 步进 x 米
    step_bais = 6.0 # 从 bais 米开始
    # rail_part_d = [4,9,13,17,21]
    rail_cross = 2
    delta_k = 0.1
    rail_part_d = [6,10,19,32,49,70]
    rail_part_range = [[rail_part_d[i],rail_part_d[i+1]] for i in range(5)]

    fit_para = []
    for i, part_range in enumerate(rail_part_range):
        part_idx = (rail_points[:,0]>=part_range[0]) & (rail_points[:,0]<part_range[1])
        rail_points_part = rail_points[part_idx]

        if len(rail_points_part) < 3:
            return "error1" #error1 表示段落内小于3个点，难以拟合

        x_poly = rail_points_part[:, 0:1] #[N,1] x
        y_poly = rail_points_part[:, 1:2] #[N,1] y

        X_poly = poly_features.fit_transform(x_poly)
        lin_reg.fit(X_poly, y_poly)
        fit_para.append([part_range[0],part_range[1],0,lin_reg.coef_[0][0],lin_reg.intercept_[0]]) #r1,r2,a,b,c; ax2+bx+c

        if i > 0: #从1开始和上一次比较斜率b
            if (np.abs(fit_para[i][3] - fit_para[i-1][3]) > delta_k) \
                    or (np.abs(fit_para[i][4] - fit_para[i-1][4]) > 0.1): #如果超过，则对后续的点二次拟合
                poly_features2 = PolynomialFeatures(degree=2, include_bias=False)
                part_idx2 = (rail_points[:, 0] >= fit_para[i-1][0]) & (rail_points[:, 0] < 70)
                rail_points_part = rail_points[part_idx2]

                x_poly = rail_points_part[:, 0:1] #[N,1] x
                y_poly = rail_points_part[:, 1:2] #[N,1] y

                X_poly = poly_features2.fit_transform(x_poly)
                lin_reg.fit(X_poly, y_poly)
                fit_para.pop() #删除当前段参数，因为它与上一段差距大。将上一段用2次函数代替
                fit_para[i-1] = [fit_para[i-1][0],70,lin_reg.coef_[0][1],lin_reg.coef_[0][0],lin_reg.intercept_[0]]

                break
    print("fit_para: ",fit_para)
    #计算iou：
    for i, para in enumerate(fit_para):
        idx_part = (points[:,0] >= para[0]) & (points[:,0] < para[1])  #取idx

        points_part = points[idx_part]
        labels_part = labels[idx_part]

        y_bias = 0.95
        y = para[4] + para[3] * points_part[:, 0] +para[2] * (
                    points_part[:, 0] ** 2)
        y_left = y - y_bias
        y_right = y + y_bias

        h_max = np.max(rail_points[:, 2]) + 0.1
        h_min = np.min(rail_points[:, 2]) - 0.1

        idx_iou = (points_part[:, 1] > y_left) & (points_part[:, 1] < y_right) \
              & (points_part[:, 0] > 6) & (points_part[:, 0] < 70) & (points_part[:, 2] > h_min) & (points_part[:, 2] < h_max)

        pred_labels = labels_part[idx_iou]
        pred_labels_rev = labels_part[~idx_iou]
        # print("rail, bkg:",pred_labels.shape,pred_labels_rev.shape)
        # print(pred_labels[pred_labels == 1].shape, pred_labels_rev[pred_labels_rev == 0].shape)
        # exit()
        iou_res = len(pred_labels[pred_labels == 1]) / (len(pred_labels[pred_labels == 0]) +
                                                        len(pred_labels_rev[pred_labels_rev == 1]) + len(
                    pred_labels[pred_labels == 1]))
        print("iou: ", iou_res)
        return iou_res


if __name__ == '__main__':
    print(1)
    #实现功能：论文第4章实验部分。
    #1.轨道线拟合，求IoU
    seq_root = "D:/code/dataset/pc_npy_a9"
    seq_names =  os.listdir(seq_root)

    iou_res_all = []
    for seq_name in seq_names:
        if seq_name == "a1" or seq_name == "a2" or seq_name == "a6" or seq_name == "a7": #
        # if seq_name == "a5a" or seq_name == "a5b" or seq_name == "a5c" or seq_name == "errorlabel": #
        # if seq_name == "a3a" or seq_name == "a3b": #

            # file_root = "D:/code/dataset/pc_npy_a9/a1"
            file_root = os.path.join(seq_root,seq_name)
            files = os.listdir(file_root)

            for i, file_name in enumerate(files):
                npy_path = os.path.join(file_root,file_name)
                raw_arr = np.load(npy_path)
                points_arr = raw_arr[:, (2, 1, 0)]
                labels_arr = raw_arr[:, -1].astype(np.int)
                iou_res = poly_fit_iou_ours(points_arr, labels_arr)
                if iou_res != 'error1':
                    iou_res_all.append(iou_res)

    print(len(iou_res_all), np.mean(iou_res_all))
    exit()
    #2.安全行车区域
    #3.路面分割
    file_path = "../datareal/npyfile/a1/1587955555044845.npy"
    raw_array = np.load(file_path)
    points_array = raw_array[:, (2, 1, 0)]  # turn to x-y-z
    labels_array = raw_array[:, -1]

    # step 1. 提取出轨道的点云
    rail_array = points_array[labels_array == 1]
    # step 2. 步进的距离，
    step_x = 5.0 # 步进 x 米
    step_bais = 6.0 # 从 bais 米开始
    rail_part_d = [4,9,13,17,21]
    rail_cross = 2
    rail_part_range = [6,10,19,32,49,70]
    print("max and min of rail : ",np.max(rail_array,axis=0))
    print("min and min of rail : ",np.min(rail_array,axis=0))

    pcd_box = []
    # for i, dist in enumerate(rail_part_range):
    #     if i <= 4:
    #         part_idx = (rail_array[:, 0] >= rail_part_range[i]-rail_cross) & (rail_array[:, 0] < rail_part_range[i + 1])
    #         part_rail_points = rail_array[part_idx]
    #         part_box = poly_fit_x(part_rail_points)
    #         pcd_box.append(part_box)

    # pcd_box = []
    # for i in range(0,10):
    #     part_idx = (rail_array[:,0] >= (step_bais + i*step_x)) & (rail_array[:,0] <= (step_bais + (i+1)*step_x))
    #     part_rail_points = rail_array[part_idx]
    part_box = poly_fit_x(rail_array)
    pcd_box.append(part_box)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(label2color((labels_array.astype(np.int))))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(2, center=mesh.get_center())
    pcd_box.append(pcd)
    pcd_box.append(mesh)


    o3d.visualization.draw_geometries(pcd_box)