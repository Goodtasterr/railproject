"""
用于现实轨道的 3D 框
属性：铁路场景点云，轨道标签
输入：点，标签；
方法：
    1. 拟合中心线，二次函数
    2. 使用步进，拟合
代码：
    1. pointclouds 表示原始数据矩阵，有x-y-z-i-lable
        points [x,y,z]; lable [l]...

存在问题：同时分离出障碍物和轨道，轨道点云需要分离轨道和障碍物；
解决：使用段与段的特征匹配，轨道与障碍物的水平高度是不同的；
可以放到 <第二章>
"""
import open3d as o3d
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time
def label2color(labels):
    """
    将label 转为 open3d colors 的矩阵
    :param labels: shape=[N,]
    :return: clors: shape=[N,3] 值的范围[0,1]
    """
    list = np.asarray([
        [127,127,127],
        [0, 255, 0],
        [255,0,0],
        [0,0,255],
        [127, 0, 127],
        [127, 127, 0]
    ],dtype=np.float)
    return list[labels]/255


class Show3DRail(object):
    def __init__(self, points, lables):
        self.points = points #[N,3]
        self.lables = lables #[N,]

        # degree=2 y =         c*x^2 + b*x + a
        # degree=3 y = d*x^3 + c*x^2 + b*x + a
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.lin_reg = LinearRegression()


    def poly_fit(self):
        print(self.points.shape,self.lables.shape)
        rail_points = self.points[self.lables == 1]
        x_poly = rail_points[:, 0:1] #[N,1] x
        y_poly = rail_points[:, 1:2] #[N,1] y

        t0 = time.time()
        X_poly = self.poly_features.fit_transform(x_poly)
        self.lin_reg.fit(X_poly, y_poly)
        poly_para = [self.lin_reg.intercept_, self.lin_reg.coef_]
        print("poly fit time cost:",time.time()-t0)
        print("poly params: ",poly_para)
        min_x = np.min(rail_points[:,0])
        x = np.arange(min_x, np.max(rail_points[:,0]), 0.1).reshape(-1, 1)
        n_sample = (x).shape[0]

        poly_para = poly_para
        y = poly_para[0] + poly_para[1][0][0] * x + poly_para[1][0][1] * x ** 2  # + poly_para[1][0][2]*x**3
        h = np.max(rail_points[:,2])

        line_points_mid = np.concatenate((h * np.ones([n_sample, 1]), y, x), axis=1)
        line_points_rd = np.concatenate((h * np.ones([n_sample, 1]), y + 1.7, x), axis=1)
        line_points_ld = np.concatenate((h * np.ones([n_sample, 1]), y - 1.7, x), axis=1)
        line_points_rt = np.concatenate((h * np.ones([n_sample, 1]) + 4, y + 1.7, x), axis=1)
        line_points_lt = np.concatenate((h * np.ones([n_sample, 1]) + 4, y - 1.7, x), axis=1)
        line_points = np.concatenate((line_points_rd, line_points_ld,
                                      line_points_rt, line_points_lt), axis=0)
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

        line_lines = np.concatenate((line_lines, line_lines_add), axis=0)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector(line_lines),
        )

        # line_color = [[0, 0, 0] for i in range(len(line_lines))]
        # line_set.colors = o3d.utility.Vector3dVector(line_color)
        return line_set
    def show(self):
        box = self.poly_fit()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(label2color(self.lables.astype(np.int)))
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh.scale(3, center=mesh.get_center())

        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(self.points[self.lables == 1])
        aabb = pcd_part.get_axis_aligned_bounding_box()  # 矩形框

        o3d.visualization.draw_geometries([pcd, mesh,box])