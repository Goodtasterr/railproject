import open3d as o3d
import numpy as np
import sys
import os
import math
import copy
sys.path.append('..')
import time

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class RailwayPointCloud():
    def __init__(self,  file_path):
        self.file_path = file_path
        data_list = []
        file = open(file_path)
        for line in file.readlines():
            if(len(line) > 1):
                data = line[:-1].split(' ') #[:-1] 去掉换行符
                data_list.append(data)
        file.close()
        self.vec = np.asarray(data_list, dtype=np.float32)
        self.vec[:,(1,2)] *= -1
        print("point cloud shape: ",self.vec.shape)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.vec)

    def pc_show(self,name,rotate_para):
        save_flag = True
        rotate_check_flag = True
        show_flag = False
        theta = rotate_para # 绕z轴偏转角度，转为弧度。 旋转矩阵相乘。
        rotate_matrix = np.asarray([[math.cos(theta), -math.sin(theta), 0],
                                    [math.sin(theta), math.cos(theta), 0],
                                    [0,0,1]])
        rotate_vec = np.transpose(np.matmul(rotate_matrix,(np.transpose(self.vec))))
        if rotate_check_flag:
            index = rotate_vec[:,0] > 0
            beta = rotate_vec[index,1]/rotate_vec[index,0]
            print("max+min: ", (np.max(beta)+np.min(beta))/2)
        if save_flag:
            short_name = name.split(".")[0]
            np.save(os.path.join("../dataunreal/npyfile1/",short_name),rotate_vec)
        if show_flag:
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh.scale(3, center=mesh.get_center())
            self.pcd.points = o3d.utility.Vector3dVector(rotate_vec)
            o3d.visualization.draw_geometries([mesh, self.pcd],window_name=name)

    def voxel_downsample(self,voxel_size, save_flag=False):
        print("origin point cloud size: ",self.vec.shape)
        downpcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        downvec = np.asarray(downpcd.points)
        print("after downsample point cloud size: ", downvec.shape)
        if save_flag:
            np.savetxt(self.file_path,downvec,fmt='%f',delimiter=' ')

    def display_inlier_outlier(self, cloud, ind):
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
    def segment_ground(self):
        plane_model, inliers = self.pcd.segment_plane(distance_threshold=0.1,
                                                 ransac_n=10,
                                                 num_iterations=1000)
        print("inliers : ",(inliers[:10]))
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        self.display_inlier_outlier(self.pcd,inliers)

        inlier_cloud = self.pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = self.pcd.select_by_index(inliers, invert=True)
        o3d.visualization.draw_geometries([outlier_cloud],
                                          zoom=0.8,
                                          front=[-0.4999, -0.1659, -0.8499],
                                          lookat=[2.1813, 2.0619, 2.0999],
                                          up=[0.1204, -0.9852, 0.1215])

        print("Statistical oulier removal")
        cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=20,
                                                           std_ratio=2.0)
        self.display_inlier_outlier(outlier_cloud, ind)

    def lable_rail(self):
        self.vec[:,-1] *= (-1)
        points = self.pc_range(self.vec,[3,132,3,71,-2.6,4])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

    def pc_range(self,points, range):
        '''

        :param points: nparray:[N,4]
        :param range: list:[6] x_min,x_max,y_min,y_max,z_min,z_max
        :return: [N',4]
        '''
        index_x = (points[:, 0] > (range[0])) & (points[:, 0] < range[1])
        index_y = (points[:, 1] > (range[2])) & (points[:, 1] < range[3])
        index_z = (points[:, 2] > (range[4])) & (points[:, 2] < range[5])
        index = index_x & index_y & index_z
        points_ranged = points[index]

        return points_ranged

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

class RailProcess():
    def __init__(self,  points, lables):
        self.points = points
        self.lables = lables
        self.rail_points = points[lables == 1]

    def rail_points_count(self):
        """
        预设固定步进长度，分段计算轨道点的数量，
        :return:
        """
        step = 5
        rail_start_dist = int(np.min(self.rail_points[:,0]))
        rail_end_dist = int(np.max(self.rail_points[:,0])) + 1
        # n = int((rail_end_dist - rail_start_dist)/step) + 1
        n=18
        print(rail_start_dist,rail_end_dist,n)

        count_arr = np.ones([n,2],dtype=np.int) #记录(起始距离，一段轨道点的数量)
        for i in range(n):
            start_dist = rail_start_dist + i * step
            stop_dist = start_dist + step
            idx = (self.rail_points[:,0] > start_dist) & (self.rail_points[:,0] < stop_dist)
            rail_points_part = self.rail_points[idx]
            #shape[0] 是点的数量
            points_number = rail_points_part.shape[0]
            count_arr[i][0] = start_dist
            count_arr[i][1] = points_number

        return count_arr

    def fit_rail(self):
        """
        拟合轨道曲线
        :return: [N,5] d1,d2,a,b,c
        """
        #1. 给定轨道点云
        railpoints = self.rail_points
        #2. 设定步长矩阵
        d1_start = np.min(railpoints[:,0]) #起始距离区最小x的值
        d_steps = [5,15,30,50] #步进的长度，共100米，根据情况改
        curve_res = []
        overlap_rate = 0.3
        overlap_d = 0  #首次交叉区域为0
        for step in d_steps:
            d2_stop = d1_start + step #中止距离
            part_idx = (railpoints[:, 0] > (d1_start-overlap_d)) and (railpoints[:, 0] < d2_stop)
            part_points = railpoints[part_idx]

            x_poly = part_points[:, 0:1] #[N,1] x
            y_poly = part_points[:, 1:2] #[N,1] y
            poly_features = PolynomialFeatures(degree=1, include_bias=False)
            lin_reg = LinearRegression()
            X_poly = poly_features.fit_transform(x_poly)
            lin_reg.fit(X_poly, y_poly)
            poly_para = [lin_reg.intercept_, lin_reg.coef_]
            part_curve = [d1_start, d2_stop, lin_reg.coef_[0][0],poly_para[0],0]
            curve_res.append(part_curve)  #得到结果，添加到res
            overlap_d = step * overlap_rate  # 交叉率 * 步进距离 = 交叉距离

        #得到了每次的一次函数拟合结果
        #————————需要与上一次作对比，符合流程图，斜率差距大的，判定为弯轨道，进行二次函数拟合

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3,
                                      front=[ 0.035240267694182037, -0.96557625585128182, 0.25772197746614944 ],
                                      lookat=[ 53.634064052299877, -0.09981487264696895, 13.151208585339701 ],
                                      up=[ -0.015453881623073734, 0.25732481629799092, 0.96620138504350395 ],)
    return inlier_cloud, outlier_cloud

from util.UnrealDataUtil import fit_xy,gen_line_set
if __name__ == '__main__':
    print("非规则路面分割方法")
    #1.load np array; 2.trans x-y-z to b-r-z [b=y/x, r=(x2+y2)^0.5]; 3.分配pixel坐标,二维的
    file_root = "../dataunreal/unreal_res_0111"
    # npy_paths = os.listdir(file_root)
    npy_paths = sorted(os.listdir(file_root), key=lambda s: int(s[:-4]))
    show_flag = True
    noise_flag = False
    res_list = []
    for i, npy_path in enumerate(npy_paths):
        if i>=0 and i<29:
            o3dshow = []
            file_path = os.path.join(file_root,npy_path)
            print(file_path)
            npy_vec = np.load(file_path)

            print("npy_vec shape: ",npy_vec.shape)
            rail_idx = (npy_vec[:,-1]) % 2==1
            rail_x = npy_vec[rail_idx,:1]
            rail_y = npy_vec[rail_idx,1:2]
            rail_z = npy_vec[rail_idx,2:3]

            line_params = fit_xy(degree=2,x=rail_x,y=rail_y)
            # 框出安全区域内的点
            safe_bais = 1.2
            vec_x = npy_vec[:,:1]
            vec_y = npy_vec[:,1:2]
            vec_z = npy_vec[:,2:3]
            up_y = line_params[0] + line_params[1][0][0] * vec_x + line_params[1][0][1] * vec_x ** 2 + 1.7 + safe_bais
            down_y = line_params[0] + line_params[1][0][0] * vec_x + line_params[1][0][1] * vec_x ** 2 - 1.7 - safe_bais
            inner_idx = (vec_y>down_y) & (vec_y<up_y) & (vec_z > -2.9) & (vec_z < 2)
            print(vec_y.shape,down_y.shape)
            line_set = gen_line_set(np.min(rail_x),np.max(rail_x),line_params,2)
            o3dshow.append(line_set)

            #1.pcd_outer 安全范围之外的点，灰色表示
            outer_idx = ~inner_idx.squeeze()#(inner_idx*(-1)+1).astype(np.bool).squeeze()
            print(outer_idx.shape)
            pcd_outer = o3d.geometry.PointCloud()
            pcd_outer.points = o3d.utility.Vector3dVector(npy_vec[outer_idx, :3])
            pcd_outer.colors = o3d.utility.Vector3dVector(label2color(npy_vec[outer_idx,-1].astype(np.int)))
            o3dshow.append(pcd_outer)

            #2.pcd_inner 安全范围之内的点，绿色表示
            pcd_inner = o3d.geometry.PointCloud()
            pcd_inner.points = o3d.utility.Vector3dVector(npy_vec[inner_idx.squeeze(), :3])
            inner_color = np.ones(inner_idx.shape[0],dtype=np.int)*8
            pcd_inner.colors = o3d.utility.Vector3dVector(label2color(inner_color[inner_idx.squeeze()]))
            # o3dshow.append(pcd_inner)

            #2.5. 将pcd_inner分段。根据line_params和x值算y，比较y与y‘的大小得到idx。


            #3.对pcd_inner分割平面
            plane_model, inliers = pcd_inner.segment_plane(distance_threshold=0.3,
                                                          ransac_n=7,
                                                          num_iterations=2000)
            #4.获得剩余点云 outlier_cloud , 使用DBSCAN
            inlier_cloud, outlier_cloud = display_inlier_outlier(pcd_inner,inliers)
            o3dshow.append(inlier_cloud) #加入地平面点

            labels = np.array( #对 outlier_cloud 剩余点做聚类
                outlier_cloud.cluster_dbscan(eps=0.9, min_points=7, print_progress=True))
            print("labels::  ",labels)
            #5.判断DBSCAN的结果
            if (labels != []) and (len(labels) != 0):
                n_class = np.max(labels) + 1  #取最大类别
                n_color = np.zeros(labels.shape,dtype=np.int)
                n_class_res = 0
                for i in range(n_class):
                    points_count = np.sum((labels==i).astype(np.int))
                    print("points_count: ",points_count)
                    if(points_count > 50):
                        n_class_res += 1
                        n_color[labels==i] = n_class_res
                        #增加据障碍物矩形框，取np
                        obstacle_arr = np.asarray(outlier_cloud.points)
                        pcd_part = o3d.geometry.PointCloud()
                        pcd_part.points = o3d.utility.Vector3dVector(obstacle_arr[labels == i])
                        aabb = pcd_part.get_axis_aligned_bounding_box()
                        o3dshow.append(aabb)

                outlier_cloud.colors = o3d.utility.Vector3dVector(label2color(n_color))
                o3dshow.append(outlier_cloud)
                o3d.visualization.draw_geometries([outlier_cloud],window_name=npy_path)

            if noise_flag:
                noise_x = (np.random.rand(npy_vec.shape[0],1)-0.5) * 0.05  # 正态分布[0~1]-0.5 再放大n倍
                noise_y = (np.random.rand(npy_vec.shape[0],1)-0.5) * 0.02  # 正态分布[0~1]-0.5 再放大n倍
                noise_z = (np.random.rand(npy_vec.shape[0],1)-0.5) * 0.03  # 正态分布[0~1]-0.5 再放大n倍
                noise = np.concatenate((noise_x,noise_y,noise_z),axis=1)
                npy_vec[:,:3] += noise

            if show_flag:

                mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                mesh.scale(2, center=mesh.get_center())
                o3dshow.append(mesh)
                o3d.visualization.draw_geometries(o3dshow,window_name=npy_path,
                                                  zoom=0.3,
                                                  front=[ 0.096314076554831055, 0.020511804979741765, 0.99513962061303918 ],
                                                  lookat=[ 53.229220539861529, -2.0228925385089158, 5.6663705489609022 ],
                                                  up=[ -0.0034982659029790973, 0.99978844152479429, -0.020269048549330322 ],
                                                  )





