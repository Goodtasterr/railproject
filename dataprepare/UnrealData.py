import open3d as o3d
import numpy as np
import sys
import os
import math
import copy
sys.path.append('..')

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
        railwayPC = RailwayPointCloud(os.path.join(file_root,pc_name))
        railwayPC.pc_show(name=pc_name,rotate_para=rotate_paras[i])

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

def pc_colors(arr):
    list = np.asarray([
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [127, 0, 127],
        [127, 127, 0]
    ])
    colors = []
    for i in arr:
        colors.append(list[i])
    return np.asarray(colors)/255

def lable_unrealdata():
    '''
    思路：
        1.直轨道：使用矩形框；
        2.左转/右转轨道：先预设二次函数[b,a]，精确调整[上，下]，找到一根铁轨表面部分点云（曲线形状），
                        对这些点二次拟合获得准确的[b,a]并更新。
    return:
    '''
    #lable params
    rail_ranges =[6, 100, .1, 0.87,  -2.74, -2.2] # 前后，左右，上下  #[-2.5,-2.73] 轨道位置
    parameter_lr = np.asarray([ 0.00114602, -0.00112116]).astype(np.float32)    # ax^2 + bx

    #lable 过程
    file_root = "../dataunreal/npyfile2/"
    npy_paths = os.listdir(file_root)
    show_flag = True
    save_flag = True
    for i, npy_path in enumerate(npy_paths):
        if i>8 and i<10:
            file_path = os.path.join(file_root,npy_path)
            print(file_path)
            npy_vec = np.load(file_path)
            noise = (np.random.rand(npy_vec.shape[0],npy_vec.shape[1])-0.5) * 0.03  # 正态分布[0~1]-0.5 再放大n倍
            print("npy_vec.shape: ", npy_vec.shape)
            # npy_vec = npy_vec + noise
            label,idx = label_points(npy_vec,parameter_lr,rail_ranges)

            new_points = np.concatenate((npy_vec,label[:,np.newaxis]),axis=1)
            print(new_points.shape)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(new_points[:,:3])
            pcd.colors = o3d.utility.Vector3dVector(pc_colors(new_points[:,-1].astype(np.int)))
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
if __name__ == '__main__':
    print()



    # railwayPC.lable_rail()
    # railwayPC.segment_ground()
    # 1. 把xyz调好， 2.计算角度偏置并矫正角度， 3.轨道打标签




