import open3d as o3d
import numpy as np
import sys
import os
sys.path.append('..')

class RailwayPointCloud():
    def __init__(self,  file_path):
        self.file_path = file_path
        self.vec = np.load(self.file_path)
        print(self.vec.shape)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.vec[:,:3])

    def voxel_downsample(self,voxel_size, save_flag=False):
        print("origin point cloud size: ",self.vec.shape)
        downpcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        o3d.visualization.draw_geometries([downpcd])
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

def o3d_edition():
    file_path = "../datareal/npyfile/a3/1587957663079948.npy"
    railwayPC = RailwayPointCloud(file_path)
    print("origin points shape: ",railwayPC.vec.shape)
    rail_points = railwayPC.vec[railwayPC.vec[:,-1] == 1]
    print(rail_points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rail_points[:, :3])
    railwayPC.segment_ground()


def pc_colors(arr):
    list = np.asarray([
        [255,0,0],
        [0,0,255],
        [127, 127, 127],
        [127, 0, 127],
        [127, 127, 0]
    ])
    colors = []
    for i in arr:
        colors.append(list[i])
    return np.asarray(colors)/255


if __name__ == '__main__':
    file_path = "../datareal/npyfile/a3/1587957663079948.npy"
    vec_raw = np.load(file_path)
    vec = vec_raw[:,(2,1,0)]  # turn to x-y-z
    beta = vec[:,1]/vec[:,0]
    r_dist = (vec[:,0]**2 + vec[:,1]**2)**0.5
    vec_hw = np.concatenate((r_dist[:,np.newaxis],beta[:,np.newaxis]),axis=1)
    print(np.max(vec_hw))
    # 分网格
    # get grid index
    max_bound = np.asarray([120,0.392699075])
    min_bound = np.asarray([0,-0.392699075])
    grid_size = np.asarray([120,45])
    crop_range = max_bound - min_bound
    cur_grid_size = grid_size
    intervals = crop_range / (cur_grid_size - 1)  #[120,(45°),6]/[480,360,32]
    print(intervals)

    if (intervals == 0).any(): print("Zero interval!")
    # [N,2] 坐标值 --> 网格编号
    grid_ind = (np.floor((np.clip(vec_hw, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
    print("grid_ind: ",grid_ind.shape)
    print("vec shape : ", vec.shape)
    vec_idx = np.arange(vec.shape[0])
    vec_idx = vec_idx[:,np.newaxis]
    points = np.concatenate((grid_ind,vec_idx),axis=1) # [h,w,i]
    points_all = np.concatenate((vec,vec_hw,grid_ind,vec_idx),axis=1) # [x,y,z,r,b,h,w,i]
    points = points[vec[:,2]<-2.2]
    sorted_idx = np.lexsort((points[:,1],points[:,0])) # 先用[:,0]排序，再用[:,1]排序
    new_points = points[sorted_idx]
    # print(new_points[:20])
    ground_idx = []
    for i in range(new_points.shape[0]):
        new_point = new_points[i]
        pixel_h = new_point[0]
        pixel_w = new_point[1]
        contain = [new_point[-1]]
        for j in range(i+1,new_points.shape[0]):
            temp_point = new_points[j]
            temp_h = temp_point[0]
            temp_w = temp_point[1]
            if (temp_h == pixel_h) and (temp_w == pixel_w):
                # 如果相等，放到容器中
                contain.append(temp_point[-1])
            else:
                # 如果不等，处理上一个容器，并退出循环
                if len(contain) != 0:
                    # print("deal the last contain h = {};  w = {}.".format(temp_h,temp_w))
                    contain_z_i = points_all[contain][:,(2,7)]  # 取出 z,i 数据， 按 z 排序
                    idx = np.lexsort((contain_z_i[:,1],contain_z_i[:,0]))
                    contain_z_i_sorted = contain_z_i[idx]

                    #判定：以 0.1米 为步进
                    if contain_z_i_sorted.shape[0]<=2:
                        for c in range(len(contain_z_i_sorted)):
                            ground_idx.append(contain_z_i_sorted[c][1])

                    else:
                        ground_z = contain_z_i_sorted[2][0]+0.06
                        ground_points = contain_z_i_sorted[contain_z_i_sorted[:,0]<ground_z]
                        for c in range(len(ground_points)):
                            ground_idx.append(ground_points[c][1])
                break

    points_color = np.zeros((vec.shape[0],1),dtype=np.int)
    npy_idx = np.asarray(ground_idx,dtype=np.int)
    points_color[npy_idx] = 2
    print(points_color.shape)
    # exit()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vec[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pc_colors(points_color[:,-1].astype(np.int)))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(2, center=mesh.get_center())
    o3d.visualization.draw_geometries([pcd,mesh])


