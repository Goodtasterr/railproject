import open3d as o3d
import numpy as np
import sys
import os
sys.path.append('..')

class VoxelDownSample(object):
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.pcd = o3d.geometry.PointCloud()

    def from_txt2numpy(self, file_path, save_flag=False):
        data_list = []

        file = open(file_path)
        for line in file.readlines():
            if(len(line) > 1):
                data = line[:-1].split(' ') #[:-1] 去掉换行
                data_list.append(data)
        # vec = np.genfromtxt(file_path, delimiter=' ', dtype=np.float32)
        file.close()
        vec = np.asarray(data_list, dtype=np.float32)
        print("origin point cloud size: ",vec.shape)
        self.pcd.points = o3d.utility.Vector3dVector(vec)
        downpcd = self.pcd.voxel_down_sample(voxel_size=self.voxel_size)
        downvec = np.asarray(downpcd.points)
        print("after downsample point cloud size: ", downvec.shape)
        if save_flag:
            np.savetxt(file_path,downvec,fmt='%f',delimiter=' ')

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

if __name__ == '__main__':
    file_path = "../unrealdata/s1_stright.txt"
    pcd = o3d.io.read_point_cloud(file_path, format='xyz')
    o3d.visualization.draw_geometries([pcd])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                             ransac_n=10,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([ outlier_cloud],
                                      zoom=0.8,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])

    print("Statistical oulier removal")
    cl, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
    display_inlier_outlier(outlier_cloud, ind)