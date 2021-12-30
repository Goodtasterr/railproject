import open3d as o3d
import numpy as np
import sys
import os
import math
import copy
sys.path.append('..')
import random


def pc_colors(arr):
    list = np.asarray([
        [127,127,127],
        [0, 255, 0],
        [255, 0, 0],
        [0, 0, 255],
        [0, 255, 255]
    ])
    colors = []
    for i in arr:
        colors.append(list[i])
    return np.asarray(colors)/255

class GenPointCloudData():
    def __init__(self,  file_path):
        self.file_path = file_path
        self.vec = np.load(file_path)
        print("point cloud shape: ",self.vec.shape)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.vec[:,:3])

        _, self.inliers = self.pcd.segment_plane(distance_threshold=0.1,
                                                 ransac_n=10,
                                                 num_iterations=1000)
    def generate_data(self,code,save_path):
        ground_idx = copy.deepcopy(self.inliers) #地面点的idx
        random.shuffle(ground_idx) #打乱
        #取丢弃点的idx
        drop_idx = ground_idx[:int(len(ground_idx)*0.08*(1+code/3))]
        #保留点的idx
        left_idx = np.ones((self.vec.shape[0]),dtype=bool)
        left_idx[drop_idx] = False
        left_points = self.vec[left_idx]
        #生成noise
        noise = (np.random.rand(left_points.shape[0], 1) - 0.5) * 0.02
        #添加到源数据
        left_points[:,(code%3):(code%3)+1] += noise #依次加到x-y-z轴
        random.shuffle(left_points)
        print("test:  ",noise.shape, left_points.shape)


        save_flag = True
        if save_flag:
            np.save(save_path,left_points)
        show_flag = False
        if show_flag:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(left_points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pc_colors(left_points[:, -1].astype(np.int)))
            pcd.points = o3d.utility.Vector3dVector(left_points[:,:3])
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh.scale(3, center=mesh.get_center())
            o3d.visualization.draw_geometries([pcd, mesh])
    def self_gen(self):
        file_root = "../dataunreal/npyfile1/"
        npy_paths = os.listdir(file_root)
        for i, npy_path in enumerate(npy_paths):
            file_path = os.path.join(file_root, npy_path)
            print(file_path)
            generatorPC = GenPointCloudData(file_path)
            file_name = npy_path.split('.')[0]
            for i in range(9):
                generatorPC.generate_data(i, os.path.join("../dataunreal/gendata/", file_name + 'g' + str(i)))

def count_points_number():
    file_root = "../dataunreal/gendata/"
    seq_names = os.listdir(file_root)
    rail_points_sum = 0
    background_points_sum = 0
    for i, seq_name in enumerate(seq_names): # path: a**
        seq_paths = os.path.join(file_root, seq_name)

        file_names = os.listdir(seq_paths)
        # print(seq_paths)
        rail_points_sum_seq = 0
        background_points_sum_seq = 0
        for j, file_name in enumerate(file_names): # path: *.npy
            file_path = os.path.join(seq_paths, file_name)
            # print(file_path)
            vec = np.load(file_path)
            background_idx = vec[:,-1]==0
            rail_idx = vec[:,-1]==1
            background_points = vec[background_idx]
            rail_points = vec[rail_idx]
            rail_points_sum += rail_points.shape[0]
            background_points_sum += background_points.shape[0]

            rail_points_sum_seq += rail_points.shape[0]
            background_points_sum_seq += background_points.shape[0]
            # print('test:',vec.shape,background_points.shape[0],rail_points.shape[0])
            # exit()
        print("rail_points_seq: {} ;  bg_points_seq: {} ;".format(rail_points_sum_seq,background_points_sum_seq))
    print("rail_points_sum,background_points_sum",rail_points_sum,background_points_sum)
## rail_points_sum: 311716   background_points_sum:4518028
if __name__ == '__main__':
    print()
    file_root = "../dataunreal/unreal_res_1230"
    npy_paths = sorted(os.listdir(file_root),key=lambda s: int(s[:-4]))
    # npy_paths = sorted(os.listdir(file_root))
    # print(npy_paths)
    sum_tp = 0  #==3
    sum_fp = 0  #==1
    sum_fn = 0  #==2

    sum_result_flag = False
    show_flag = True
    for i, npy_path in enumerate(npy_paths):
        if i >= 20:
            file_path = os.path.join(file_root, npy_path)
            vec = np.load(file_path)
            # print(np.max(vec,axis=0))
            # print(np.min(vec,axis=0))
            if show_flag:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vec[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(pc_colors(vec[:, -1].astype(np.int)))
                pcd.points = o3d.utility.Vector3dVector(vec[:,:3])
                mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                mesh.scale(3, center=mesh.get_center())

                o3d.visualization.draw_geometries([pcd, mesh],
                                                  zoom=0.34,
                                                  front=[ -0.31564211951980803, -0.049484015586810912, 0.94758713825507768 ],
                                                  lookat=[ 41.424667981818729, 1.2425143526741407, -4.1332853722269771 ],
                                                  up=[ -0.013715723560478914, 0.99877286484398098, 0.047588269337724282 ])


            if sum_result_flag:
                sum_tp += np.sum((vec[:,-1]==3).astype(np.int))
                sum_fp += np.sum((vec[:,-1]==1).astype(np.int))
                sum_fn += np.sum((vec[:,-1]==2).astype(np.int))

                if(i%10==9):
                    print("i = {};  sum_tp = {};  sum_fp = {};  sum_tn = {};".format(i,sum_tp,sum_fp,sum_fn))
                    iou = sum_tp*100.0/(sum_tp+sum_fp+sum_fn)
                    precision = sum_tp*100.0/(sum_tp+sum_fp)
                    recall = sum_tp*100.0/(sum_tp+sum_fn)
                    print("iou = {};  precision = {};  recall = {}".format(iou,precision,recall))

                    sum_tp = 0  # ==3
                    sum_fp = 0  # ==1
                    sum_fn = 0  # ==2
# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" :
# 	[
# 		{
# 			"boundingbox_max" : [ 99.049998643118386, 45.390001143475359, 5.1161642074584961 ],
# 			"boundingbox_min" : [ -0.28335097001763665, -46.032932290085718, -3.6534450054168701 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.31564211951980803, -0.049484015586810912, 0.94758713825507768 ],
# 			"lookat" : [ 41.424667981818729, 1.2425143526741407, -4.1332853722269771 ],
# 			"up" : [ -0.013715723560478914, 0.99877286484398098, 0.047588269337724282 ],
# 			"zoom" : 0.33999999999999964
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }