import open3d as o3d
import numpy as np
import sys
import os
sys.path.append('..')
import math
import copy
'''
将原始数据转为可训练的数据集
1. 体素降采样，百万个点->3万个点左右
2. 调整xyz，调整偏置角度，把轨道对准列车正前方
3. 标label
'''
def from_txt2numpy_downsample(file_path, voxel_size, save_flag=False):
    '''
    1. 处理仿真环境录制的数据，降采样-另存为npy，
    '''
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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vec)

    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(2, center=mesh.get_center())
    o3d.visualization.draw_geometries([pcd,mesh])


    downvec = np.asarray(downpcd.points)
    print("after downsample point cloud size: ", downvec.shape)
    if save_flag:
        np.savetxt(file_path,downvec,fmt='%f',delimiter=' ')

def adjust_xyz_bias(file_path,rotate_angle,save_flag=False):
    '''
    2. 调整xyz，调整偏置角度，把轨道对准列车正前方
    :param file_path: .txt文件路径
    :param rotate_angle:  设置偏置角
    :param save_flag:
    :return:
    '''
    # ra_angle_bias_matrix = [2.4943680, 2.49436800, 2.110000, 2.041080, 2.041080, 1.98798, 1.98798, 1.900721, 1.9131385,
    #                         1.865817,
    #                         2.4240655, 2.38940478, 2.337538, 2.336803, 2.30239638, 2.301901843, 2.23233302, 2.232568,
    #                         2.179728, 2.17996554,
    #                         2.14505854, ]
    # la_angle_bias_matrix = [-0.21138631173489697, -0.5429909340274038, -0.5686781883627849, -0.5866248295598208,
    #                         -0.6128044548499887, -0.6472098181822128, -0.7167797316491197, -0.7519308179431905,
    #                         -0.7698794304081646, -0.8392074605631741,
    #                         -0.9087662534392228, -0.2634972536528782, -0.9613770487817996, -0.961375072317369,
    #                         -0.9960391363992125, -0.996279021378593, -1.0309415768417154, -1.101257, -1.153357,
    #                         -1.1531126,
    #                         -1.240874, -1.240627,
    #                         -0.2979058837206351, -0.3240901067438193, -0.3592395003585013, -0.35924482200743224,
    #                         -0.38541791449189855, -0.4552315117115814, -0.4642035968276257, -0.49012726942948337,
    #                         -0.507828960096424, 2.371457]
    data_list = []
    file = open(file_path)
    for line in file.readlines():
        if(len(line) > 1):
            data = line[:-1].split(' ') #[:-1] 去掉换行
            data_list.append(data)
    # vec = np.genfromtxt(file_path, delimiter=' ', dtype=np.float32)
    file.close()
    vec = np.asarray(data_list, dtype=np.float32)
    vec[:, (1, 2)] *= -1
    print("origin point cloud size: ",vec.shape)

    theta = rotate_angle
    rotate_matrix = np.asarray([[math.cos(theta), -math.sin(theta), 0],
                                [math.sin(theta), math.cos(theta), 0],
                                [0, 0, 1]])
    rotated_vec = np.transpose(np.matmul(rotate_matrix, (np.transpose(vec))))
    vec = np.transpose(np.matmul(rotate_matrix, (np.transpose(vec))))

    # 获得偏置角的最大和最小值
    beta_r2 = (vec[:,0]**2 + vec[:,1]**2)**0.5 #计算半径
    beta_idx = (beta_r2 > 10) & (beta_r2 < 100)
    betas = np.arctan(vec[beta_idx,1]/vec[beta_idx,0]) #半径内的角度
    bias_angle = - (np.max(betas) + np.min(betas))/2
    print("betas: ",np.max(betas),np.min(betas),bias_angle)

    # 可视化：
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vec[beta_idx])
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(2, center=mesh.get_center())
    o3d.visualization.draw_geometries([pcd, mesh])

    # 另存为npy文件
    if save_flag:
        print(file_path[:-4])
        np.save(file_path[:-4],rotated_vec)
    return bias_angle

def label2color(labels):
    """
    将label 转为 open3d colors 的矩阵
    :param labels: shape=[n,]
    :return: clors: shape=[n,3] 值的范围[0,1]
    """
    list = np.asarray([
        # [127,127,127],
        # [255,0,0],
        # [0,0,255],
        # [127, 0, 127],
        # [127, 127, 0]
        [127, 127, 127], [0, 255, 0], [255, 0, 255],
        # [127,127,127],
        [128, 0, 128], [255, 0, 255],

    ],dtype=np.float)
    return list[labels]/255

def label_points(points,parameter,range):
    '''
    根据给定的点云，轨道的参数，输出label矩阵和轨道点的索引
    :param points:
    :param parameter:
    :param range:
    :return:
    '''
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

def label_unreal_rail(file_path,parameter,range):
    npy_vec = np.load(file_path)
    print("npy_vec.shape: ", npy_vec.shape)
    label,idx = label_points(npy_vec,parameter,range) #获得label矩阵
    new_points = np.concatenate((npy_vec[:,:3], label[:, np.newaxis]), axis=1)
    print("new_points.shape: ",new_points.shape)

    #可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(new_points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(label2color(new_points[:, -1].astype(np.int)))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(3, center=mesh.get_center())

    #拟合曲线
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    rail_points = new_points[idx]
    x_poly = rail_points[:, 0]
    x_poly = x_poly[:, np.newaxis]
    y_poly = rail_points[:, 1]
    y_poly = y_poly[:, np.newaxis]

    # degree=2 y =         c*x^2 + b*x + a
    # degree=3 y = d*x^3 + c*x^2 + b*x + a
    poly_features = PolynomialFeatures(degree=1, include_bias=False)
    X_poly = poly_features.fit_transform(x_poly)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y_poly)
    poly_para = [lin_reg.intercept_, lin_reg.coef_]
    print('poly_para : ', poly_para)

    show_flag = True
    save_flag = True

    if show_flag:
        o3d.visualization.draw_geometries([pcd, mesh], window_name=file_path,
                                          zoom=0.3,
                                          front=[ 0.096314076554831055, 0.020511804979741765, 0.99513962061303918 ],
                                          lookat=[ 53.229220539861529, -2.0228925385089158, 5.6663705489609022 ],
                                          up=[ -0.0034982659029790973, 0.99978844152479429, -0.020269048549330322 ],

                                          )
    if save_flag:
        np.save(file_path, new_points)

def label_unreal_obstacle(file_path,obs_range):
    npy_vec = np.load(file_path)
    print("npy_vec.shape: ", npy_vec.shape)
    points,label = npy_vec[:,:3],npy_vec[:,3] #获得label矩阵

    o3dshow = []
    for i, ranges in enumerate(obs_range):

        idx_x = (points[:, 0] > ranges[0]) & (points[:, 0] < ranges[3])
        idx_y = (points[:, 1] > ranges[1]) & (points[:, 1] < ranges[4])
        idx_z = (points[:, 2] > ranges[2]) & (points[:, 2] < ranges[5])
        idx = idx_x & idx_y & idx_z
        label[idx] = 3 + i ## obstale_0 =>3, obstale_1 =>4 ...

        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(points[idx])
        if len(points[idx]) > 0:
            print("obstacle: ", i," ",points[idx].shape, " min&max:", np.min(points[idx], axis=0),
                  " ", np.max(points[idx], axis=0))
        else:
            print("error range !.................")
            exit()
        aabb = pcd_part.get_axis_aligned_bounding_box()
        o3dshow.append(aabb)

    new_points = np.concatenate((points[:,:3], label[:, np.newaxis]), axis=1)

    #可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(label2color(label.astype(np.int)))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(3, center=mesh.get_center())
    o3dshow.append(pcd)
    o3dshow.append(mesh)
    show_flag = True
    save_flag = True
    if show_flag:
        o3d.visualization.draw_geometries(o3dshow, window_name=file_path,
                                          zoom=0.3,
                                          front=[ 0.096314076554831055, 0.020511804979741765, 0.99513962061303918 ],
                                          lookat=[ 53.229220539861529, -2.0228925385089158, 5.6663705489609022 ],
                                          up=[ -0.0034982659029790973, 0.99978844152479429, -0.020269048549330322 ],
                                          )
    if save_flag:
        np.save(file_path, new_points)

def check_label(file_path):
    npy_vec = np.load(file_path)
    print("npy_vec.shape: ", npy_vec.shape)
    points,label = npy_vec[:,:3],npy_vec[:,-1] #获得label矩阵
    #可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(label2color(label.astype(np.int)))
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(3, center=mesh.get_center())

    o3d.visualization.draw_geometries([pcd, mesh], window_name=file_path,
                                      zoom=0.3,
                                      front=[0.096314076554831055, 0.020511804979741765, 0.99513962061303918],
                                      lookat=[53.229220539861529, -2.0228925385089158, 5.6663705489609022],
                                      up=[-0.0034982659029790973, 0.99978844152479429, -0.020269048549330322],
                                          )
if __name__ == '__main__':
    '''
    处理仿真环境录制的数据，降采样-另存为npy，
    '''
    file_root = '../dataunreal/npyfile2'
    file_names = sorted(os.listdir(file_root))
    # 先按文件名排序
    # la_file_names = sorted(os.listdir(file_root)[:31], key=lambda s: (int(s[2:3]) if s[3] is '.' or s[3] is '-' else int(s[2:4])))
    # ra_file_names = sorted(os.listdir(file_root)[31:53], key=lambda s: (int(s[2:3]) if s[3] is '.' or s[3] is '-' else int(s[2:4])))
    # sa_file_names = sorted(os.listdir(file_root)[53:], key=lambda s: (int(s[2:3]) if s[3] is '.' or s[3] is '-' else int(s[2:4])))

    for i, file_name in enumerate(file_names) :
        if i>=0 and len(file_name)>6:
            print('i =', i , "  ", file_name)
            file_path = os.path.join(file_root,file_name)
            #1. 降采样
            # from_txt2numpy_downsample(file_path,0.03,False)
            #2. rotate
            # totate_angle = adjust_xyz_bias(file_path,la_angle_bias_matrix[i],True)
            #3. label
            # label params
            # rail_ranges = [6, 100, -0.4, 1.2,  -2.86, -2.2]  # 前后，左右，上下  #[-2.5,-2.73] 轨道位置
            # # rail_ranges = [6, 100, -.4, 2,  -2.5, -2.2]  # 前后，左右，上下  #[-2.5,-2.73] 轨道位置
            # parameter_lr = np.asarray([ 0.00415167, 0]).astype(np.float32)  # ax^2 + bx
            # label_unreal_rail(file_path,parameter_lr,rail_ranges)

            #4. label obstacles
            # min_xyz, max_xyz
            obstacle_ranges = [[32,-2.7,-2.5,34,-0.3,-0.7],
                               [52,1.0,-2.6,55,2.2,1.0]]
            # label_unreal_obstacle(file_path,obstacle_ranges)


            #end. check label
            check_label(file_path)