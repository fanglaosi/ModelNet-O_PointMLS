import open3d as o3d
from numpy.linalg import inv
import scipy.io as sio
import numpy as np
import os
import argparse
import time

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--point-output-dir', type=str, default="../ModelNet40_singleview_2048/", help="generated_single_view_pcd")
    parser.add_argument('--depth-output-dir', type=str, default="../ModelNet40_singleview_depth_2048/", help="generated_single_view_depth")
    parser.add_argument('--input-dir', type=str, default="../ModelNet40_off_norm/", help="generated_single_view_pcd")
    parser.add_argument('--number-points', type=int, default=2048, help="generated_single_view_pcd")
    args = parser.parse_args()

    classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
               'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
               'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
               'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
               'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    modes = ["/train/", "/test/"]

    # 导入相机投影矩阵，包含20个视角
    camera_pose = "camera_position.mat"
    posdata = sio.loadmat(camera_pose)
    poset = np.array(posdata['transformCell'])

    failed_list = list()
    amount_processed = np.zeros((2, 40))
    amount_ground = np.zeros((2, 40))

    start_time = time.time()
    for i, mode in enumerate(modes):
        start_time1 = time.time()
        for j, Class in enumerate(classes):
            print("------------------------>Processing {}<----------------------".format(Class + mode))
            point_input_dir = args.input_dir + Class + mode # "../ModelNet40_off_norm/train/airplane_0001.off"
            point_output_dir = args.point_output_dir + Class + mode # "../ModelNet40_singleview/train/"
            depth_output_dir = args.depth_output_dir + Class + mode # "../ModelNet40_singleview_depth/train/"
            file_name_list = os.listdir(point_input_dir) # ["airplane_0001.off", "airplane_0002.off", ...]

            num = 0

            if not os.path.exists(point_output_dir):
                os.makedirs(point_output_dir)
            if not os.path.exists(depth_output_dir):
                os.makedirs(depth_output_dir)

            start_time2 = time.time()
            for k, file_name in enumerate(file_name_list):
                input_path = point_input_dir + file_name

                pcd = o3d.io.read_triangle_mesh(input_path)
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(pcd)
                cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
                start_time3 = time.time()
                for n in range(poset.shape[0]):

                    depth_output_path = os.path.join(depth_output_dir, '{}_{:03d}_depth.png'.format(file_name[:-4], n+1))
                    point_output_path = os.path.join(point_output_dir, '{}_{:03d}.xyz'.format(file_name[:-4], n+1))

                    try:
                        number1 = poset[n][0]
                        number = inv(number1)
                        cam.extrinsic = number
                        vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                        vis.poll_events()
                        vis.update_renderer()

                        depth = vis.capture_depth_float_buffer(False)
                        image = vis.capture_screen_float_buffer(False)

                        # capture and save depth map
                        depth_save = vis.capture_depth_image(depth_output_path)
                        depth_raw = o3d.io.read_image(depth_output_path)

                        # generate point cloud from depth map
                        pc = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, cam.intrinsic, cam.extrinsic)
                        write_raw_successfully = o3d.io.write_point_cloud(point_output_path, pc)


                        # sample 2048 points
                        read_mesh = o3d.io.read_point_cloud(point_output_path)
                        pcd_points = np.asarray(read_mesh.points)
                        point_set = farthest_point_sample(pcd_points, args.number_points)
                        read_mesh.points = o3d.utility.Vector3dVector(point_set)
                        write_successfully = o3d.io.write_point_cloud(point_output_path, read_mesh)
                    except:
                        print("Error!!!!Error!!!!Error!!!!Error!!!!:" + point_output_path)
                        failed_list.append(point_output_path)

                    num += 1
                end_time3 = time.time()
                total_time3 = end_time3 - start_time3
                print(Class + ":{}/{}, cost_time:{:.2f}s".format(k+1, len(file_name_list), total_time3))
            end_time2 = time.time()
            total_time2 = end_time2 - start_time2
            print(Class + ", cost_time:{:.2f}s".format(total_time2))

            amount_processed[i][j] = num
            amount_ground[i][j] = len(file_name_list) * 20

            np.save('amount_processed_singleview.npy', amount_processed)
            print(amount_processed)
            np.save('amount_ground_singleview.npy', amount_ground)
            print(amount_ground)

        end_time1 = time.time()
        total_time1 = end_time1 - start_time1
        print(mode + ", cost_time:{:.2f}s".format(total_time1))

    end_time = time.time()
    total_time = end_time - start_time
    print("Total cost_time:{:.2f}s".format(total_time))

    with open('failed_list.txt', 'w', encoding='utf-8') as f:
        for failed in failed_list:
            f.write(failed + '\n')
