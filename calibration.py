# Perform Eye-on-base calibration
import time
import sys
import os
import yaml
import cv2
import numpy as np
from numpy.linalg import inv

import ar_marker

from datetime import datetime
from realsense_wapper import realsense
from franka.FrankaController import FrankaController

def read_cali_cfg(path):
    with open(path, 'r') as stream:
        out = yaml.safe_load(stream)
    return out

def load_cali_matrix(file_path):
    data = np.load(file_path)

    cam_T_marker_mats_R = data['arr_0']
    cam_T_marker_mats_t = data['arr_1']

    EE_T_base_mats_R = data['arr_2']
    EE_T_base_mats_t = data['arr_3']

    base_T_cam_R, base_T_cam_t = cv2.calibrateHandEye(EE_T_base_mats_R, EE_T_base_mats_t, cam_T_marker_mats_R, cam_T_marker_mats_t)
    return base_T_cam_R, base_T_cam_t

def random_sample(lower_limit, upper_limit, shape):
    size = upper_limit - lower_limit
    return size * np.random.random_sample(shape) + lower_limit

def get_H_from_R_t(R, t):
    padding = np.array([0, 0, 0, 1])

    # Stack arrays to get transformation matrix H
    H = np.vstack((np.hstack((R, t)), padding))

    return H

def check_trans_matrix_from_file(file_path):

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt

    data = np.load(file_path)
    cam_pts = data['arr_0']
    arm_base_pts = data['arr_1']
    R, t = get_rigid_transform(cam_pts, arm_base_pts)
    # H = np.concatenate([np.concatenate([R,t.reshape([3,1])],axis=1),np.array([0, 0, 0, 1]).reshape(1,4)])

    x_errs = []
    y_errs = []
    z_errs = []
    abs_errs = []

    trans_pts = []

    for i, cam_pt in enumerate(cam_pts):
        print("Point in cam: ", i , " ", cam_pt)
        print("Point in base: ", i, " ", arm_base_pts[i])
        trans_pt = np.dot(R, cam_pt) + t
        trans_pts.append(trans_pt)
        print("Transed point: ", trans_pt)

        x_errs.append(abs(trans_pt[0] - arm_base_pts[i][0]))
        y_errs.append(abs(trans_pt[1] - arm_base_pts[i][1]))
        z_errs.append(abs(trans_pt[2] - arm_base_pts[i][2]))
        abs_errs.append(np.sqrt(np.square(trans_pt[0] - arm_base_pts[i][0]) + np.square(trans_pt[1] - arm_base_pts[i][1]) + np.square(trans_pt[2] - arm_base_pts[i][2])))

    print("X-axis error: ", "max: ", max(x_errs), "min: ", min(x_errs), "mean: ", sum(x_errs)/len(x_errs))
    print("Y-axis error: ", "max: ", max(y_errs), "min: ", min(y_errs), "mean: ", sum(y_errs)/len(y_errs))
    print("Z-axis error: ", "max: ", max(z_errs), "min: ", min(z_errs), "mean: ", sum(z_errs)/len(z_errs))
    print("Abs error: ", "max: ", max(abs_errs), "min: ", min(abs_errs), "mean: ", sum(abs_errs)/len(abs_errs))

    abs_errs = np.sort(abs_errs)
    plt.figure(1)
    plt.plot(range(len(abs_errs)), abs_errs)

    fig_3d = plt.figure(2)
    ax = fig_3d.add_subplot(111, projection='3d')

    for pt in arm_base_pts:
        ax.scatter(pt[0], pt[1], pt[2], marker='^')

    for pt in trans_pts:
        ax.scatter(pt[0], pt[1], pt[2], marker='.')

    plt.show()

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT)

    arm = FrankaController(ROOT + '/config/franka.yaml')
    cam = realsense(frame_width = 1280, frame_height = 720, fps = 30)
    cfg = read_cali_cfg("config/calibration.yaml")

    initial_pose = cfg['initial_position']

    cube_size = cfg['sample_cube_size']

    maker_size = cfg['marker_size']

    x_step = cfg['x_stride']
    y_step = cfg['y_stride']
    z_step = cfg['z_stride']

    rotation_upper_limit = cfg['rotation_upper_limit']
    rotation_lower_limit = cfg['rotation_lower_limit']

    x_offset = cfg['board_offset_x']
    y_offset = cfg['board_offset_y']
    z_offset = cfg['board_offset_z']

    arm.move_p(initial_pose)
    x = initial_pose[0]
    y = initial_pose[1]
    z = initial_pose[2]
    roll = initial_pose[3]   # Rotation around x-axis
    pitch = initial_pose[4]  # Rotation around y-axis
    yaw = initial_pose[5]    # Rotation around z-axis

    # cam_pts = []
    # arm_base_pts = []

    cam_T_marker_mats_R = []
    cam_T_marker_mats_t = []

    EE_T_base_mats_R = []
    EE_T_base_mats_t = []

    print("Waiting for camera steady auto exposure...")
    for i in range(20):
        cam.get_frame_cv()
    print("Waiting for camera steady auto exposure... Done")

    cv2.namedWindow('visualize_img', cv2.WINDOW_AUTOSIZE)

    for i in range(cube_size):
        for j in range(cube_size):
            for k in range(cube_size):
                arm_target_x = round(x + x_step * i, 5)
                arm_target_y = round(y + y_step * j, 5)
                arm_target_z = round(z + z_step * k, 5)

                rotation_sample = random_sample(rotation_lower_limit, rotation_upper_limit, 3)
                print("random sample: ", rotation_sample)

                arm_target_roll = round(roll + rotation_sample[0], 5)
                arm_target_pitch = round(pitch + rotation_sample[1], 5)
                arm_target_yaw = round(yaw + rotation_sample[2], 5)
                print("move target: ", [arm_target_x, arm_target_y, arm_target_z, arm_target_roll, arm_target_pitch, arm_target_yaw])

                arm.move_p([arm_target_x, arm_target_y, arm_target_z, arm_target_roll, arm_target_pitch, arm_target_yaw])
                
                _, color_frame = cam.get_frame_cv()

                cam_T_marker_R, cam_T_marker_t, visualize_img = ar_marker.get_mat_cam_T_marker(color_frame, maker_size, cam.get_intrinsics_matrix(), cam.get_distortion_coeffs())
                base_T_EE_R, base_T_EE_t = arm.getMatrixO_T_EE()

                # print("End Effector in base frame:\n", "R:\n", base_T_EE_R, "t:\n", base_T_EE_t)
                # print("End Effector in base frame:\n", arm_base_pt)

                cv2.imshow("visualize_img", visualize_img)
                cv2.waitKey(5)

                if len(cam_T_marker_R) != 0:

                    EE_T_base_R = inv(base_T_EE_R)
                    EE_T_base_t = -inv(base_T_EE_R).dot(base_T_EE_t)
                    
                    cam_T_marker_mats_R.append(cam_T_marker_R)
                    cam_T_marker_mats_t.append(cam_T_marker_t)

                    EE_T_base_mats_R.append(EE_T_base_R)
                    EE_T_base_mats_t.append(EE_T_base_t)

                    print("Marker in camera frame\n", "R:\n", cam_T_marker_R.shape, cam_T_marker_R, "t:\n", cam_T_marker_t.shape, cam_T_marker_t)

                    # print("EE in base frame\n", "R:\n", base_T_EE_R.shape, base_T_EE_R, "t:\n", base_T_EE_t.shape, base_T_EE_t)
                    print("Base in EE frame\n", "R:\n", EE_T_base_R.shape, EE_T_base_R, "t:\n", EE_T_base_t.shape, EE_T_base_t)

                    # cam_pts.append(cam_pt)
                    # arm_base_pts.append(arm_base_pt)
                else:
                    print("No marker detected in this frame!")

    cv2.destroyAllWindows()

    print("Performing calibration...")
    base_T_cam_R, base_T_cam_t = cv2.calibrateHandEye(EE_T_base_mats_R, EE_T_base_mats_t, cam_T_marker_mats_R, cam_T_marker_mats_t)
    print("Performing calibration... Done")

    print("From base to camera:\n", "R:\n", base_T_cam_R, "\nt:\n", base_T_cam_t)

    # Stack arrays to get transformation matrix H
    base_T_cam_H = get_H_from_R_t(base_T_cam_R, base_T_cam_t)
    print("H:\n", base_T_cam_H)
    # cam_T_marker_mats_filename = ROOT + cfg['save_dir'] + "cam_T_marker" + str(datetime.now()).replace(' ', '-') + ".csv"
    filename_csv = ROOT + cfg['save_dir'] + str(datetime.now()).replace(' ', '-') + ".csv"   
    filename_npz = ROOT + cfg['save_dir'] + str(datetime.now()).replace(' ', '-') + ".npz"
    
    np.savez(filename_npz, cam_T_marker_mats_R, cam_T_marker_mats_t, EE_T_base_mats_R, EE_T_base_mats_t)
    np.savetxt(filename_csv, base_T_cam_H, delimiter=',')

    print("------ Test load ------")
    load_R, load_t = load_cali_matrix(filename_npz)
    print("Loaded R:\n", load_R)
    print("Loaded t:\n", load_t)
    print("-------- End ----------")
