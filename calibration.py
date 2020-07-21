# Perform Eye-on-base calibration
import time
import sys
import os
import yaml
import cv2
import numpy as np

from datetime import datetime
from realsense_wapper import realsense
from franka.FrankaController import FrankaController

def read_cali_cfg(path):
    with open(path, 'r') as stream:
        out = yaml.safe_load(stream)
    return out

def image_callback(color_image, depth_image, intrinsics):
    checkerboard_size = (3, 3)
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    checkerboard_found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None,
                                                            cv2.CALIB_CB_ADAPTIVE_THRESH)
    if checkerboard_found:
        corners_refined = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), refine_criteria)

        # Get observed checkerboard center 3D point in camera space
        checkerboard_pix = np.round(corners_refined[4, 0, :]).astype(int)
        checkerboard_z = np.mean(np.mean(depth_image[checkerboard_pix[1] - 20:checkerboard_pix[1] + 20,
                                         checkerboard_pix[0] - 20:checkerboard_pix[0] + 20])) / 1000.0
        checkerboard_x = np.multiply(checkerboard_pix[0] - cx, checkerboard_z / fx)  # 1920, 1080
        checkerboard_y = np.multiply(checkerboard_pix[1] - cy, checkerboard_z / fy)  # 1920, 1080
        print("Found checkerboard, X,Y,Z = ", [checkerboard_x, checkerboard_y, checkerboard_z])
        if checkerboard_z > 0:
            # Save calibration point and observed checkerboard center
            observed_pt = np.array([checkerboard_x, checkerboard_y, checkerboard_z])
            return observed_pt
    return []

def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0]  # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:  # Special reflection case
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t

def load_cali_matrix(file_path):
    data = np.load(file_path)
    cam_pts = data['arr_0']
    arm_base_pts = data['arr_1']
    R, t = get_rigid_transform(cam_pts, arm_base_pts)
    H = np.concatenate([np.concatenate([R,t.reshape([3,1])],axis=1),np.array([0, 0, 0, 1]).reshape(1,4)])
    return H

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT)

    arm = FrankaController(ROOT + '/config/franka.yaml')
    cam = realsense()
    cfg = read_cali_cfg("config/calibration.yaml")

    initial_pose = cfg['initial_position']

    x_step = cfg['x_stride']
    y_step = cfg['y_stride']
    z_step = cfg['z_stride']

    x_offset = cfg['board_offset_x']
    y_offset = cfg['board_offset_y']
    z_offset = cfg['board_offset_z']

    arm.move_p(initial_pose)
    x = initial_pose[0]
    y = initial_pose[1]
    z = initial_pose[2]

    cam_pts = []
    arm_base_pts = []

    for i in range(4):
        for j in range(4):
            for k in range(4):
                arm_target_x = round(x + x_step * i, 3)
                arm_target_y = round(y + y_step * j, 3)
                arm_target_z = round(z + z_step * k, 3)

                arm.move_p([arm_target_x, arm_target_y, arm_target_z,
                            round(initial_pose[3], 3), round(initial_pose[4], 3), round(initial_pose[5], 3)])
                
                depth_frame, color_frame = cam.get_frame_cv()
                cam_pt = image_callback(color_frame, depth_frame, cam.get_color_intrinsics())
                arm_base_pt = [arm_target_x + x_offset, arm_target_y + y_offset, arm_target_z + z_offset]
                print("Point in base frame:")
                print(arm_base_pt)

                if len(cam_pt) != 0:
                    cam_pts.append(cam_pt)
                    arm_base_pts.append(arm_base_pt)

    filename = ROOT + cfg['save_dir'] + str(datetime.now().replace(' ', '-')) + ".npz"           
    np.savez(filename, cam_pts, arm_base_pts)

    R, t = get_rigid_transform(cam_pts, arm_base_pts)
    H = np.concatenate([np.concatenate([R,t.reshape([3,1])],axis=1),np.array([0, 0, 0, 1]).reshape(1,4)])

    print("Transformation matrix from arm base to camera:")
    print(H)
