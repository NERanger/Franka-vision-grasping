# Perform Eye-on-base calibration
import time
import sys
import os
import yaml
import cv2
import numpy as np

import ar_marker

from datetime import datetime
from realsense_wapper import realsense
from franka.FrankaController import FrankaController

def read_cali_cfg(path):
    with open(path, 'r') as stream:
        out = yaml.safe_load(stream)
    return out

def image_callback(color_image, depth_image, intrinsics, depth_scale):
    checkerboard_size = (4, 3)
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
                                         checkerboard_pix[0] - 20:checkerboard_pix[0] + 20])) * depth_scale
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

def check_trans_matrix_from_file(file_path):
    data = np.load(file_path)
    cam_pts = data['arr_0']
    arm_base_pts = data['arr_1']
    R, t = get_rigid_transform(cam_pts, arm_base_pts)
    # H = np.concatenate([np.concatenate([R,t.reshape([3,1])],axis=1),np.array([0, 0, 0, 1]).reshape(1,4)])

    x_errs = []
    y_errs = []
    z_errs = []

    for i, cam_pt in enumerate(cam_pts):
        print("Point in cam: ", i , " ", cam_pt)
        print("Point in base: ", i, " ", arm_base_pts[i])
        trans_pt = np.dot(R, cam_pt) + t
        print("Transed point: ", trans_pt)

        x_errs.append(abs(trans_pt[0] - arm_base_pts[i][0]))
        y_errs.append(abs(trans_pt[1] - arm_base_pts[i][1]))
        z_errs.append(abs(trans_pt[2] - arm_base_pts[i][2]))

    print("X-axis error: ", "max: ", max(x_errs), "min: ", min(x_errs), "mean: ", sum(x_errs)/len(x_errs))
    print("Y-axis error: ", "max: ", max(y_errs), "min: ", min(y_errs), "mean: ", sum(y_errs)/len(y_errs))
    print("Z-axis error: ", "max: ", max(z_errs), "min: ", min(z_errs), "mean: ", sum(z_errs)/len(z_errs))

def save_mat_to_file(cam_T_marker_mats, base_T_EE_mats, cam_T_marker_mats_filename, base_T_EE_mats_filename):
    with open(cam_T_marker_mats_filename, 'w') as out:
        for mat in cam_T_marker_mats:
            np.savetxt(out, mat, delimiter=',')

    with open(base_T_EE_mats_filename, 'w') as out:
        for mat in base_T_EE_mats:
            np.savetxt(out, mat, delimiter=',')




if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT)

    arm = FrankaController(ROOT + '/config/franka.yaml')
    cam = realsense(frame_width = 1280, frame_height = 720, fps = 30)
    cfg = read_cali_cfg("config/calibration.yaml")

    initial_pose = cfg['initial_position']
    second_pose = cfg['second_position']

    cube_size = cfg['sample_cube_size']

    maker_size = cfg['marker_size']

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

    #base_pts = []

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

                if(j == 0):
                    arm.move_p([arm_target_x, arm_target_y, arm_target_z, initial_pose[3], initial_pose[4], initial_pose[5]])
                else:
                    arm.move_p([arm_target_x, arm_target_y, arm_target_z, second_pose[3], second_pose[4], second_pose[5]])
                
                _, color_frame = cam.get_frame_cv()

                # cam_pt = image_callback(color_frame, depth_frame, cam.intrinsics, cam.depth_scale)
                #arm_base_pt = [arm_target_x + x_offset, arm_target_y + y_offset, arm_target_z + z_offset]
                cam_T_marker_R, cam_T_marker_t, visualize_img = ar_marker.get_mat_cam_T_marker(color_frame, maker_size, cam.get_intrinsics_matrix(), cam.get_distortion_coeffs())
                base_T_EE_R, base_T_EE_t = arm.getMatrixO_T_EE()

                print("End Effector in base frame:\n", "R:\n", base_T_EE_R, "t:\n", base_T_EE_t)

                cv2.imshow("visualize_img", visualize_img)
                cv2.waitKey(20)

                if len(cam_T_marker_R) != 0:
                    cam_T_marker_mats_R.append(cam_T_marker_R.transpose())
                    cam_T_marker_mats_t.append(-np.dot(cam_T_marker_R.transpose(), cam_T_marker_t))
                    EE_T_base_mats_R.append(base_T_EE_R.transpose())
                    EE_T_base_mats_t.append(-np.dot(base_T_EE_R.transpose(), base_T_EE_t))
                    print("Found marker in camera frame\n", "R:\n", cam_T_marker_R, "t:\n", cam_T_marker_t)
                else:
                    print("No marker detected in this frame!")

    cv2.destroyAllWindows()

    print("Performing calibration...")
    base_T_cam_R, base_T_cam_t= cv2.calibrateHandEye(EE_T_base_mats_R, EE_T_base_mats_t, cam_T_marker_mats_R, cam_T_marker_mats_t, method = cv2.CALIB_HAND_EYE_TSAI)
    print("Performing calibration... Done")

    print("From arm base to camera:\n", "R:\n", base_T_cam_R, "t:\n", base_T_cam_t)

    #cam_T_marker_mats_filename = ROOT + cfg['save_dir'] + "cam_T_marker" + str(datetime.now()).replace(' ', '-') + ".csv"
    #base_T_EE_mats_filename = ROOT + cfg['save_dir'] + "base_T_EE" + str(datetime.now()).replace(' ', '-') + ".csv"            
    
    # np.savez(filename, cam_pts, arm_base_pts)
    #save_mat_to_file(cam_T_marker_mats, base_T_EE_mats, cam_T_marker_mats_filename, base_T_EE_mats_filename)

    # #R, t = get_rigid_transform(cam_pts, arm_base_pts)
    # #H = np.concatenate([np.concatenate([R,t.reshape([3,1])],axis=1),np.array([0, 0, 0, 1]).reshape(1,4)])

    # H = load_cali_matrix(filename)

    # print("Transformation matrix from camera to arm base:")
    # #print(R, t)
    # print(H)

    # check_trans_matrix(filename)
