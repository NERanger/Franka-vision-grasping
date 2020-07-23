import cv2 as cv
import numpy as np

def detect_ar_marker(color_frame, intr_matrix, dist_coeff):
    #Load the dictionary that was used to generate the markers.
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_7X7_1000)

    # Initialize the detector parameters using default values
    parameters =  cv.aruco.DetectorParameters_create()

    # Detect the markers in the image
    marker_corners, marker_ids, rejected_candidates = cv.aruco.detectMarkers(color_frame, dictionary, parameters=parameters, cameraMatrix = intr_matrix, distCoeff = dist_coeff)

    # target_corners = []
    # target_ids = []

    # Get desired marker with id
    # if marker_ids != None:
    #     for i, marker_id in enumerate(marker_ids):
    #         if marker_id == target_id:
    #             target_ids.append(marker_id)
    #             target_corners.append(target_corners[i])

    return marker_corners, marker_ids

def get_mat_cam_T_marker(color_frame, maker_size, intr_matrix, dist_coeff):
    corners, ids = detect_ar_marker(color_frame, intr_matrix, dist_coeff)

    if len(corners) > 0:
        R, t, _ = cv.aruco.estimatePoseSingleMarkers(corners, maker_size, intr_matrix, dist_coeff)

        visualize_img = cv.aruco.drawAxis(color_frame, intr_matrix, dist_coeff, R, t, 0.03)

        # convert from 3x1 rotation vector to 3x3 rotation matrix
        R, _ = cv.Rodrigues(R)
        # Squeeze t for stacking
        t = np.squeeze(t, 1).transpose()

        #padding = np.array([0, 0, 0, 1])

        # Stack arrays to get transformation matrix H
        #H = np.vstack((np.hstack((R, t)), padding))
        #print("Found marker, H = ", "\n", H)

        return R, t, visualize_img

    #print("No marker found in this frame !")
    return [], [], color_frame

if __name__ == '__main__':

    from realsense_wapper import realsense

    cam = realsense(frame_width = 1280, frame_height = 720, fps = 30)

    while(True):
        _, color_frame = cam.get_frame_cv()

        corners, ids = detect_ar_marker(color_frame, cam.get_intrinsics_matrix(), cam.get_distortion_coeffs())

        # print("Cornors:", corners)
        # print("IDs： ", ids)

        visualize_img = color_frame

        if(len(corners) > 0):
            R, t, _ = cv.aruco.estimatePoseSingleMarkers(corners, 0.05, cam.get_intrinsics_matrix(), cam.get_distortion_coeffs())

            R_mat, _ = cv.Rodrigues(R)
            #R_mat = np.squeeze(R_mat, 0)
            t = np.squeeze(t, 1).transpose()

            print("R: ", type(R_mat), "Shape: ", R_mat.shape, "\n", R_mat)
            print("t: ", type(t), "Shape: ", t.shape, "\n", t)

            padding = np.array([0, 0, 0, 1])
            tmp = np.hstack((R_mat, t))

            H = np.vstack((tmp, padding))

            print("H: ", type(H), "Shape: ", H.shape, "\n", H)

            visualize_img = cv.aruco.drawDetectedMarkers(color_frame, corners, ids)
            visualize_img = cv.aruco.drawAxis(visualize_img, cam.get_intrinsics_matrix(), cam.get_distortion_coeffs(), R, t, 0.03)

        cv.imshow("ar marker detection", visualize_img)
        cv.waitKey(1)
