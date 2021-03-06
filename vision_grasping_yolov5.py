import yaml
import os
import sys

import numpy as np
import cv2 as cv

import yolov5_module
import gripper_control as gripper

from realsense_wapper import realsense
from franka.FrankaController import FrankaController

def read_cfg(path):
    with open(path, 'r') as stream:
        out = yaml.safe_load(stream)
    return out

def load_cam_T_base_matrix(file_path):
	H =  np.loadtxt(file_path, delimiter = ',')
	
	cam_T_base_R = H[:3, :3]
	cam_T_base_t = H[:3, 3:].squeeze(1)

	return cam_T_base_R, cam_T_base_t

def get_rect(img, bbox, input_w, input_h):
    l = 0
    r = 0
    t = 0
    b = 0

    r_w = input_w / (img.shape[1] * 1.0)
    r_h = input_h / (img.shape[0] * 1.0)

    if(r_h > r_w):
        l = bbox[0] - bbox[2] / 2.0
        r = bbox[0] + bbox[2] / 2.0
        t = bbox[1] - bbox[3] / 2.0 - (input_h - r_w * img.shape[0]) / 2
        b = bbox[1] + bbox[3] / 2.0 - (input_h - r_w * img.shape[0]) / 2

        l = l / r_w
        r = r / r_w
        t = t / r_w
        b = b / r_w
    else:
        l = bbox[0] - bbox[2] / 2.0 - (input_w - r_h * img.shape[1]) / 2
        r = bbox[0] + bbox[2] / 2.0 - (input_w - r_h * img.shape[1]) / 2
        t = bbox[1] - bbox[3] / 2.0
        b = bbox[1] + bbox[3] / 2.0
        
        l = l / r_h
        r = r / r_h
        t = t / r_h
        b = b / r_h
    
    return (int(l), int(t), int(r-l), int(b-t))

def draw_bbox(res: "one result form returned result list", 
              rect: "rectangular extracted from get_rect()", 
              img: "color img"):

    center_x = res.bbox[0]
    center_y = res.bbox[1]
    width = res.bbox[2]
    height = res.bbox[3]

    text = 'classid: %d, conf: %f' % (int(r.classid), r.conf)

    # bbox_start = (int(center_x - width / 2), int(center_y - height / 2))
    # bbox_end = (int(center_x + width / 2), int(center_y + height / 2))

    # rect = get_rect(img, r.bbox, input_w, input_h)

    # Bounding box color in BGR
    bbox_color = (255, 0, 0)

    text_color = (255, 255, 255)

    thickness = 2

    cv.rectangle(img, rect, bbox_color, thickness)
    cv.putText(img, text, (rect[0], rect[1]), cv.FONT_HERSHEY_PLAIN, 1.2, text_color)

def print_detection_info(res):
    print("Detection number: ", res.size())
    print("----------------------------------")
    for r in res:
        print("classid: ", r.classid)


if __name__ == '__main__':

    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT)

    cfg = read_cfg(ROOT + '/config/grasping_yolov5.yaml')

    arm = FrankaController(ROOT + '/config/franka.yaml')
    cam = realsense(frame_width = cfg['width'], frame_height = cfg['height'], fps = cfg['fps'])

    # yolov5 config
    engine_path = cfg['engine_path'] 
    input_w = cfg['input_width']
    input_h = cfg['input_height']

    # grasping config
    initial_pose = cfg['initial_position']
    drop_position = cfg['drop_position']
    conf_threshold = cfg['conf_threshold']

    grasp_pre_offset = cfg['grasp_prepare_offset']
    effector_offset = cfg['effector_offset']

    detection_turncation = cfg['detection_turncation']

    # Load calibration matrix
    R, t = load_cam_T_base_matrix(cfg['matrix_path'])
    print("Load R, t from file:\nR:\n", R, "\nt:\n", t)

    print("Moving to initial position...")
    arm.move_p(initial_pose)
    print("Moving to initial position... Done")

    print("Initializing yolov5 engine...")
    yolov5_module.init_inference(engine_path)
    print("Initializing yolov5 engine... Done")

    stored_exception = None

    while True:
        try:
            if stored_exception:
                break

            depth_img, color_img = cam.get_frame_cv()
            img_inf = color_img.copy()

            res = yolov5_module.image_inference(img_inf)

            for r in res:
                rect = get_rect(color_img, r.bbox, input_w, input_h)
                draw_bbox(r, rect, color_img)

                # Visualization
                cv.imshow('result', color_img)
                cv.waitKey(5)

                if int(r.classid) == 39 and r.conf > conf_threshold:
                    obj_center_row = int(rect[1] + rect[3] / 2)
                    obj_center_col = int(rect[0] + rect[2] / 2)
                    #print("row: {}, col: {}".format(obj_center_row, obj_center_col))

                    if(obj_center_col < detection_turncation):
                        print("Discard detection result: in turncation area")
                        continue

                    # compute target coordinate in camera frame
                    target_in_cam_z = depth_img[obj_center_row, obj_center_col] * cam.depth_scale
                    target_in_cam_x = np.multiply(obj_center_col - cam.intrinsics['cx'], target_in_cam_z / cam.intrinsics['fx'])
                    target_in_cam_y = np.multiply(obj_center_row - cam.intrinsics['cy'], target_in_cam_z / cam.intrinsics['fy'])

                    print("Target in camera frame:\n", [target_in_cam_x, target_in_cam_y, target_in_cam_z])

                    target_in_cam = np.array([target_in_cam_x, target_in_cam_y, target_in_cam_z])
                    target_in_base = R.dot(target_in_cam) + t

                    print("Target in base frame:\n", target_in_base)

                    prepare_pos = [target_in_base[0], target_in_base[1], target_in_base[2] + grasp_pre_offset + effector_offset, 3.14, 0, 0]
                    arm.move_p(prepare_pos)

                    gripper.gripper_open()
                    arm.move_p([target_in_base[0], target_in_base[1], target_in_base[2] + effector_offset, 3.14, 0, 0])
                    gripper.gripper_close()
                    
                    # Move to drop position and drop object
                    arm.move_p(initial_pose)
                    arm.move_p(drop_position)
                    gripper.gripper_open()

                    # Back to initial position
                    arm.move_p(initial_pose)

        except KeyboardInterrupt:
            stored_exception = sys.exc_info()

    cv.destroyAllWindows()
    yolov5_module.destory_inference()
