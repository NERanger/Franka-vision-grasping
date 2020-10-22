import yaml
import os
import sys
import time
import csv
from datetime import datetime

import numpy as np
import cv2 as cv

import gripper_control as gripper

from realsense_wapper import realsense
from franka.FrankaController import FrankaController
from get_obj_by_color import get_obj_bbox, check_gripper_bbox

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

    cfg = read_cfg(ROOT + '/config/grasping _colorseg.yaml')

    arm = FrankaController(ROOT + '/config/franka.yaml')
    cam = realsense(frame_width = cfg['width'], frame_height = cfg['height'], fps = cfg['fps'])

    # grasping config
    initial_pose = cfg['initial_position']
    check_position = cfg['check_position']
    drop_position = cfg['drop_position']

    grasp_pre_offset = cfg['grasp_prepare_offset']
    effector_offset = cfg['effector_offset']

    #check_threshold = cfg['check_threshold']

    attmp_num = cfg['attmp_num']

    # Load calibration matrix
    R, t = load_cam_T_base_matrix(cfg['matrix_path'])
    print("Load R, t from file:\nR:\n", R, "\nt:\n", t)

    print("Moving to initial position...")
    arm.move_p(initial_pose)
    print("Moving to initial position... Done")

    stored_exception = None

    arm.move_p(check_position) # Test
    while True:
        _, color_img = cam.get_frame_cv()
        bbox = check_gripper_bbox(color_img)
        print("Area: {}".format(bbox[2]*bbox[3]))
        cv.rectangle(color_img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)
        cv.imshow('result', color_img)
        cv.waitKey(10)
