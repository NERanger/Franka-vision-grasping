import argparse
import sys
import os
import time
import yaml

import numpy as np
import cv2 as cv

import yolov5_module

from realsense_wapper import realsense

def read_cfg(path: str):
    with open(path, 'r') as stream:
        out = yaml.safe_load(stream)
    return out

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

def draw_bbox(res, img, input_w, input_h):
    for r in res:
        center_x = r.bbox[0]
        center_y = r.bbox[1]
        width = r.bbox[2]
        height = r.bbox[3]

        text = 'classid: %f, conf: %f' % (r.classid, r.conf)

        # bbox_start = (int(center_x - width / 2), int(center_y - height / 2))
        # bbox_end = (int(center_x + width / 2), int(center_y + height / 2))

        rect = get_rect(img, r.bbox, input_w, input_h)

        # Bounding box color in BGR
        bbox_color = (255, 0, 0)

        text_color = (255, 255, 255)

        thickness = 2

        cv.rectangle(img, rect, bbox_color, thickness)
        cv.putText(img, text, (rect[0], rect[1]), cv.FONT_HERSHEY_PLAIN, 1.2, text_color)

    return img

if __name__ == '__main__':

    cfg = read_cfg('config/test_yolov5.yaml')
    cam = realsense(frame_width = cfg['width'], frame_height = cfg['height'], fps = cfg['fps'])

    engine_path = cfg['engine_path']    # Path to yolov5 tensorrt engine file
    input_w = cfg['input_width']
    input_h = cfg['input_height']

    yolov5_module.init_inference(engine_path)

    stored_exception = None

    while True:
        try:
            if stored_exception:
                break

            _, color_img = cam.get_frame_cv()
            img_inf = color_img.copy()

            res = yolov5_module.image_inference(img_inf)

            img_visualize = draw_bbox(res, color_img, input_w, input_h)

            cv.imshow('result', img_visualize)
            cv.waitKey(10)

        except KeyboardInterrupt:
            stored_exception=sys.exc_info()

    cv.destroyAllWindows()
    yolov5_module.destory_inference()
        