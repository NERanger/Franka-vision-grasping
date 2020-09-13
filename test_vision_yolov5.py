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

def draw_bbox(res, img):
    for r in res:
        center_x = r.bbox[0]
        center_y = r.bbox[1]
        width = r.bbox[2]
        height = r.bbox[3]

        text = 'classid: %f, conf: %f' % (r.classid, r.conf)

        bbox_start = (int(center_x - width / 2), int(center_y - height / 2))
        bbox_end = (int(center_x + width / 2), int(center_y + height / 2))

        # Bounding box color in BGR
        bbox_color = (255, 0, 0)

        text_color = (255, 255, 255)

        thickness = 2

        cv.rectangle(img, bbox_start, bbox_end, bbox_color, thickness)
        cv.putText(img, text, bbox_start, cv.FONT_HERSHEY_PLAIN, 1.2, text_color)

    return img

if __name__ == '__main__':

    cfg = read_cfg('config/test_yolov5.yaml')
    cam = realsense(frame_width = cfg['width'], frame_height = cfg['height'], fps = cfg['fps'])

    engine_path = cfg['engine_path']    # Path to yolov5 tensorrt engine file

    yolov5_module.init_inference(engine_path)

    stored_exception = None

    while True:
        try:
            if stored_exception:
                break

            _, color_img = cam.get_frame_cv()
            img_inf = color_img.copy()

            res = yolov5_module.image_inference(img_inf)

            img_visualize = draw_bbox(res, color_img)

            cv.imshow('result', img_visualize)
            cv.waitKey(10)

        except KeyboardInterrupt:
            stored_exception=sys.exc_info()

    cv.destroyAllWindows()
    yolov5_module.destory_inference()
        