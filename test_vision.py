import argparse
import sys
import os
import time
import yaml

import jetson.inference
import jetson.utils

import numpy as np
import cv2 as cv

from datetime import datetime
from realsense_wapper import realsense

def read_cfg(path):
    with open(path, 'r') as stream:
        out = yaml.safe_load(stream)
    return out

if __name__ == '__main__':

	# parse the command line
	parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

	parser.add_argument("--network", type=str, default="coco-bottle", help="pre-trained model to load (see below for options)")
	parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
	parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

	is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

	try:
		opt = parser.parse_known_args()[0]
	except:
		print("")
		parser.print_help()
		sys.exit(0)

	ROOT = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(ROOT)

	cfg = read_cfg('config/grasping.yaml')
	cam = realsense(frame_width = cfg['width'], frame_height = cfg['height'], fps = cfg['fps'])
	net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)  # load the object detection network

	time_evaluate = cfg['time_evaluate']
	conf_threshold = cfg['conf_threshold']
	is_logging = cfg['log']

	detection_turncation = cfg['detection_turncation']

	if(is_logging):
		current_log_dir = ROOT + '/log/' + str(datetime.now()).replace(' ', '-')
		os.mkdir(current_log_dir)
		print("Set log dir to " + current_log_dir)

	while(True):

		if(time_evaluate):
			t0 = time.time()

		# Get img from realsense in Numpy array format
		depth_img, color_img = cam.get_frame_cv()

		# Numpy array can only be accessed by cpu
		# Copy color img to GPU for network inference
		color_img_cuda = jetson.utils.cudaFromNumpy(color_img)

		# allocate gpu memory for network input image as rgba32f, with the same width/height as the color frame
		network_input_img = jetson.utils.cudaAllocMapped(width = cam.color_frame_width, height = cam.color_frame_height, format='rgba32f')

		# convert from rgb8 (default format for realsense color frame in this program) to rgba32f
		jetson.utils.cudaConvertColor(color_img_cuda, network_input_img)

		if(time_evaluate):
			print("Time to convert from numpy array to cuda: ", time.time() - t0)

		# detect objects in the image (with overlay)
		detections = net.Detect(network_input_img, cam.color_frame_width, cam.color_frame_height, opt.overlay)

		visual_img = cv.cvtColor(jetson.utils.cudaToNumpy(network_input_img), cv.COLOR_RGBA2BGR)
		visual_img = visual_img.astype(np.uint8)

		print(visual_img)

		cv.imshow("Result", visual_img)
		cv.waitKey(100)

		# print the detections
		print("detected {:d} objects in image".format(len(detections)))

		for detection in detections:
			print(detection)

		# print out performance info
		if(time_evaluate):
			net.PrintProfilerTimes()