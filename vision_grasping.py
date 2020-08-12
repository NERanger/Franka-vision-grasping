import argparse
import sys
import signal
import os
import time
import yaml

import jetson.inference
import jetson.utils

import numpy as np
import cv2 as cv
import gripper_control as gripper

from datetime import datetime
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
	arm = FrankaController(ROOT + '/config/franka.yaml')
	cam = realsense(frame_width = cfg['width'], frame_height = cfg['height'], fps = cfg['fps'])
	net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)  # load the object detection network

	time_evaluate = cfg['time_evaluate']
	initial_pose = cfg['initial_position']
	drop_position = cfg['drop_position']
	conf_threshold = cfg['conf_threshold']
	is_logging = cfg['log']

	R, t = load_cam_T_base_matrix(cfg['matrix_path'])
	print("Load R, t from file:\nR:\n", R, "\nt:\n", t)

	grasp_pre_offset = cfg['grasp_prepare_offset']
	effector_offset = cfg['effector_offset']

	detection_turncation = cfg['detection_turncation']

	grasp_width = cfg['grasp_width']

	display = jetson.utils.glDisplay()

	print("Moving to initial position...")
	arm.move_p(initial_pose)
	print("Moving to initial position... Done")

	if(is_logging):
		current_log_dir = ROOT + '/log/' + str(datetime.now()).replace(' ', '-')
		os.mkdir(current_log_dir)
		print("Set log dir to " + current_log_dir)

	while display.IsOpen():

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

		# print the detections
		print("detected {:d} objects in image".format(len(detections)))

		#for detection in detections:
			#print(detection)

		# render the image
		display.RenderOnce(network_input_img, cam.color_frame_width, cam.color_frame_height)

		# update the title bar
		display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

		# print out performance info
		if(time_evaluate):
			net.PrintProfilerTimes()

		if(len(detections) != 0):
			for detection in detections:
				if detection.Confidence > conf_threshold:
					print(detection)
					obj_center_row = int(detection.Center[1])
					obj_center_col = int(detection.Center[0])

					if(obj_center_col < detection_turncation):
						print("Discard detection result: in turncation area")
						continue

					if(is_logging):
						log_img = cv.cvtColor(color_img, cv.COLOR_RGB2BGR)
						cv.circle(log_img, (obj_center_col, obj_center_row), 20, (0, 0, 255), thickness = 3)
						cv.imwrite(current_log_dir + '/' + str(time.time()) + '.jpg', log_img)

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

		# pause for a bit
		# time.sleep(3)
