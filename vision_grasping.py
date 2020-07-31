import argparse
import sys
import os
import yaml

import jetson.inference
import jetson.utils

import numpy as np

from realsense_wapper import realsense

from franka.FrankaController import FrankaController

def read_cfg(path):
    with open(path, 'r') as stream:
        out = yaml.safe_load(stream)
    return out

def load_base_T_cam_matrix(file_path):
	H =  np.loadtxt(file_path, delimiter = ',')
	
	base_T_cam_R = H[:3, :3]
	base_T_cam_t = H[:3, 3:].squeeze(1)

	cam_T_base_R = base_T_cam_R.transpose()
	cam_T_base_t = -cam_T_base_R.dot(base_T_cam_t)

	return R, t

if __name__ == '__main__':

	ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT)

    arm = FrankaController(ROOT + '/config/franka.yaml')
	
	opt = read_cfg(config/grasping.yaml)

	R, t = load_base_T_cam_matrix(opt['matrix_path'])

	print("Load R, t from file:\nR:\n", R, "\nt:\n", t)

	cam = realsense(frame_width = opt['width'], frame_height = opt['height'], fps = opt['fps'])

	# load the object detection network
	net = jetson.inference.detectNet(opt['network'], sys.argv, opt['threshold'])

	display = jetson.utils.glDisplay()

	# process frames until user exits
	while display.IsOpen():

		# Get img from realsense in Numpy array format
		depth_img, color_img = cam.get_frame_cv()

		# Numpy array can only be accessed by cpu
		# Copy color img to GPU for network inference
		color_img_cuda = jetson.utils.cudaFromNumpy(color_img)

		# allocate gpu memory for network input image as rgba32f, with the same width/height as the color frame
		network_input_img = jetson.utils.cudaAllocMapped(width = cam.color_frame_width, height = cam.color_frame_height, format='rgba32f')

		# convert from rgb8 (default format for realsense color frame in this program) to rgba32f
		jetson.utils.cudaConvertColor(color_img_cuda, network_input_img)

		# detect objects in the image (with overlay)
		detections = net.Detect(network_input_img, cam.color_frame_width, cam.color_frame_height, opt['overlay'])

		# print the detections
		print("detected {:d} objects in image".format(len(detections)))

		for detection in detections:
			print(detection)

		# render the image
		display.RenderOnce(network_input_img, cam.color_frame_width, cam.color_frame_height)

		# update the title bar
		display.SetTitle("{:s} | Network {:.0f} FPS".format(opt['network'], net.GetNetworkFPS()))

		# print out performance info
		# net.PrintProfilerTimes()

		if(len(detections) != 0):
			for detection in detections:
				if detection.Confidence > 0.90:
					obj_center = int(detection.Center)
					# compute target coordinate in camera frame
					target_in_cam_z = depth_img[obj_center[0], obj_center[1]]
					target_in_cam_x = np.multiply(obj_center[0] - cam.intrinsics['cx'], target_in_cam_z / cam.intrinsics['fx'])
					target_in_cam_y = np.multiply(obj_center[1] - cam.intrinsics['cy'], target_in_cam_z / cam.intrinsics['fy'])

					print("Target in camera frame:\n", [target_in_cam_x, target_in_cam_y, target_in_cam_z])

					target_in_cam = np.array([target_in_cam_x, target_in_cam_y, target_in_cam_z])
					target_in_base = R.dot(target_in_cam) + t

					arm.move_p([target_in_base[0], target_in_base[1], target_in_base[2], 3.14, 0, 0])




