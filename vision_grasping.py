import jetson.inference
import jetson.utils

from realsense_wapper import realsense

import argparse
import sys

# convert colorspace
def convert_color(img, output_format):
	converted_img = jetson.utils.cudaAllocMapped(width=img.width, height=img.height, format=output_format)
	jetson.utils.cudaConvertColor(img, converted_img)
	return converted_img

if __name__ == '__main__':
	# parse the command line
	parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
							formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

	parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
	parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
	parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
	# parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
	parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
	parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
	parser.add_argument("--fps", type=int, default=30, help="desired frame rate for camera")

	try:
		opt = parser.parse_known_args()[0]
	except:
		print("")
		parser.print_help()
		sys.exit(0)

	cam = realsense(frame_width = opt.width, frame_height = opt.height, fps = opt.fps)

	# load the object detection network
	net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

	# # create the camera and display
	# camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
	display = jetson.utils.glDisplay()

	# process frames until user exits
	while display.IsOpen():

		# Get img from realsense in Numpy array format
		depth_img, color_img = cam.get_frame_cv()

		# Numpy array can only be accessed by cpu
		# Copy color img to GPU for network inference
		color_img_cuda = jetson.utils.cudaFromNumpy(color_img)

		# allocate gpu memory for network input image as rgba32f, with the same width/height as the color frame
		network_input_img = jetson.utils.cudaAllocMapped(width=1280, height=720, format='rgba32f')

		# convert from bgr8 (default format for realsense color frame) to rgba32f
		jetson.utils.cudaConvertColor(color_img_cuda, network_input_img)

		# detect objects in the image (with overlay)
		detections = net.Detect(network_input_img, cam.color_frame_width, cam.color_frame_height, opt.overlay)

		# print the detections
		print("detected {:d} objects in image".format(len(detections)))

		for detection in detections:
			print(detection)

		# render the image
		display.RenderOnce(network_input_img, cam.color_frame_width, cam.color_frame_height)

		# update the title bar
		display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

		# print out performance info
		net.PrintProfilerTimes()
