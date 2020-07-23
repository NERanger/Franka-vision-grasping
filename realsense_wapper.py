import pyrealsense2 as rs
import numpy as np
import cv2

class realsense(object):
    def __init__(self, frame_width = 640, frame_height = 480, fps = 30):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)

        # Start streaming
        self.cfg = self.pipeline.start(config)

        # Align depth frame to color frame
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        profile = self.cfg.get_stream(rs.stream.color) # Fetch stream profile for color stream
        intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
        self.intrinsics = {'cx' : intr.ppx, 'cy' : intr.ppy, 'fx' : intr.fx, 'fy' : intr.fy}

        depth_sensor = self.cfg.get_device().first_depth_sensor()
        self.depth_scale = round(depth_sensor.get_depth_scale(), 4)

        print("------ Realsense start info ------ ")
        print("Frame size: " + str(frame_width) + "*" + str(frame_height))
        print("FPS: " + str(fps))
        print("Depth scale: " + str(self.depth_scale))
        print("Intrinsics: " + str(self.intrinsics))
        print("---------------------------------- ")

    # Get a frame that can be processed in opencv
    def get_frame_cv(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Convert images to numpy arrays ( can be read with opencv )
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

if __name__ == '__main__':
    cam = realsense(1280, 720, 30)
    #depth, color = cam.get_frame_cv()
    #cv2.imwrite('test.jpg', color)