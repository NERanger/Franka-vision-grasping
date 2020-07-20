import pyrealsense2 as rs
import numpy as np

class realsense(object):
    def __init__(self, frame_width, frame_height, fps):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)

        # Start streaming
        self.cfg = self.pipeline.start(config)

    def get_color_intrinsics(self):
        profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for color stream
        intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

        # print("ppx/cx: " + str(intr.ppx) + "\nppy/cy: " + str(intr.ppy))
        # print("fx: " + str(intr.fx) + "\nfy: " + str(intr.fy))
        # print("distortion coeffs: " + str(intr.coeffs))
        # print("height: " + str(intr.height))
        # print("width: " + str(intr.width))

        return {'cx' : intr.ppx, 'cy' : intr.ppy, 'fx' : intr.fx, 'fy' : intr.fy}

    # Get a frame that can be processed in opencv
    def get_frame_cv(self):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert images to numpy arrays ( can be read with opencv )
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

if __name__ == '__main__':
    cam = realsense()
    intr = cam.get_color_intrinsics()
    print(intr)
    depth, color = cam.get_frame_cv()
    cv.imwrite('test.jpg', color)