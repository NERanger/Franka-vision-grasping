# Franka vision grasping

This repo is for grasping task on the following setup.

**Arm**: Franka Emika Panda 

**Vision Sensor**: Intel RealSense D435

**Computing Platform**: Jetson AGX Xavier 

## Requirements

* [Jetpack 4.4 with Realtime Kernel](https://orenbell.com/?p=436)
* [Set user task priority for realtime kernel](https://frankaemika.github.io/docs/installation_linux.html)
* [libfranka version == 0.7.1](https://frankaemika.github.io/docs/installation_linux.html)
* [opencv with contrib (for calibration with aruco marker)](https://github.com/AastaNV/JEP/blob/master/script/install_opencv4.3.0_Jetson.sh)
* [jetson-inference](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md)
* [librealsense2](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md)
* [pyrealsense2 (build from source)](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)
* [pybind11](https://github.com/pybind/pybind11)
* Eigen3
* Python3
  * numpy
  * yaml
  * matplotlib (for calibration visualization)
  * pytransform3d (for calibration visualization)

