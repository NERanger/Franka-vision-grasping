# Franka vision grasping

This repo is for grasping task on the following setup.

**Arm**: Franka Emika Panda 

**Vision Sensor**: Intel RealSense D435

**Computing Platform**: Jetson AGX Xavier 

## Requirements

* [Jetpack 4.4 with Realtime Kernel](./docs/Jetpack4.4-with-realtime-kernel.md)
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

## Troubleshooting

* After flash the Jetson Xavier with JetPack 4.4 and realtime kernel patch, `sudo jetson_clocks` may not able to turn on the fan, which will lead to severe heat accumulation. You can use `echo 255 | sudo tee /sys/devices/pwm-fan/target-pwm` to manually set the fan at maximum speed.

  > Reference: https://forums.developer.nvidia.com/t/fan-management-xavier/70166/9