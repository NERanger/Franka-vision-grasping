cmake_minimum_required(VERSION 3.5)

project(FrankaDriver)

#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /home/bionicdl-Mega/guoning/GN/lib/cmake/opencv4 /home/bionicdl-Mega/guoning/GN/lib/cmake/Franka /home/bionicdl-Mega/guoning/GN/share/pcl-1.9 /home/bionicdl-Mega/guoning/GN/lib/cmake/visp)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#set( CMAKE_BUILD_TYPE "Debug" )
FIND_PACKAGE(pybind11)

find_package(Franka)
find_package(Eigen3 REQUIRED)


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})
LINK_LIBRARIES(Franka::Franka Threads::Threads ${EIGEN3_LIBRARIES} -lPocoNet -lPocoFoundation)

pybind11_add_module(RobotDriver Bionic/Wrapper.cpp Bionic/FrankaDriver.h Bionic/FrankaDriver.cpp Bionic/JointPosTraj.cpp Bionic/JointPosTraj.h Bionic/VelTraj.h Bionic/VelTraj.cpp)


