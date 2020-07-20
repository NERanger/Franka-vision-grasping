#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <sstream>
#include <vector>
#include "FrankaDriver.h"


namespace py = pybind11;
using namespace Bionic::RobotDriver;


PYBIND11_MODULE(RobotDriver, m) {
    // FrankaDriver
    py::class_<FrankaDriver>(m, "FrankaDriver")
        .def(py::init<std::string>(),"Contructor of Class with ip address")
        .def("connect", &FrankaDriver::connect,"Connect to Robot, No need to Call")
        .def("setMaxAttempt", &FrankaDriver::setMaxAttempt,"Set Max Attempt times to recover error from move -- Default : 5")
        .def("setPositioningVelocity", &FrankaDriver::setPositioningVelocity,"Set speed factor [0 ~ 100]")
        .def("getJointPos", &FrankaDriver::getJointPos,"Get Current Joint Pos")
        .def("getJointMin", &FrankaDriver::getJointMin,"Get Min Limit of each Joint")
        .def("getJointMax", &FrankaDriver::getJointMax,"Get Max Limit of each Joint")
        .def("getO_T_EE", &FrankaDriver::getO_T_EE,"Get Cartisian Pose 4*4 Matrix from Base to EE !!Column-Major!! ")
        .def("getF_T_EE", &FrankaDriver::getF_T_EE,"Get Cartisian Pose 4*4 Matrix from Flange to EE !!Column-Major!! ")
        .def("setJointPos", &FrankaDriver::setJointPos,"Move Joint to desired position")
        .def("gripperClose", &FrankaDriver::gripperClose,"Close Gripper")
        .def("gripperGrasp", &FrankaDriver::gripperGrasp,"Grasp with  2 parameters")
        .def("gripperHoming", &FrankaDriver::gripperHoming,"Gripper Move to Home width")
        .def("gripperMove", &FrankaDriver::gripperMove,"Gripper move to desired width")
        .def("gripperOpen", &FrankaDriver::gripperOpen,"Open Gripper")
        .def("gripperRelease", &FrankaDriver::gripperRelease,"Release Object")
        .def("speedJ",&FrankaDriver::setJointVelocity,"Set Speed in Joint Space")
        .def("getAllState",&FrankaDriver::getAllState,"Get All State of Robot")
        .def("stopMotion",&FrankaDriver::stopMotion,"Stop Move in Velocity State");
//        .def("speedL",&FrankaDriver::setCartesianVelocity,"Set Speed in Cartesian Space");
}










