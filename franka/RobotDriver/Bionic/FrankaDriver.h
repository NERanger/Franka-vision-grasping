#ifndef BIONIC_FRANKA_DRIVER_H
#define BIONIC_FRANKA_DRIVER_H


#include <cmath>
#include <iostream>

#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

#include <stdio.h>
#include "JointPosTraj.h"

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/model.h>
#include <franka/gripper.h>

namespace Bionic
{
namespace RobotDriver
{

class FrankaDriver
{
public:
    /*!
      Robot control states.
    */
    typedef enum {
        STATE_STOP,                 //!< Stops robot motion especially in velocity and acceleration control.
        STATE_VELOCITY_CONTROL,     //!< Initialize the velocity controller.
        STATE_POSITION_CONTROL,     //!< Initialize the position controller.
        STATE_ACCELERATION_CONTROL, //!< Initialize the acceleration controller.
        STATE_FORCE_TORQUE_CONTROL  //!< Initialize the force/torque controller.
    } RobotStateType;

    /*!
      Robot control frames.
    */
    typedef enum {
        REFERENCE_FRAME,
        JOINT_STATE, /*!< Corresponds to the joint state. */
        END_EFFECTOR_FRAME,    /*!< Corresponds to robot end-effector frame. */
    } ControlFrameType;

private:
  /*!
    Copy constructor not allowed.
   */
  FrankaDriver(const FrankaDriver &robot) = delete;

  ///
  /// \brief init
  /// Initilise the default limits of Joints' Pos, Vel, Acc
  void init();

  franka::Robot *m_handler; //!< Robot handler
  franka::Gripper *m_gripper; //!< Gripper handler


  RobotStateType stateRobot;


  // Velocity controller
  std::thread m_velControlThread;
  std::atomic_bool m_velControlThreadIsRunning;
  std::atomic_bool m_velControlThreadStopAsked;
  std::array<double, 7> m_dq_des;             // Desired joint velocity
  std::array<double, 6> m_v_cart_des;         // Desired cartesian velocity either in reference, end-effector, camera, or tool frame


  franka::RobotState m_robot_state; // Robot state protected by mutex
  std::mutex m_mutex;               // Mutex to protect m_robot_state


  std::array<double, 7> m_q_min;    // Joint min position
  std::array<double, 7> m_q_max;    // Joint max position
  std::array<double, 7> m_dq_max;   // Joint max velocity
  std::array<double, 7> m_ddq_max;  // Joint max acceleration



  ///
  /// \brief m_positionningVelocity
  /// Speed Factor [0 -- 100]
  double m_positionningVelocity;

  ///
  /// \brief m_max_attempt
  ///  Max Attempt times try to recover from move error and continue to move
  int m_max_attempt;

  ///
  /// \brief m_franka_address
  /// IP Address of Franka Robot
  std::string m_franka_address;

  ///
  /// \brief getRobotInternalState
  /// \return franka state read from the controller
  franka::RobotState getRobotInternalState();

public:

  /**
   *  You have two ways to instantiate this class
   *  1\
   *    FrankaDriver franka;
   *    franka.connect("192.168.0.1");
   *
   *  2\
   *    FrankaDriver franka("192.168.0.1")
   * **/

  FrankaDriver();

  FrankaDriver(std::string franka_address);

  virtual ~FrankaDriver();

  void connect(std::string franka_address);

  /*!
   * Get gripper handler to access native libfranka functions.
   *
   * \return Robot handler if it exists, an exception otherwise.
   */
  franka::Gripper *getGripperHandler() {
    if (!m_gripper)
    {
        std::exception e;
        std::cerr<<"Gripper is not connected! "<<std::endl;
        throw e;

    }

    return m_gripper;
  }

  /*!
   * Get robot handler to access native libfranka functions.
   *
   * \return Robot handler if it exists, an exception otherwise.
   */
  franka::Robot *getHandler() {
    if (!m_handler)
    {
        std::exception e;
        std::cerr<<"Robot is not connected! "<<std::endl;
        throw e;
    }

    return m_handler;
  }


  RobotStateType getRobotState(void) const { return stateRobot; }

  RobotStateType setRobotState(RobotStateType newState);

  ///
  /// \brief setMaxAttempt
  /// \param attemp
  /// Default is 5
  void setMaxAttempt(int attemp);

  ///
  /// \brief setPositioningVelocity
  /// \param velocity
  ///  Default is 20, which means  move at 20% of max joint speed
  void setPositioningVelocity(double velocity);

  ///
  /// \brief getJointPos
  /// \return Current Joint Angles from robot
  ///
  std::array<double,7> getJointPos();

  ///
  /// \brief getJointMin
  /// \return Default limits of Joints
  ///
  std::array<double,7> getJointMin();


  std::array<double,7> getJointMax();

  // Base   ->    Flange   ->   EE

  ///
  /// \brief getO_T_EE
  /// \return Current Cartisian Pose Matrix  from Base To EndEffector
  ///
  /// The order of 16 numbers is arranged as following
  ///
  /// 1,  5,  9,   13,
  /// 2,  6,  10,  14,
  /// 3,  7,  11,  15,
  /// 4,  8,  12,  16
  ///
  std::array<double, 16> getO_T_EE();

  ///
  /// \brief getF_T_EE
  /// \return Current Cartisian Pose Matrix  from  Flange  To EndEffector
  ///
  std::array<double, 16> getF_T_EE();


  ///
  /// \brief setJointPos
  /// \param q_d
  ///
  /// Set Speed Factor first
  /// Set Max Attempts to increase the robustness
  ///
  ///            FrankaDriver franka("192.168.0.1");
  ///            franka.setPositioningVelocity(80.);
  ///            franka.setsetMaxAttempt
  ///            Vector7d TargetQ;
  ///            TargetQ<<M_pi,M_pi,M_pi,M_pi,M_pi,M_pi,M_pi;
  ///            franka.setJointPos(TargetQ);
  ///
  void setJointPos(std::array<double,7> q_d);

  ///
  /// \brief SetVelocity In Joint Space
  /// \param vel_j , rad/s
  ///
  void setJointVelocity(std::array<double,7> vel_j);


  ///
  /// \brief SetVelocity In Cartesian Space
  /// \param vel_c,  twist in the end_effector, vel_c {v_x,v_y,v_z,omega_x,omega_y,omega_z} m/s, rad/s
  ///
  void setCartesianVelocity(std::array<double,6> vel_c, std::string ref);


  int gripperClose();

  int gripperGrasp(double grasping_width, double force=60.);

  void gripperHoming();

  int gripperMove(double width);

  int gripperOpen();

  void gripperRelease();

  void stopMotion();

  std::string getAllState(); //todo

};




}

}




#endif
