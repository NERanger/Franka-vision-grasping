#ifndef BIONIC_FRANKA_VEL_TRAJ_H
#define BIONIC_FRANKA_VEL_TRAJ_H

#include <array>
#include <vector>
#include <iostream>
#include <atomic>

#include <franka/exception.h>
#include <franka/robot.h>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/robot_state.h>

#include <Eigen/Core>
#include <Eigen/Geometry>


#include "FrankaDriver.h"

namespace Bionic
{
namespace RobotDriver
{

using Vector7d = Eigen::Matrix<double, 7, 1, Eigen::ColMajor>;
using Vector7i = Eigen::Matrix<int, 7, 1, Eigen::ColMajor>;
using HomogeneousMatrix = Eigen::Affine3d;



class VelTrajGenerator {
public:
  VelTrajGenerator() :
    m_status(), m_delta_q(), m_delta_q_max(), m_delta_q_acc(),
    m_q_final(), m_sign(), m_q_cmd(), m_q_cmd_prev(), m_dist_AD(), m_dq_des(), m_dq_des_prev(), m_dist_to_final(),
    m_flagSpeed(),
    m_q_min(), m_q_max(), m_dq_max(), m_ddq_max(), m_njoints(7), m_delta_t(0.001), m_flagJointLimit(false) {}
  virtual ~VelTrajGenerator() {}

  void applyVel(const std::array<double, 7> &dq_des,
                std::array<double, 7> &q_cmd,
                std::array<double, 7> &dq_cmd);

  void control_thread(franka::Robot *robot, std::atomic_bool &stop,
                      const FrankaDriver::ControlFrameType &frame,
                      const HomogeneousMatrix &eMc,
                      const std::array<double, 6> &v_cart_des,
                      const std::array<double, 7> &dq_des,
                      const std::array<double, 7> &q_min,
                      const std::array<double, 7> &q_max,
                      const std::array<double, 7> &dq_max,
                      const std::array<double, 7> &ddq_max,
                      franka::RobotState &robot_state,
                      std::mutex &mutex);

  void init (const std::array<double, 7> &q,
             const std::array<double, 7> &q_min,
             const std::array<double, 7> &q_max,
             const std::array<double, 7> &dq_max,
             const std::array<double, 7> &ddq_max,
             double delta_t);

  std::array<double, 7> limitRate(const std::array<double, 7>& max_derivatives,
                                  const std::array<double, 7>& desired_values,
                                  const std::array<double, 7>& last_desired_values);

private:
  typedef enum {
    FLAGACC,	// Axis in acceleration
    FLAGCTE,  // Axis at constant velocity
    FLAGDEC,	// Axis in deceleration
    FLAGSTO 	// Axis stopped
  } status_t;

  std::array<status_t, 7>	m_status;	     // Axis status
  std::array<double, 7> m_delta_q;		   // Current position increment
  std::array<double, 7> m_delta_q_max;   // Max position increment
  std::array<double, 7> m_delta_q_acc;	 // Increment related to acceleration
  std::array<double, 7> m_q_final;       // Final position before joint limit
  std::array<int, 7>    m_sign;	         // Displacement sign: +1 = increment position
  std::array<double, 7> m_q_cmd;         // Joint position command
  std::array<double, 7> m_q_cmd_prev;    // Previous joint position command
  std::array<double, 7> m_dist_AD;       // Distance required to accelerate or decelerate
  std::array<double, 7> m_dq_des;		     // Desired velocity
  std::array<double, 7> m_dq_des_prev;   // Previous desired velocity
  std::array<double, 7> m_dist_to_final; // Distance between current joint position and final
  std::array<bool, 7>   m_flagSpeed;

  // Constant
  std::array<double, 7> m_q_min;
  std::array<double, 7> m_q_max;
  std::array<double, 7> m_dq_max;
  std::array<double, 7> m_ddq_max;

  size_t m_njoints;
  double m_delta_t;

  bool m_flagJointLimit;

  const double m_offset_joint_limit = 1; // stop before joint limit (rad)
  const double m_delta_q_min =	1e-9;	                // Delta q minimum (rad)
};

}
}
#endif
