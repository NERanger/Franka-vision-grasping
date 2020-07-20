#ifndef BIONIC_FRANKA_JOINT_POS_TRAJ_H
#define BIONIC_FRANKA_JOINT_POS_TRAJ_H


#include <cmath>
#include <iostream>
#include <Eigen/Core>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/robot_state.h>


namespace Bionic
{
namespace RobotDriver
{

using Vector7d = Eigen::Matrix<double, 7, 1, Eigen::ColMajor>;
using Vector7i = Eigen::Matrix<int, 7, 1, Eigen::ColMajor>;

class JointPosTrajGenerator
{
public:
    /**
   * Creates a new MotionGenerator instance for a target q.
   *
   * @param[in] speed_factor General speed factor in range [0, 1].
   * @param[in] q_goal Target joint positions.
   */
    JointPosTrajGenerator(double speed_factor, const std::array<double, 7> &q_goal);

    /**
   * Sends joint position calculations
   *
   * @param[in] robot_state Current state of the robot.
   * @param[in] period Duration of execution.
   *
   * @return Joint positions for use inside a control loop.
   */
    franka::JointPositions operator()(const franka::RobotState& robot_state, franka::Duration period);

private:
    bool calculateDesiredValues(double t, Vector7d &delta_q_d) const;
    void calculateSynchronizedValues();

    static constexpr double kDeltaQMotionFinished = 1e-6;
    Vector7d m_q_goal;
    Vector7d m_q_start;
    Vector7d m_delta_q;
    Vector7d m_dq_max_sync;
    Vector7d m_t_1_sync;
    Vector7d m_t_2_sync;
    Vector7d m_t_f_sync;
    Vector7d m_q_1;

    Vector7d m_dq_max;
    Vector7d m_ddq_max_start;
    Vector7d m_ddq_max_goal;

    double m_time = 0.0;
};



}

}




#endif
