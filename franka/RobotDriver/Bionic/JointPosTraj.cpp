#include "JointPosTraj.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

#include <franka/exception.h>
#include <franka/robot.h>

namespace Bionic
{
namespace RobotDriver
{

int getsign(double x)
{
    if (fabs(x) < std::numeric_limits<double>::epsilon())
        return 0;
    else
    {
        if (x < 0)
            return -1;
        else
            return 1;
    }
}

double getmax(Vector7d vec)
{
    double *dataptr = vec.data();
    double max = *dataptr;
    dataptr++;
    for (unsigned int i = 0; i < vec.size() - 1; i++)
    {
        if (*dataptr > max)
        {
            max = *dataptr;
        }
        dataptr++;
    }
    return max;
}


JointPosTrajGenerator::JointPosTrajGenerator(double speed_factor, const std::array<double, 7> &q_goal)
    : m_q_goal(7), m_q_start(7), m_delta_q(7), m_dq_max_sync(7), m_t_1_sync(7), m_t_2_sync(7), m_t_f_sync(7), m_q_1(7),
      m_dq_max(7), m_ddq_max_start(7), m_ddq_max_goal(7)
{
    for (size_t i = 0; i< 7; i++)
    {
        m_q_goal[i] = q_goal[i];
        m_dq_max[i] = 2.5;
        m_ddq_max_start[i] = 5.;
        m_ddq_max_goal[i] = 5.;
    }

    m_dq_max *= speed_factor;
    m_ddq_max_start *= speed_factor;
    m_ddq_max_goal *= speed_factor;
}

bool JointPosTrajGenerator::calculateDesiredValues(double t, Vector7d &delta_q_d) const
{
    std::vector<int> sign_delta_q(7);
    for (size_t i = 0; i< 7; i++) {
        sign_delta_q[i] = getsign(m_delta_q[i]);
    }

    Vector7d t_d = m_t_2_sync - m_t_1_sync;
    Vector7d delta_t_2_sync = m_t_f_sync - m_t_2_sync;
    std::array<bool, 7> joint_motion_finished{};

    for (size_t i = 0; i < 7; i++)
    {
        if (std::abs(m_delta_q[i]) < kDeltaQMotionFinished)
        {
            delta_q_d[i] = 0.;
            joint_motion_finished[i] = true;
        }
        else
        {
            if (t < m_t_1_sync[i])
            {
                delta_q_d[i] = -1.0 / std::pow(m_t_1_sync[i], 3.0) * m_dq_max_sync[i] * sign_delta_q[i] *
                        (0.5 * t - m_t_1_sync[i]) * std::pow(t, 3.0);
            }
            else if (t >= m_t_1_sync[i] && t < m_t_2_sync[i])
            {
                delta_q_d[i] = m_q_1[i] + (t - m_t_1_sync[i]) * m_dq_max_sync[i] * sign_delta_q[i];
            }
            else if (t >= m_t_2_sync[i] && t < m_t_f_sync[i])
            {
                delta_q_d[i] = m_delta_q[i] +
                        0.5 *
                        (1.0 / std::pow(delta_t_2_sync[i], 3.0) *
                         (t - m_t_1_sync[i] - 2.0 * delta_t_2_sync[i] - t_d[i]) *
                         std::pow((t - m_t_1_sync[i] - t_d[i]), 3.0) +
                         (2.0 * t - 2.0 * m_t_1_sync[i] - delta_t_2_sync[i] - 2.0 * t_d[i])) *
                        m_dq_max_sync[i] * sign_delta_q[i];
            }
            else
            {
                delta_q_d[i] = m_delta_q[i];
                joint_motion_finished[i] = true;
            }
        }
    }
    return std::all_of(joint_motion_finished.cbegin(), joint_motion_finished.cend(),
                       [](bool x) { return x; });
}

void JointPosTrajGenerator::calculateSynchronizedValues()
{
    Vector7d dq_max_reach(m_dq_max);
    Vector7d t_f(7);
    Vector7d delta_t_2(7);
    Vector7d t_1(7);
    Vector7d delta_t_2_sync(7);
    std::vector<int> sign_delta_q(7);
    for (size_t i = 0; i< 7; i++) {
        sign_delta_q[i] = getsign(m_delta_q[i]);
    }

    for (size_t i = 0; i < 7; i++) {
        if (std::abs(m_delta_q[i]) > kDeltaQMotionFinished) {
            if (std::abs(m_delta_q[i]) < (3.0 / 4.0 * (std::pow(m_dq_max[i], 2.0) / m_ddq_max_start[i]) +
                                          3.0 / 4.0 * (std::pow(m_dq_max[i], 2.0) / m_ddq_max_goal[i]))) {
                dq_max_reach[i] = std::sqrt(4.0 / 3.0 * m_delta_q[i] * sign_delta_q[i] *
                                            (m_ddq_max_start[i] * m_ddq_max_goal[i]) /
                                            (m_ddq_max_start[i] + m_ddq_max_goal[i]));
            }
            t_1[i] = 1.5 * dq_max_reach[i] / m_ddq_max_start[i];
            delta_t_2[i] = 1.5 * dq_max_reach[i] / m_ddq_max_goal[i];
            t_f[i] = t_1[i] / 2.0 + delta_t_2[i] / 2.0 + std::abs(m_delta_q[i]) / dq_max_reach[i];
        }
    }
    double max_t_f = getmax(t_f);
    for (size_t i = 0; i < 7; i++)
    {
        if (std::abs(m_delta_q[i]) > kDeltaQMotionFinished)
        {
            double a = 1.5 / 2.0 * (m_ddq_max_goal[i] + m_ddq_max_start[i]);
            double b = -1.0 * max_t_f * m_ddq_max_goal[i] * m_ddq_max_start[i];
            double c = std::abs(m_delta_q[i]) * m_ddq_max_goal[i] * m_ddq_max_start[i];
            double delta = b * b - 4.0 * a * c;
            if (delta < 0.0) {
                delta = 0.0;
            }
            m_dq_max_sync[i] = (-1.0 * b - std::sqrt(delta)) / (2.0 * a);
            m_t_1_sync[i] = 1.5 * m_dq_max_sync[i] / m_ddq_max_start[i];
            delta_t_2_sync[i] = 1.5 * m_dq_max_sync[i] / m_ddq_max_goal[i];
            m_t_f_sync[i] =
                    (m_t_1_sync)[i] / 2.0 + delta_t_2_sync[i] / 2.0 + std::abs(m_delta_q[i] / m_dq_max_sync[i]);
            m_t_2_sync[i] = (m_t_f_sync)[i] - delta_t_2_sync[i];
            m_q_1[i] = m_dq_max_sync[i] * sign_delta_q[i] * (0.5 * m_t_1_sync[i]);
        }
    }
}

franka::JointPositions JointPosTrajGenerator::operator()(const franka::RobotState& robot_state,
                                                         franka::Duration period) {
    m_time += period.toSec();

    if (m_time == 0.0) {
        for (size_t i = 0; i < 7; i++) {
            m_q_start[i] = robot_state.q_d[i];
        }
        m_delta_q = m_q_goal - m_q_start;

        calculateSynchronizedValues();
    }

    Vector7d delta_q_d(7);
    bool motion_finished = calculateDesiredValues(m_time, delta_q_d);

    std::array<double, 7> joint_positions;
    Vector7d q_des = m_q_start + delta_q_d;
    for (size_t i = 0; i < 7; i++) {
        joint_positions[i] = q_des[i];
    }
    franka::JointPositions output(joint_positions);
    output.motion_finished = motion_finished;
    return output;
}
}
}
