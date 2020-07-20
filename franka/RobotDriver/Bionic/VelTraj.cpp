#include "VelTraj.h"
#include <cmath>
#include <iomanip>
#include <algorithm>

#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/model.h>

//#include <visp3/core/vpException.h>
//#include <visp3/core/vpTime.h>
//#include <visp3/core/vpMatrix.h>


namespace Bionic
{
namespace RobotDriver
{


void skew(Eigen::Matrix3d& M, const Eigen::Vector3d& t)
{
    M(0,0) = 0;
    M(0,1) = -t[2];
    M(0,2) = t[1];
    M(1,0) = t[2];
    M(1,1) = 0;
    M(1,2) = -t[0];
    M(2,0) = -t[1];
    M(2,1) = t[0];
    M(2,2) = 0;
}


void buildFrom(Eigen::Matrix<double,6,6>& cVe,const Eigen::Affine3d eMc)
{
    Eigen::Matrix3d rotation;
    rotation = eMc.rotation();
    Eigen::Vector3d translation;
    translation = eMc.translation();

    Eigen::Matrix3d skew_vector;
    skew(skew_vector,translation);


    Eigen::Matrix3d skewaR;
    skewaR = skew_vector * rotation;

    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        cVe(i,j) = rotation(i,j);
        cVe(i + 3,j + 3) = rotation(i,j);
        cVe(i,j + 3) = skewaR(i,j);
      }
    }
}


void transpose(Eigen::Matrix<double,7,6>& out, const Eigen::Matrix<double,6,7>& in)
{

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 7; j++)
        {
            out(j,i) = in(i,j);

        }
    }

}


void svdEigen3(Eigen::Matrix<double,7,6>& in, Eigen::Matrix<double,6,1> &w,
               Eigen::Matrix<double,6,6> &V)
{

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(in, Eigen::ComputeThinU | Eigen::ComputeThinV);

    for(int i = 0; i < svd.singularValues().cols(); i++)
    {
        w(i) = svd.singularValues()(i);
    }

    for(int i = 0; i < 7; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            in(i,j) = svd.matrixU()(i,j);
        }
    }

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            V(i,j) = svd.matrixV()(i,j);
        }

    }
}




void compute_pseudo_inverse(const Eigen::Matrix<double,7,6> &a, const Eigen::Matrix<double,6,1> &sv,
                            const Eigen::Matrix<double,6,6> &v, double svThreshold,
                            Eigen::Matrix<double,7,6> &Ap, unsigned int &rank)
{

  Eigen::Matrix<double,6,7> a1;

  // compute the highest singular value and the rank of h
  double maxsv = 0;
  for (unsigned int i = 0; i < 6; i++)
  {
    if (fabs(sv[i]) > maxsv)
      maxsv = fabs(sv[i]);
  }

  rank = 0;

  for (unsigned int i = 0; i < 6; i++)
  {
    if (fabs(sv[i]) > maxsv * svThreshold)
    {
      rank++;
    }

    for (unsigned int j = 0; j < 7; j++)
    {
      a1(i,j) = 0.0;

      for (unsigned int k = 0; k < 6; k++)
      {
        if (fabs(sv[k]) > maxsv * svThreshold)
        {
          a1(i,j) += v(i,k) * a(j,k) / sv[k];
        }
      }
    }
  }


  for(int i = 0; i < 6; i++)
  {
      for(int j = 0; j < 7; j++)
      {
          Ap(j,i) = a1(i,j);
      }
  }


}


void pseudoInverseEigen3(Eigen::Matrix<double,7,6>& out ,const Eigen::Matrix<double,6,7>& in)
{
    double svThreshold = 1e-6;

    unsigned int nrows, ncols;
    unsigned int nrows_orig = in.rows();
    unsigned int ncols_orig = in.cols();
    nrows = 7;
    ncols = 6;

    Eigen::Matrix<double,7,6> U;
    Eigen::Matrix<double,6,6> V;

    Eigen::Matrix<double,6,1> sv;

    transpose(U,in);

    svdEigen3(U,sv,V);

    unsigned int rank;

    compute_pseudo_inverse(U, sv, V, svThreshold, out, rank);

}

void VelTrajGenerator::control_thread(franka::Robot *robot,
                                             std::atomic_bool &stop,
                                             const FrankaDriver::ControlFrameType &frame,
                                             const HomogeneousMatrix &eMc,
                                             const std::array<double, 6> &v_cart_des, // end-effector velocity
                                             const std::array<double, 7> &dq_des, // joint velocity
                                             const std::array<double, 7> &q_min,
                                             const std::array<double, 7> &q_max,
                                             const std::array<double, 7> &dq_max,
                                             const std::array<double, 7> &ddq_max,
                                             franka::RobotState &robot_state,
                                             std::mutex &mutex)
{
  double time = 0.0;
  double delta_t = 0.001;
  std::array<double, 7> q_prev;
  franka::Model model = robot->loadModel();

  Eigen::Matrix<double,6,7> eJe,fJe;

  Eigen::Matrix<double,6,6> cVe;
  buildFrom(cVe,eMc);




  auto joint_velocity_callback = [=, &time, &q_prev, &dq_des, &stop, &robot_state, &mutex]
      (const franka::RobotState& state, franka::Duration period) -> franka::JointVelocities
  {



    time += period.toSec();

    static VelTrajGenerator joint_vel_traj_generator;

    if (time == 0.0)
    {

      q_prev = state.q_d;
      joint_vel_traj_generator.init(state.q_d, q_min, q_max, dq_max, ddq_max, delta_t);
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      robot_state = state;
    }

    std::array<double, 7> q_cmd;
    std::array<double, 7> dq_cmd;

    auto dq_des_ = dq_des;
    if (stop)
    { // Stop asked
      for (auto & dq_ : dq_des_)
      {
        dq_ = 0.0;
      }
    }

    joint_vel_traj_generator.applyVel(dq_des_, q_cmd, dq_cmd);

    franka::JointVelocities velocities = {dq_cmd[0], dq_cmd[1], dq_cmd[2], dq_cmd[3], dq_cmd[4], dq_cmd[5], dq_cmd[6]};


    if (stop)
    {
      unsigned int nb_joint_stop = 0;
      const double q_eps = 1e-6; // Motion finished
      for(size_t i=0; i < 7; i++)
      {
        if (std::abs(state.q_d[i] - q_prev[i]) < q_eps)
        {
          nb_joint_stop ++;
        }
      }

      if (nb_joint_stop == 7)
      {
        return franka::MotionFinished(velocities);
      }
    }

    q_prev = state.q_d;

    // state.q_d contains the last joint velocity command received by the robot.
    // In case of packet loss due to bad connection or due to a slow control loop
    // not reaching the 1kHz rate, even if your desired velocity trajectory
    // is smooth, discontinuities might occur.
    // Saturating the acceleration computed with respect to the last command received
    // by the robot will prevent from getting discontinuity errors.
    // Note that if the robot does not receive a command it will try to extrapolate
    // the desired behavior assuming a constant acceleration model
    return limitRate(ddq_max, velocities.dq, state.dq_d);

  };




  auto cartesian_velocity_callback = [=, &time, &model, &q_prev, &v_cart_des, &stop, &robot_state, &mutex, &eJe, &fJe]
      (const franka::RobotState& state, franka::Duration period) -> franka::JointVelocities
  {

    time += period.toSec();

    static VelTrajGenerator joint_vel_traj_generator;

    if (time == 0.0)
    {

      q_prev = state.q_d;
      joint_vel_traj_generator.init(state.q_d, q_min, q_max, dq_max, ddq_max, delta_t);
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      robot_state = state;
    }

    // Get robot Jacobian
    if (frame == FrankaDriver::END_EFFECTOR_FRAME)
    {
      std::array<double, 42> jacobian = model.bodyJacobian(franka::Frame::kEndEffector, state);
      // Convert row-major to col-major
      for (size_t i = 0; i < 6; i ++) { // TODO make a function
        for (size_t j = 0; j < 7; j ++) {
          eJe(i,j) = jacobian[j*6 + i];
        }
      }
    }
    else if (frame == FrankaDriver::REFERENCE_FRAME) {
      std::array<double, 42> jacobian = model.zeroJacobian(franka::Frame::kEndEffector, state);
      // Convert row-major to col-major
      for (size_t i = 0; i < 6; i ++) { // TODO make a function
        for (size_t j = 0; j < 7; j ++) {
          fJe(i,j) = jacobian[j*6 + i];
        }
      }
    }
///////////////////////



//////////////


    // Compute joint velocity
    Vector7d q_dot;
    if (frame == FrankaDriver::END_EFFECTOR_FRAME)
    {
        Eigen::Matrix<double,7,6> ieJe;
        pseudoInverseEigen3(ieJe,eJe);


        Eigen::Matrix<double,6,1> v_des;
        for(int ll = 0; ll < 6 ;ll++)
        {
            v_des[ll] = v_cart_des[ll];
        }
        q_dot = ieJe * v_des; // TODO introduce try catch
    }
    else if (frame == FrankaDriver::REFERENCE_FRAME)
    {
        Eigen::Matrix<double,6,7> cfJe;
        cfJe = cVe * fJe;
        Eigen::Matrix<double,7,6> icfJe;
        pseudoInverseEigen3(icfJe,cfJe);


        Eigen::Matrix<double,6,1> v_des;
        for(int ll = 0; ll < 6 ;ll++)
        {
            v_des[ll] = v_cart_des[ll];
        }

        q_dot = icfJe * v_des; // TODO introduce try catch
    }


    std::array<double, 7> dq_des_eigen;
    for (size_t i = 0; i < 7; i++) // TODO create a function to convert
      dq_des_eigen[i] = q_dot[i];

    std::array<double, 7> q_cmd;
    std::array<double, 7> dq_cmd;

    auto dq_des_ = dq_des_eigen;
    if (stop)
    { // Stop asked
      for (auto & dq_ : dq_des_)
      {
        dq_ = 0.0;
      }
    }

    joint_vel_traj_generator.applyVel(dq_des_, q_cmd, dq_cmd);

    franka::JointVelocities velocities = {dq_cmd[0], dq_cmd[1], dq_cmd[2], dq_cmd[3], dq_cmd[4], dq_cmd[5], dq_cmd[6]};

//    std::cout<<dq_des[0]<<dq_des[1]<<dq_des[2]<<dq_des[3]<<dq_des[4]<<std::endl;
//    std::cout<<dq_des_eigen[0]<<dq_des_eigen[1]<<dq_des_eigen[2]<<dq_des_eigen[3]<<dq_des_eigen[4]<<dq_des_eigen[5]<<std::endl;
    if (stop)
    {
      unsigned int nb_joint_stop = 0;
      const double q_eps = 1e-6; // Motion finished
      for(size_t i=0; i < 7; i++)
      {
        if (std::abs(state.q_d[i] - q_prev[i]) < q_eps)
        {
          nb_joint_stop ++;
        }
      }
      if (nb_joint_stop == 7)
      {
        return franka::MotionFinished(velocities);
      }
    }

    q_prev = state.q_d;

    // state.q_d contains the last joint velocity command received by the robot.
    // In case of packet loss due to bad connection or due to a slow control loop
    // not reaching the 1kHz rate, even if your desired velocity trajectory
    // is smooth, discontinuities might occur.
    // Saturating the acceleration computed with respect to the last command received
    // by the robot will prevent from getting discontinuity errors.
    // Note that if the robot does not receive a command it will try to extrapolate
    // the desired behavior assuming a constant acceleration model
    return limitRate(ddq_max, velocities.dq, state.dq_d);

  };


  switch (frame)
  {
  case FrankaDriver::JOINT_STATE:
  {
      int nbAttempts = 10;
      for (int attempt = 1; attempt <= nbAttempts; attempt++)
      {
          try
          {

              robot->control(joint_velocity_callback);

              break;
          }
          catch (const franka::ControlException &e)
          {
              std::cerr << "Warning: communication error: " << e.what() << "\nRetry attempt: " << attempt << std::endl;
              robot->automaticErrorRecovery();
              if (attempt == nbAttempts)
                  throw;
          }
      }
      break;
  }
  case FrankaDriver::REFERENCE_FRAME:
  case FrankaDriver::END_EFFECTOR_FRAME:
  {
      int nbAttempts = 10;
      for (int attempt = 1; attempt <= nbAttempts; attempt++)
      {
          try
          {
              std::cout<<"in"<<std::endl;
              robot->control(cartesian_velocity_callback);

              break;
          }
          catch (const franka::ControlException &e)
          {
              std::cerr << "Warning: communication error: " << e.what() << "\nRetry attempt: " << attempt << std::endl;
              robot->automaticErrorRecovery();
              if (attempt == nbAttempts)
                  throw;
          }
      }
      break;
  }
  }
}

void VelTrajGenerator::init(const std::array<double, 7> &q,
                                   const std::array<double, 7> &q_min,
                                   const std::array<double, 7> &q_max,
                                   const std::array<double, 7> &dq_max,
                                   const std::array<double, 7> &ddq_max,
                                   double delta_t)
{
  if (m_njoints != q_min.size() || m_njoints != q_max.size()
      || m_njoints != dq_max.size() || m_njoints != ddq_max.size())
  {
      std::exception e;
      std::cerr<<"Inconsistent number of joints! "<<std::endl;
      throw e;
  }
  m_q_min = q_min;
  m_q_max = q_max;
  m_dq_max = dq_max;
  m_ddq_max = ddq_max;

  m_delta_t = delta_t;

  m_q_final = m_q_cmd = q;
  m_q_cmd_prev = m_q_cmd;

  m_dq_des        = {0, 0, 0, 0, 0, 0, 0};
  m_dq_des_prev   = {0, 0, 0, 0, 0, 0, 0};
  m_delta_q       = {0, 0, 0, 0, 0, 0, 0};
  m_delta_q_max   = {0, 0, 0, 0, 0, 0, 0};
  m_sign          = {0, 0, 0, 0, 0, 0, 0};
  m_dist_to_final = {0, 0, 0, 0, 0, 0, 0};
  m_dist_AD       = {0, 0, 0, 0, 0, 0, 0};
  m_flagSpeed     = {false, false, false, false, false, false, false};
  m_status        = {FLAGSTO, FLAGSTO, FLAGSTO, FLAGSTO, FLAGSTO, FLAGSTO, FLAGSTO};

  for (size_t i=0; i<m_njoints; i++)
  {
    m_delta_q_acc[i] = m_ddq_max[i] * m_delta_t * m_delta_t;
  }
}


/*!
 * Compute the joint position and velocity to reach desired joint velocity.
 * \param dq_des : Desired joint velocity
 * \param q_cmd : Position to apply.
 * \param dq_cmd : Velocity to apply.
 */
void VelTrajGenerator::applyVel(const std::array<double, 7> &dq_des,
                                       std::array<double, 7> &q_cmd,
                                       std::array<double, 7> &dq_cmd)
{
  for (size_t i=0; i < m_njoints; i++)
  {
    m_dq_des[i] = dq_des[i];

    if (m_dq_des[i] != m_dq_des_prev[i])
    {

      m_flagJointLimit = false;

      if (m_dq_des[i] > m_dq_max[i])
      {
        m_dq_des[i] = m_dq_max[i];
      }
      else if (m_dq_des[i] < (-m_dq_max[i]))
      {
        m_dq_des[i] = -m_dq_max[i];
      }

      if (m_flagSpeed[i] == false)
      {
        // Change from stop to new vel with acc control
        if ( m_status[i] == FLAGSTO) // If stop
        {
          if (m_dq_des[i] > 0)
          {
            m_delta_q_max[i] = m_dq_des[i]*m_delta_t;
            m_sign[i] = 1;
            m_q_final[i] = m_q_max[i] - m_offset_joint_limit;
            m_delta_q[i] = 0;
            m_status[i] = FLAGACC;
          }
          else if (m_dq_des[i] < 0)
          {
            m_delta_q_max[i] = - m_dq_des[i]*m_delta_t;
            m_sign[i] = -1;
            m_q_final[i] = m_q_min[i] + m_offset_joint_limit;
            m_delta_q[i] = 0;
            m_status[i] = FLAGACC;
          }
        }

        // Change of direction
        else if ( (m_dq_des[i] * m_sign[i]) < 0)
        {
          m_flagSpeed[i] = true;
          m_status[i] = FLAGDEC;
          m_delta_q_max[i] = 0;
        }
        else
        {
          // Acceleration or deceleration
          if ( m_sign[i] == 1)
          {
            if ( m_dq_des[i] > m_dq_des_prev[i])
              m_status[i] = FLAGACC;
            else
              m_status[i] = FLAGDEC;
            m_delta_q_max[i] = m_dq_des[i]*m_delta_t;
          }
          else
          {
            if ( m_dq_des[i] > m_dq_des_prev[i])
              m_status[i] = FLAGDEC;
            else
              m_status[i] = FLAGACC;
            m_delta_q_max[i] = - m_dq_des[i]*m_delta_t;
          }
        }

        // Update distance to accelerate or decelerate
        int n = (int) (m_delta_q_max[i] / m_delta_q_acc[i]);
        m_dist_AD[i]=n*(m_delta_q_max[i]-(n+1)*m_delta_q_acc[i]/2);
      }
      m_dq_des_prev[i] = m_dq_des[i];
    }
  }

  /*
   * Compute new command in case of
   *		- acceleration
   *		- deceleration
   *		- stop
   */
  for (size_t i=0; i < m_njoints; i++) {
    // Security joint limit
    m_dist_to_final[i] = ( m_q_final[i] - m_q_cmd[i]) * m_sign[i];
    if ((m_dist_to_final[i] - m_delta_q_max[i]) <=  m_dist_AD[i]) {
      if (m_dist_AD[i] > 0) {
        if (!m_flagJointLimit) printf("Joint limit flag axis %lu\n", (unsigned long)i);
        m_flagJointLimit = true;
        for(size_t k=0; k < m_njoints; k++)
        {
          if (m_status[k] != FLAGSTO) m_status[k] = FLAGDEC;
          m_delta_q_max[k] = 0;
        }
      }
    }
    /*
     * Deceleration.
     */
    if ( m_status[i] == FLAGDEC) {
      m_delta_q[i] -=  m_delta_q_acc[i];
      if (m_delta_q[i] <=  m_delta_q_max[i]) {
        if (m_delta_q_max[i] < m_delta_q_min)  {
          m_status[i] = FLAGSTO;
          m_delta_q[i] = 0.0;
          // Test if change of direction
          if (m_flagSpeed[i] == true) {
            if (m_dq_des[i] > 0) {
              m_delta_q_max[i] = m_dq_des[i]*m_delta_t;
              m_sign[i] = 1;
              m_q_final[i] = m_q_max[i] - m_offset_joint_limit;
            }
            else if (m_dq_des[i] < 0) {
              m_delta_q_max[i] = -m_dq_des[i]*m_delta_t;
              m_sign[i] = -1;
              m_q_final[i] = m_q_min[i] + m_offset_joint_limit;
            }
            m_status[i] = FLAGACC;
            m_flagSpeed[i] = false;

            int n = (int) (m_delta_q_max[i] / m_delta_q_acc[i]);
            m_dist_AD[i]=n*(m_delta_q_max[i]-(n+1)*m_delta_q_acc[i]/2);
          }
        }
        else if ((m_delta_q_max[i] > 0) && !m_flagJointLimit)  {
          if (m_delta_q_max[i] < (m_delta_q[i] + 2*m_delta_q_acc[i])) {
            m_delta_q[i] = m_delta_q_max[i];
            m_status[i] = FLAGCTE;
          }
          else if (!m_flagJointLimit) {
            /* acceleration moins rapide*/
            m_delta_q[i] += (2*m_delta_q_acc[i]);
            m_status[i] = FLAGACC;
          }
        }
      }
    }
    /*
     * Acceleration.
     */
    else if (m_status[i] == FLAGACC) {
      m_delta_q[i] += m_delta_q_acc[i];

      if (m_delta_q[i] >= m_delta_q_max[i]) {
        m_delta_q[i] = m_delta_q_max[i];
        m_status[i] = FLAGCTE;
      }
    }
    /*
     * Constant velocity
     */
    m_q_cmd[i] += m_sign[i] * m_delta_q[i];
  }

  // Test si un axe arrive pres des butees. Si oui, arret de tous les axes
  for (size_t i=0; i < m_njoints;i++) {
    double butee = m_q_min[i] + m_offset_joint_limit;
    if (m_q_cmd[i] < butee) {
      for (size_t j=0; j < m_njoints;j++) {
        m_q_cmd[j] -= m_sign[j]*m_delta_q[j];
      }
      m_q_cmd[i] = butee;
      printf("Joint limit axis %lu\n", (unsigned long)i);
      break;
    }
    butee = (float) (m_q_max[i] - m_offset_joint_limit);
    if (m_q_cmd[i] > butee) {
      for (size_t j=0; j < m_njoints; j++) {
        m_q_cmd[j] -= m_sign[j]*m_delta_q[j];
      }
      m_q_cmd[i] = butee;
      printf("Joint limit axis %lu\n", (unsigned long)i);
      break;
    }
  }

  q_cmd = m_q_cmd;

  // Compute velocity command
  for (size_t i = 0; i < m_q_cmd.size(); i++) {
    dq_cmd[i] = (m_q_cmd[i] - m_q_cmd_prev[i]) / m_delta_t;
  }

  m_q_cmd_prev = m_q_cmd;
}

/**
 * Limits the rate of an input vector of per-joint commands considering the maximum allowed time
 * derivatives.
 *
 * @param[in] max_derivatives Per-joint maximum allowed time derivative.
 * @param[in] desired_values Desired values of the current time step.
 * @param[in] last_desired_values Desired values of the previous time step.
 *
 * @return Rate-limited vector of desired values.
 */
std::array<double, 7> VelTrajGenerator::limitRate(const std::array<double, 7>& max_derivatives,
                                                         const std::array<double, 7>& desired_values,
                                                         const std::array<double, 7>& last_desired_values) {
  std::array<double, 7> limited_values{};
  for (size_t i = 0; i < 7; i++)
  {
    double desired_difference = (desired_values[i] - last_desired_values[i]) / 1e-3;
    limited_values[i] =
        last_desired_values[i] +
        std::max(std::min(desired_difference, max_derivatives[i]), -max_derivatives[i]) * 1e-3;
  }
  return limited_values;
}

}
}

