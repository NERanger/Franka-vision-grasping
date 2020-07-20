#include "FrankaDriver.h"
#include "VelTraj.h"
#include <strstream>

namespace Bionic
{
namespace RobotDriver
{

FrankaDriver::FrankaDriver()
    : m_handler(nullptr), m_gripper(nullptr),
      stateRobot(RobotStateType::STATE_STOP),m_velControlThread(),
      m_velControlThreadIsRunning(false), m_velControlThreadStopAsked(false),
      m_dq_des(), m_v_cart_des(),m_robot_state(),m_mutex(),
      m_q_min(), m_q_max(), m_dq_max(), m_ddq_max(), m_positionningVelocity(20.),
      m_max_attempt(5), m_franka_address()
{
    init();
}

/*!
 * Establishes a connection with the robot.
 * \param[in] franka_address IP/hostname of the robot.
 * \param[in] realtime_config If set to kEnforce, an exception will be thrown if realtime priority cannot
 * be set when required. Setting realtime_config to kIgnore disables this behavior.
 */
FrankaDriver::FrankaDriver(std::string franka_address)
    :  m_handler(nullptr), m_gripper(nullptr),
      stateRobot(RobotStateType::STATE_STOP),m_velControlThread(),
      m_velControlThreadIsRunning(false), m_velControlThreadStopAsked(false),
      m_dq_des(), m_v_cart_des(),m_robot_state(),m_mutex(),
      m_q_min(), m_q_max(), m_dq_max(), m_ddq_max(), m_positionningVelocity(20.),
      m_max_attempt(5), m_franka_address()
{
    init();
    connect(franka_address);
}

/*!
 * Initialize internal vars, such as min, max joint positions, max velocity and acceleration.
 */
void FrankaDriver::init()
{

  m_q_min   = std::array<double, 7> {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
  m_q_max   = std::array<double, 7> {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
  m_dq_max  = std::array<double, 7> {2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100};
  m_ddq_max = std::array<double, 7> {15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0};

}

/*!

  Destructor.

*/
FrankaDriver::~FrankaDriver()
{
  if (m_handler)
    delete m_handler;

  if (m_gripper)
  {
    std::cerr << "Grasped object, will release it now." << std::endl;
    m_gripper->stop();
    delete m_gripper;
  }

}

/*!
 * Establishes a connection with the robot and set default behavior.
 * \param[in] franka_address IP/hostname of the robot.
 * \param[in] realtime_config If set to kEnforce, an exception will be thrown if realtime priority cannot
 * be set when required. Setting realtime_config to kIgnore disables this behavior.
 */
void FrankaDriver::connect(std::string franka_address)
{
    init();

    if(franka_address.empty())
    {
        std::exception e;
        std::cerr<<"Cannot connect Franka robot: IP/hostname is not set! "<<std::endl;
        throw e;
    }
    if (m_handler)
        delete m_handler;

    m_franka_address = franka_address;
    m_handler = new franka::Robot(franka_address, franka::RealtimeConfig::kEnforce);

    std::array<double, 7> lower_torque_thresholds_nominal{
        {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.}};
    std::array<double, 7> upper_torque_thresholds_nominal{
        {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0}};
    std::array<double, 7> lower_torque_thresholds_acceleration{
        {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0}};
    std::array<double, 7> upper_torque_thresholds_acceleration{
        {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0}};

    std::array<double, 6> lower_force_thresholds_nominal{{30.0, 30.0, 30.0, 25.0, 25.0, 25.0}};
    std::array<double, 6> upper_force_thresholds_nominal{{40.0, 40.0, 40.0, 35.0, 35.0, 35.0}};
    std::array<double, 6> lower_force_thresholds_acceleration{{30.0, 30.0, 30.0, 25.0, 25.0, 25.0}};
    std::array<double, 6> upper_force_thresholds_acceleration{{40.0, 40.0, 40.0, 35.0, 35.0, 35.0}};

    m_handler->setCollisionBehavior(
                lower_torque_thresholds_acceleration, upper_torque_thresholds_acceleration,
                lower_torque_thresholds_nominal, upper_torque_thresholds_nominal,
                lower_force_thresholds_acceleration, upper_force_thresholds_acceleration,
                lower_force_thresholds_nominal, upper_force_thresholds_nominal);

  m_handler->setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
  m_handler->setCartesianImpedance({{3000, 3000, 3000, 300, 300, 300}});



//  m_handler->setFilters(10, 10, 10, 10, 10);

}


std::array<double,7> FrankaDriver::getJointPos()
{
    if (!m_handler)
    {
        std::exception e;
        std::cerr<<"Robot is not connected! "<<std::endl;
        throw e;
    }

    franka::RobotState robot_state = getRobotInternalState();

    std::array<double,7> q {};
    for (int i=0; i < 7; i++)
    {
        q[i] = robot_state.q[i];
    }
    return q;
}



FrankaDriver::RobotStateType FrankaDriver::setRobotState(RobotStateType newState)
{
  switch (newState)
  {
  case STATE_STOP:
  {
    // Start primitive STOP only if the current state is velocity or force/torque
    if (STATE_VELOCITY_CONTROL == getRobotState())
    {
      // Stop the robot
      m_velControlThreadStopAsked = true;
      if(m_velControlThread.joinable())
      {
        m_velControlThread.join();
        m_velControlThreadStopAsked = false;
        m_velControlThreadIsRunning = false;
      }
    }
    break;
  }
  case STATE_POSITION_CONTROL:
  {
    if (STATE_VELOCITY_CONTROL == getRobotState())
    {
      std::cout << "Change the control mode from velocity to position control.\n";
      // Stop the robot
      m_velControlThreadStopAsked = true;
      if(m_velControlThread.joinable())
      {
        m_velControlThread.join();
        m_velControlThreadStopAsked = false;
        m_velControlThreadIsRunning = false;
      }
    }
    break;
  }
  case STATE_VELOCITY_CONTROL:
  {
    if (STATE_STOP == getRobotState())
    {
      std::cout << "Change the control mode from stop to velocity control.\n";
    }
    else if (STATE_POSITION_CONTROL == getRobotState())
    {
      std::cout << "Change the control mode from position to velocity control.\n";
    }

    break;
  }

  default:
    break;
  }

  stateRobot = newState;

  return newState;
}







std::array<double, 16> FrankaDriver::getO_T_EE()
{
    if (!m_handler)
    {
        std::exception e;
        std::cerr<<"Robot is not connected! "<<std::endl;
        throw e;
    }

    franka::RobotState robot_state = getRobotInternalState();

    std::array<double,16> out {};
    for (int i=0; i < 16; i++)
    {
        out[i] = robot_state.O_T_EE[i];
    }
    return out;
}


std::array<double, 16> FrankaDriver::getF_T_EE()
{
    if (!m_handler)
    {
        std::exception e;
        std::cerr<<"Robot is not connected! "<<std::endl;
        throw e;
    }

    franka::RobotState robot_state = getRobotInternalState();

    std::array<double,16> out {};
    for (int i=0; i < 16; i++)
    {
        out[i] = robot_state.F_T_EE[i];
    }
    return out;

}




void FrankaDriver::setMaxAttempt(int attemp)
{
    m_max_attempt = attemp;
}

/*!
 * Set robot position. This function is blocking; it returns when the desired position is reached.
 * \param[in] position : This is a 7-dim vector that corresponds to the robot joint positions expressed in rad.
 */
void FrankaDriver:: setJointPos(std::array<double,7> position)
{
    if (!m_handler)
    {
        std::exception e;
        std::cerr<<"Robot is not connected! "<<std::endl;
        throw e;
    }
    if (STATE_POSITION_CONTROL != getRobotState())
    {
      std::cout << "Robot now in position-based control. "<< std::endl;
      setRobotState(STATE_POSITION_CONTROL);
    }

    double speed_factor = m_positionningVelocity / 100.;

    std::array<double, 7> q_goal {};
    for (size_t i = 0; i < 7; i++)
    {
        q_goal[i] = position[i];
    }

    JointPosTrajGenerator joint_pos_traj_generator(speed_factor, q_goal);

    int nbAttempts = m_max_attempt;
    for (int attempt = 1; attempt <= nbAttempts; attempt++)
    {
        try
        {
            m_handler->control(joint_pos_traj_generator);
            break;
        }
        catch (const franka::ControlException &e)
        {
            std::cerr << "Warning: communication error: " << e.what() << "\nRetry attempt: " << attempt << std::endl;
            m_handler->automaticErrorRecovery();
            if (attempt == nbAttempts)
                throw;
        }
    }
}


void FrankaDriver::setJointVelocity(std::array<double,7> vel_j)
{
    setRobotState(STATE_VELOCITY_CONTROL);

    if (STATE_VELOCITY_CONTROL != getRobotState())
    {
        std::exception e;
        std::cerr<<"Cannot send a velocity to the robot! "<<std::endl;
        throw e;
    }

    // Saturation in joint space
    if (vel_j.size() != 7)
    {
        std::exception e;
        std::cerr<<"Joint velocity vector is not of size 7! "<<std::endl;
        throw e;
    }


    for (size_t i = 0; i < m_dq_des.size(); i++)
    { // TODO create a function to convert
        m_dq_des[i] = fabs(vel_j[i]) < fabs(m_dq_max[i]) ? vel_j[i] : m_dq_max[i];
    }

    if(! m_velControlThreadIsRunning)
    {
      Eigen::Affine3d m_eMc;
      m_eMc.setIdentity();
      m_velControlThreadIsRunning = true;
      m_velControlThread = std::thread(&VelTrajGenerator::control_thread, VelTrajGenerator(),
                                    std::ref(m_handler), std::ref(m_velControlThreadStopAsked),
                                    JOINT_STATE, m_eMc, std::ref(m_v_cart_des), std::ref(m_dq_des),
                                    std::cref(m_q_min), std::cref(m_q_max), std::cref(m_dq_max), std::cref(m_ddq_max),
                                    std::ref(m_robot_state), std::ref(m_mutex));

    }

}

void FrankaDriver::setCartesianVelocity(std::array<double,6> vel_c, std::string ref)
{
    setRobotState(STATE_VELOCITY_CONTROL);

    if (STATE_VELOCITY_CONTROL != getRobotState())
    {
        std::exception e;
        std::cerr<<"Cannot send a velocity to the robot! "<<std::endl;
        throw e;
    }

    ControlFrameType frame;
    frame = END_EFFECTOR_FRAME;
      // Saturation in cartesian space
    if(ref == "base")
    {
        if (vel_c.size() != 6)
        {
            std::exception e;
            std::cerr<<"Cartesian velocity vector is not of size 6! "<<std::endl;
            throw e;
        }


        for (size_t i = 0; i < 3; i++)
        { // TODO create a function to convert
            m_v_cart_des[i] = fabs(vel_c[i]) < 0.2 ? vel_c[i] : 0.2;
        }

        for (size_t i = 3; i < 6; i++)
        {
            m_v_cart_des[i] = fabs(vel_c[i]) < 0.7 ? vel_c[i] : 0.7;
        }

        frame = REFERENCE_FRAME;
    }

    if(ref == "ee")
    {

        if (vel_c.size() != 6)
        {
            std::exception e;
            std::cerr<<"Cartesian velocity vector is not of size 6! "<<std::endl;
            throw e;
        }


        for (size_t i = 0; i < 3; i++)
        { // TODO create a function to convert
            m_v_cart_des[i] = fabs(vel_c[i]) < 0.2 ? vel_c[i] : 0.2;
        }

        for (size_t i = 3; i < 6; i++)
        {
            m_v_cart_des[i] = fabs(vel_c[i]) < 0.7 ? vel_c[i] : 0.7;
        }

        frame = END_EFFECTOR_FRAME;

    }


    if(! m_velControlThreadIsRunning)
    {
        Eigen::Affine3d m_eMc;
        m_eMc.setIdentity();
      m_velControlThreadIsRunning = true;
      m_velControlThread = std::thread(&VelTrajGenerator::control_thread, VelTrajGenerator(),
                                    std::ref(m_handler), std::ref(m_velControlThreadStopAsked),
                                    frame, m_eMc, std::ref(m_v_cart_des), std::ref(m_dq_des),
                                    std::cref(m_q_min), std::cref(m_q_max), std::cref(m_dq_max), std::cref(m_ddq_max),
                                    std::ref(m_robot_state), std::ref(m_mutex));
    }


}


/*!

  Set the maximal velocity percentage to use for a position control.

  \param[in] velocity : Percentage of the maximal velocity. Values should
  be in ]0:100].

*/
void FrankaDriver::setPositioningVelocity(double velocity)
{
    m_positionningVelocity = velocity;
}



franka::RobotState FrankaDriver::getRobotInternalState()
{
    if (!m_handler)
    {
        std::exception e;
        std::cerr<<"Robot is not connected! "<<std::endl;
        throw e;
    }

    franka::RobotState robot_state;


    if (! m_velControlThreadIsRunning )
    {
        robot_state = m_handler->readOnce();

        std::lock_guard<std::mutex> lock(m_mutex);
        m_robot_state = robot_state;
    }
    else { // robot_state is updated in the velocity control thread
        std::lock_guard<std::mutex> lock(m_mutex);
        robot_state = m_robot_state;
    }

    return robot_state;
}

/*!
  Gets minimal joint values.
  \return A 7-dimension vector that contains the minimal joint values for the 7 dof.
  All the values are expressed in radians.
 */
std::array<double,7> FrankaDriver::getJointMin()
{
    std::array<double,7> q_min {};
  for (size_t i = 0; i < m_q_min.size(); i ++)
    q_min[i] = m_q_min[i];

  return q_min;
}
/*!
  Gets maximum joint values.
  \return A 7-dimension vector that contains the maximum joint values for the 7 dof.
  All the values are expressed in radians.
 */
std::array<double,7> FrankaDriver::getJointMax()
{
    std::array<double,7> q_max {};
  for (size_t i = 0; i < m_q_max.size(); i ++)
    q_max[i] = m_q_max[i];

  return q_max;
}

/*!

  Performing a gripper homing.

  \sa gripperGrasp(), gripperMove(), gripperRelease()
*/
void FrankaDriver::gripperHoming()
{
  if (m_franka_address.empty())
  {
      std::exception e;
      std::cerr<<"Cannot connect Franka robot: IP/hostname is not set! "<<std::endl;
      throw e;
  }
  if (m_gripper == nullptr)
    m_gripper = new franka::Gripper(m_franka_address);

  m_gripper->homing();
}

/*!

  Moves the gripper fingers to a specified width.
  @param[in] width : Intended opening width. [m]

  \return EXIT_SUCCESS if the success, EXIT_FAILURE otherwise.

  \sa gripperHoming(), gripperGrasp(), gripperRelease()
*/
int FrankaDriver::gripperMove(double width)
{
  if (m_franka_address.empty())
  {
      std::exception e;
      std::cerr<<"Cannot connect Franka robot: IP/hostname is not set! "<<std::endl;
      throw e;
  }
  if (m_gripper == nullptr)
    m_gripper = new franka::Gripper(m_franka_address);

  // Check for the maximum grasping width.
  franka::GripperState gripper_state = m_gripper->readOnce();

  if (gripper_state.max_width < width) {
    std::cout << "Finger width request is too large for the current fingers on the gripper."
              << "Maximum possible width is " << gripper_state.max_width << std::endl;
    return EXIT_FAILURE;
  }

  m_gripper->move(width, 0.1);
  return EXIT_SUCCESS;
}

/*!

  Closes the gripper.

  \return EXIT_SUCCESS if the success, EXIT_FAILURE otherwise.

  \sa gripperHoming(), gripperGrasp(), gripperRelease(), gripperOpen()
*/
int FrankaDriver::gripperClose()
{
  return gripperMove(0);
}

/*!

  Closes the gripper.

  \return EXIT_SUCCESS if the success, EXIT_FAILURE otherwise.

  \sa gripperHoming(), gripperGrasp(), gripperRelease(), gripperOpen()
*/
int FrankaDriver::gripperOpen()
{
  if (m_franka_address.empty())
  {
      std::exception e;
      std::cerr<<"Cannot connect Franka robot: IP/hostname is not set! "<<std::endl;
      throw e;
  }
  if (m_gripper == nullptr)
    m_gripper = new franka::Gripper(m_franka_address);

  // Check for the maximum grasping width.
  franka::GripperState gripper_state = m_gripper->readOnce();

  m_gripper->move(gripper_state.max_width, 0.1);
  return EXIT_SUCCESS;
}

/*!

  Release an object that is grasped.

  \sa gripperHoming(), gripperMove(), gripperRelease()
*/
void FrankaDriver::gripperRelease()
{
    if (m_franka_address.empty())
    {
        std::exception e;
        std::cerr<<"Cannot connect Franka robot: IP/hostname is not set! "<<std::endl;
        throw e;
    }
    if (m_gripper == nullptr)
        m_gripper = new franka::Gripper(m_franka_address);

    m_gripper->stop();
}

/*!

  Grasp an object that has a given width.

  An object is considered grasped if the distance \e d between the gripper fingers satisfies
  \e grasping_width - 0.005 < d < \e grasping_width + 0.005.

  \param[in] grasping_width : Size of the object to grasp. [m]
  \param[in] force : Grasping force. [N]

  \return EXIT_SUCCESS if grasping succeed, EXIT_FAILURE otherwise.

  \sa gripperHoming(), gripperOpen(), gripperRelease()
*/
int FrankaDriver::gripperGrasp(double grasping_width, double force)
{
    if (m_gripper == nullptr)
        m_gripper = new franka::Gripper(m_franka_address);

    // Check for the maximum grasping width.
    franka::GripperState gripper_state = m_gripper->readOnce();
    std::cout << "Gripper max witdh: " << gripper_state.max_width << std::endl;
    if (gripper_state.max_width < grasping_width) {
        std::cout << "Object is too large for the current fingers on the gripper."
                  << "Maximum possible width is " << gripper_state.max_width << std::endl;
        return EXIT_FAILURE;
    }

    // Grasp the object.
    if (!m_gripper->grasp(grasping_width, 0.1, force,0.05,0.05)) {
        std::cout << "Failed to grasp object." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

std::string FrankaDriver::getAllState()
{
    franka::RobotState state = getRobotInternalState();
    std::stringstream ss;
    ss<<state;
    return ss.str();

}

void FrankaDriver::stopMotion()
{
  if (getRobotState() == STATE_VELOCITY_CONTROL)
  {
    std::array<double,7> q{0,0,0,0,0,0,0};
    setJointVelocity(q);
  }
  setRobotState(STATE_STOP);
}

}

}



























//int main(int argc, char *argv[])
//{
//    Vector7d i(7);

//    i<<1,2,3,4,5,6,7;

//   std::cout<< getmax(i)<<std::endl;

//    return 0;
//}
