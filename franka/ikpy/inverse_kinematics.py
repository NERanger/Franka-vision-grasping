# coding= utf8
import scipy.optimize
import numpy as np
import math
from . import logs


def inverse_kinematic_optimization(chain, target_frame, starting_nodes_angles, regularization_parameter=None, max_iter=None):
    """
    Computes the inverse kinematic on the specified target with an optimization method

    Parameters
    ----------
    chain: ikpy.chain.Chain
        The chain used for the Inverse kinematics.
    target_frame: numpy.array
        The desired target.
    starting_nodes_angles: numpy.array
        The initial pose of your chain.
    regularization_parameter: float
        The coefficient of the regularization.
    max_iter: int
        Maximum number of iterations for the optimisation algorithm.
    """
    # # Only get the position
    # target = target_frame[:3, 3]
    # get the orientation and postion
    target_orientation = target_frame[0:3, 0:3]
    target_position = target_frame[:3, 3]

    if starting_nodes_angles is None:
        raise ValueError("starting_nodes_angles must be specified")

    # Compute squared distance to target
    # def optimize_target(x):
    #     # y = np.append(starting_nodes_angles[:chain.first_active_joint], x)
    #     y = chain.active_to_full(x, starting_nodes_angles)
    #     squared_distance = np.linalg.norm(chain.forward_kinematics(y)[:3, -1] - target_position)
    #     return squared_distance
        # Compute squared distance to target
    def optimize_target(x):
        # y = np.append(starting_nodes_angles[:chain.first_active_joint], x)
        y = chain.active_to_full(x, starting_nodes_angles)
        posi = np.linalg.norm(chain.forward_kinematics(y)[:3, -1] - target_position)

        iter_rotation = chain.forward_kinematics(y)[0:3, 0:3]
        diff_r = np.dot(iter_rotation,np.linalg.pinv(target_orientation))
        angle_error = math.acos((np.trace(diff_r)-1)/2)
        angle_error = np.linalg.norm(angle_error)

        squared_distance = 0.8*posi + 0.2*angle_error
        return squared_distance


    # If a regularization is selected
    if regularization_parameter is not None:
        def optimize_total(x):
            regularization = np.linalg.norm(x - starting_nodes_angles[chain.first_active_joint:])
            return optimize_target(x) + regularization_parameter * regularization
    else:
        def optimize_total(x):
            return optimize_target(x)

    # Compute bounds
    real_bounds = [link.bounds for link in chain.links]
    # real_bounds = real_bounds[chain.first_active_joint:]
    real_bounds = chain.active_from_full(real_bounds)

    options = {}
    # Manage iterations maximum
    if max_iter is not None:
        options["maxiter"] = max_iter

    # Utilisation d'une optimisation L-BFGS-B
    res = scipy.optimize.minimize(optimize_total, chain.active_from_full(starting_nodes_angles), method='L-BFGS-B', bounds=real_bounds, options=options)

    # # optimize orientation
    # # Compute squared distance to target
    # def optimize_target(x):
    #     # y = np.append(starting_nodes_angles[:chain.first_active_joint], x)
    #     y = chain.active_to_full(x, starting_nodes_angles)
    #     iter_rotation = chain.forward_kinematics(y)[0:3, 0:3]
    #     diff_r = np.dot(iter_rotation,np.linalg.inv(target_orientation))
    #     angle_error = math.acos((np.trace(diff_r)-1)/2)
    #     squared_distance = np.linalg.norm(angle_error)
    #     return squared_distance
    #
    # # If a regularization is selected
    # if regularization_parameter is not None:
    #     def optimize_total(x):
    #         regularization = np.linalg.norm(x - starting_nodes_angles[chain.first_active_joint:])
    #         return optimize_target(x) + regularization_parameter * regularization
    # else:
    #     def optimize_total(x):
    #         return optimize_target(x)
    # # Utilisation d'une optimisation L-BFGS-B
    # res = scipy.optimize.minimize(optimize_total, chain.active_from_full(res_posi.x), method='L-BFGS-B', bounds=real_bounds, options=options)



    logs.logger.info("Inverse kinematic optimisation OK, done in {} iterations".format(res.nit))

    return chain.active_to_full(res.x, starting_nodes_angles)
    # return(np.append(starting_nodes_angles[:chain.first_active_joint], res.x))
