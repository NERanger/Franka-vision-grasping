import yaml
import os
import sys
import time

import numpy as np
import cv2 as cv

from franka.FrankaController import FrankaController

def read_cfg(path):
    with open(path, 'r') as stream:
        out = yaml.safe_load(stream)
    return out

if __name__ == '__main__':

    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(ROOT)

    cfg = read_cfg(ROOT + '/config/grasping _colorseg.yaml')

    arm = FrankaController(ROOT + '/config/franka.yaml')

    # grasping config
    initial_pose = cfg['initial_position']
    initial_pose[2] -= 0.3
    check_position = cfg['check_position']
    drop_position = cfg['drop_position']

    grasp_pre_offset = cfg['grasp_prepare_offset']
    effector_offset = cfg['effector_offset']

    check_threshold = cfg['check_threshold']

    attmp_num = cfg['attmp_num']

    print("Moving to initial position...")
    arm.move_p(initial_pose)
    print("Moving to initial position... Done")

    stored_exception = None

    arm.move_p(initial_pose)

    current_num = 0
    while current_num < attmp_num:
        try:
            if stored_exception:
                break

            target_in_base = drop_position.copy()
            target_in_base[2] -= 0.37

            prepare_pos = [target_in_base[0], target_in_base[1], target_in_base[2] + grasp_pre_offset + effector_offset, 3.14, 0, 0]
            arm.move_p(prepare_pos)

            arm.gripperOpen()
            arm.move_p([target_in_base[0], target_in_base[1], target_in_base[2] + effector_offset, 3.14, 0, 0])
            arm.gripperGrasp(width=0.05, force=2)
            time.sleep(0.5)

            # Move to check position
            # arm.move_p(check_position)
            arm.move_p(initial_pose)
            
            # Move to drop position and drop object
            arm.move_p(drop_position)
            arm.gripperOpen()

            # Back to initial position
            arm.move_p(initial_pose)

            current_num += 1

        except KeyboardInterrupt:
            stored_exception = sys.exc_info()

    cv.destroyAllWindows()
