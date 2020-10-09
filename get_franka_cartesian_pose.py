import os
import sys

from franka.FrankaController import FrankaController

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

arm = FrankaController(ROOT + '/config/franka.yaml')

t, euler = arm.getCartesianPose()

print("Current Cartesian Pose: {}, {}".format(t, euler))
