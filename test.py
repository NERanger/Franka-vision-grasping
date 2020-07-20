import time
import sys
import os
import numpy as np

from franka.FrankaController import FrankaController

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

FC = FrankaController(ROOT + '/config/franka.yaml')

print("Joint Pos:")
print(FC.getJoint())

print("Cartesian Pose:")
print(FC.getCartesianPose())

allState = FC.getState()
print("All State:")
print(allState['F_T_EE'])
print(allState['O_T_EE'])

print("------Start Move Test------")

print("Move to joint target:")
joint_target = np.array([-0.0137566,0.0150639,0.06416,-2.50988,-0.00736516,2.80153,-1.8411])
print(joint_target)
print("Executing...")
FC.move_j(joint_target)
print("Executing... Done")

print("Move to pose target:")
pose = [0.5,0,0.4,3.14,0.0,0.0]
print(pose)
print("Executing...")
FC.move_p(pose)
print("Executing... Done")

# Load model library failed when performing joint speed control test
# speed_j 
# print("Joint speed control:")
# joint_speed = [0,0,0,0,0,0,0.1]
# print(joint_speed)
# print("Executing...")
# FC.speed_j(joint_speed)
# time.sleep(2)
# FC.stopSpeed()
# print("Executing... Done")

print("Mutiple pose target:")
print("Executing...")
FC.move_p([0.5,0,0.3,3.14,0.0,0.0])
FC.move_p([0.6,0,0.4,3.14,0.0,1.0])
FC.move_p([0.6,0.3,0.2,3.14,0.0,1.0])
print("Executing... Done")
