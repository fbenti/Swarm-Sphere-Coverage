""" Helper function for the crazyswarm package """
# External
from pycrazyswarm import Crazyswarm
from pycrazyswarm.crazyflie import TimeHelper
from crazyswarm.msg import GenericLogData
import rospy
from math import cos,sin
import numpy as np

# Internal

HOVER_HEIGHT = 0.3


def takeoff(swarm : Crazyswarm, timeHelper: TimeHelper, groupMask: int = 0):
    """ Swarm Take-Off """
    for cf in swarm.crazyflies:
        cf.takeoff(targetHeight=HOVER_HEIGHT, duration=1.0+1, groupMask=groupMask)
    timeHelper.sleep(1.5+1)

def land(swarm : Crazyswarm, timeHelper: TimeHelper, groupMask: int = 0):
    """ Swarm Land """
    for cf in swarm.crazyflies:
        cf.land(targetHeight=0.02, duration=2.5, groupMask=groupMask)
    timeHelper.sleep(1.5+1)


""" Helper function for general purpose """
def posCallback(data,cf_prefix):
    print(cf_prefix)
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.values)

def posEst_subscriber(cf_prefix=None):
    """ Wait for pose message """
    if cf_prefix is None:
        print("Error - topic name not defined")
    node_name = 'listener_to_' + cf_prefix
    # rospy.init_node(node_name)
    topic_name = '/' + cf_prefix + '/posEst'
    msg = rospy.wait_for_message(topic_name, GenericLogData, timeout=2)
    print(cf_prefix, msg.values)
    # rospy.Subscriber(topic_name, GenericLogData, posCallback,cf_prefix)
    # rospy.spinOnce()

# def Jack(p,c):
#     phi = p[0]
#     s = p[1:3]
#     # t = p[3:]

#     J = np.array([
#         [-sin(phi)*s[0]*c[0] - cos(phi)*s[1]*c[1], cos(phi)*c[0], -sin(phi)*c[1], 1, 0],
#         [cos(phi)*s[0]*c[0] - sin(phi)*s[1]*c[1], sin(phi)*c[0],  cos(phi)*c[1], 0, 1]])
#     return J