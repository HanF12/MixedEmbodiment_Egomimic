#!/usr/bin/env python
# This line tells the OS to execute the script using the python interpreter

import rospy
# Import the JointState message type
from sensor_msgs.msg import JointState
import numpy as np

sampling_interval = 30 # Currently at 30Hz

# Global variable to store the last received joint state message
last_joint_state_msg = None

def slave_left_joint_states_callback(msg):
    """
    Callback function for receiving robot arm joint state feedback for slave_left.
    Samples the data and stores it in an HDF5 file.
    """
    global last_joint_state_msg # Declare that we are using the global variable
    last_joint_state_msg = msg # Store the entire message

    current_time = msg.header.stamp.to_sec() # Get the timestamp of the message in seconds
    # You can process msg.position and msg.velocity here if needed
    # msg.position
    # msg.velocity


def get_current_slave_left_positions():
    """
    Returns the joint positions from the last received JointState message for slave_left.
    Returns None if no message has been received yet.
    """
    global last_joint_state_msg
    if last_joint_state_msg is not None:
        return (last_joint_state_msg.position,last_joint_state_msg.velocity)
    else:
        rospy.logwarn("No slave_left joint state message received yet.")
        return None

def joint_state_listener(topic: str = "/joint_states_slave_left"):
    """
    Register the subscriber used by inference.

    Note: this does NOT call `rospy.init_node()` and does NOT block/spin.
    """
    rospy.Subscriber(topic, JointState, slave_left_joint_states_callback)
    rospy.loginfo(f"Subscriber created for topic: {topic}")