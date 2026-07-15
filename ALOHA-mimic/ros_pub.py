#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
from typing import Iterable, Optional, Sequence, Union

import numpy as np

_arm_pub: Optional[rospy.Publisher] = None
_gripper_pub: Optional[rospy.Publisher] = None
_gripper_msg_type = None
_joint_names: Sequence[str] = tuple(f"joint{i}" for i in range(1, 7))
_frame_id: str = "world"


def init_publishers(
    topic_arm: str = "/arm_joint_target_position",
    topic_gripper: Optional[str] = None,
    *,
    joint_names: Optional[Sequence[str]] = None,
    frame_id: str = "world",
    queue_size: int = 10,
):
    """
    Create ROS publishers used by inference.

    Note: this does NOT call `rospy.init_node()`; do that in the top-level script.
    """
    global _arm_pub, _gripper_pub, _joint_names, _frame_id, _gripper_msg_type
    _joint_names = joint_names if joint_names is not None else _joint_names
    _frame_id = frame_id
    _arm_pub = rospy.Publisher(topic_arm, JointState, queue_size=queue_size)
    _gripper_pub = None
    _gripper_msg_type = None
    if topic_gripper:
        try:
            from signal_arm.msg import gripper_position_control  # type: ignore

            _gripper_msg_type = gripper_position_control
            _gripper_pub = rospy.Publisher(topic_gripper, gripper_position_control, queue_size=queue_size)
        except Exception as e:
            rospy.logwarn(
                f"Gripper publisher disabled: could not import signal_arm/gripper_position_control ({e}). "
                f"Topic was {topic_gripper!r}."
            )


def publish_joint_positions(
    positions: Union[Sequence[float], np.ndarray],
    *,
    joint_names: Optional[Sequence[str]] = None,
):
    """Publish a single step. If 7D, split arm(6) + gripper(1)."""
    if _arm_pub is None:
        raise RuntimeError("Publishers not initialized. Call init_publishers() first.")

    pos = np.asarray(positions, dtype=np.float32).reshape(-1)
    arm_pos = pos[:6] if pos.shape[0] >= 6 else pos

    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = _frame_id
    msg.name = list(joint_names if joint_names is not None else _joint_names)
    msg.position = arm_pos.tolist()
    _arm_pub.publish(msg)

    # Optional gripper stroke (7th element).
    if _gripper_pub is not None and _gripper_msg_type is not None and pos.shape[0] >= 7:
        grip = float(pos[6])
        grip_msg = _gripper_msg_type()
        grip_msg.header.stamp = msg.header.stamp
        grip_msg.header.frame_id = _frame_id
        grip_msg.gripper_stroke = grip
        _gripper_pub.publish(grip_msg)

def publish_trajectory(
    trajectory: Union[np.ndarray, Iterable[Sequence[float]]],
    *,
    joint_names: Optional[Sequence[str]] = None,
):
    """
    Backwards-compatible helper used by existing inference scripts.

    Accepts either:
    - a single joint vector shaped (J,) or (1, J), or
    - an iterable of joint vectors shaped (J,)
    """
    arr = np.asarray(trajectory, dtype=np.float32)
    if arr.ndim == 1:
        publish_joint_positions(arr, joint_names=joint_names)
        return
    if arr.ndim == 2 and arr.shape[0] == 1:
        publish_joint_positions(arr[0], joint_names=joint_names)
        return

    for step in arr:
        if rospy.is_shutdown():
            break
        publish_joint_positions(step, joint_names=joint_names)

# ---------------------------------------------------------------------------#
# Example usage (uncomment to run directly):
#
# if __name__ == '__main__':
#     import numpy as np
#     # 1-second, 250-step trajectory that drives joint1 from 0 → 0.5 rad
#     steps = 250
#     traj = np.zeros((steps, 6))
#     traj[:, 0] = np.linspace(0.0, 0.5, steps)
#     try:
#         publish_trajectory(traj, rate_hz=250)
#     except rospy.ROSInterruptException:
#         pass
