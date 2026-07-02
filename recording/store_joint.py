#!/usr/bin/env python
"""Record host left + right joint states to HDF5 and npy."""

import argparse
import os
from datetime import datetime

import h5py
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from recording_paths import under_recording

SAMPLING_FREQUENCY = 30.0
SAMPLING_INTERVAL = 1.0 / SAMPLING_FREQUENCY

BASE_OUT_DIR = under_recording("joint-data")
NPY_DIR = os.path.join(BASE_OUT_DIR, "npy")
os.makedirs(NPY_DIR, exist_ok=True)


class ArmRecorder:
    def __init__(
        self,
        label: str,
        joint_states_topic: str,
        hdf5_filename: str,
        position_npy_name: str,
        timestamp_npy_name: str,
    ) -> None:
        self.label = label
        self.joint_states_topic = joint_states_topic
        self.hdf5_path = os.path.join(os.path.expanduser("~"), "ros_data", hdf5_filename)
        self.position_npy_name = position_npy_name
        self.timestamp_npy_name = timestamp_npy_name

        self.hdf5_file = None
        self.positions_dset = None
        self.velocities_dset = None
        self.timestamps_dset = None
        self.num_joints = 0
        self.last_sampled_time = 0.0
        self.time_arr: list[float] = []
        self.position_arr: list = []

    def open_hdf5(self) -> bool:
        data_dir = os.path.dirname(self.hdf5_path)
        os.makedirs(data_dir, exist_ok=True)
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, "w")
            rospy.loginfo(f"[{self.label}] HDF5 opened: {self.hdf5_path}")
            return True
        except Exception as exc:
            rospy.logerr(f"[{self.label}] Error opening HDF5: {exc}")
            self.hdf5_file = None
            return False

    def callback(self, msg: JointState) -> None:
        current_time = msg.header.stamp.to_sec()
        if current_time - self.last_sampled_time < SAMPLING_INTERVAL:
            return
        self.last_sampled_time = current_time

        if self.hdf5_file is None:
            return

        if len(msg.position) != len(msg.velocity) or (
            self.num_joints > 0 and len(msg.position) != self.num_joints
        ):
            rospy.logwarn(
                f"[{self.label}] Inconsistent JointState lengths "
                f"({len(msg.position)} pos, {len(msg.velocity)} vel). Skipping."
            )
            return

        if self.num_joints == 0:
            self.num_joints = len(msg.position)
            rospy.loginfo(f"[{self.label}] Detected {self.num_joints} joints.")
            try:
                self.positions_dset = self.hdf5_file.create_dataset(
                    "positions",
                    shape=(0, self.num_joints),
                    maxshape=(None, self.num_joints),
                    dtype="float32",
                    chunks=True,
                )
                self.velocities_dset = self.hdf5_file.create_dataset(
                    "velocities",
                    shape=(0, self.num_joints),
                    maxshape=(None, self.num_joints),
                    dtype="float32",
                    chunks=True,
                )
                self.timestamps_dset = self.hdf5_file.create_dataset(
                    "timestamps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float64",
                    chunks=True,
                )
                self.hdf5_file.create_dataset(
                    "joint_names", data=np.array(msg.name, dtype="S")
                )
            except Exception as exc:
                rospy.logerr(f"[{self.label}] Error creating HDF5 datasets: {exc}")
                self.close()
                return

        try:
            current_size = self.timestamps_dset.shape[0]
            self.positions_dset.resize((current_size + 1, self.num_joints))
            self.velocities_dset.resize((current_size + 1, self.num_joints))
            self.timestamps_dset.resize((current_size + 1,))

            self.positions_dset[current_size, :] = msg.position
            self.velocities_dset[current_size, :] = msg.velocity
            self.timestamps_dset[current_size] = current_time
            self.time_arr.append(current_time)
            self.position_arr.append(msg.position)
        except Exception as exc:
            rospy.logerr(f"[{self.label}] Error appending HDF5 data: {exc}")
            self.close()

    def close(self) -> None:
        if self.hdf5_file is not None:
            try:
                self.hdf5_file.close()
                rospy.loginfo(f"[{self.label}] HDF5 closed: {self.hdf5_path}")
            except Exception as exc:
                rospy.logerr(f"[{self.label}] Error closing HDF5: {exc}")
            self.hdf5_file = None
            self.positions_dset = None
            self.velocities_dset = None
            self.timestamps_dset = None

        if self.time_arr:
            np.save(os.path.join(NPY_DIR, self.timestamp_npy_name), self.time_arr)
            np.save(os.path.join(NPY_DIR, self.position_npy_name), self.position_arr)
            rospy.loginfo(
                f"[{self.label}] Saved {len(self.time_arr)} samples to "
                f"{NPY_DIR}/{self.position_npy_name}"
            )
        else:
            rospy.logwarn(
                f"[{self.label}] No samples recorded — npy not written. "
                f"Is {self.joint_states_topic} publishing?"
            )


def build_recorders(datetime_id: str, arms: str = "both") -> list[ArmRecorder]:
    specs = [
        (
            "left",
            "/joint_states_host_left",
            f"host_left_joint_states_{datetime_id}.hdf5",
            f"joint_position_{datetime_id}.npy",
            f"joint_timestamp_{datetime_id}.npy",
        ),
        (
            "right",
            "/joint_states_host_right",
            f"host_right_joint_states_{datetime_id}.hdf5",
            f"joint_position_right_{datetime_id}.npy",
            f"joint_timestamp_right_{datetime_id}.npy",
        ),
    ]
    selected = {arms} if arms in ("left", "right") else {"left", "right"}
    return [
        ArmRecorder(label=label, joint_states_topic=topic, hdf5_filename=hdf5, position_npy_name=pos, timestamp_npy_name=ts)
        for label, topic, hdf5, pos, ts in specs
        if label in selected
    ]


def joint_state_listener(datetime_id: str, arms: str = "both") -> None:
    recorders = build_recorders(datetime_id, arms=arms)

    rospy.init_node("bimanual_joint_state_recorder", anonymous=True)
    rospy.loginfo("Bimanual joint state recorder initialized.")

    for recorder in recorders:
        if not recorder.open_hdf5():
            rospy.logerr(f"Failed to open HDF5 for {recorder.label} arm. Exiting.")
            return

    def shutdown_hook() -> None:
        for recorder in recorders:
            recorder.close()

    rospy.on_shutdown(shutdown_hook)

    for recorder in recorders:
        rospy.Subscriber(
            recorder.joint_states_topic,
            JointState,
            recorder.callback,
            queue_size=100,
        )
        rospy.loginfo(
            f"Subscribed [{recorder.label}] -> {recorder.joint_states_topic}"
        )

    rospy.loginfo("Spinning...")
    rospy.spin()
    rospy.loginfo("Recorder finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record host left + right joint states to HDF5/npy.",
    )
    parser.add_argument(
        "--datetime-id",
        type=str,
        default=None,
        help="Shared timestamp id for output filenames (default: now).",
    )
    parser.add_argument(
        "--arms",
        choices=("left", "right", "both"),
        default="both",
        help="Which robot arm joint streams to record (default: both).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datetime_id = args.datetime_id or datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        joint_state_listener(datetime_id, arms=args.arms)
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down.")
    except Exception as exc:
        rospy.logerr(f"Unexpected error: {exc}")
