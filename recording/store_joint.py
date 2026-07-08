#!/usr/bin/env python
"""Record host left + right joint states to HDF5 and npy."""

import argparse
import os
import threading
import time
from datetime import datetime

import h5py
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from recording_paths import under_recording
from recording_sync import read_recording_start, wait_for_recording_go

SAMPLING_FREQUENCY = 30.0
SAMPLING_INTERVAL = 1.0 / SAMPLING_FREQUENCY

BASE_OUT_DIR = under_recording("joint-data")


def arm_npy_dirs(label: str) -> tuple[str, str]:
    """Return (position_dir, time_dir) for left or right arm."""
    arm_root = os.path.join(BASE_OUT_DIR, label)
    position_dir = os.path.join(arm_root, "position")
    time_dir = os.path.join(arm_root, "time")
    os.makedirs(position_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)
    return position_dir, time_dir


class ArmRecorder:
    def __init__(
        self,
        label: str,
        joint_states_topic: str,
        hdf5_filename: str,
        position_npy_name: str,
        timestamp_npy_name: str,
        position_dir: str,
        time_dir: str,
    ) -> None:
        self.label = label
        self.joint_states_topic = joint_states_topic
        self.hdf5_path = os.path.join(os.path.expanduser("~"), "ros_data", hdf5_filename)
        self.position_npy_name = position_npy_name
        self.timestamp_npy_name = timestamp_npy_name
        self.position_dir = position_dir
        self.time_dir = time_dir

        self.num_joints = 0
        self.last_sampled_time = 0.0
        self.recording_enabled = False
        self.joint_names: list[str] | None = None
        self.time_arr: list[float] = []
        self.position_arr: list[list[float]] = []
        self.velocity_arr: list[list[float]] = []
        self._lock = threading.Lock()
        self._closed = False

    def callback(self, msg: JointState) -> None:
        if not self.recording_enabled:
            return

        # Wall-clock stamp to match camera recorders (time.time()), not ROS header time.
        current_time = time.time()
        if current_time - self.last_sampled_time < SAMPLING_INTERVAL:
            return
        self.last_sampled_time = current_time

        if len(msg.position) != len(msg.velocity):
            rospy.logwarn(
                f"[{self.label}] Inconsistent JointState lengths "
                f"({len(msg.position)} pos, {len(msg.velocity)} vel). Skipping."
            )
            return

        with self._lock:
            if self._closed:
                return

            if self.num_joints == 0:
                self.num_joints = len(msg.position)
                self.joint_names = list(msg.name)
                rospy.loginfo(f"[{self.label}] Detected {self.num_joints} joints.")
            elif len(msg.position) != self.num_joints:
                rospy.logwarn(
                    f"[{self.label}] Joint count changed "
                    f"({self.num_joints} -> {len(msg.position)}). Skipping sample."
                )
                return

            self.time_arr.append(current_time)
            self.position_arr.append(list(msg.position))
            self.velocity_arr.append(list(msg.velocity))

    def _write_hdf5(self) -> None:
        if not self.time_arr:
            return
        data_dir = os.path.dirname(self.hdf5_path)
        os.makedirs(data_dir, exist_ok=True)
        positions = np.asarray(self.position_arr, dtype=np.float32)
        velocities = np.asarray(self.velocity_arr, dtype=np.float32)
        timestamps = np.asarray(self.time_arr, dtype=np.float64)
        try:
            with h5py.File(self.hdf5_path, "w") as hdf5_file:
                hdf5_file.create_dataset("positions", data=positions)
                hdf5_file.create_dataset("velocities", data=velocities)
                hdf5_file.create_dataset("timestamps", data=timestamps)
                if self.joint_names:
                    hdf5_file.create_dataset(
                        "joint_names", data=np.array(self.joint_names, dtype="S")
                    )
            rospy.loginfo(f"[{self.label}] HDF5 saved: {self.hdf5_path}")
        except Exception as exc:
            rospy.logerr(f"[{self.label}] Error writing HDF5: {exc}")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.recording_enabled = False

        if self.time_arr:
            pos_path = os.path.join(self.position_dir, self.position_npy_name)
            ts_path = os.path.join(self.time_dir, self.timestamp_npy_name)
            np.save(ts_path, np.asarray(self.time_arr, dtype=np.float64))
            np.save(pos_path, np.asarray(self.position_arr, dtype=np.float64))
            rospy.loginfo(
                f"[{self.label}] Saved {len(self.time_arr)} samples to {pos_path}"
            )
            self._write_hdf5()
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
            f"joint_position_{datetime_id}.npy",
            f"joint_timestamp_{datetime_id}.npy",
        ),
    ]
    selected = {arms} if arms in ("left", "right") else {"left", "right"}
    recorders = []
    for label, topic, hdf5, pos, ts in specs:
        if label not in selected:
            continue
        position_dir, time_dir = arm_npy_dirs(label)
        recorders.append(
            ArmRecorder(
                label=label,
                joint_states_topic=topic,
                hdf5_filename=hdf5,
                position_npy_name=pos,
                timestamp_npy_name=ts,
                position_dir=position_dir,
                time_dir=time_dir,
            )
        )
    return recorders


def drain_joint_subscribers(duration_s: float = 0.05) -> None:
    """Process and discard joint messages queued during the boot countdown."""
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline and not rospy.is_shutdown():
        rospy.sleep(0.01)
    print(f"[joints] Drained subscriber queue ({duration_s:.2f}s).", flush=True)


def joint_state_listener(datetime_id: str, arms: str = "both", wait_for_go: bool = False) -> None:
    recorders = build_recorders(datetime_id, arms=arms)

    rospy.init_node("bimanual_joint_state_recorder", anonymous=True)
    rospy.loginfo("Bimanual joint state recorder initialized.")

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

    if wait_for_go and not wait_for_recording_go(datetime_id, label="joints"):
        return

    recording_t0 = read_recording_start(datetime_id)
    drain_joint_subscribers()
    # Brief spin so joint callbacks catch up with camera drain finishing.
    rospy.sleep(0.05)
    for recorder in recorders:
        recorder.recording_enabled = True
        recorder.last_sampled_time = 0.0
    if recording_t0 is not None:
        lag_ms = (time.time() - recording_t0) * 1000.0
        print(f"[joints] Shared recording t0={recording_t0:.3f} (enabled {lag_ms:.0f}ms after go)", flush=True)

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
    parser.add_argument(
        "--wait-for-go",
        action="store_true",
        help="Wait for record_pedal.py sync signal before saving samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datetime_id = args.datetime_id or datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        joint_state_listener(datetime_id, arms=args.arms, wait_for_go=args.wait_for_go)
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt — shutting down.")
    except Exception as exc:
        rospy.logerr(f"Unexpected error: {exc}")
