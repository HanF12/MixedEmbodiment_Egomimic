#!/usr/bin/env python3
"""
Replay recorded joint + gripper trajectories from recording/joint-data/npy.

Single arm (default left slave topics for bimanual tabletop):
  python replay_joint.py --datetime-id 20260702122247 --dual

Publishes 6 arm joints + gripper per arm to the mobiman slave topics.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _first_existing_dir(*candidates: str) -> str:
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return candidates[0] if candidates else ""


# Back-compat default (old layout). If it doesn't exist, prefer the newer session layout.
DEFAULT_DATA_DIR = _first_existing_dir(
    os.path.join(SCRIPT_DIR, "joint-data", "npy"),
    os.path.join(SCRIPT_DIR, "sessions", "teleop_bimanual", "joint-data"),
)

DEFAULT_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))

# Bimanual tabletop teleop: replay on slave (follower) arms.
# iarm_node_slave_* subscribes to /arm_joint_command_slave_* (signal_arm/arm_control).
# The older /arm_joint_target_position_slave_* (sensor_msgs/JointState) is typically used by
# tracker/bridge nodes, not the controller itself.
DEFAULT_LEFT_ARM_TOPIC = "/arm_joint_target_position_slave_left"
DEFAULT_LEFT_GRIPPER_TOPIC = "/gripper_position_control_slave_left"
DEFAULT_RIGHT_ARM_TOPIC = "/arm_joint_target_position_slave_right"
DEFAULT_RIGHT_GRIPPER_TOPIC = "/gripper_position_control_slave_right"

# Direct controller command topics (bypass jointTracker).
DEFAULT_LEFT_ARM_COMMAND_TOPIC = "/arm_joint_command_slave_left"
DEFAULT_RIGHT_ARM_COMMAND_TOPIC = "/arm_joint_command_slave_right"

# Single-arm fallback (A1_SDK style).
DEFAULT_SINGLE_ARM_TOPIC = "/arm_joint_target_position"
DEFAULT_SINGLE_GRIPPER_TOPIC = "/gripper_position_control_host"


@dataclass
class ArmTrajectory:
    label: str
    positions: np.ndarray
    timestamps: Optional[np.ndarray]
    position_path: str


def _left_position_name(rec_id: str) -> str:
    return f"joint_position_{rec_id}.npy"


def _right_position_name(rec_id: str) -> str:
    return f"joint_position_right_{rec_id}.npy"


def _left_timestamp_name(rec_id: str) -> str:
    return f"joint_timestamp_{rec_id}.npy"


def _right_timestamp_name(rec_id: str) -> str:
    return f"joint_timestamp_right_{rec_id}.npy"


def _looks_like_session_jointdata_dir(data_dir: str) -> bool:
    # New layout:
    #   <...>/joint-data/{left,right}/{position,time}/joint_position_<id>.npy
    return (
        os.path.isdir(os.path.join(data_dir, "left", "position"))
        and os.path.isdir(os.path.join(data_dir, "right", "position"))
    )


def _normalize_data_dir(data_dir: str) -> str:
    """
    Accept any of:
      - old: .../joint-data/npy
      - new: .../joint-data
      - session root: .../sessions/<session-name>
    and normalize to a concrete directory used by resolution helpers.
    """
    data_dir = os.path.abspath(data_dir)
    if _looks_like_session_jointdata_dir(data_dir):
        return data_dir
    maybe_joint_data = os.path.join(data_dir, "joint-data")
    if _looks_like_session_jointdata_dir(maybe_joint_data):
        return maybe_joint_data
    return data_dir


def _infer_rec_id_from_position_path(position_path: str) -> Optional[str]:
    base = os.path.basename(position_path)
    if base.startswith("joint_position_") and base.endswith(".npy"):
        return base.removeprefix("joint_position_").removesuffix(".npy")
    if base.startswith("joint_position_right_") and base.endswith(".npy"):
        return base.removeprefix("joint_position_right_").removesuffix(".npy")
    return None


def _infer_timestamp_path(position_path: str) -> Optional[str]:
    """
    Best-effort inference for timestamp path from a position path.
    Supports both old flat layout and new session layout.
    """
    pos_base = os.path.basename(position_path)
    ts_base = None
    if pos_base.startswith("joint_position_right_"):
        ts_base = pos_base.replace("joint_position_right_", "joint_timestamp_right_", 1)
    elif pos_base.startswith("joint_position_"):
        ts_base = pos_base.replace("joint_position_", "joint_timestamp_", 1)
    if ts_base is None:
        return None

    # New layout: replace /position/ with /time/ if present
    parts = position_path.split(os.sep)
    try:
        idx = parts.index("position")
    except ValueError:
        idx = -1
    if idx != -1:
        parts[idx] = "time"
        candidate = os.sep.join(parts[:-1] + [ts_base])
        if os.path.isfile(candidate):
            return candidate

    # Old flat layout (same directory)
    candidate = os.path.join(os.path.dirname(position_path), ts_base)
    if os.path.isfile(candidate):
        return candidate
    return None


def list_available_recordings(data_dir: str) -> list[str]:
    """Return ids with at least one position file (left or right)."""
    data_dir = _normalize_data_dir(data_dir)
    ids: set[str] = set()

    if _looks_like_session_jointdata_dir(data_dir):
        for side in ("left", "right"):
            pos_dir = os.path.join(data_dir, side, "position")
            for path in glob.glob(os.path.join(pos_dir, "joint_position_*.npy")):
                name = os.path.basename(path)
                ids.add(name.removeprefix("joint_position_").removesuffix(".npy"))
        return sorted(ids)

    for path in glob.glob(os.path.join(data_dir, "joint_position_*.npy")):
        name = os.path.basename(path)
        if name.startswith("joint_position_right_"):
            ids.add(name.removeprefix("joint_position_right_").removesuffix(".npy"))
        else:
            ids.add(name.removeprefix("joint_position_").removesuffix(".npy"))
    return sorted(ids)


def has_right_recording(data_dir: str, rec_id: str) -> bool:
    data_dir = _normalize_data_dir(data_dir)
    if _looks_like_session_jointdata_dir(data_dir):
        right_pos_dir = os.path.join(data_dir, "right", "position")
        return os.path.isfile(os.path.join(right_pos_dir, _left_position_name(rec_id)))
    # Old flat layout: accept either naming.
    return os.path.isfile(os.path.join(data_dir, _right_position_name(rec_id))) or os.path.isfile(
        os.path.join(data_dir, _left_position_name(rec_id))
    )


def resolve_arm_paths(
    data_dir: str,
    rec_id: str,
    side: str,
    position_file: Optional[str],
    timestamp_file: Optional[str],
) -> Tuple[str, Optional[str]]:
    if position_file:
        pos_path = os.path.abspath(position_file)
        if not os.path.isfile(pos_path):
            raise FileNotFoundError(f"{side} position file not found: {pos_path}")

        if timestamp_file:
            ts_path = os.path.abspath(timestamp_file)
            if not os.path.isfile(ts_path):
                raise FileNotFoundError(f"{side} timestamp file not found: {ts_path}")
            return pos_path, ts_path

        inferred = _infer_timestamp_path(pos_path)
        return pos_path, os.path.abspath(inferred) if inferred else None

    data_dir = _normalize_data_dir(data_dir)

    if _looks_like_session_jointdata_dir(data_dir):
        pos_dir = os.path.join(data_dir, side, "position")
        ts_dir = os.path.join(data_dir, side, "time")
        pos_name = _left_position_name(rec_id)  # new layout uses same naming for both
        ts_name = _left_timestamp_name(rec_id)
        pos_path = os.path.join(pos_dir, pos_name)
        ts_path = os.path.join(ts_dir, ts_name)
        if not os.path.isfile(pos_path):
            raise FileNotFoundError(f"{side} position file not found: {pos_path}")
        if not os.path.isfile(ts_path):
            return pos_path, None
        return pos_path, ts_path

    # Old flat layout: right arm sometimes stored as joint_position_right_<id>.npy.
    if side == "right":
        pos_candidates = [_right_position_name(rec_id), _left_position_name(rec_id)]
        ts_candidates = [_right_timestamp_name(rec_id), _left_timestamp_name(rec_id)]
    else:
        pos_candidates = [_left_position_name(rec_id)]
        ts_candidates = [_left_timestamp_name(rec_id)]

    pos_path = None
    for name in pos_candidates:
        candidate = os.path.join(data_dir, name)
        if os.path.isfile(candidate):
            pos_path = candidate
            break
    if pos_path is None:
        raise FileNotFoundError(
            f"{side} position file not found in {data_dir}; tried: {pos_candidates}"
        )

    ts_path = None
    for name in ts_candidates:
        candidate = os.path.join(data_dir, name)
        if os.path.isfile(candidate):
            ts_path = candidate
            break
    return pos_path, ts_path


def load_trajectory(position_path: str, timestamp_path: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    positions = np.asarray(np.load(position_path), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 7:
        raise ValueError(
            f"Expected positions shaped (N, 7), got {positions.shape} from {position_path}"
        )

    timestamps = None
    if timestamp_path is not None:
        timestamps = np.asarray(np.load(timestamp_path), dtype=np.float64).reshape(-1)
        if timestamps.shape[0] != positions.shape[0]:
            raise ValueError(
                f"Timestamp count ({timestamps.shape[0]}) does not match "
                f"position count ({positions.shape[0]}) in {position_path}"
            )
    return positions, timestamps


def preprocess_step(
    pos: np.ndarray,
    *,
    gripper_scale: float,
    gripper_max: Optional[float],
    clamp_joint6: bool,
    joint6_min: float,
    joint6_max: float,
) -> np.ndarray:
    out = pos.astype(np.float64, copy=True)
    if clamp_joint6:
        out[-2] = np.clip(out[-2], joint6_min, joint6_max)
    if gripper_scale != 1.0:
        out[-1] *= gripper_scale
    if gripper_max is not None:
        out[-1] = min(out[-1], gripper_max)
    return out


def estimate_motion_start(positions: np.ndarray, threshold: float = 0.05) -> int:
    if positions.shape[0] <= 1:
        return 0
    delta = np.linalg.norm(positions[:, :6] - positions[0, :6], axis=1)
    motion = np.where(delta > threshold)[0]
    return int(motion[0]) if motion.size else 0


class JointReplayer:
    def __init__(
        self,
        label: str,
        arm_topic: str,
        gripper_topic: str,
        joint_names: Sequence[str],
        frame_id: str,
        queue_size: int,
        *,
        arm_msg_type: str = "joint_state",
        arm_kp: float = 10.0,
        arm_kd: float = 1.0,
        arm_mode: int = 0,
    ) -> None:
        import rospy
        from sensor_msgs.msg import JointState
        from signal_arm.msg import gripper_position_control

        self.label = label
        self._rospy = rospy
        self._JointState = JointState
        self._gripper_position_control = gripper_position_control
        self._arm_msg_type = str(arm_msg_type)
        self._arm_kp = float(arm_kp)
        self._arm_kd = float(arm_kd)
        self._arm_mode = int(arm_mode)

        self._arm_control = None
        if self._arm_msg_type == "arm_control":
            try:
                from signal_arm.msg import arm_control
            except Exception as exc:
                raise RuntimeError(
                    "arm_control publishing requested, but signal_arm/arm_control could not be imported. "
                    "Source Tabletop_Teleoperation_SDK install/setup.bash in this shell."
                ) from exc
            self._arm_control = arm_control
            self._arm_pub = rospy.Publisher(arm_topic, arm_control, queue_size=queue_size)
        else:
            self._arm_pub = rospy.Publisher(arm_topic, JointState, queue_size=queue_size)

        self._gripper_pub = rospy.Publisher(
            gripper_topic, gripper_position_control, queue_size=queue_size
        )
        self._joint_names = list(joint_names)
        self._frame_id = frame_id
        self._arm_topic = arm_topic
        self._gripper_topic = gripper_topic

    def wait_for_subscribers(self, timeout_sec: float) -> bool:
        rospy = self._rospy
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and not rospy.is_shutdown():
            arm_n = self._arm_pub.get_num_connections()
            grip_n = self._gripper_pub.get_num_connections()
            if arm_n > 0 and grip_n > 0:
                rospy.loginfo(
                    f"[{self.label}] subscribers connected "
                    f"(arm={arm_n}, gripper={grip_n}) on "
                    f"{self._arm_topic} / {self._gripper_topic}"
                )
                return True
            rospy.sleep(0.1)
        rospy.logwarn(
            f"[{self.label}] no subscribers before timeout on "
            f"{self._arm_topic} / {self._gripper_topic}"
        )
        return False

    def publish_step(self, positions_7d: np.ndarray) -> None:
        rospy = self._rospy
        gripper_position_control = self._gripper_position_control

        stamp = rospy.Time.now()

        if self._arm_msg_type == "arm_control":
            arm_control = self._arm_control
            if arm_control is None:
                raise RuntimeError("arm_control message type not available.")
            arm_msg = arm_control()
            arm_msg.header.stamp = stamp
            arm_msg.header.frame_id = self._frame_id
            arm_msg.p_des = positions_7d[:6].astype(np.float32).tolist()
            arm_msg.v_des = [0.0] * 6
            arm_msg.kp = [float(self._arm_kp)] * 6
            arm_msg.kd = [float(self._arm_kd)] * 6
            arm_msg.t_ff = [0.0] * 6
            arm_msg.mode = int(self._arm_mode)
        else:
            JointState = self._JointState
            arm_msg = JointState()
            arm_msg.header.stamp = stamp
            arm_msg.header.frame_id = self._frame_id
            arm_msg.name = self._joint_names
            arm_msg.position = positions_7d[:6].astype(np.float32).tolist()
            arm_msg.velocity = []
            arm_msg.effort = []

        gripper_msg = gripper_position_control()
        gripper_msg.header.stamp = stamp
        gripper_msg.header.frame_id = self._frame_id
        gripper_msg.gripper_stroke = float(positions_7d[6])

        self._arm_pub.publish(arm_msg)
        self._gripper_pub.publish(gripper_msg)


def replay(
    arms: list[Tuple[ArmTrajectory, JointReplayer]],
    *,
    rate_hz: float,
    use_recorded_timing: bool,
    loop: bool,
    gripper_scale: float,
    gripper_max: Optional[float],
    clamp_joint6: bool,
    joint6_min: float,
    joint6_max: float,
    warmup_sec: float,
    start_step: int,
    progress_interval_sec: float,
) -> None:
    import rospy

    if warmup_sec > 0:
        rospy.loginfo(f"Warmup: waiting {warmup_sec:.1f}s before publishing...")
        rospy.sleep(warmup_sec)

    start_steps = []
    n_steps_list = []
    for traj, _ in arms:
        n = traj.positions.shape[0]
        s = max(0, min(start_step, n - 1))
        start_steps.append(s)
        n_steps_list.append(n - s)

        motion_start = estimate_motion_start(traj.positions)
        if motion_start > 0 and start_step == 0:
            est_sec = motion_start / rate_hz
            rospy.logwarn(
                f"[{traj.label}] mostly still for first ~{motion_start} steps "
                f"(~{est_sec:.0f}s at {rate_hz:g} Hz). "
                f"Use --start-step {motion_start} to skip."
            )

    n_steps = min(n_steps_list)
    if len(arms) > 1:
        lengths = [traj.positions.shape[0] for traj, _ in arms]
        if len(set(lengths)) != 1:
            rospy.logwarn(
                f"Arm trajectories differ in length {lengths}; "
                f"replaying first {n_steps} synchronized steps."
            )

    labels = ", ".join(traj.label for traj, _ in arms)
    rospy.loginfo(f"Replaying {n_steps} steps on: {labels}")
    rospy.loginfo("Watch the SLAVE (follower) arms — host leaders are not commanded.")

    next_progress = time.monotonic() + progress_interval_sec if progress_interval_sec > 0 else float("inf")

    while not rospy.is_shutdown():
        loop_start = time.perf_counter()
        for step_idx in range(n_steps):
            if rospy.is_shutdown():
                return

            sleep_s = 1.0 / rate_hz
            for arm_i, (traj, replayer) in enumerate(arms):
                i = start_steps[arm_i] + step_idx
                step = preprocess_step(
                    traj.positions[i],
                    gripper_scale=gripper_scale,
                    gripper_max=gripper_max,
                    clamp_joint6=clamp_joint6,
                    joint6_min=joint6_min,
                    joint6_max=joint6_max,
                )
                replayer.publish_step(step)

                if (
                    use_recorded_timing
                    and traj.timestamps is not None
                    and i + 1 < traj.positions.shape[0]
                ):
                    sleep_s = max(
                        sleep_s,
                        float(traj.timestamps[i + 1] - traj.timestamps[i]),
                    )

            now = time.monotonic()
            if now >= next_progress:
                pct = 100.0 * (step_idx + 1) / n_steps
                left_i = start_steps[0] + step_idx
                rospy.loginfo(
                    f"Replay progress: {step_idx + 1}/{n_steps} ({pct:.0f}%) "
                    f"{arms[0][0].label} j1={arms[0][0].positions[left_i, 1]:.3f}"
                )
                next_progress = now + progress_interval_sec

            if sleep_s > 0:
                rospy.sleep(sleep_s)

        elapsed = time.perf_counter() - loop_start
        rospy.loginfo(f"Replay finished in {elapsed:.2f}s")
        if not loop:
            break
        rospy.loginfo("Looping recording...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay joint-data recordings to arm and gripper ROS topics.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--datetime-id", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--use-recorded-timing", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument("--gripper-scale", type=float, default=30.2)
    parser.add_argument("--gripper-max", type=float, default=60.0)
    parser.add_argument("--no-clamp-joint6", action="store_true")
    parser.add_argument("--joint6-min", type=float, default=-2.8)
    parser.add_argument("--joint6-max", type=float, default=2.8)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--subscriber-timeout", type=float, default=10.0)
    parser.add_argument("--progress-interval-sec", type=float, default=5.0)

    parser.add_argument(
        "--arm-msg-type",
        choices=("arm_control", "joint_state"),
        default="joint_state",
        help="Arm command message type. iarm_node_slave_* expects signal_arm/arm_control.",
    )
    parser.add_argument("--arm-kp", type=float, default=10.0, help="arm_control kp gain (per joint).")
    parser.add_argument("--arm-kd", type=float, default=1.0, help="arm_control kd gain (per joint).")
    parser.add_argument("--arm-mode", type=int, default=0, help="arm_control mode (0=position).")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dual",
        action="store_true",
        help="Replay both arms (default when right recording exists).",
    )
    mode.add_argument(
        "--left-only",
        action="store_true",
        help="Replay left arm only.",
    )
    mode.add_argument(
        "--right-only",
        action="store_true",
        help="Replay right arm only.",
    )

    parser.add_argument("--left-position-file", default=None)
    parser.add_argument("--left-timestamp-file", default=None)
    parser.add_argument("--right-position-file", default=None)
    parser.add_argument("--right-timestamp-file", default=None)

    parser.add_argument("--left-arm-topic", default=DEFAULT_LEFT_ARM_TOPIC)
    parser.add_argument("--left-gripper-topic", default=DEFAULT_LEFT_GRIPPER_TOPIC)
    parser.add_argument("--right-arm-topic", default=DEFAULT_RIGHT_ARM_TOPIC)
    parser.add_argument("--right-gripper-topic", default=DEFAULT_RIGHT_GRIPPER_TOPIC)

    # Back-compat aliases for single-arm usage.
    parser.add_argument("--arm-topic", default=None)
    parser.add_argument("--gripper-topic", default=None)
    parser.add_argument("--position-file", default=None)
    parser.add_argument("--timestamp-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _normalize_data_dir(args.data_dir)

    if args.list:
        ids = list_available_recordings(data_dir)
        if not ids:
            print(f"No recordings in {data_dir}")
            return
        print("Available recording ids:")
        for rec_id in ids:
            left_path = os.path.join(data_dir, _left_position_name(rec_id))
            right_path = os.path.join(data_dir, _right_position_name(rec_id))
            left_n = np.load(left_path).shape[0] if os.path.isfile(left_path) else None
            right_n = (
                np.load(right_path).shape[0] if os.path.isfile(right_path) else None
            )
            if left_n is not None and right_n is not None:
                print(f"  {rec_id}  (left={left_n}, right={right_n} steps) [dual]")
            elif left_n is not None:
                print(f"  {rec_id}  (left={left_n} steps)")
            else:
                print(f"  {rec_id}  (right={right_n} steps)")
        return

    try:
        import rospy
    except ImportError as exc:
        print("Source ROS first: /opt/ros/noetic/setup.bash + SDK setup.bash", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        from signal_arm.msg import gripper_position_control  # noqa: F401
    except ImportError as exc:
        print("Source SDK workspace for signal_arm.", file=sys.stderr)
        raise SystemExit(1) from exc

    # If explicit position files are provided, we can replay without discovering a recording id
    # in data_dir (useful with session-style folder structures).
    rec_id = args.datetime_id
    inferred_id = _infer_rec_id_from_position_path(args.right_position_file) if args.right_position_file else None
    if not rec_id and inferred_id:
        rec_id = inferred_id

    left_pos_file = args.left_position_file or args.position_file
    left_ts_file = args.left_timestamp_file or args.timestamp_file

    replay_left = not args.right_only
    replay_right = args.dual or args.right_only
    if not args.dual and not args.left_only and not args.right_only:
        replay_right = has_right_recording(data_dir, rec_id) if rec_id else False

    needs_rec_id = (replay_left and not left_pos_file) or (replay_right and not args.right_position_file)
    if not rec_id and needs_rec_id:
        available = list_available_recordings(data_dir)
        if not available:
            raise SystemExit(
                f"No recordings found in {data_dir}. "
                "Either pass --datetime-id (matching the .npy suffix), "
                "or pass explicit --left-position-file/--right-position-file."
            )
        rec_id = available[-1]
        print(f"Using latest recording id: {rec_id}")

    arms: list[Tuple[ArmTrajectory, JointReplayer]] = []

    rospy.init_node("joint_data_replayer", anonymous=True)

    # If user wants to publish arm_control and did not override topics, switch to the controller topics.
    if args.arm_msg_type == "arm_control":
        if args.left_arm_topic == DEFAULT_LEFT_ARM_TOPIC:
            args.left_arm_topic = DEFAULT_LEFT_ARM_COMMAND_TOPIC
        if args.right_arm_topic == DEFAULT_RIGHT_ARM_TOPIC:
            args.right_arm_topic = DEFAULT_RIGHT_ARM_COMMAND_TOPIC

    if replay_left:
        pos_path, ts_path = resolve_arm_paths(
            data_dir, rec_id, "left", left_pos_file, left_ts_file
        )
        positions, timestamps = load_trajectory(pos_path, ts_path)
        left_arm_topic = args.arm_topic or args.left_arm_topic
        left_gripper_topic = args.gripper_topic or args.left_gripper_topic
        left_replayer = JointReplayer(
            "left",
            left_arm_topic,
            left_gripper_topic,
            DEFAULT_JOINT_NAMES,
            "world",
            1000,
            arm_msg_type=args.arm_msg_type,
            arm_kp=args.arm_kp,
            arm_kd=args.arm_kd,
            arm_mode=args.arm_mode,
        )
        arms.append(
            (
                ArmTrajectory("left", positions, timestamps, pos_path),
                left_replayer,
            )
        )

    if replay_right:
        if not has_right_recording(data_dir, rec_id) and not args.right_position_file:
            raise SystemExit(
                f"No right-arm recording for id {rec_id}. "
                f"Expected {os.path.join(data_dir, _right_position_name(rec_id))}. "
                "Re-record with the updated store_joint.py, or use --left-only."
            )
        pos_path, ts_path = resolve_arm_paths(
            data_dir,
            rec_id,
            "right",
            args.right_position_file,
            args.right_timestamp_file,
        )
        positions, timestamps = load_trajectory(pos_path, ts_path)
        right_replayer = JointReplayer(
            "right",
            args.right_arm_topic,
            args.right_gripper_topic,
            DEFAULT_JOINT_NAMES,
            "world",
            1000,
            arm_msg_type=args.arm_msg_type,
            arm_kp=args.arm_kp,
            arm_kd=args.arm_kd,
            arm_mode=args.arm_mode,
        )
        arms.append(
            (
                ArmTrajectory("right", positions, timestamps, pos_path),
                right_replayer,
            )
        )

    if not arms:
        raise SystemExit("Nothing to replay — specify --left-only, --right-only, or --dual.")

    gripper_max = None if args.gripper_max < 0 else args.gripper_max
    use_recorded_timing = args.use_recorded_timing

    for traj, _ in arms:
        rospy.loginfo(f"[{traj.label}] {traj.position_path} ({traj.positions.shape[0]} steps)")
    rospy.loginfo(
        f"Timing: {'recorded' if use_recorded_timing else f'{args.rate_hz} Hz'}"
    )

    for _, replayer in arms:
        replayer.wait_for_subscribers(args.subscriber_timeout)

    try:
        replay(
            arms,
            rate_hz=args.rate_hz,
            use_recorded_timing=use_recorded_timing,
            loop=args.loop,
            gripper_scale=args.gripper_scale,
            gripper_max=gripper_max,
            clamp_joint6=not args.no_clamp_joint6,
            joint6_min=args.joint6_min,
            joint6_max=args.joint6_max,
            warmup_sec=args.warmup_sec,
            start_step=args.start_step,
            progress_interval_sec=args.progress_interval_sec,
        )
    except rospy.ROSInterruptException:
        rospy.loginfo("Replay interrupted (Ctrl+C).")


if __name__ == "__main__":
    main()
