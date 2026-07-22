#!/usr/bin/env python3
"""
Slowly move slave arm(s) back to a safe home pose via mobiman joint targets.

Uses the same ROS topics as replay_joint.py (jointTracker + gripper control).
Inspired by the interpolation pattern in ROSTry.py.

Typical usage (bimanual tabletop, dual.launch running):
  source /opt/ros/noetic/setup.bash
  cd ~/Tabletop_Teleoperation_SDK/install && source setup.bash
  cd ~/MixedEmbodiment_Egomimic/recording

  # Capture home while arms are physically at home:
  python3 go_home.py --print-current --save-home

  # Return to saved home later:
  python3 go_home.py --yes

Ctrl+C stops motion immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from replay_joint import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_LEFT_ARM_TOPIC,
    DEFAULT_LEFT_GRIPPER_TOPIC,
    DEFAULT_RIGHT_ARM_TOPIC,
    DEFAULT_RIGHT_GRIPPER_TOPIC,
    JointReplayer,
)

SCRIPT_DIR = Path(__file__).resolve().parent
HOME_POSE_FILE = SCRIPT_DIR / "joint-data" / "home_pose.json"

# Fallback if home_pose.json does not exist yet.
DEFAULT_HOME_LEFT = np.array(
    [-0.03, 0.002, -0.007, -0.002, 0.021, -1.655, -0.244], dtype=np.float64
)
DEFAULT_HOME_RIGHT = np.array(
    [-0.014, 0.003, -0.008, -0.077, -0.053, 2.878, -0.136], dtype=np.float64
)

JOINT_LIMITS_MIN = np.array([-3.14, -2.0, -3.14, -0.1, -3.14, -3.14], dtype=np.float64)
JOINT_LIMITS_MAX = np.array([3.14, 2.5, 3.14, 3.14, 3.14, 3.14], dtype=np.float64)


def parse_joint_list(text: str, expected: int = 6) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {len(values)}")
    return np.array(values, dtype=np.float64)


def build_home_pose(joints_6: np.ndarray, gripper_raw: float) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float64)
    pose[:6] = joints_6
    pose[6] = gripper_raw
    return pose


def shortest_joint_delta(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Shortest signed angle delta for each revolute joint (avoids long rotations)."""
    delta = goal - start
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def read_current_pose(joint_states_topic: str, timeout: float, label: str) -> np.ndarray:
    import rospy
    from sensor_msgs.msg import JointState

    rospy.loginfo(f"[{label}] Reading current pose from {joint_states_topic}...")
    msg = rospy.wait_for_message(joint_states_topic, JointState, timeout=timeout)
    if len(msg.position) < 6:
        raise RuntimeError(
            f"[{label}] Expected >=6 joint positions, got {len(msg.position)}"
        )
    pose = np.zeros(7, dtype=np.float64)
    pose[:6] = np.asarray(msg.position[:6], dtype=np.float64)
    if len(msg.position) >= 7:
        pose[6] = float(msg.position[6])
    return pose


def clamp_home_joints(pose_7d: np.ndarray) -> np.ndarray:
    out = pose_7d.copy()
    out[:6] = np.clip(out[:6], JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)
    return out


def interpolate_pose_shortest(start: np.ndarray, goal: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate using shortest angular path for arm joints."""
    w = smoothstep(alpha)
    out = start.copy()
    out[:6] = start[:6] + shortest_joint_delta(start[:6], goal[:6]) * w
    out[6] = start[6] + (goal[6] - start[6]) * w
    return out


def apply_gripper_command(
    pose_7d: np.ndarray,
    *,
    gripper_scale: float,
    gripper_max: Optional[float],
) -> np.ndarray:
    out = pose_7d.astype(np.float64, copy=True)
    if gripper_scale != 1.0:
        out[6] *= gripper_scale
    if gripper_max is not None:
        out[6] = min(out[6], gripper_max)
    return out


def per_joint_travel(start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.abs(shortest_joint_delta(start[:6], goal[:6]))


def load_saved_home() -> dict[str, np.ndarray]:
    if not HOME_POSE_FILE.is_file():
        return {}
    with open(HOME_POSE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        if side not in data:
            continue
        entry = data[side]
        joints = np.asarray(entry["joints"], dtype=np.float64)
        gripper_raw = float(entry.get("gripper_raw", 0.051))
        out[side] = build_home_pose(joints, gripper_raw)
    return out


def save_home_pose(left: np.ndarray, right: np.ndarray) -> None:
    HOME_POSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "left": {"joints": left[:6].tolist(), "gripper_raw": float(left[6])},
        "right": {"joints": right[:6].tolist(), "gripper_raw": float(right[6])},
    }
    with open(HOME_POSE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved home pose to {HOME_POSE_FILE}")


def max_joint_speed(start: np.ndarray, goal: np.ndarray, duration_sec: float) -> float:
    if duration_sec <= 0:
        return float("inf")
    return float(np.max(per_joint_travel(start, goal)) / duration_sec)


def log_travel_warnings(label: str, start: np.ndarray, goal: np.ndarray) -> None:
    import rospy

    travel = per_joint_travel(start, goal)
    joint_names = [f"joint{i + 1}" for i in range(6)]
    for i, (name, delta) in enumerate(zip(joint_names, travel)):
        if delta > 1.0:
            rospy.logwarn(
                f"[{label}] Large move on {name}: {delta:.3f} rad "
                f"(start={start[i]:.3f} -> goal={goal[i]:.3f})"
            )
    wrist_delta = travel[5]
    if wrist_delta > 0.5:
        rospy.logwarn(
            f"[{label}] Wrist (joint6) travel is {wrist_delta:.3f} rad — "
            "verify home pose is correct for this arm."
        )


def go_home_arm(
    label: str,
    replayer: JointReplayer,
    start_pose: np.ndarray,
    home_pose: np.ndarray,
    *,
    duration_sec: float,
    rate_hz: float,
    hold_sec: float,
    gripper_scale: float,
    gripper_max: Optional[float],
    max_speed_rad_s: float,
    at_home_tol: float,
) -> None:
    import rospy

    home_pose = clamp_home_joints(home_pose)
    delta = float(np.linalg.norm(shortest_joint_delta(start_pose[:6], home_pose[:6])))
    if delta < at_home_tol:
        rospy.loginfo(f"[{label}] Already at home (delta={delta:.4f} rad). Skipping.")
        return

    log_travel_warnings(label, start_pose, home_pose)

    speed = max_joint_speed(start_pose, home_pose, duration_sec)
    if speed > max_speed_rad_s:
        suggested = float(np.max(per_joint_travel(start_pose, home_pose))) / max_speed_rad_s
        rospy.logwarn(
            f"[{label}] Requested move may be fast ({speed:.2f} rad/s). "
            f"Consider --duration-sec {suggested:.1f} or higher."
        )

    steps = max(1, int(round(duration_sec * rate_hz)))
    rospy.loginfo(
        f"[{label}] Homing over {steps} steps ({duration_sec:.1f}s @ {rate_hz:g} Hz), "
        f"joint travel={delta:.3f} rad (shortest path)"
    )
    rospy.loginfo(f"[{label}] start arm = {np.round(start_pose[:6], 4).tolist()}")
    rospy.loginfo(f"[{label}] goal  arm = {np.round(home_pose[:6], 4).tolist()}")

    rate = rospy.Rate(rate_hz)
    for step in range(steps):
        if rospy.is_shutdown():
            rospy.loginfo(f"[{label}] Homing interrupted.")
            return
        alpha = (step + 1) / steps
        raw_pose = interpolate_pose_shortest(start_pose, home_pose, alpha)
        cmd = apply_gripper_command(
            raw_pose,
            gripper_scale=gripper_scale,
            gripper_max=gripper_max,
        )
        replayer.publish_step(cmd)
        rate.sleep()

    final_cmd = apply_gripper_command(
        home_pose,
        gripper_scale=gripper_scale,
        gripper_max=gripper_max,
    )
    hold_steps = max(1, int(hold_sec * rate_hz))
    for _ in range(hold_steps):
        if rospy.is_shutdown():
            return
        replayer.publish_step(final_cmd)
        rate.sleep()

    rospy.loginfo(f"[{label}] Home pose reached.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slowly move slave arm(s) to a safe home configuration.",
    )
    parser.add_argument("--dual", action="store_true", help="Home both slave arms (default).")
    parser.add_argument("--left-only", action="store_true", help="Home left slave arm only.")
    parser.add_argument("--right-only", action="store_true", help="Home right slave arm only.")

    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--hold-sec", type=float, default=2.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--subscriber-timeout", type=float, default=10.0)
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--max-speed-rad-s", type=float, default=0.2)
    parser.add_argument("--at-home-tol", type=float, default=0.02)

    parser.add_argument("--gripper-scale", type=float, default=30.2)
    parser.add_argument("--gripper-max", type=float, default=60.0)
    parser.add_argument("--gripper-raw", type=float, default=0.051)

    parser.add_argument("--home-joints", type=str, default=None)
    parser.add_argument("--home-joints-left", type=str, default=None)
    parser.add_argument("--home-joints-right", type=str, default=None)
    parser.add_argument(
        "--home-file",
        type=str,
        default=str(HOME_POSE_FILE),
        help="JSON file with per-arm home poses (written by --save-home).",
    )

    parser.add_argument("--left-arm-topic", default=DEFAULT_LEFT_ARM_TOPIC)
    parser.add_argument("--left-gripper-topic", default=DEFAULT_LEFT_GRIPPER_TOPIC)
    parser.add_argument("--right-arm-topic", default=DEFAULT_RIGHT_ARM_TOPIC)
    parser.add_argument("--right-gripper-topic", default=DEFAULT_RIGHT_GRIPPER_TOPIC)
    parser.add_argument("--left-joint-states-topic", default="/joint_states_slave_left")
    parser.add_argument("--right-joint-states-topic", default="/joint_states_slave_right")

    parser.add_argument("--print-current", action="store_true")
    parser.add_argument(
        "--save-home",
        action="store_true",
        help="With --print-current, save poses to joint-data/home_pose.json.",
    )
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def resolve_home_pose(
    args: argparse.Namespace,
    side: str,
    saved: dict[str, np.ndarray],
) -> np.ndarray:
    if side == "left":
        if args.home_joints_left:
            joints = parse_joint_list(args.home_joints_left, 6)
        elif args.home_joints:
            joints = parse_joint_list(args.home_joints, 6)
        elif side in saved:
            return saved[side].copy()
        else:
            return DEFAULT_HOME_LEFT.copy()
        return build_home_pose(joints, args.gripper_raw)

    if args.home_joints_right:
        joints = parse_joint_list(args.home_joints_right, 6)
    elif args.home_joints:
        joints = parse_joint_list(args.home_joints, 6)
    elif side in saved:
        return saved[side].copy()
    else:
        return DEFAULT_HOME_RIGHT.copy()
    return build_home_pose(joints, args.gripper_raw)


def main() -> None:
    args = parse_args()

    if args.left_only and args.right_only:
        raise SystemExit("Use only one of --left-only or --right-only.")

    if args.left_only:
        do_left, do_right = True, False
    elif args.right_only:
        do_left, do_right = False, True
    else:
        do_left, do_right = True, True

    try:
        import rospy
    except ImportError as exc:
        print("Source ROS + SDK setup.bash first.", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        from signal_arm.msg import gripper_position_control  # noqa: F401
    except ImportError as exc:
        print("Source Tabletop_Teleoperation_SDK setup.bash first.", file=sys.stderr)
        raise SystemExit(1) from exc

    global HOME_POSE_FILE
    HOME_POSE_FILE = Path(args.home_file)
    saved_home = load_saved_home()

    rospy.init_node("go_home", anonymous=True)

    if args.print_current:
        left_pose = right_pose = None
        if do_left:
            left_pose = read_current_pose(
                args.left_joint_states_topic, args.joint_state_timeout, "left"
            )
            print("left home joints + gripper raw:")
            print(",".join(f"{v:.6f}" for v in left_pose))
            print(
                f'  --home-joints-left "{",".join(f"{v:.6f}" for v in left_pose[:6])}"'
            )
        if do_right:
            right_pose = read_current_pose(
                args.right_joint_states_topic, args.joint_state_timeout, "right"
            )
            print("right home joints + gripper raw:")
            print(",".join(f"{v:.6f}" for v in right_pose))
            print(
                f'  --home-joints-right "{",".join(f"{v:.6f}" for v in right_pose[:6])}"'
            )
        if args.save_home and left_pose is not None and right_pose is not None:
            save_home_pose(left_pose, right_pose)
        elif args.save_home:
            print("Need both arms to --save-home. Use default dual mode.", file=sys.stderr)
        return

    if args.warmup_sec > 0:
        rospy.loginfo(f"Warmup {args.warmup_sec:.1f}s — Ctrl+C to abort before motion.")
        rospy.sleep(args.warmup_sec)

    arms_plan: list[tuple[str, JointReplayer, np.ndarray, np.ndarray]] = []

    if do_left:
        start = read_current_pose(args.left_joint_states_topic, args.joint_state_timeout, "left")
        home = resolve_home_pose(args, "left", saved_home)
        replayer = JointReplayer(
            "left",
            args.left_arm_topic,
            args.left_gripper_topic,
            DEFAULT_JOINT_NAMES,
            "world",
            1000,
        )
        arms_plan.append(("left", replayer, start, home))

    if do_right:
        start = read_current_pose(args.right_joint_states_topic, args.joint_state_timeout, "right")
        home = resolve_home_pose(args, "right", saved_home)
        replayer = JointReplayer(
            "right",
            args.right_arm_topic,
            args.right_gripper_topic,
            DEFAULT_JOINT_NAMES,
            "world",
            1000,
        )
        arms_plan.append(("right", replayer, start, home))

    if not arms_plan:
        raise SystemExit("Nothing to home.")

    gripper_max = None if args.gripper_max < 0 else args.gripper_max

    for label, replayer, start, home in arms_plan:
        replayer.wait_for_subscribers(args.subscriber_timeout)
        delta = float(np.linalg.norm(shortest_joint_delta(start[:6], home[:6])))
        rospy.loginfo(f"[{label}] delta to home: {delta:.4f} rad (shortest path)")

    if not args.yes:
        labels = ", ".join(x[0] for x in arms_plan)
        print(f"\nAbout to home: {labels}")
        print(f"Duration: {args.duration_sec:.1f}s   Ctrl+C to abort during motion.")
        for label, _, start, home in arms_plan:
            print(f"  {label} wrist: {start[5]:.3f} -> {home[5]:.3f} rad")
        try:
            answer = input("Proceed? [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in ("y", "yes"):
            print("Aborted.")
            return

    try:
        for label, replayer, start, home in arms_plan:
            go_home_arm(
                label,
                replayer,
                start,
                home,
                duration_sec=args.duration_sec,
                rate_hz=args.rate_hz,
                hold_sec=args.hold_sec,
                gripper_scale=args.gripper_scale,
                gripper_max=gripper_max,
                max_speed_rad_s=args.max_speed_rad_s,
                at_home_tol=args.at_home_tol,
            )
    except rospy.ROSInterruptException:
        rospy.loginfo("Homing interrupted (Ctrl+C).")


if __name__ == "__main__":
    main()
