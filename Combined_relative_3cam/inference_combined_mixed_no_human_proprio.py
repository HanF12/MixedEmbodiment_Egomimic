#!/usr/bin/env python
"""
Combined-relative-3cam ACT inference for *mixed* embodiment:
  - one robot arm (left OR right)
  - one human hand (the opposite side; present only in the bird view)

This script intentionally runs the model through the ROBOT joint pathway
(`embodiment=EMBODIMENT_ROBOT`) so we get `joint_pred` and can command the robot.
The "mixed" aspect is implemented purely by:
  - packing a *single-arm* joint_state into the 14D vector (inactive arm = zeros)
  - masking cameras to keep bird + active wrist only
  - publishing joint targets to the active arm only

It is compatible with "no human proprio" checkpoints (no human_input_* / human_cvae_state_*).
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import torch
import torch.nn as nn
from sensor_msgs.msg import JointState

_PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PKG_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_PKG_DIR))

from config import (  # noqa: E402
    CAMERA_ORDER,
    DEFAULT_NUM_QUERIES,
    EMBODIMENT_ROBOT,
    GRIPPER_INDICES,
    LEFT_ARM_SLICE,
    MODEL_CAMERA_NAMES,
    POSE_DIM,
    RIGHT_ARM_SLICE,
    ROBOT_JOINT_DIM,
    stack_camera_tensors,
    validate_run_metadata,
    load_run_metadata,
)
from core import build  # noqa: E402

ALOHA_DIR = (Path(__file__).resolve().parents[1] / "ALOHA-mimic").resolve()
if str(ALOHA_DIR) not in sys.path:
    sys.path.insert(0, str(ALOHA_DIR))

from joint_lisener import (  # type: ignore  # noqa: E402
    get_current_slave_left_positions,
    get_current_slave_right_positions,
    joint_state_listener,
)


RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

# Reasonable defaults for fold-3 mixed checkpoints in this repo.
DEFAULT_CHECKPOINT = "mixed_act_epoch_9000_fold_3.pth"
DEFAULT_MIXED_NORM = "normalization_stats_mixed_fold_3.npz"


def convert_model_to_no_human_proprio(model: nn.Module) -> nn.Module:
    """
    Drop human proprio / CVAE-state adapters so state_dict matches checkpoints
    that have neither pose Linears nor learned constants.
    """
    for name in (
        "human_input_proj",
        "human_cvae_state_proj",
        "human_input_const",
        "human_cvae_state_const",
    ):
        if hasattr(model, name):
            delattr(model, name)
    return model


def resolve_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path


def serial_matches(requested: str, device_serial: str) -> bool:
    return requested == device_serial or requested in device_serial or device_serial.endswith(requested)


def load_camera_map() -> dict[str, str]:
    try:
        from recording.hand_pose_track import CAMERA_MAP  # type: ignore

        return dict(CAMERA_MAP)
    except Exception:
        return {}


def connected_realsense_devices():
    ctx = rs.context()
    return list(ctx.query_devices())


def pick_realsense_serial(*, role: str | None, serial_arg: str | None) -> str:
    devices = connected_realsense_devices()
    if not devices:
        raise RuntimeError("No RealSense devices found.")

    if serial_arg:
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            if serial_matches(serial_arg, serial):
                return serial
        raise RuntimeError(f"Requested RealSense serial {serial_arg!r} not found.")

    camera_map = load_camera_map()
    if role:
        for mapped_serial, mapped_role in camera_map.items():
            if mapped_role != role:
                continue
            for device in devices:
                serial = device.get_info(rs.camera_info.serial_number)
                if serial_matches(mapped_serial, serial):
                    return serial

    if len(devices) == 1:
        return devices[0].get_info(rs.camera_info.serial_number)
    connected = [device.get_info(rs.camera_info.serial_number) for device in devices]
    raise RuntimeError(f"Multiple RealSense devices detected; please pass a serial or role. Connected: {connected}")


def to_resnet_norm_rgb_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).to(dtype=torch.float32).div_(255.0)
    return (tensor - RESNET_MEAN) / RESNET_STD


def poll_color_bgr(pipeline: rs.pipeline, *, timeout_ms: int = 120) -> Optional[np.ndarray]:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        frames = pipeline.poll_for_frames()
        if frames:
            color_frame = frames.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
        time.sleep(0.002)
    return None


def maybe_resize(img_bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def annotate(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img_bgr.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22
    return out


def stack_preview_3cam(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """bird | left_wrist on top row; right_wrist alone on bottom (letterboxed left)."""
    top = np.concatenate([a, b], axis=1)
    pad = np.zeros_like(c)
    bottom = np.concatenate([c, pad], axis=1)
    return np.concatenate([top, bottom], axis=0)


def load_joint_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(str(path), allow_pickle=True)
    if "joint_mean" in stats.files and "joint_std" in stats.files:
        mean = np.asarray(stats["joint_mean"], dtype=np.float32)
        std = np.asarray(stats["joint_std"], dtype=np.float32)
    elif "qpos_mean" in stats.files and "qpos_std" in stats.files:
        mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
        std = np.asarray(stats["qpos_std"], dtype=np.float32)
    else:
        raise KeyError(
            f"{path} missing joint_mean/joint_std or qpos_mean/qpos_std (keys={stats.files})."
        )
    if mean.shape != (ROBOT_JOINT_DIM,) or std.shape != (ROBOT_JOINT_DIM,):
        raise ValueError(
            f"Expected joint norm shape ({ROBOT_JOINT_DIM},), got mean={mean.shape} std={std.shape}"
        )
    return mean, std


def mixed_camera_mask(robot_side: str) -> torch.Tensor:
    """
    Return float mask [3] aligned with CAMERA_ORDER=(bird,left_wrist,right_wrist).
    """
    if robot_side == "left":
        return torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    if robot_side == "right":
        return torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    raise ValueError(f"robot_side must be 'left' or 'right', got {robot_side!r}")


def pack_single_arm_qpos(
    arm_qpos: np.ndarray,
    *,
    robot_side: str,
    binarize_gripper: bool,
    gripper_threshold: float,
) -> np.ndarray:
    """
    Build a 14D joint vector with inactive arm = zeros.
    """
    q = np.zeros((ROBOT_JOINT_DIM,), dtype=np.float32)
    arm = np.asarray(arm_qpos, dtype=np.float32).reshape(-1)
    if arm.size < 7:
        raise ValueError(f"Expected >=7 joints, got {arm.size}")
    if robot_side == "left":
        q[LEFT_ARM_SLICE] = arm[:7]
        grip_idx = GRIPPER_INDICES[0]
    elif robot_side == "right":
        q[RIGHT_ARM_SLICE] = arm[:7]
        grip_idx = GRIPPER_INDICES[1]
    else:
        raise ValueError(f"robot_side must be 'left' or 'right', got {robot_side!r}")
    if binarize_gripper:
        thr = float(gripper_threshold)
        q[grip_idx] = 1.0 if float(q[grip_idx]) >= thr else 0.0
    return q


class ArmPublishers:
    def __init__(self, arm_topic: str, gripper_topic: str, frame_id: str = "world"):
        self.arm_pub = rospy.Publisher(arm_topic, JointState, queue_size=10)
        self.gripper_pub = None
        self.gripper_msg_type = None
        self.frame_id = frame_id
        try:
            from signal_arm.msg import gripper_position_control  # type: ignore

            self.gripper_msg_type = gripper_position_control
            self.gripper_pub = rospy.Publisher(gripper_topic, gripper_position_control, queue_size=10)
        except Exception as exc:
            rospy.logwarn(f"Gripper publisher disabled for {gripper_topic}: {exc}")

    def publish(self, positions: np.ndarray) -> None:
        arm_msg = JointState()
        arm_msg.header.stamp = rospy.Time.now()
        arm_msg.header.frame_id = self.frame_id
        arm_msg.name = [f"joint{i}" for i in range(1, 7)]
        arm_msg.position = positions[:6].tolist()
        self.arm_pub.publish(arm_msg)

        if self.gripper_pub is not None and self.gripper_msg_type is not None and len(positions) >= 7:
            grip_msg = self.gripper_msg_type()
            grip_msg.header.stamp = arm_msg.header.stamp
            grip_msg.header.frame_id = self.frame_id
            grip_msg.gripper_stroke = float(positions[6])
            self.gripper_pub.publish(grip_msg)


class Args:
    def __init__(self, num_queries: int):
        self.num_queries = int(num_queries)
        self.camera_names = list(MODEL_CAMERA_NAMES)
        self.hidden_dim = 512
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False
        self.position_embedding = "sine"
        self.backbone = "resnet18"
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False


parser = argparse.ArgumentParser(
    description=(
        "Combined-relative-3cam ACT inference for mixed one-arm embodiment "
        "(no human proprio checkpoints; absolute joint targets)"
    )
)
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
parser.add_argument(
    "--robot_side",
    choices=("left", "right"),
    required=True,
    help="Which side is the robot arm to command (the other side is the human hand).",
)
parser.add_argument(
    "--normalization_path",
    type=str,
    default=DEFAULT_MIXED_NORM,
    help="Mixed normalization npz (must include joint_mean/joint_std or qpos_mean/qpos_std; shape (14,))",
)
parser.add_argument("--num_queries", type=int, default=DEFAULT_NUM_QUERIES)
parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--display_scale", type=float, default=0.5)
parser.add_argument("--display_max_fps", type=float, default=15.0)
parser.add_argument("--chunking", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--inference_fps", type=float, default=15.0)
parser.add_argument("--resize_factor", type=float, default=1.0)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--aggregation_horizon", type=int, default=None)
parser.add_argument("--left_joint_topic", type=str, default="/joint_states_slave_left")
parser.add_argument("--right_joint_topic", type=str, default="/joint_states_slave_right")
parser.add_argument("--topic_arm_left", type=str, default="/arm_joint_target_position_slave_left")
parser.add_argument("--topic_gripper_left", type=str, default="/gripper_position_control_slave_left")
parser.add_argument("--topic_arm_right", type=str, default="/arm_joint_target_position_slave_right")
parser.add_argument("--topic_gripper_right", type=str, default="/gripper_position_control_slave_right")
parser.add_argument(
    "--binarize_input_gripper",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If set, binarize the active-arm joint-state gripper channel to {0,1} before normalization/model input.",
)
parser.add_argument(
    "--input_gripper_binarize_threshold",
    type=float,
    # Training default in this repo (see Combined_relative_3cam/config.py)
    default=0.7,
    help="Threshold for --binarize_input_gripper (value >= thr → 1, else 0).",
)
parser.add_argument(
    "--gripper_mode",
    choices=("binary", "continuous"),
    default="binary",
    help="binary: threshold denorm pred to 0/70; continuous: scale*pred then clamp to --gripper_max",
)
parser.add_argument(
    "--gripper_threshold",
    type=float,
    default=0.55,
    help="[binary] Active gripper: denormalized pred < threshold → 0, else 70",
)
parser.add_argument(
    "--gripper_scale",
    type=float,
    default=65.0,
    help="[continuous] Multiply denormalized gripper pred by this before publishing",
)
parser.add_argument(
    "--gripper_max",
    type=float,
    default=80.0,
    help="[continuous] Clamp scaled gripper cmd to this (set <0 to disable)",
)
parser.add_argument("--max_joint_speed", type=float, default=0.5)
parser.add_argument("--max_gripper_speed", type=float, default=100000)
parser.add_argument("--bird_role", choices=("left", "right", "center", "front"), default="center")
parser.add_argument("--bird_serial", type=str, default=None)
parser.add_argument("--bird_color_fps", type=int, default=15)
parser.add_argument("--left_wrist_role", choices=("left", "right", "center", "front"), default="left")
parser.add_argument("--left_wrist_serial", type=str, default=None)
parser.add_argument("--left_wrist_color_fps", type=int, default=15)
parser.add_argument("--right_wrist_role", choices=("left", "right", "center", "front"), default="right")
parser.add_argument("--right_wrist_serial", type=str, default=None)
parser.add_argument("--right_wrist_color_fps", type=int, default=15)
cli = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device} cuda_available={torch.cuda.is_available()}")
print(
    f"mode=mixed robot_side={cli.robot_side} "
    f"safety: max_joint_speed={cli.max_joint_speed:g} rad/s, max_gripper_speed={cli.max_gripper_speed:g}, "
    f"gripper_mode={cli.gripper_mode}"
)
print(f"cameras={list(CAMERA_ORDER)} (true 3-cam, pose_dim={POSE_DIM})")
print(f"camera_mask={mixed_camera_mask(str(cli.robot_side)).tolist()} (bird + active wrist)")

checkpoint_path = resolve_path(cli.checkpoint)
metadata = load_run_metadata(checkpoint_path.parent)
if metadata is not None:
    try:
        validate_run_metadata(metadata, num_queries=cli.num_queries)
        print(f"run_metadata: pose_action_space={metadata.get('pose_action_space')} (joint control uses joint_pred)")
    except ValueError as exc:
        print(f"Warning: skipping run_metadata validation: {exc}")

model = build(Args(cli.num_queries)).to(device)
convert_model_to_no_human_proprio(model)
state_dict = torch.load(str(checkpoint_path), map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"Loaded no-human-proprio checkpoint: {checkpoint_path}")

norm_path = resolve_path(cli.normalization_path)
joint_mean_np, joint_std_np = load_joint_norm_stats(norm_path)
qpos_mean = torch.from_numpy(joint_mean_np.reshape(1, ROBOT_JOINT_DIM)).to(device)
qpos_std = torch.from_numpy(joint_std_np.reshape(1, ROBOT_JOINT_DIM)).to(device)
print(f"Joint norms: {norm_path}")

cam_mask = mixed_camera_mask(str(cli.robot_side)).unsqueeze(0).to(device)  # [1,3]
# Dummy pose_state for API; no-human-proprio checkpoints ignore human state adapters anyway.
dummy_pose_state = torch.zeros(1, POSE_DIM, dtype=torch.float32, device=device)

rospy.init_node("combined_relative_act_inference_mixed_no_human_proprio", anonymous=True)
joint_state_listener(topic=str(cli.left_joint_topic), side="left")
joint_state_listener(topic=str(cli.right_joint_topic), side="right")
left_publishers = ArmPublishers(str(cli.topic_arm_left), str(cli.topic_gripper_left))
right_publishers = ArmPublishers(str(cli.topic_arm_right), str(cli.topic_gripper_right))

color_width = int(cli.width)
color_height = int(cli.height)
camera_specs = [
    (CAMERA_ORDER[0], str(cli.bird_role), cli.bird_serial, int(cli.bird_color_fps)),
    (CAMERA_ORDER[1], str(cli.left_wrist_role), cli.left_wrist_serial, int(cli.left_wrist_color_fps)),
    (CAMERA_ORDER[2], str(cli.right_wrist_role), cli.right_wrist_serial, int(cli.right_wrist_color_fps)),
]

pipelines: list[tuple[str, str, rs.pipeline]] = []
for label, role, serial_arg, fps in camera_specs:
    serial = pick_realsense_serial(role=role, serial_arg=serial_arg)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
    pipeline.start(config)
    pipelines.append((label, serial, pipeline))
    cam_id = list(CAMERA_ORDER).index(label)
    print(f"Started {label} -> model cam{cam_id}/backbone{cam_id} on serial {serial} (role={role})")

prediction_horizon = int(cli.num_queries)
aggregation_horizon = int(cli.aggregation_horizon) if cli.aggregation_horizon is not None else prediction_horizon
decay = 0.075
past_predictions_buffer = collections.deque(maxlen=aggregation_horizon)
loop_rate = rospy.Rate(float(cli.inference_fps))
last_cmd: Optional[np.ndarray] = None
last_cmd_t: Optional[float] = None
last_preview_t = 0.0

robot_side = str(cli.robot_side)
active_offset = 0 if robot_side == "left" else 7
active_grip_idx = GRIPPER_INDICES[0] if robot_side == "left" else GRIPPER_INDICES[1]

try:
    while not rospy.is_shutdown():
        frames = [poll_color_bgr(pipeline, timeout_ms=90) for _, _, pipeline in pipelines]
        left_state = get_current_slave_left_positions()
        right_state = get_current_slave_right_positions()
        if any(frame is None for frame in frames) or left_state is None or right_state is None:
            loop_rate.sleep()
            continue

        if robot_side == "left":
            active_arm = np.asarray(list(left_state[0]), dtype=np.float32)
        else:
            active_arm = np.asarray(list(right_state[0]), dtype=np.float32)

        qpos_np = pack_single_arm_qpos(
            active_arm,
            robot_side=robot_side,
            binarize_gripper=bool(cli.binarize_input_gripper),
            gripper_threshold=float(cli.input_gripper_binarize_threshold),
        )
        joint_state = torch.from_numpy(qpos_np).unsqueeze(0).to(device)
        joint_state = (joint_state - qpos_mean) / qpos_std

        if float(cli.resize_factor) != 1.0:
            frames = [maybe_resize(frame, float(cli.resize_factor)) for frame in frames]

        # Stack by label so order always matches training:
        # CAMERA_ORDER = (bird, left_wrist, right_wrist) -> cam0/cam1/cam2.
        frame_by_label = {label: frame for (label, _, _), frame in zip(pipelines, frames)}
        stacked_images = stack_camera_tensors(
            to_resnet_norm_rgb_tensor(frame_by_label["bird"]),
            to_resnet_norm_rgb_tensor(frame_by_label["left_wrist"]),
            to_resnet_norm_rgb_tensor(frame_by_label["right_wrist"]),
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(
                pose_state=dummy_pose_state,
                images=stacked_images,
                embodiment=EMBODIMENT_ROBOT,
                joint_state=joint_state,
                camera_mask=cam_mask,
            )
            pred = out["joint_pred"]
            if pred is None:
                raise RuntimeError("joint_pred is None (expected in robot joint path)")

        predicted_trajectory = pred[0] * qpos_std.squeeze(0) + qpos_mean.squeeze(0)
        predicted_trajectory_np = predicted_trajectory.cpu().numpy()
        past_predictions_buffer.append(predicted_trajectory_np)

        positions_to_publish = np.zeros((ROBOT_JOINT_DIM,), dtype=np.float32)
        if bool(cli.chunking):
            wsum = 0.0
            for i in range(len(past_predictions_buffer)):
                buffered_pred = past_predictions_buffer[-(i + 1)]
                if i < buffered_pred.shape[0]:
                    weight = float(np.exp(-decay * i))
                    positions_to_publish += buffered_pred[i] * weight
                    wsum += weight
            if wsum > 0:
                positions_to_publish /= wsum
            else:
                positions_to_publish = predicted_trajectory_np[0].astype(np.float32)
        else:
            positions_to_publish = predicted_trajectory_np[0].astype(np.float32)

        # Post-process ACTIVE gripper only; inactive arm remains unused (and unpublished).
        raw_grip = float(positions_to_publish[active_grip_idx])
        if cli.gripper_mode == "binary":
            cmd_grip = 70.0 if raw_grip >= float(cli.gripper_threshold) else 0.0
        else:
            cmd_grip = raw_grip * float(cli.gripper_scale)
            if float(cli.gripper_max) >= 0:
                cmd_grip = min(cmd_grip, float(cli.gripper_max))
        positions_to_publish[active_grip_idx] = float(cmd_grip)

        desired = positions_to_publish.astype(np.float32)
        now_t = time.monotonic()
        if last_cmd is None or last_cmd_t is None:
            last_cmd = qpos_np.astype(np.float32).copy()
            # Seed gripper command in the active slot
            if cli.gripper_mode == "binary":
                last_cmd[active_grip_idx] = 70.0 if float(last_cmd[active_grip_idx]) >= float(cli.gripper_threshold) else 0.0
            else:
                last_cmd[active_grip_idx] = float(last_cmd[active_grip_idx]) * float(cli.gripper_scale)
                if float(cli.gripper_max) >= 0:
                    last_cmd[active_grip_idx] = min(float(last_cmd[active_grip_idx]), float(cli.gripper_max))
            last_cmd_t = now_t
        else:
            dt_nom = 1.0 / max(1e-3, float(cli.inference_fps))
            dt = min(max(1e-3, float(now_t - last_cmd_t)), dt_nom)
            max_dq = float(cli.max_joint_speed) * dt
            max_dg = float(cli.max_gripper_speed) * dt
            cmd = last_cmd.copy()

            offset = int(active_offset)
            dq = desired[offset : offset + 6] - cmd[offset : offset + 6]
            dq = np.clip(dq, -max_dq, max_dq)
            cmd[offset : offset + 6] += dq
            dg = float(np.clip(desired[offset + 6] - cmd[offset + 6], -max_dg, max_dg))
            cmd[offset + 6] += dg

            last_cmd = cmd.astype(np.float32)
            last_cmd_t = now_t

        if robot_side == "left":
            left_publishers.publish(last_cmd[LEFT_ARM_SLICE])
        else:
            right_publishers.publish(last_cmd[RIGHT_ARM_SLICE])

        if bool(cli.display):
            wall = time.time()
            min_dt = 1.0 / max(1e-3, float(cli.display_max_fps))
            if wall - last_preview_t >= min_dt:
                last_preview_t = wall
                grip_overlay = (
                    f"robot_side={robot_side} "
                    f"raw_grip={raw_grip:.3f} -> cmd_grip={float(cmd_grip):.1f} "
                    f"(cam_mask={cam_mask.squeeze(0).tolist()})"
                )
                print(f"gripper {grip_overlay}", flush=True)
                preview_order = ("bird", "left_wrist", "right_wrist")
                serial_by_label = {label: serial for label, serial, _ in pipelines}
                shown = []
                for cam_id, label in enumerate(preview_order):
                    shown.append(
                        annotate(
                            frame_by_label[label],
                            [
                                f"cam{cam_id}={label} serial={serial_by_label[label]}",
                                grip_overlay,
                            ],
                        )
                    )
                shown = [maybe_resize(frame, float(cli.display_scale)) for frame in shown]
                h = min(frame.shape[0] for frame in shown)
                w = min(frame.shape[1] for frame in shown)
                preview = stack_preview_3cam(
                    shown[0][:h, :w],  # bird (cam0)
                    shown[1][:h, :w],  # left_wrist (cam1)
                    shown[2][:h, :w],  # right_wrist (cam2)
                )
                cv2.imshow("Combined-relative-3cam ACT Inference (mixed, no human proprio)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        loop_rate.sleep()
except KeyboardInterrupt:
    print("Inference stopped by user.")
finally:
    for _, _, pipeline in pipelines:
        try:
            pipeline.stop()
        except Exception:
            pass
    cv2.destroyAllWindows()

