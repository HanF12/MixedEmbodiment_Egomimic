#!/usr/bin/env python

from __future__ import annotations

import rospy
from sensor_msgs.msg import JointState # Explicitly import JointState
from ros_pub import * # Explicitly import function
from joint_lisener import (
    joint_state_listener,
    get_current_slave_left_positions,
    get_current_slave_right_positions,
) # Explicitly import functions
import numpy as np
import os
import sys
import time
import torch
import cv2
import pyrealsense2 as rs
import torch.nn.functional as F # For image resizing
import argparse
import warnings
from core import build
import collections # Import collections for deque
from pathlib import Path
from typing import Optional

RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

# --- Global Configuration ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path


def _serial_matches(requested: str, device_serial: str) -> bool:
    # Allow passing suffixes / substrings (matches recording scripts).
    return requested == device_serial or requested in device_serial or device_serial.endswith(requested)


def _load_camera_map() -> dict[str, str] | None:
    try:
        from recording.hand_pose_track import CAMERA_MAP  # type: ignore

        return dict(CAMERA_MAP)
    except Exception:
        return None


def _connected_realsense_devices():
    ctx = rs.context()
    return list(ctx.query_devices())


def _pick_realsense_serial(*, role: str | None, serial_arg: str | None) -> str:
    devices = _connected_realsense_devices()
    if not devices:
        raise RuntimeError("No RealSense devices found.")

    # 1) Explicit serial wins.
    if serial_arg:
        for d in devices:
            s = d.get_info(rs.camera_info.serial_number)
            if _serial_matches(serial_arg, s):
                return s
        raise RuntimeError(
            f"Requested RealSense serial {serial_arg!r} not found. "
            f"Connected: {[d.get_info(rs.camera_info.serial_number) for d in devices]}"
        )

    # 2) Role-based selection using CAMERA_MAP.
    if role:
        camera_map = _load_camera_map() or {}
        # Exact match
        for s, mapped_role in camera_map.items():
            if mapped_role == role:
                # Map key might be a shortened serial; match against connected
                for d in devices:
                    dev_s = d.get_info(rs.camera_info.serial_number)
                    if _serial_matches(s, dev_s):
                        return dev_s
        # If role not found or not connected, fall through.

    # 3) If only one device, pick it (best-effort).
    if len(devices) == 1:
        return devices[0].get_info(rs.camera_info.serial_number)

    # 4) Otherwise, force the user to pick (prevents silently selecting wrong camera).
    camera_map = _load_camera_map() or {}
    lines = []
    for d in devices:
        s = d.get_info(rs.camera_info.serial_number)
        lines.append(f"  {s}  (role: {camera_map.get(s, 'unassigned')})")
    raise RuntimeError(
        "Multiple RealSense devices detected; please specify --wrist_serial/--wrist_role "
        "and (if using bird realsense) --bird_serial/--bird_role.\n"
        + "\n".join(lines)
    )


parser = argparse.ArgumentParser(description="ACT single-arm inference controller (ROS + RealSense + bird cam)")
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    default="Vinilla_ACT.pth",
    help="Path to a .pth state_dict (relative paths are resolved from this script's directory).",
)
parser.add_argument(
    "-q",
    "--num_queries",
    type=int,
    default=100,
    help="Must match training (training_single.py default: 100).",
)
parser.add_argument("--arm_side", choices=("left", "right"), default="right")
parser.add_argument(
    "--normalization_path",
    type=str,
    default="normalization_stats_direct.npz",
    help="npz with qpos_mean/qpos_std saved by training_single.py",
)
parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--display_scale", type=float, default=0.6)
parser.add_argument("--display_max_fps", type=float, default=15.0)
parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--chunking", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--inference_fps", type=float, default=15.0)
parser.add_argument("--resize_factor", type=float, default=1.0)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--aggregation_horizon", type=int, default=None, help="Chunking buffer length (default: num_queries).")
parser.add_argument("--joint_topic", type=str, default=None)
parser.add_argument("--topic_arm", type=str, default=None)
parser.add_argument("--topic_gripper", type=str, default=None)
parser.add_argument(
    "--gripper_scale",
    type=float,
    default=30.2,
    help="Scale the model's gripper output before publishing (match your controller units).",
)
parser.add_argument(
    "--gripper_max",
    type=float,
    default=60.0,
    help="Clamp published gripper target to this max (use <0 to disable).",
)
parser.add_argument(
    "--max_joint_speed",
    type=float,
    default=0.25,
    help="Rate-limit arm joint targets (rad/s) to avoid jerky motion at low Hz.",
)
parser.add_argument(
    "--max_gripper_speed",
    type=float,
    default=0.10,
    help="Rate-limit gripper target (units/s in the published gripper units).",
)

# Wrist RealSense selection (3-cam training expects: cam0=left wrist, cam1=right wrist)
parser.add_argument("--left_wrist_role", choices=("left", "right", "center"), default="left")
parser.add_argument("--left_wrist_serial", type=str, default=None)
parser.add_argument("--left_wrist_color_fps", type=int, default=15)
parser.add_argument("--right_wrist_role", choices=("left", "right", "center"), default="right")
parser.add_argument("--right_wrist_serial", type=str, default=None)
parser.add_argument("--right_wrist_color_fps", type=int, default=15)

# Backwards-compat aliases for earlier 2-cam inference (treated as LEFT wrist)
parser.add_argument("--wrist_role", choices=("left", "right", "center"), default=None)
parser.add_argument("--wrist_serial", type=str, default=None)
parser.add_argument("--wrist_color_fps", type=int, default=None)

# Bird (cam2) selection:
# - webcam mode: matches recording/bird_record.py conventions (index or /dev/videoX).
# - realsense mode: matches recording/realsense_bird_record.py conventions (role=center).
parser.add_argument("--bird_source", choices=("webcam", "realsense"), default="realsense")
parser.add_argument("--bird_role", choices=("left", "right", "center"), default="center")
parser.add_argument("--bird_serial", type=str, default=None)
parser.add_argument("--bird_color_fps", type=int, default=15)
parser.add_argument("--bird_device_path", type=str, default=None, help="Webcam path like /dev/video6 (optional).")
parser.add_argument(
    "--bird_device",
    "--bird-camera",
    dest="bird_device",
    type=int,
    default=6,
    help="Webcam index (only used if --bird_source=webcam).",
)
cli = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

if cli.joint_topic is None:
    cli.joint_topic = "/joint_states_slave_right" if cli.arm_side == "right" else "/joint_states_slave_left"

if cli.topic_arm is None:
    cli.topic_arm = f"/arm_joint_target_position_slave_{cli.arm_side}"
if cli.topic_gripper is None:
    cli.topic_gripper = f"/gripper_position_control_slave_{cli.arm_side}"

DEBUG = bool(cli.debug)
CHUNKING = bool(cli.chunking) # Change to true to do the real ACT.

K_PREDICTION_HORIZON = int(cli.num_queries)
K_AGGREGATION_HORIZON = int(cli.aggregation_horizon) if cli.aggregation_horizon is not None else int(cli.num_queries)
m = 0.075 #Decay factor for chunking

INFERENCE_FPS = float(cli.inference_fps)
target_interval_ms = (1.0 / INFERENCE_FPS) * 1000

resize_factor = float(cli.resize_factor) # If you want to resize images, change this. Set to 1.0 means no resizing for now.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Global buffer for ACT temporal chunking
# Stores (K_PREDICTION_HORIZON, 7) numpy arrays of predicted trajectories
past_predictions_buffer = collections.deque(maxlen=K_AGGREGATION_HORIZON)


class Args:
    def __init__(self):
        self.num_queries = int(cli.num_queries)
        # MUST match training_single.py + dataloader_3cam.py stacking:
        # cam0=left wrist, cam1=right wrist, cam2=bird
        self.camera_names = ["cam0", "cam1", "cam2"]
        self.hidden_dim = 512
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False

        # Backbone/DETR args
        self.position_embedding = "sine"
        self.backbone = "resnet18"
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False

        # Custom for your use
        self.state_dim = 7


model = build(Args())
model.to(device)

# --- Joint normalization (must match dataloader_3cam.py) ---
try:
    norm_path = _resolve_path(cli.normalization_path)
    stats = np.load(str(norm_path))
    qpos_mean_np = np.asarray(stats["qpos_mean"], dtype=np.float32).reshape(1, 7)
    qpos_std_np = np.asarray(stats["qpos_std"], dtype=np.float32).reshape(1, 7)
    qpos_mean = torch.from_numpy(qpos_mean_np).to(device)
    qpos_std = torch.from_numpy(qpos_std_np).to(device)
    print(f"Loaded normalization stats: {norm_path}")
except Exception as e:
    print(f"Error loading normalization stats: {e}")
    time.sleep(2)
    exit()

# --- Model Loading ---
try:
    ckpt_path = _resolve_path(cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # Set model to evaluation mode
    print(f"Model Load Success: {ckpt_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    time.sleep(5)
    exit()

print(f"\nStarting inference loop at {INFERENCE_FPS} Hz. Press 'q' to quit.")

# --- ROS Node Initialization and Subscriber Setup ---
# IMPORTANT: joint_state_listener() MUST NOT call rospy.spin() if this script
# is to run an inference loop. It should only initialize the node and subscriber.
# If it calls rospy.spin(), this line will block forever.
try:
    rospy.init_node('act_inference_controller', anonymous=True) # Initialize ROS node here
    init_publishers(topic_arm=str(cli.topic_arm), topic_gripper=str(cli.topic_gripper))
    joint_state_listener(topic=str(cli.joint_topic), side=str(cli.arm_side)) # Assuming this only sets up the subscriber, not blocks.
except rospy.ROSInitException as e:
    print(f"Failed to initialize ROS node: {e}")
    exit()


def _wait_for_subscribers(timeout_sec: float = 10.0) -> None:
    """
    Mirror go_home.py behavior: wait until controllers subscribe.
    Otherwise, publishing once in a while can look like "no movement".
    """
    try:
        arm_pub = _arm_pub  # from ros_pub import *
        grip_pub = _gripper_pub
    except Exception:
        return

    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline and not rospy.is_shutdown():
        arm_n = arm_pub.get_num_connections() if arm_pub is not None else 0
        grip_n = grip_pub.get_num_connections() if grip_pub is not None else 0
        if arm_n > 0 and (grip_pub is None or grip_n > 0):
            rospy.loginfo(
                f"Subscribers connected: arm={arm_n} gripper={grip_n} "
                f"on {cli.topic_arm} / {cli.topic_gripper}"
            )
            return
        rospy.sleep(0.1)
    rospy.logwarn(
        f"No subscribers before timeout on {cli.topic_arm} / {cli.topic_gripper} "
        f"(arm={arm_pub.get_num_connections() if arm_pub is not None else 0}, "
        f"gripper={grip_pub.get_num_connections() if grip_pub is not None else 0})."
    )


_wait_for_subscribers(timeout_sec=10.0)

# --- GPU Information ---
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s):")
    for i in range(n_gpus):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs detected, falling back to CPU.")

# --- RealSense Camera Setup (Outside the loop) ---
left_pipeline = rs.pipeline()
left_config = rs.config()
right_pipeline = rs.pipeline()
right_config = rs.config()
color_width = int(cli.width)
color_height = int(cli.height)

# Back-compat: allow --wrist_* to fill in left wrist settings.
left_role = str(cli.left_wrist_role)
left_serial_arg = cli.left_wrist_serial
left_fps = int(cli.left_wrist_color_fps)
if cli.wrist_role is not None:
    left_role = str(cli.wrist_role)
if cli.wrist_serial is not None:
    left_serial_arg = cli.wrist_serial
if cli.wrist_color_fps is not None:
    left_fps = int(cli.wrist_color_fps)

try:
    left_serial = _pick_realsense_serial(role=left_role, serial_arg=left_serial_arg)
    right_serial = _pick_realsense_serial(role=str(cli.right_wrist_role), serial_arg=cli.right_wrist_serial)
    left_config.enable_device(left_serial)
    right_config.enable_device(right_serial)
except Exception as e:
    print(f"Error selecting wrist RealSense cameras: {e}")
    exit()

left_config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, left_fps)
right_config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, int(cli.right_wrist_color_fps))
try:
    left_profile = left_pipeline.start(left_config)
    right_profile = right_pipeline.start(right_config)
except RuntimeError as e:
    print(f"Error starting RealSense pipeline: {e}")
    print("Please ensure the camera is connected and not in use by another application.")
    exit()

# --- Bird Camera Setup (Outside the loop) ---
bird_cap = None
bird_pipeline = None
bird_profile = None

if str(cli.bird_source) == "realsense":
    try:
        bird_serial = _pick_realsense_serial(role=str(cli.bird_role), serial_arg=cli.bird_serial)
    except Exception as e:
        print(f"Error selecting bird RealSense: {e}")
        left_pipeline.stop()
        right_pipeline.stop()
        exit()

    bird_pipeline = rs.pipeline()
    bird_config = rs.config()
    bird_config.enable_device(bird_serial)
    bird_config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, int(cli.bird_color_fps))
    try:
        bird_profile = bird_pipeline.start(bird_config)
    except RuntimeError as e:
        print(f"Error starting bird RealSense pipeline: {e}")
        print("Please ensure the camera is connected and not in use by another application.")
        left_pipeline.stop()
        right_pipeline.stop()
        exit()
else:
    # IMPORTANT: Move cv2.VideoCapture(...) OUTSIDE the loop!
    bird_src = cli.bird_device_path if cli.bird_device_path else int(cli.bird_device)
    bird_cap = cv2.VideoCapture(bird_src)
    if not bird_cap.isOpened():
        devs = sorted(Path("/dev").glob("video*"))
        print(f"Error: Unable to open bird webcam source {bird_src!r}.")
        if devs:
            print(f"Available /dev/video*: {[str(p) for p in devs]}")
        left_pipeline.stop()
        right_pipeline.stop()
        exit()

action_number = 0 # Counter for the number of inference steps

def _to_resnet_norm_rgb_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """
    Match dataloader_3cam.py transform='resnet_normalization':
      ToTensor() (=> float32 [0,1], CHW) + Normalize(mean,std).
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).to(dtype=torch.float32).div_(255.0)
    return (t - RESNET_MEAN) / RESNET_STD


def _poll_color_bgr(pipeline: rs.pipeline, *, timeout_ms: int = 120):
    """Non-blocking-ish RealSense read like recording/realsense_utils.py."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        frames = pipeline.poll_for_frames()
        if frames:
            cf = frames.get_color_frame()
            if cf:
                return np.asanyarray(cf.get_data())
        time.sleep(0.002)
    return None


def _annotate(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img_bgr
    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22
    return out


def _maybe_resize(img_bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _stack_preview(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    # All images must have same size.
    top = np.concatenate([a, b], axis=1)
    bottom = np.concatenate([c, np.zeros_like(c)], axis=1)
    return np.concatenate([top, bottom], axis=0)


_last_preview_t = 0.0


try:
    loop_rate = rospy.Rate(float(INFERENCE_FPS))
    last_cmd: Optional[np.ndarray] = None
    last_cmd_t: Optional[float] = None

    while not rospy.is_shutdown(): # Use rospy.is_shutdown() for ROS node termination
        # --- 1. Capture Frames (best-effort) ---
        left_bgr = _poll_color_bgr(left_pipeline, timeout_ms=60)
        right_bgr = _poll_color_bgr(right_pipeline, timeout_ms=60)
        if bird_pipeline is not None:
            bird_bgr = _poll_color_bgr(bird_pipeline, timeout_ms=90)
        else:
            ret, bird_bgr = bird_cap.read()
            bird_bgr = bird_bgr if ret else None

        pos_vel_tuple = (
            get_current_slave_right_positions()
            if str(cli.arm_side) == "right"
            else get_current_slave_left_positions()
        )

        if left_bgr is not None and right_bgr is not None and bird_bgr is not None and pos_vel_tuple is not None:
            current_joint_data_np = np.asarray(list(pos_vel_tuple[0]), dtype=np.float32)
            current_joint_data_tensor = torch.from_numpy(current_joint_data_np).to(device).unsqueeze(0)
            current_joint_data_tensor = (current_joint_data_tensor - qpos_mean) / qpos_std

            # Resize if necessary
            if resize_factor != 1.0:
                h_rs, w_rs = left_bgr.shape[:2]
                left_bgr = cv2.resize(left_bgr, (int(w_rs * resize_factor), int(h_rs * resize_factor)), interpolation=cv2.INTER_AREA)
                h_r, w_r = right_bgr.shape[:2]
                right_bgr = cv2.resize(right_bgr, (int(w_r * resize_factor), int(h_r * resize_factor)), interpolation=cv2.INTER_AREA)
                h_b, w_b = bird_bgr.shape[:2]
                bird_bgr = cv2.resize(bird_bgr, (int(w_b * resize_factor), int(h_b * resize_factor)), interpolation=cv2.INTER_AREA)

            stacked_images = torch.stack(
                [
                    _to_resnet_norm_rgb_tensor(left_bgr),
                    _to_resnet_norm_rgb_tensor(right_bgr),
                    _to_resnet_norm_rgb_tensor(bird_bgr),
                ],
                dim=0,
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                a_hat, _, _ = model(current_joint_data_tensor, stacked_images, None)

            predicted_trajectory_norm = a_hat[0]
            predicted_trajectory = predicted_trajectory_norm * qpos_std.squeeze(0) + qpos_mean.squeeze(0)
            predicted_trajectory_np = predicted_trajectory.cpu().numpy()
            past_predictions_buffer.append(predicted_trajectory_np)

            # Optional preview (driven by loop cadence)
            if bool(cli.display):
                wall = time.time()
                min_dt = 1.0 / max(1e-3, float(cli.display_max_fps))
                if wall - _last_preview_t >= min_dt:
                    _last_preview_t = wall
                    l2 = float(np.linalg.norm(predicted_trajectory_np[0] - current_joint_data_np))
                    left_show = _annotate(left_bgr.copy(), [f"cam0 left wrist ({left_serial})"])
                    right_show = _annotate(right_bgr.copy(), [f"cam1 right wrist ({right_serial})"])
                    bird_label = f"cam2 bird ({bird_serial})" if bird_pipeline is not None else f"cam2 bird webcam ({cli.bird_device_path or cli.bird_device})"
                    bird_show = _annotate(
                        bird_bgr.copy(),
                        [bird_label, f"arm_side={cli.arm_side}  ||a0-q||={l2:.4f}"],
                    )
                    s = float(cli.display_scale)
                    left_show = _maybe_resize(left_show, s)
                    right_show = _maybe_resize(right_show, s)
                    bird_show = _maybe_resize(bird_show, s)
                    h = min(left_show.shape[0], right_show.shape[0], bird_show.shape[0])
                    w = min(left_show.shape[1], right_show.shape[1], bird_show.shape[1])
                    preview = _stack_preview(left_show[:h, :w], right_show[:h, :w], bird_show[:h, :w])
                    cv2.imshow("ACT Inference (cam0/cam1/cam2)", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # --- Aggregate to a single command ---
            positions_to_publish = np.zeros((7,), dtype=np.float32)
            if CHUNKING:
                wsum = 0.0
                for i in range(len(past_predictions_buffer)):
                    pred = past_predictions_buffer[-(i + 1)]
                    if i < pred.shape[0]:
                        wc = float(np.exp(-m * i))
                        positions_to_publish += pred[i] * wc
                        wsum += wc
                if wsum > 0:
                    positions_to_publish /= wsum
                else:
                    positions_to_publish = predicted_trajectory_np[0].astype(np.float32)
            else:
                positions_to_publish = predicted_trajectory_np[0].astype(np.float32)

            raw_gripper = float(positions_to_publish[6])

            # Gripper scaling/clamp (controller units)
            if float(cli.gripper_scale) != 1.0:
                positions_to_publish[-1] *= float(cli.gripper_scale)
            if float(cli.gripper_max) >= 0:
                positions_to_publish[-1] = min(float(positions_to_publish[-1]), float(cli.gripper_max))

            scaled_gripper = float(positions_to_publish[6])

            # Rate-limit to avoid big jumps when inference/update Hz is low.
            desired = positions_to_publish.astype(np.float32)
            now_t = time.monotonic()
            if last_cmd is None or last_cmd_t is None:
                # Initialize from the current measured state to avoid a first-step "snap"
                # when the model's initial desired target is far away.
                last_cmd = current_joint_data_np.astype(np.float32).copy()
                last_cmd_t = now_t
            else:
                dt = max(1e-3, float(now_t - last_cmd_t))
                max_dq = float(cli.max_joint_speed) * dt
                max_dg = float(cli.max_gripper_speed) * dt

                cmd = last_cmd.copy()
                dq = desired[:6] - cmd[:6]
                dq = np.clip(dq, -max_dq, max_dq)
                cmd[:6] = cmd[:6] + dq

                dg = float(desired[6] - cmd[6])
                dg = float(np.clip(dg, -max_dg, max_dg))
                cmd[6] = float(cmd[6] + dg)

                last_cmd = cmd.astype(np.float32)
                last_cmd_t = now_t

            if DEBUG:
                # Print a compact view of where the gripper command gets flattened:
                #   model/raw -> scaled/clamped -> published(after rate limit)
                pub_gripper = float(last_cmd[6]) if last_cmd is not None else float("nan")
                rospy.loginfo(
                    f"[gripper] raw={raw_gripper:.6f}  "
                    f"scaled={scaled_gripper:.3f} (scale={float(cli.gripper_scale):g}, max={float(cli.gripper_max):g})  "
                    f"published={pub_gripper:.3f}  max_dg/step={float(cli.max_gripper_speed) * max(1e-3, float(now_t - (last_cmd_t or now_t))):.3f}"
                )

        # Publish at the same cadence as inference updates.
        if last_cmd is not None:
            publish_trajectory(np.asarray([last_cmd], dtype=np.float32))

        loop_rate.sleep()

except KeyboardInterrupt:
    print("Inference stopped by user (KeyboardInterrupt).")
except rospy.ROSInterruptException:
    print("ROS node interrupted (ROSInterruptException).")
finally:
    # --- Clean Up ---
    print("Cleaning up resources...")
    try:
        left_pipeline.stop()
    except Exception:
        pass
    try:
        right_pipeline.stop()
    except Exception:
        pass
    if bird_pipeline is not None:
        try:
            bird_pipeline.stop()
        except Exception:
            pass
    if bird_cap is not None:
        bird_cap.release() # Release Bird Camera
    cv2.destroyAllWindows() # Close OpenCV windows
    print("Cameras released and windows closed.")

# The script implicitly exits after the try-finally block or if `rospy.is_shutdown()` becomes true.