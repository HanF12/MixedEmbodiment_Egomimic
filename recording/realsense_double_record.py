import cv2
import pyrealsense2 as rs
import numpy as np
import os
import signal
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

from hand_pose_track import CAMERA_MAP
from realsense_utils import drain_pipeline, poll_for_frames, serial_for_role, warmup_pipeline
from recording_paths import under_recording
from recording_sync import read_recording_start, wait_for_recording_go, wrist_ready_path

stop_recording = False

def request_stop(signum, frame):
    global stop_recording
    stop_recording = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record one wrist RealSense color stream (one process per camera)."
    )
    parser.add_argument(
        "--fps", "-f",
        action="store_true",
        help="If set, print the approximate FPS for each frame.",
    )
    parser.add_argument(
        "--color-fps",
        type=int,
        default=15,
        help="Color stream FPS (default: 15 for USB bandwidth with 3 cameras).",
    )
    parser.add_argument(
        "--datetime-id",
        type=str,
        default=None,
        help="Shared timestamp id for output filenames (default: now).",
    )
    parser.add_argument(
        "--arms",
        choices=("left", "right"),
        required=True,
        help="Which wrist RealSense camera to record.",
    )
    parser.add_argument(
        "--wait-for-go",
        action="store_true",
        help="Wait for record_pedal.py sync signal before saving frames.",
    )
    return parser.parse_args()


def find_arm_device(devices, role: str):
    """Return (device, serial) for left/right role, or (None, None)."""
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        if CAMERA_MAP.get(serial) == role:
            return device, serial
    return None, None


def main(args):
    global stop_recording
    should_stop = lambda: stop_recording
    arm_type = args.arms

    ctx = rs.context()
    devices = list(ctx.query_devices())
    print(f"Found {len(devices)} RealSense cameras.")
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        role = CAMERA_MAP.get(serial, "unassigned")
        print(f"  {serial}  ->  {role}")

    device, serial_number = find_arm_device(devices, arm_type)
    if device is None:
        expected = serial_for_role(CAMERA_MAP, arm_type) or arm_type
        print(f"ERROR: {arm_type.capitalize()} arm RealSense ({expected}) is not connected / not visible on USB.")
        connected = [d.get_info(rs.camera_info.serial_number) for d in devices]
        print(f"  Currently visible RealSense serials: {connected or '(none)'}")
        sys.exit(1)

    print(f"Configuring camera with serial number: {serial_number} (Mapped to {arm_type} arm)")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, args.color_fps)

    current_datetime_id = args.datetime_id or datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        profile = pipeline.start(config)
        warmup_pipeline(pipeline, should_stop=should_stop)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(
            f"Camera {serial_number} ({arm_type}) Color Stream: "
            f"Resolution {color_profile.width()}x{color_profile.height()}, FPS: {color_profile.fps()}"
        )
    except RuntimeError as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("Please ensure the requested camera is connected and not in use by another application.")
        sys.exit(1)

    mp4_dir = os.path.join(under_recording("aloha-data"), arm_type, "mp4")
    npy_dir = os.path.join(under_recording("aloha-data"), arm_type, "npy")
    os.makedirs(mp4_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    mp4_path = os.path.join(mp4_dir, f"video_recording_realsense_{arm_type}#{current_datetime_id}.mp4")
    npy_path = os.path.join(npy_dir, f"video_recording_realsense_{arm_type}#{current_datetime_id}.npy")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    actual_color_width = color_profile.width()
    actual_color_height = color_profile.height()
    actual_color_fps = color_profile.fps()

    video_writer = cv2.VideoWriter(
        mp4_path,
        fourcc,
        actual_color_fps,
        (actual_color_width, actual_color_height),
    )

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for camera {serial_number} ({arm_type}) for {mp4_path}.")
        pipeline.stop()
        sys.exit(1)

    frame_array = []
    print(f"Recording video from Intel RealSense camera {serial_number} ({arm_type}) to {mp4_path}.")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    ready_path = wrist_ready_path(current_datetime_id, arm_type)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.touch()
    print(f"{arm_type.capitalize()} wrist ready (signal: {ready_path})", flush=True)

    if args.wait_for_go and not wait_for_recording_go(
        current_datetime_id, label=f"wrist-{arm_type}", should_stop=should_stop
    ):
        pipeline.stop()
        video_writer.release()
        return

    recording_t0 = read_recording_start(current_datetime_id)
    drained = drain_pipeline(pipeline, should_stop=should_stop)
    print(f"[wrist-{arm_type}] Drained {drained} stale frame(s) from pipeline buffer.", flush=True)

    print("Press 'q' to stop recording.", flush=True)

    start_time = time.time()
    if recording_t0 is not None:
        print(f"Shared recording t0={recording_t0:.3f} (local now={start_time:.3f})", flush=True)
    prev_time = time.time()

    try:
        while not stop_recording:
            frames = poll_for_frames(pipeline, timeout_ms=100, should_stop=should_stop)
            if frames is None:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_t = time.time()
            color_image = np.asanyarray(color_frame.get_data())
            video_writer.write(color_image)
            cv2.imshow(
                f'RealSense Video Recording - {arm_type.capitalize()} Arm ({serial_number})',
                color_image,
            )
            frame_array.append(frame_t)

            if args.fps:
                current_time = time.time()
                dt = current_time - prev_time
                if dt > 0:
                    fps = 1.0 / dt
                    print(f"Camera {serial_number} ({arm_type}) Approximate FPS: {fps:.2f}")
                prev_time = current_time

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            ready_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        video_writer.release()
        cv2.destroyAllWindows()

        np.save(npy_path, np.asarray(frame_array, dtype=np.float64))
        n_frames = len(frame_array)
        print(f"Saved {n_frames} frames from {arm_type} arm ({serial_number})", flush=True)
        if n_frames == 0:
            expected = serial_for_role(CAMERA_MAP, arm_type) or arm_type
            print(
                f"ERROR: {arm_type.capitalize()} arm camera recorded ZERO frames — video file will be empty. "
                f"Check USB for serial {expected}.",
                flush=True,
            )

    end_time = time.time()
    duration = end_time - start_time

    print(f"Recording stopped. Total Duration: {duration:.2f} seconds.")
    print(f"Camera {serial_number} ({arm_type}) — {len(frame_array)} frames — video: '{mp4_path}'")


if __name__ == "__main__":
    args = parse_args()
    main(args)
