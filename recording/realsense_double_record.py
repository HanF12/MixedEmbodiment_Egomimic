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
from realsense_utils import (
    drain_pipelines,
    poll_for_frames,
    serial_for_role,
    start_pipelines_parallel,
)
from recording_paths import under_recording
from recording_sync import arms_ready_path, read_recording_start, wait_for_recording_go

stop_recording = False

def request_stop(signum, frame):
    global stop_recording
    stop_recording = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record RealSense color stream to disk with optional FPS logging and camera selection."
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
        choices=("left", "right", "both"),
        default="both",
        help="Which wrist RealSense cameras to record (default: both).",
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

    ctx = rs.context()
    devices = list(ctx.query_devices())
    num_devices = len(devices)
    print(f"Found {num_devices} RealSense cameras.")
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        role = CAMERA_MAP.get(serial, "unassigned")
        print(f"  {serial}  ->  {role}")

    arm_roles = ("left", "right")
    if args.arms in arm_roles:
        arm_roles = (args.arms,)
    camera_info = {}

    for role in arm_roles:
        device, serial_number = find_arm_device(devices, role)
        if device is None:
            print(f"Warning: no RealSense mapped as {role} arm is connected.")
            continue

        print(f"Configuring camera with serial number: {serial_number} (Mapped to {role} arm)")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, args.color_fps)

        camera_info[serial_number] = {
            "pipeline": pipeline,
            "config": config,
            "arm_type": role,
            "profile": None,
        }

    if not camera_info:
        print(f"Error: no RealSense camera found for --arms {args.arms!r}.")
        sys.exit(1)

    for role in arm_roles:
        if not any(info["arm_type"] == role for info in camera_info.values()):
            expected = serial_for_role(CAMERA_MAP, role) or role
            print(f"ERROR: {role.capitalize()} arm RealSense ({expected}) is not connected / not visible on USB.")
            connected = [d.get_info(rs.camera_info.serial_number) for d in devices]
            print(f"  Currently visible RealSense serials: {connected or '(none)'}")
            sys.exit(1)

    if args.arms == "both" and len(camera_info) < 2:
        print(f"Warning: only {len(camera_info)} arm camera(s) found; continuing with available arm(s).")

    serial_numbers = [serial for serial, info in camera_info.items() if info["arm_type"] == "left"]
    serial_numbers += [serial for serial, info in camera_info.items() if info["arm_type"] == "right"]

    current_datetime_id = args.datetime_id or datetime.now().strftime("%Y%m%d%H%M%S")

    try:
        start_entries = [
            (serial_number, info["pipeline"], info["config"])
            for serial_number, info in sorted(
                camera_info.items(),
                key=lambda item: 0 if item[1]["arm_type"] == "left" else 1,
            )
        ]
        profiles = start_pipelines_parallel(start_entries, should_stop=should_stop)
        for serial_number, profile in profiles.items():
            info = camera_info[serial_number]
            info["profile"] = profile
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            print(
                f"Camera {serial_number} ({info['arm_type']}) Color Stream: "
                f"Resolution {color_profile.width()}x{color_profile.height()}, FPS: {color_profile.fps()}"
            )
    except RuntimeError as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("Please ensure the requested camera(s) are connected and not in use by another application.")
        for info in camera_info.values():
            try:
                info["pipeline"].stop()
            except Exception:
                pass
        exit()

    base_output_dir = under_recording("aloha-data")
    output_paths = {}
    for serial_number, info in camera_info.items():
        arm_type = info["arm_type"]
        mp4_dir = os.path.join(base_output_dir, arm_type, "mp4")
        npy_dir = os.path.join(base_output_dir, arm_type, "npy")

        os.makedirs(mp4_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)

        output_paths[serial_number] = {
            "mp4_path": os.path.join(
                mp4_dir, f"video_recording_realsense_{arm_type}#{current_datetime_id}.mp4"
            ),
            "npy_path": os.path.join(
                npy_dir, f"video_recording_realsense_{arm_type}#{current_datetime_id}.npy"
            ),
        }

    video_writers = {}
    frame_arrays = {}

    for serial_number, info in camera_info.items():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        color_profile = info["profile"].get_stream(rs.stream.color).as_video_stream_profile()
        actual_color_width = color_profile.width()
        actual_color_height = color_profile.height()
        actual_color_fps = color_profile.fps()

        out = cv2.VideoWriter(
            output_paths[serial_number]["mp4_path"],
            fourcc,
            actual_color_fps,
            (actual_color_width, actual_color_height),
        )

        if not out.isOpened():
            print(
                f"Error: Could not open video writer for camera {serial_number} "
                f"({info['arm_type']}) for {output_paths[serial_number]['mp4_path']}."
            )
            for p_info in camera_info.values():
                try:
                    p_info["pipeline"].stop()
                except Exception:
                    pass
            exit()

        video_writers[serial_number] = out
        frame_arrays[serial_number] = []
        print(
            f"Recording video from Intel RealSense camera {serial_number} ({info['arm_type']}) "
            f"to {output_paths[serial_number]['mp4_path']}."
        )

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    ready_dir = Path(".recording")
    ready_dir.mkdir(exist_ok=True)
    ready_path = arms_ready_path(current_datetime_id)
    ready_path.touch()
    print(f"Arm cameras ready (signal: {ready_path})", flush=True)

    if args.wait_for_go and not wait_for_recording_go(
        current_datetime_id, label="wrist", should_stop=should_stop
    ):
        return

    recording_t0 = read_recording_start(current_datetime_id)
    drain_pipelines(
        [camera_info[sn]["pipeline"] for sn in serial_numbers],
        label="wrist",
        should_stop=should_stop,
    )

    print("Press 'q' to stop recording.", flush=True)

    start_time = time.time()
    if recording_t0 is not None:
        print(f"Shared recording t0={recording_t0:.3f} (local now={start_time:.3f})", flush=True)
    prev_times = {sn: time.time() for sn in serial_numbers}

    try:
        while not stop_recording:
            got_frame = False
            capture_t = time.time()
            for serial_number in serial_numbers:
                info = camera_info[serial_number]
                frames = poll_for_frames(info["pipeline"], timeout_ms=100, should_stop=should_stop)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                got_frame = True
                color_image = np.asanyarray(color_frame.get_data())
                video_writers[serial_number].write(color_image)
                cv2.imshow(
                    f'RealSense Video Recording - {info["arm_type"].capitalize()} Arm ({serial_number})',
                    color_image,
                )
                frame_arrays[serial_number].append(capture_t)

                if args.fps:
                    current_time = time.time()
                    dt = current_time - prev_times[serial_number]
                    if dt > 0:
                        fps = 1.0 / dt
                        print(
                            f"Camera {serial_number} ({info['arm_type']}) Approximate FPS: {fps:.2f}"
                        )
                    prev_times[serial_number] = current_time

            if cv2.waitKey(1 if got_frame else 5) & 0xFF == ord("q"):
                break

    finally:
        try:
            ready_path.unlink(missing_ok=True)
        except OSError:
            pass
        for info in camera_info.values():
            try:
                info["pipeline"].stop()
            except Exception:
                pass
        for out in video_writers.values():
            out.release()
        cv2.destroyAllWindows()

        for serial_number, frame_array in frame_arrays.items():
            np.save(output_paths[serial_number]["npy_path"], np.asarray(frame_array, dtype=np.float64))
            n_frames = len(frame_array)
            arm = camera_info[serial_number]["arm_type"]
            print(f"Saved {n_frames} frames from {arm} arm ({serial_number})", flush=True)
            if n_frames == 0:
                expected = serial_for_role(CAMERA_MAP, arm) or arm
                print(
                    f"ERROR: {arm.capitalize()} arm camera recorded ZERO frames — video file will be empty. "
                    f"Check USB for serial {expected}.",
                    flush=True,
                )

    end_time = time.time()
    duration = end_time - start_time

    print(f"Recording stopped. Total Duration: {duration:.2f} seconds.")
    for serial_number, info in camera_info.items():
        n_frames = len(frame_arrays.get(serial_number, []))
        print(
            f"Camera {serial_number} ({info['arm_type']}) — {n_frames} frames — "
            f"video: '{output_paths[serial_number]['mp4_path']}'"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
