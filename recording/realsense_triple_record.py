#!/usr/bin/env python3
"""
Record left wrist, right wrist, and bird RealSense streams in ONE process.

All cameras share a single aligned poll loop so timestamps are stamped together
(no cross-process phase offset between bird and wrists).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from hand_pose_track import (
    CAMERA_MAP,
    HandPoseRecorder,
    HandPoseTracker,
    get_color_intrinsics,
)
from realsense_utils import (
    drain_pipelines,
    poll_aligned_frame_sets,
    serial_for_role,
    start_pipelines_parallel,
)
from recording_paths import under_recording
from recording_sync import (
    arms_ready_path,
    bird_ready_path,
    read_recording_start,
    wait_for_recording_go,
)

BIRD_ROLE = "center"
HAND_POSE_DIR = under_recording("hand-pose-data")

stop_recording = False


def request_stop(signum, frame):
    global stop_recording
    stop_recording = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record wrist + bird RealSense cameras with aligned timestamps."
    )
    parser.add_argument("--color-fps", type=int, default=15)
    parser.add_argument("--datetime-id", type=str, default=None)
    parser.add_argument(
        "--arms",
        choices=("left", "right", "both"),
        default="both",
        help="Which wrist cameras to include (default: both).",
    )
    parser.add_argument("--serial", type=str, default=None, help="Bird RealSense serial override.")
    parser.add_argument("--no-hand-pose", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--track-hand",
        choices=("left", "right", "both"),
        default="both",
    )
    parser.add_argument("--wait-for-go", action="store_true")
    return parser.parse_args()


def find_device(devices, role: str, serial_arg: str | None = None):
    if role == BIRD_ROLE and serial_arg:
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            if serial_arg in serial or serial.endswith(serial_arg):
                return device, serial
    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        if CAMERA_MAP.get(serial) == role:
            return device, serial
    return None, None


def filter_detections_by_hand(detections, track_hand: str):
    if track_hand == "both":
        return detections
    want = "Left" if track_hand == "left" else "Right"
    return [d for d in detections if d.get("handedness") == want]


def main(args):
    global stop_recording
    should_stop = lambda: stop_recording

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("Error: no RealSense devices found.")
        sys.exit(1)

    roles: list[str] = []
    if args.arms in ("left", "both"):
        roles.append("left")
    if args.arms in ("right", "both"):
        roles.append("right")
    roles.append(BIRD_ROLE)

    session_id = args.datetime_id or datetime.now().strftime("%Y%m%d%H%M%S")
    hand_pose_enabled = not args.no_hand_pose
    show_preview = not args.no_display
    track_hand = args.track_hand
    num_hands = 1 if track_hand in ("left", "right") else 2

    # role -> {pipeline, config, serial, align?, arm_type for wrists}
    streams: dict = {}

    for role in roles:
        serial_arg = args.serial if role == BIRD_ROLE else None
        device, serial = find_device(devices, role, serial_arg)
        if device is None:
            label = "bird" if role == BIRD_ROLE else f"{role} wrist"
            expected = serial_for_role(CAMERA_MAP, role) or role
            print(f"ERROR: {label} RealSense ({expected}) not connected.")
            sys.exit(1)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, args.color_fps)
        enable_depth = hand_pose_enabled and role == BIRD_ROLE
        if enable_depth:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.color_fps)

        streams[role] = {
            "pipeline": pipeline,
            "config": config,
            "serial": serial,
            "enable_depth": enable_depth,
            "profile": None,
            "align": None,
        }
        print(f"Configured {role} camera: {serial}")

    try:
        start_entries = [(role, streams[role]["pipeline"], streams[role]["config"]) for role in roles]
        profiles = start_pipelines_parallel(start_entries, should_stop=should_stop)
    except RuntimeError as exc:
        print(f"Error starting RealSense pipelines: {exc}")
        sys.exit(1)

    for role, profile in profiles.items():
        streams[role]["profile"] = profile
        if streams[role]["enable_depth"]:
            streams[role]["align"] = rs.align(rs.stream.color)
        cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(f"  {role}: {cp.width()}x{cp.height()} @ {cp.fps()} FPS")

    # Output paths and writers
    aloha_root = under_recording("aloha-data")
    bird_root = under_recording("bird-realsense-data")
    Path(HAND_POSE_DIR).mkdir(parents=True, exist_ok=True)

    writers: dict = {}
    timestamp_arrays: dict = {role: [] for role in roles}

    for role in roles:
        serial = streams[role]["serial"]
        profile = streams[role]["profile"]
        cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        w, h, fps = cp.width(), cp.height(), cp.fps()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if role == BIRD_ROLE:
            mp4_dir = os.path.join(bird_root, "mp4")
            npy_dir = os.path.join(bird_root, "npy")
            mp4_name = f"video_recording_bird_realsense_{serial}#{session_id}.mp4"
            npy_name = f"video_recording_bird_realsense_{serial}#{session_id}.npy"
        else:
            mp4_dir = os.path.join(aloha_root, role, "mp4")
            npy_dir = os.path.join(aloha_root, role, "npy")
            mp4_name = f"video_recording_realsense_{role}#{session_id}.mp4"
            npy_name = f"video_recording_realsense_{role}#{session_id}.npy"

        os.makedirs(mp4_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)
        mp4_path = os.path.join(mp4_dir, mp4_name)
        npy_path = os.path.join(npy_dir, npy_name)

        out = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
        if not out.isOpened():
            print(f"Error: could not open video writer for {mp4_path}")
            sys.exit(1)

        writers[role] = {"writer": out, "npy_path": npy_path, "mp4_path": mp4_path}
        print(f"  -> {mp4_path}")

    bird_serial = streams[BIRD_ROLE]["serial"]
    hand_raw_path = Path(HAND_POSE_DIR) / f"hand_pose_{bird_serial}#{session_id}.npz"
    tracker = None
    recorder = None
    intrinsics = None
    if hand_pose_enabled:
        intrinsics = get_color_intrinsics(streams[BIRD_ROLE]["profile"])
        tracker = HandPoseTracker(num_hands=num_hands)
        recorder = HandPoseRecorder(max_hands=num_hands)
        print(f"Hand pose -> {hand_raw_path}")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # Ready signals (same process owns wrists + bird)
    arms_ready_path(session_id).parent.mkdir(parents=True, exist_ok=True)
    if any(r in roles for r in ("left", "right")):
        arms_ready_path(session_id).touch()
        print(f"Arm cameras ready (signal: {arms_ready_path(session_id)})", flush=True)
    bird_ready_path(session_id).touch()
    print(f"Bird RealSense ready (signal: {bird_ready_path(session_id)})", flush=True)

    if args.wait_for_go and not wait_for_recording_go(session_id, label="triple", should_stop=should_stop):
        return

    recording_t0 = read_recording_start(session_id)
    pipelines = [streams[role]["pipeline"] for role in roles]
    drain_pipelines(pipelines, label="triple", should_stop=should_stop)
    if recording_t0 is not None:
        print(f"[triple] Shared recording t0={recording_t0:.3f}", flush=True)

    poll_map = {role: streams[role]["pipeline"] for role in roles}
    print("Press 'q' to stop recording.", flush=True)

    try:
        while not stop_recording:
            frame_sets = poll_aligned_frame_sets(poll_map, timeout_ms=150, should_stop=should_stop)
            if frame_sets is None:
                continue

            capture_t = time.time()

            for role, frames in frame_sets.items():
                if streams[role]["enable_depth"]:
                    frames = streams[role]["align"].process(frames)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())
                writers[role]["writer"].write(image)
                timestamp_arrays[role].append(capture_t)

                if show_preview:
                    title = f"{role} ({streams[role]['serial']})"
                    display = image
                    if role == BIRD_ROLE and hand_pose_enabled and tracker is not None:
                        depth_frame = frames.get_depth_frame()
                        detections, annotated = tracker.process(image, depth_frame, intrinsics)
                        detections = filter_detections_by_hand(detections, track_hand)
                        if recorder is not None:
                            recorder.add_frame(capture_t, detections)
                        display = annotated
                    cv2.imshow(title, display)

            if show_preview and cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if tracker is not None:
            tracker.close()
        for role in roles:
            try:
                streams[role]["pipeline"].stop()
            except RuntimeError:
                pass
        for info in writers.values():
            info["writer"].release()
        cv2.destroyAllWindows()

        for role in roles:
            arr = np.asarray(timestamp_arrays[role], dtype=np.float64)
            np.save(writers[role]["npy_path"], arr)
            print(f"Saved {len(arr)} frames [{role}] -> {writers[role]['mp4_path']}", flush=True)

        if recorder is not None and recorder.timestamps:
            recorder.save(hand_raw_path)
        elif hand_pose_enabled:
            print("No hand pose frames recorded.")

        try:
            arms_ready_path(session_id).unlink(missing_ok=True)
            bird_ready_path(session_id).unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main(parse_args())
