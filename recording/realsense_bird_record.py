#!/usr/bin/env python3
"""
Record bird-view RealSense: color MP4 + timestamps, plus RGBD hand 6DOF pose.

Outputs (separate from webcam bird-data/):
  bird-realsense-data/mp4/video_recording_bird_realsense_<serial>#<id>.mp4
  bird-realsense-data/npy/video_recording_bird_realsense_<serial>#<id>.npy
  hand-pose-data/hand_pose_<serial>#<id>.npz          (raw)
  hand-pose-data/hand_pose_<serial>#<id>_processed.npy (smoothed)
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
    start_pipeline,
)
from realsense_utils import drain_pipeline, poll_for_frames, warmup_pipeline
from recording_paths import under_recording
from recording_sync import bird_ready_path, read_recording_start, wait_for_recording_go

SCRIPT_DIR = Path(__file__).resolve().parent
BIRD_ROLE = "center"
BASE_OUT_DIR = under_recording("bird-realsense-data")
HAND_POSE_DIR = under_recording("hand-pose-data")

stop_recording = False


def request_stop(signum, frame):
    global stop_recording
    stop_recording = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record bird-view RealSense video + RGBD hand pose."
    )
    parser.add_argument(
        "--datetime-id",
        type=str,
        default=None,
        help="Shared demo id for output filenames (default: now).",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="RealSense serial (default: device mapped as 'center' in CAMERA_MAP).",
    )
    parser.add_argument(
        "--width", type=int, default=640,
    )
    parser.add_argument(
        "--height", type=int, default=480,
    )
    parser.add_argument(
        "--fps", type=int, default=15,
    )
    parser.add_argument(
        "--num-hands", type=int, default=2,
    )
    parser.add_argument(
        "--no-hand-pose",
        action="store_true",
        help="Record video only (no RGBD hand tracking).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=True,
        help="Show preview window while recording (default: on).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable preview window.",
    )
    parser.add_argument(
        "--track-hand",
        choices=("left", "right", "both"),
        default="both",
        help="Which hand(s) to keep in pose output (default: both).",
    )
    parser.add_argument(
        "--wait-for-go",
        action="store_true",
        help="Wait for record_pedal.py sync signal before saving frames.",
    )
    return parser.parse_args()


def filter_detections_by_hand(detections, track_hand: str):
    if track_hand == "both":
        return detections
    want = "Left" if track_hand == "left" else "Right"
    return [d for d in detections if d.get("handedness") == want]


def _serial_matches(requested: str, device_serial: str) -> bool:
    return requested == device_serial or requested in device_serial or device_serial.endswith(requested)


def find_bird_device(devices, serial_arg: str | None):
    if serial_arg:
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            if _serial_matches(serial_arg, serial):
                return device, serial
        return None, None

    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        if CAMERA_MAP.get(serial) == BIRD_ROLE:
            return device, serial

    for device in devices:
        serial = device.get_info(rs.camera_info.serial_number)
        for key, role in CAMERA_MAP.items():
            if role == BIRD_ROLE and _serial_matches(key, serial):
                return device, serial

    return None, None


def main(args):
    global stop_recording
    should_stop = lambda: stop_recording

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("Error: no RealSense devices found.")
        sys.exit(1)

    device, serial = find_bird_device(devices, args.serial)
    if device is None:
        print("Error: bird-view RealSense not found.")
        print(f"  Looking for role={BIRD_ROLE!r} in CAMERA_MAP or --serial")
        print("  Connected devices:")
        for d in devices:
            s = d.get_info(rs.camera_info.serial_number)
            print(f"    {s}  (map role: {CAMERA_MAP.get(s, 'unassigned')})")
        sys.exit(1)

    session_id = args.datetime_id or datetime.now().strftime("%Y%m%d%H%M%S")
    hand_pose_enabled = not args.no_hand_pose
    show_preview = args.display and not args.no_display
    track_hand = args.track_hand
    num_hands = 1 if track_hand in ("left", "right") else args.num_hands

    mp4_dir = Path(BASE_OUT_DIR) / "mp4"
    npy_dir = Path(BASE_OUT_DIR) / "npy"
    mp4_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    Path(HAND_POSE_DIR).mkdir(parents=True, exist_ok=True)

    video_path = mp4_dir / f"video_recording_bird_realsense_{serial}#{session_id}.mp4"
    ts_path = npy_dir / f"video_recording_bird_realsense_{serial}#{session_id}.npy"
    hand_raw_path = Path(HAND_POSE_DIR) / f"hand_pose_{serial}#{session_id}.npz"

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    pipeline = None
    tracker = None
    recorder = HandPoseRecorder(max_hands=num_hands) if hand_pose_enabled else None
    video_writer = None
    timestamps = []

    print(f"Bird RealSense {serial} -> {video_path}")
    if hand_pose_enabled:
        print(f"Hand pose (RGBD) -> {hand_raw_path} (+ _processed.npy)")

    try:
        pipeline, profile, _, align = start_pipeline(
            device,
            args.width,
            args.height,
            args.fps,
            enable_depth=hand_pose_enabled,
        )
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        w, h, fps = color_profile.width(), color_profile.height(), color_profile.fps()
        print(f"Stream: {w}x{h} @ {fps} FPS  hand_pose={'on' if hand_pose_enabled else 'off'}")

        warmup_pipeline(pipeline, should_stop=should_stop)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
        if not video_writer.isOpened():
            print(f"Error: could not open video writer for {video_path}")
            sys.exit(1)

        intrinsics = get_color_intrinsics(profile) if hand_pose_enabled else None
        if hand_pose_enabled:
            tracker = HandPoseTracker(num_hands=num_hands)

        ready_path = bird_ready_path(session_id)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.touch()
        print(f"Bird RealSense ready (signal: {ready_path})", flush=True)

        if args.wait_for_go and not wait_for_recording_go(
            session_id, label="bird", should_stop=should_stop
        ):
            return

        recording_t0 = read_recording_start(session_id)
        drained = drain_pipeline(pipeline, should_stop=should_stop)
        print(f"[bird] Drained {drained} stale frame(s) from pipeline buffer.", flush=True)
        if recording_t0 is not None:
            print(f"[bird] Shared recording t0={recording_t0:.3f}", flush=True)

        while not stop_recording:
            frames = poll_for_frames(pipeline, timeout_ms=100, should_stop=should_stop)
            if frames is None:
                continue

            if hand_pose_enabled:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame_t = time.time()
            frame = np.asanyarray(color_frame.get_data())
            video_writer.write(frame)
            timestamps.append(frame_t)

            if hand_pose_enabled and tracker is not None and recorder is not None:
                depth_frame = frames.get_depth_frame()
                detections, annotated = tracker.process(frame, depth_frame, intrinsics)
                detections = filter_detections_by_hand(detections, track_hand)
                recorder.add_frame(timestamps[-1], detections)
                if show_preview:
                    cv2.imshow(f"Bird RealSense {serial}", annotated)
            elif show_preview:
                cv2.imshow(f"Bird RealSense {serial}", frame)

            if show_preview and cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if tracker is not None:
            tracker.close()
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

        np.save(ts_path, np.asarray(timestamps, dtype=np.float64))
        print(f"Saved {len(timestamps)} frames -> {video_path}")
        print(f"Timestamps -> {ts_path}")

        if recorder is not None and recorder.timestamps:
            recorder.save(hand_raw_path)
        elif hand_pose_enabled:
            print("No hand pose frames recorded.")

    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    print(f"Bird RealSense recording done ({duration:.1f}s).")


if __name__ == "__main__":
    os.chdir(SCRIPT_DIR)
    main(parse_args())
