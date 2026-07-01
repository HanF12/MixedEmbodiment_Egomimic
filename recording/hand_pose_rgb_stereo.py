#!/usr/bin/env python3
"""
RealSense hand 6DOF tracking: RGB detection (MediaPipe) + IR stereo triangulation.

Unlike hand_pose_track.py (RGBD depth sampling), this version:
  - detects hands on the RGB color stream only
  - triangulates 3D wrist/palm points from the RealSense left/right IR stereo pair (SGBM)
  - runs the same post-processing pipeline on exit
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from hand_pose_stereo import (
    RealSenseStereoCalibration,
    StereoMatcher,
    estimate_hand_pose_6dof_stereo,
)
from hand_pose_track import (
    CAMERA_MAP,
    HandPoseRecorder,
    HandPoseTracker,
    ROLE_KEYS,
    choose_device,
    draw_overlay,
    draw_pose_axes,
    ensure_hand_model,
    get_color_intrinsics,
    landmark_to_pixel,
    list_devices,
    print_camera_map,
    WRIST,
)

stop_preview = False


def request_stop(signum, frame):
    global stop_preview
    stop_preview = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="RGB hand detection + RealSense IR stereo 6DOF tracking."
    )
    parser.add_argument("--serial", "-s", type=str, default=None)
    parser.add_argument("--index", "-i", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hand-pose-data",
        help="Directory for raw .npz and processed .npy files.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Preview without saving.",
    )
    parser.add_argument(
        "--show-disparity",
        action="store_true",
        help="Show IR stereo disparity map in a second window.",
    )
    return parser.parse_args()


def start_stereo_pipeline(device, width, height, fps):
    serial = device.get_info(rs.camera_info.serial_number)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    profile = pipeline.start(config)
    return pipeline, profile, serial


class RGBStereoHandTracker:
    """MediaPipe on RGB + 6DOF from IR stereo triangulation."""

    def __init__(self, calib: RealSenseStereoCalibration, num_hands=1):
        self.calib = calib
        self.stereo = StereoMatcher()
        self.mp_tracker = HandPoseTracker(num_hands=num_hands)

    def process(self, color_bgr, ir_left, ir_right, color_intrinsics):
        disp = self.stereo.compute_disparity(ir_left, ir_right)
        height, width = color_bgr.shape[:2]

        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(self.mp_tracker._frame_idx * 1000 / 30)
        self.mp_tracker._frame_idx += 1
        result = self.mp_tracker._detector.detect_for_video(mp_image, timestamp_ms)

        detections = []
        if not result.hand_landmarks:
            return detections, color_bgr, disp

        annotated = color_bgr.copy()
        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            pose = estimate_hand_pose_6dof_stereo(
                landmarks, disp, self.calib, width, height
            )
            if pose is None:
                continue

            position, quaternion = pose
            handedness = "Unknown"
            if result.handedness and hand_idx < len(result.handedness):
                categories = result.handedness[hand_idx]
                if categories:
                    handedness = categories[0].category_name

            detections.append(
                {
                    "position": position,
                    "quaternion": quaternion,
                    "handedness": handedness,
                }
            )

            wrist_u, wrist_v = landmark_to_pixel(landmarks[WRIST], width, height)
            cv2.circle(annotated, (int(wrist_u), int(wrist_v)), 6, (255, 128, 0), -1)
            draw_pose_axes(annotated, color_intrinsics, position, quaternion)

            pos_text = (
                f"{handedness}: [{position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}] m"
            )
            y = 60 + hand_idx * 44
            cv2.putText(annotated, pos_text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(annotated, pos_text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)

        return detections, annotated, disp

    def close(self):
        self.mp_tracker.close()


def save_stereo_recording(recorder: HandPoseRecorder, output_path: Path):
    """Save raw npz + processed npy with stereo method tag."""
    if not recorder.timestamps:
        print("No hand pose frames recorded.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        timestamps=np.asarray(recorder.timestamps, dtype=np.float64),
        positions=np.asarray(recorder.positions, dtype=np.float64),
        quaternions=np.asarray(recorder.quaternions, dtype=np.float64),
        handedness=np.asarray(recorder.handedness, dtype=object),
        valid=np.asarray(recorder.valid, dtype=bool),
        frame="color",
        method="rgb_stereo",
        representation=(
            "MediaPipe RGB landmarks + RealSense IR stereo triangulation; "
            "position_xyz_m + quaternion_wxyz in color camera frame"
        ),
    )
    print(f"Saved {len(recorder.timestamps)} raw stereo hand pose frames -> {output_path}")

    from hand_pose_postprocess import postprocess_hand_pose, save_processed_npy

    processed = postprocess_hand_pose(
        np.asarray(recorder.timestamps, dtype=np.float64),
        np.asarray(recorder.positions, dtype=np.float64),
        np.asarray(recorder.quaternions, dtype=np.float64),
        np.asarray(recorder.valid, dtype=bool),
        np.asarray(recorder.handedness, dtype=object),
    )
    processed["method"] = "rgb_stereo"
    processed_path = output_path.with_name(output_path.stem + "_processed.npy")
    save_processed_npy(processed, processed_path)


def run_rgb_stereo(
    device,
    width,
    height,
    fps,
    num_hands=1,
    output_dir="hand-pose-data",
    record=True,
    show_disparity=False,
):
    global stop_preview

    pipeline = None
    tracker = None
    recorder = HandPoseRecorder(max_hands=num_hands) if record else None
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    serial = device.get_info(rs.camera_info.serial_number)
    name = device.get_info(rs.camera_info.name)
    role = CAMERA_MAP.get(serial, "unassigned")

    print(f"\nRGB + Stereo tracking: {name} ({serial})")
    print("Detection: MediaPipe on RGB | 3D: RealSense IR1/IR2 SGBM triangulation")
    if record:
        print(f"Recording to {output_dir}/")
    print("Press q to quit.\n")

    try:
        ensure_hand_model()
        pipeline, profile, serial = start_stereo_pipeline(device, width, height, fps)
        calib = RealSenseStereoCalibration(profile)
        color_intrinsics = get_color_intrinsics(profile)
        tracker = RGBStereoHandTracker(calib, num_hands=num_hands)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(
            f"Color: {color_profile.width()}x{color_profile.height()} @ {color_profile.fps()} FPS"
        )
        print(f"Stereo baseline: {calib.baseline_m * 1000:.1f} mm")

        window = f"RGB+Stereo Hand Track - {serial}"
        while not stop_preview:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            ir1_frame = frames.get_infrared_frame(1)
            ir2_frame = frames.get_infrared_frame(2)
            if not color_frame or not ir1_frame or not ir2_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            ir_left = np.asanyarray(ir1_frame.get_data())
            ir_right = np.asanyarray(ir2_frame.get_data())

            detections, display, disp = tracker.process(
                color, ir_left, ir_right, color_intrinsics
            )
            if recorder is not None:
                recorder.add_frame(time.time(), detections)

            display = draw_overlay(
                display,
                serial,
                role,
                extra_lines=["Mode: RGB detect + IR stereo 6DOF"],
            )
            cv2.imshow(window, display)

            if show_disparity:
                disp_vis = disp.copy()
                disp_vis[~np.isfinite(disp_vis)] = 0
                disp_norm = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imshow("Stereo disparity (IR1/IR2)", disp_norm.astype(np.uint8))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in ROLE_KEYS:
                role = ROLE_KEYS[key]
                if role == "unassigned":
                    CAMERA_MAP.pop(serial, None)
                else:
                    CAMERA_MAP[serial] = role

    finally:
        if tracker is not None:
            tracker.close()
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        cv2.destroyAllWindows()

        if recorder is not None:
            out_path = Path(output_dir) / f"hand_pose_stereo_{serial}#{session_id}.npz"
            save_stereo_recording(recorder, out_path)

    print_camera_map()


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    devices = list_devices()
    if not devices:
        sys.exit(1)

    device = choose_device(devices, serial=args.serial, index=args.index)
    run_rgb_stereo(
        device,
        args.width,
        args.height,
        args.fps,
        num_hands=args.num_hands,
        output_dir=args.output_dir,
        record=not args.no_record,
        show_disparity=args.show_disparity,
    )


if __name__ == "__main__":
    main()
