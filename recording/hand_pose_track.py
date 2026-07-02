#!/usr/bin/env python3
"""RealSense preview, camera labeling, and RGBD hand 6DOF tracking via MediaPipe."""

import argparse
import signal
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

# Physical setup (cameras swapped): 317222=left arm, 332522=bird/center, 317422=right arm
CAMERA_MAP = {
    "317222072157": "left",
    "317422075805": "right",
    "332522076706": "center",
}

ROLE_KEYS = {
    ord("l"): "left",
    ord("r"): "right",
    ord("c"): "center",
    ord("u"): "unassigned",
}

# MediaPipe hand landmark indices used for palm 6DOF (no per-finger detail).
WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17

stop_preview = False


def request_stop(signum, frame):
    global stop_preview
    stop_preview = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preview RealSense cameras, assign left/right/center labels, "
            "and optionally track/record 6DOF hand pose from RGBD."
        )
    )
    parser.add_argument("--serial", "-s", type=str, default=None)
    parser.add_argument("--index", "-i", type=int, default=None)
    parser.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--hand-track",
        action="store_true",
        help="Enable MediaPipe Hand Landmarker + RGBD 6DOF pose (single-camera mode).",
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=2,
        help="Maximum hands to track (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hand-pose-data",
        help="Directory for recorded hand pose .npz files.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Preview hand tracking without saving pose data.",
    )
    return parser.parse_args()


def ensure_hand_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file():
        return MODEL_PATH
    print(f"Downloading Hand Landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model download complete.")
    return MODEL_PATH


def rotation_matrix_to_quaternion(rot):
    """Return quaternion as [w, x, y, z]."""
    m = rot
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def sample_depth_m(depth_frame, u, v, radius=4):
    depth_image = np.asanyarray(depth_frame.get_data())
    h, w = depth_image.shape
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or vi < 0 or ui >= w or vi >= h:
        return None

    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    patch = depth_image[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    depth_scale = depth_frame.get_units()
    return float(np.median(valid) * depth_scale)


def deproject_point(intrinsics, u, v, depth_m):
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], depth_m)
    return np.asarray(point, dtype=np.float64)


def landmark_to_pixel(landmark, width, height):
    return landmark.x * width, landmark.y * height


def estimate_hand_pose_6dof(landmarks, depth_frame, intrinsics, width, height):
    """
    Estimate wrist 6DOF in the RealSense camera frame from RGBD.

    Position: wrist depth sample + deprojection.
    Orientation: orthonormal palm frame from wrist, index MCP, pinky MCP (all RGBD).
    Returns (position_xyz, quaternion_wxyz) or None if depth is unavailable.
    """
    wrist_u, wrist_v = landmark_to_pixel(landmarks[WRIST], width, height)
    index_u, index_v = landmark_to_pixel(landmarks[INDEX_MCP], width, height)
    pinky_u, pinky_v = landmark_to_pixel(landmarks[PINKY_MCP], width, height)

    wrist_depth = sample_depth_m(depth_frame, wrist_u, wrist_v)
    index_depth = sample_depth_m(depth_frame, index_u, index_v)
    pinky_depth = sample_depth_m(depth_frame, pinky_u, pinky_v)
    if wrist_depth is None or index_depth is None or pinky_depth is None:
        return None

    wrist = deproject_point(intrinsics, wrist_u, wrist_v, wrist_depth)
    index_mcp = deproject_point(intrinsics, index_u, index_v, index_depth)
    pinky_mcp = deproject_point(intrinsics, pinky_u, pinky_v, pinky_depth)

    x_axis = index_mcp - wrist
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return None
    x_axis /= x_norm

    palm_span = pinky_mcp - wrist
    z_axis = np.cross(x_axis, palm_span)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return None
    z_axis /= z_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    quaternion = rotation_matrix_to_quaternion(rotation)
    return wrist, quaternion


def draw_pose_axes(image, intrinsics, position, quaternion, axis_len=0.05):
    """Draw RGB XYZ axes projected onto the color image."""
    w, x, y, z = quaternion
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)

    origin = position
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X,Y,Z
    prev_px = None
    for axis_idx in range(3):
        direction = rotation[:, axis_idx] * axis_len
        tip = origin + direction
        pixels = []
        for point in (origin, tip):
            px = rs.rs2_project_point_to_pixel(intrinsics, point.tolist())
            pixels.append((int(round(px[0])), int(round(px[1]))))
        cv2.arrowedLine(image, pixels[0], pixels[1], colors[axis_idx], 2, tipLength=0.25)
        if axis_idx == 0:
            prev_px = pixels[0]
    if prev_px is not None:
        cv2.circle(image, prev_px, 4, (255, 255, 255), -1)


class HandPoseRecorder:
    """Buffers per-frame 6DOF hand poses for export."""

    def __init__(self, max_hands=2):
        self.max_hands = max_hands
        self.timestamps = []
        self.positions = []
        self.quaternions = []
        self.handedness = []
        self.valid = []

    def add_frame(self, timestamp, detections):
        self.timestamps.append(timestamp)
        positions = np.full((self.max_hands, 3), np.nan, dtype=np.float64)
        quaternions = np.full((self.max_hands, 4), np.nan, dtype=np.float64)
        labels = [""] * self.max_hands
        valid = np.zeros(self.max_hands, dtype=bool)

        for idx, det in enumerate(detections[: self.max_hands]):
            positions[idx] = det["position"]
            quaternions[idx] = det["quaternion"]
            labels[idx] = det["handedness"]
            valid[idx] = True

        self.positions.append(positions)
        self.quaternions.append(quaternions)
        self.handedness.append(labels)
        self.valid.append(valid)

    def save(self, output_path):
        if not self.timestamps:
            print("No hand pose frames recorded.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            positions=np.asarray(self.positions, dtype=np.float64),
            quaternions=np.asarray(self.quaternions, dtype=np.float64),
            handedness=np.asarray(self.handedness, dtype=object),
            valid=np.asarray(self.valid, dtype=bool),
            frame="camera",
            representation="position_xyz_m + quaternion_wxyz; 6DOF wrist pose in RealSense optical frame",
        )
        print(f"Saved {len(self.timestamps)} raw hand pose frames -> {output_path}")

        from hand_pose_postprocess import postprocess_hand_pose, save_processed_npy

        processed = postprocess_hand_pose(
            np.asarray(self.timestamps, dtype=np.float64),
            np.asarray(self.positions, dtype=np.float64),
            np.asarray(self.quaternions, dtype=np.float64),
            np.asarray(self.valid, dtype=bool),
            np.asarray(self.handedness, dtype=object),
        )
        processed_path = output_path.with_name(output_path.stem + "_processed.npy")
        save_processed_npy(processed, processed_path)


class HandPoseTracker:
    """MediaPipe Hand Landmarker (Tasks API) + RealSense RGBD 6DOF estimation."""

    def __init__(self, num_hands=2):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = ensure_hand_model()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._vision = vision
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._frame_idx = 0

    def process(self, color_bgr, depth_frame, intrinsics):
        import mediapipe as mp

        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(self._frame_idx * 1000 / 30)
        self._frame_idx += 1

        result = self._detector.detect_for_video(mp_image, timestamp_ms)
        height, width = color_bgr.shape[:2]
        detections = []

        if not result.hand_landmarks:
            return detections, color_bgr

        annotated = color_bgr.copy()
        for hand_idx, landmarks in enumerate(result.hand_landmarks):
            pose = estimate_hand_pose_6dof(landmarks, depth_frame, intrinsics, width, height)
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
                    "landmarks": landmarks,
                }
            )

            wrist_u, wrist_v = landmark_to_pixel(landmarks[WRIST], width, height)
            cv2.circle(annotated, (int(wrist_u), int(wrist_v)), 6, (0, 255, 255), -1)
            draw_pose_axes(annotated, intrinsics, position, quaternion)

            pos_text = f"{handedness}: [{position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f}] m"
            quat_text = f"q=[{quaternion[0]:+.2f}, {quaternion[1]:+.2f}, {quaternion[2]:+.2f}, {quaternion[3]:+.2f}]"
            y = 60 + hand_idx * 44
            cv2.putText(annotated, pos_text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(annotated, pos_text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(annotated, quat_text, (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(annotated, quat_text, (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return detections, annotated

    def close(self):
        self._detector.close()


def list_devices():
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("No RealSense devices found.")
        return []

    print(f"\nFound {len(devices)} RealSense device(s):\n")
    for idx, device in enumerate(devices):
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        role = CAMERA_MAP.get(serial, "unassigned")
        print(f"  [{idx}] {name}")
        print(f"       serial: {serial}")
        print(f"       role:   {role}\n")
    return devices


def choose_device(devices, serial=None, index=None):
    if not devices:
        sys.exit(1)

    if serial is not None:
        for device in devices:
            if device.get_info(rs.camera_info.serial_number) == serial:
                return device
        print(f"Error: no camera with serial '{serial}'.")
        sys.exit(1)

    if index is not None:
        if index < 0 or index >= len(devices):
            print(f"Error: index {index} out of range (0-{len(devices) - 1}).")
            sys.exit(1)
        return devices[index]

    if len(devices) == 1:
        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"Using the only connected camera [{serial}].")
        return device

    while True:
        choice = input(f"Enter device index [0-{len(devices) - 1}]: ").strip()
        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if 0 <= idx < len(devices):
            return devices[idx]
        print(f"Index must be between 0 and {len(devices) - 1}.")


def start_pipeline(device, width, height, fps, enable_depth=False):
    serial = device.get_info(rs.camera_info.serial_number)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if enable_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color) if enable_depth else None
    return pipeline, profile, serial, align


def get_color_intrinsics(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    return color_stream.get_intrinsics()


def draw_overlay(frame, serial, role, extra_lines=None):
    overlay = frame.copy()
    lines = [
        f"Serial: {serial}",
        f"Role:   {role}",
        "Keys: l=left  r=right  c=center  u=unassigned  q=quit",
    ]
    if extra_lines:
        lines = extra_lines + lines

    y = 28
    for line in lines:
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        y += 26
    return overlay


def print_camera_map():
    assigned = {serial: role for serial, role in sorted(CAMERA_MAP.items()) if role != "unassigned"}
    print("\nCurrent camera_map (paste into realsense_double_record.py):")
    print("camera_map = {")
    for serial, role in assigned.items():
        print(f'    "{serial}": "{role}",')
    print("}")


def preview_single(device, width, height, fps, hand_track=False, num_hands=2, output_dir="hand-pose-data", record=True):
    global stop_preview

    pipeline = None
    tracker = None
    recorder = HandPoseRecorder(max_hands=num_hands) if hand_track and record else None
    session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    serial = device.get_info(rs.camera_info.serial_number)
    name = device.get_info(rs.camera_info.name)
    role = CAMERA_MAP.get(serial, "unassigned")

    print(f"\nPreviewing: {name} ({serial})")
    if hand_track:
        print("Hand tracking: MediaPipe Hand Landmarker + RGBD 6DOF wrist pose.")
        if record:
            print(f"Recording hand poses to {output_dir}/")
    print("Press l/r/c/u in the preview window to set the camera role.\n")

    try:
        pipeline, profile, serial, align = start_pipeline(
            device, width, height, fps, enable_depth=hand_track
        )
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(f"Stream: {color_profile.width()}x{color_profile.height()} @ {color_profile.fps()} FPS")

        if hand_track:
            tracker = HandPoseTracker(num_hands=num_hands)
            intrinsics = get_color_intrinsics(profile)

        window = f"RealSense Preview - {serial}"
        while not stop_preview:
            frames = pipeline.wait_for_frames()
            if hand_track:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            display = frame

            if hand_track:
                depth_frame = frames.get_depth_frame()
                detections, display = tracker.process(frame, depth_frame, intrinsics)
                if recorder is not None:
                    recorder.add_frame(time.time(), detections)

            display = draw_overlay(
                display,
                serial,
                role,
                extra_lines=["Hand track: ON (RGBD 6DOF)"] if hand_track else None,
            )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in ROLE_KEYS:
                role = ROLE_KEYS[key]
                if role == "unassigned":
                    CAMERA_MAP.pop(serial, None)
                else:
                    CAMERA_MAP[serial] = role
                print(f"Set {serial} -> {role}")

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
            out_path = Path(output_dir) / f"hand_pose_{serial}#{session_id}.npz"
            recorder.save(out_path)

    print_camera_map()


def preview_all(devices, width, height, fps):
    global stop_preview

    pipelines = []
    serials = []
    selected_idx = 0

    print("\nPreviewing all cameras. Press q in any window to quit.\n")
    print("Use 0-9 to select a camera, then l/r/c/u to label it.\n")

    try:
        for device in devices:
            pipeline, _, serial, _ = start_pipeline(device, width, height, fps, enable_depth=False)
            pipelines.append(pipeline)
            serials.append(serial)

        while not stop_preview:
            for idx, (pipeline, serial) in enumerate(zip(pipelines, serials)):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                role = CAMERA_MAP.get(serial, "unassigned")
                marker = ">>>" if idx == selected_idx else "   "
                frame = np.asanyarray(color_frame.get_data())
                display = draw_overlay(
                    frame,
                    serial,
                    role,
                    extra_lines=[f"{marker} [{idx}] selected"],
                )
                cv2.imshow(f"RealSense [{idx}] {serial}", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if ord("0") <= key <= ord("9"):
                idx = key - ord("0")
                if idx < len(serials):
                    selected_idx = idx
                    print(f"Selected camera [{idx}] {serials[idx]}")
                continue

            if key in ROLE_KEYS:
                serial = serials[selected_idx]
                role = ROLE_KEYS[key]
                if role == "unassigned":
                    CAMERA_MAP.pop(serial, None)
                else:
                    CAMERA_MAP[serial] = role
                print(f"Set [{selected_idx}] {serial} -> {role}")

    finally:
        for pipeline in pipelines:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        cv2.destroyAllWindows()

    print_camera_map()


def main():
    args = parse_args()
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.hand_track and args.all:
        print("Hand tracking is only supported in single-camera mode (omit --all).")
        sys.exit(1)

    devices = list_devices()
    if not devices:
        sys.exit(1)

    if args.all:
        preview_all(devices, args.width, args.height, args.fps)
    else:
        device = choose_device(devices, serial=args.serial, index=args.index)
        preview_single(
            device,
            args.width,
            args.height,
            args.fps,
            hand_track=args.hand_track,
            num_hands=args.num_hands,
            output_dir=args.output_dir,
            record=not args.no_record,
        )


if __name__ == "__main__":
    main()
