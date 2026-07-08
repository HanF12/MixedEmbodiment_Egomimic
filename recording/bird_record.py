import cv2
import argparse
import os
import numpy as np
import signal
import time
from datetime import datetime

from recording_paths import under_recording
from recording_sync import read_recording_start, wait_for_recording_go

stop_recording = False

def request_stop(signum, frame):
    global stop_recording
    stop_recording = True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Record from a webcam and log timestamps, with optional per-frame FPS printing and camera selection."
    )
    parser.add_argument(
        "--fps", "-f",
        action="store_true",
        help="If set, print the approximate FPS for each frame."
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=8,
        help="Index of the webcam device to use (default: 0)."
    )
    parser.add_argument(
        "--datetime-id",
        type=str,
        default=None,
        help="Shared timestamp id for output filenames (default: now).",
    )
    parser.add_argument(
        "--wait-for-go",
        action="store_true",
        help="Wait for record_pedal.py sync signal before saving frames.",
    )
    return parser.parse_args()

def drain_webcam(cap, max_ms: float = 400, idle_ms: float = 60) -> int:
    discarded = 0
    idle_start = None
    deadline = time.monotonic() + max_ms / 1000.0
    while time.monotonic() < deadline:
        if cap.grab():
            discarded += 1
            idle_start = None
            continue
        if idle_start is None:
            idle_start = time.monotonic()
        elif time.monotonic() - idle_start >= idle_ms / 1000.0:
            break
        time.sleep(0.001)
    return discarded

def main(args):
    # Open the selected camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Unable to open camera device {args.camera}.")
        exit()

    # Get camera properties: resolution and FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 50  # fallback if camera doesn't report FPS

    print(f"Camera {args.camera} opened: {width}×{height} @ {fps:.2f} FPS (reported)")

    # Prepare output directories
    base_out_dir = under_recording("bird-data")
    mp4_dir = os.path.join(base_out_dir, "mp4")
    npy_dir = os.path.join(base_out_dir, "npy")
    os.makedirs(mp4_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    # Generate a timestamped ID for filenames
    id_str = args.datetime_id or datetime.now().strftime("%Y%m%d%H%M%S")

    # Prepare video writer (MP4 with mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = os.path.join(mp4_dir, f"video_recording_bird#{id_str}.mp4")
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    np_filename = os.path.join(npy_dir, f"video_recording_bird#{id_str}.npy")

    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for '{video_filename}'.")
        print("Check that OpenCV is built with the necessary codec support.")
        cap.release()
        exit()

    frame_number = 0
    frame_array = []
    prev_time = time.time()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.wait_for_go and not wait_for_recording_go(id_str, label="webcam bird"):
        cap.release()
        out.release()
        return

    recording_t0 = read_recording_start(id_str)
    drained = drain_webcam(cap)
    print(f"[webcam bird] Drained {drained} stale frame(s) from camera buffer.", flush=True)
    if recording_t0 is not None:
        print(f"[webcam bird] Shared recording t0={recording_t0:.3f}", flush=True)

    print(f"Recording from camera {args.camera}. Press 'q' to stop.")
    try:
        while not stop_recording:
            capture_t = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Write frame to video file
            out.write(frame)
            cv2.imshow(f"Webcam Recording - Camera {args.camera}", frame)
            frame_array.append(capture_t)

            # If --fps was set, compute and print per-frame FPS
            if args.fps:
                current_time = time.time()
                dt = current_time - prev_time
                if dt > 0:
                    instantaneous_fps = 1.0 / dt
                    print(f"Approximate FPS: {instantaneous_fps:.2f}")
                prev_time = current_time

            frame_number += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        np.save(np_filename, np.asarray(frame_array, dtype=np.float64))

    print(f"Recording stopped. Video saved as '{video_filename}', timestamps saved as '{np_filename}'")

if __name__ == "__main__":
    args = parse_args()
    main(args)
