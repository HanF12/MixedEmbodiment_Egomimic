import cv2
import pyrealsense2 as rs
import numpy as np
import os
import time
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Record RealSense color stream to disk with optional FPS logging and camera selection."
    )
    parser.add_argument(
        "--fps", "-f",
        action="store_true",
        help="If set, print the approximate FPS for each frame."
    )
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default=None,
        help="Serial number (or device ID) of the RealSense camera to use. If omitted, the default camera is used."
    )
    return parser.parse_args()

def main(args):
    # Configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # If a specific camera ID was provided, tell librealsense to use it
    if args.camera:
        config.enable_device(args.camera)

    # Enable the color stream (RGB)
    color_width = 640
    color_height = 480
    color_fps = 30

    config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)

    # Start streaming
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("Please ensure the camera is connected and not in use by another application.")
        exit()

    # Get the actual stream profiles
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    actual_color_width = color_profile.width()
    actual_color_height = color_profile.height()
    actual_color_fps = color_profile.fps()

    print(f"RealSense Color Stream: Resolution {actual_color_width}x{actual_color_height}, FPS: {actual_color_fps}")

    # Prepare output directory
    if not os.path.exists("aloha-data"):
        os.makedirs("aloha-data")

    # Prepare video writer
    id = datetime.now().strftime("%Y%m%d%H%M%S")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_filename = f'aloha-data/video_recording_realsense#{id}.mp4'
    out = cv2.VideoWriter(output_filename, fourcc, actual_color_fps, (actual_color_width, actual_color_height))

    np_filename = f'aloha-data/video_recording_realsense#{id}.npy'

    if not out.isOpened():
        print(f"Error: Could not open video writer with FOURCC {fourcc} for {output_filename}.")
        print("This often means OpenCV is not built with the necessary codec support (e.g., FFMPEG with H.264).")
        print("Try a different FOURCC, like cv2.VideoWriter_fourcc(*'mp4v') or cv2.VideoWriter_fourcc(*'MJPG') with a .avi extension.")
        pipeline.stop()
        exit()

    frame_number = 0
    frame_array = []
    start_time = time.time()
    prev_time = time.time()   # initialize previous-frame timestamp

    print(f"Recording video from Intel RealSense to {output_filename}. Press 'q' to stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            out.write(color_image)
            cv2.imshow('RealSense Video Recording', color_image)
            frame_array.append(time.time())

            if args.fps:
                current_time = time.time()
                dt = current_time - prev_time
                if dt > 0:
                    fps = 1.0 / dt
                    print(f"Approximate FPS: {fps:.2f}")
                prev_time = current_time

            frame_number += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()
        np.asarray(frame_array, dtype=np.float64)
        np.save(np_filename, frame_array)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Recording stopped, video saved as '{output_filename}' - Video Duration: {duration}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
