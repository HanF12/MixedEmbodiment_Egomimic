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
    return parser.parse_args()

def main(args):
    # Enumerate all connected RealSense devices
    ctx = rs.context()
    devices = ctx.query_devices()
    num_devices = len(devices)
    print(f"Found {num_devices} RealSense cameras.")

    if num_devices < 2:
        print(f"Error: Only {num_devices} RealSense cameras found. Please connect at least two cameras.")
        # exit()
    elif num_devices > 2:
        print(f"Warning: {num_devices} RealSense cameras found. Using the first two detected cameras.")

    pipelines = []
    configs = []
    serial_numbers = []
    
    # Define camera mappings
    camera_map = {
        "317422075805": "right",  # Right arm camera
        "332522076706": "left",   # Left arm camera
        "f1421276" : "center"
    }
    
    # Store camera information in a structured way
    camera_info = {} # Stores {serial_number: {"pipeline": pipeline, "config": config, "arm_type": "left/right", "profile": None}}

    # Initialize pipelines for the first two detected cameras and map them to arm types
    found_cameras = 0
    for device in devices:
        if found_cameras >= 2:
            break
        
        serial_number = device.get_info(rs.camera_info.serial_number)
        
        if serial_number not in camera_map:
            print(f"Warning: Unknown camera with serial number {serial_number}. Skipping.")
            continue # Skip cameras not in our defined map

        arm_type = camera_map[serial_number]
        print(f"Configuring camera with serial number: {serial_number} (Mapped to {arm_type} arm)")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        
        # Enable the color stream (RGB)
        color_width = 640
        color_height = 480
        color_fps = 30
        config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
        
        pipelines.append(pipeline)
        configs.append(config)
        serial_numbers.append(serial_number)
        
        camera_info[serial_number] = {
            "pipeline": pipeline,
            "config": config,
            "arm_type": arm_type,
            "profile": None
        }
        found_cameras += 1

    if found_cameras < 2:
        print("Error: Could not find both 'left' and 'right' arm cameras with the specified serial numbers.")
        exit()

    # Start streaming for both pipelines
    try:
        for serial_number, info in camera_info.items():
            profile = info["pipeline"].start(info["config"])
            info["profile"] = profile # Store the profile
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            print(f"Camera {serial_number} ({info['arm_type']}) Color Stream: Resolution {color_profile.width()}x{color_profile.height()}, FPS: {color_profile.fps()}")
    except RuntimeError as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("Please ensure both cameras are connected and not in use by another application.")
        for info in camera_info.values():
            try:
                info["pipeline"].stop()
            except:
                pass # Already stopped or never started
        exit()

    # Prepare output directories
    base_output_dir = "aloha-data"
    current_datetime_id = datetime.now().strftime("%Y%m%d%H%M%S")

    # Create specific directories for left/right and mp4/npy
    output_paths = {}
    for serial_number, info in camera_info.items():
        arm_type = info["arm_type"]
        mp4_dir = os.path.join(base_output_dir, arm_type, "mp4")
        npy_dir = os.path.join(base_output_dir, arm_type, "npy")
        
        os.makedirs(mp4_dir, exist_ok=True)
        os.makedirs(npy_dir, exist_ok=True)
        
        output_paths[serial_number] = {
            "mp4_path": os.path.join(mp4_dir, f'video_recording_realsense_{serial_number}#{current_datetime_id}.mp4'),
            "npy_path": os.path.join(npy_dir, f'video_recording_realsense_{serial_number}#{current_datetime_id}.npy')
        }

    # Prepare video writers and numpy arrays for each camera
    video_writers = {}
    frame_arrays = {} # Stores {serial_number: [timestamps]}

    for serial_number, info in camera_info.items():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        color_profile = info["profile"].get_stream(rs.stream.color).as_video_stream_profile()
        actual_color_width = color_profile.width()
        actual_color_height = color_profile.height()
        actual_color_fps = color_profile.fps()

        out = cv2.VideoWriter(output_paths[serial_number]["mp4_path"], fourcc, actual_color_fps, (actual_color_width, actual_color_height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for camera {serial_number} ({info['arm_type']}) with FOURCC {fourcc} for {output_paths[serial_number]['mp4_path']}.")
            print("This often means OpenCV is not built with the necessary codec support (e.g., FFMPEG with H.264).")
            print("Try a different FOURCC, like cv2.VideoWriter_fourcc(*'mp4v') or cv2.VideoWriter_fourcc(*'MJPG') with a .avi extension.")
            for p_info in camera_info.values():
                try:
                    p_info["pipeline"].stop()
                except:
                    pass
            exit()
        
        video_writers[serial_number] = out
        frame_arrays[serial_number] = []
        print(f"Recording video from Intel RealSense camera {serial_number} ({info['arm_type']}) to {output_paths[serial_number]['mp4_path']}.")

    print("Press 'q' to stop recording for both cameras.")

    start_time = time.time()
    prev_times = {sn: time.time() for sn in serial_numbers}

    try:
        while True:
            # Get frames for both cameras
            frames_map = {}
            for serial_number, info in camera_info.items():
                frames_map[serial_number] = info["pipeline"].wait_for_frames()

            for serial_number in serial_numbers: # Iterate in a consistent order (the order they were configured)
                frames = frames_map[serial_number]
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                video_writers[serial_number].write(color_image)
                
                # Display each camera's feed in its own window
                cv2.imshow(f'RealSense Video Recording - {camera_info[serial_number]["arm_type"].capitalize()} Arm ({serial_number})', color_image)

                frame_arrays[serial_number].append(time.time())

                if args.fps:
                    current_time = time.time()
                    dt = current_time - prev_times[serial_number]
                    if dt > 0:
                        fps = 1.0 / dt
                        print(f"Camera {serial_number} ({camera_info[serial_number]['arm_type']}) Approximate FPS: {fps:.2f}")
                    prev_times[serial_number] = current_time
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for info in camera_info.values():
            try:
                info["pipeline"].stop()
            except:
                pass
        for out in video_writers.values():
            out.release()
        cv2.destroyAllWindows()
        
        for serial_number, frame_array in frame_arrays.items():
            np.save(output_paths[serial_number]["npy_path"], np.asarray(frame_array, dtype=np.float64))

    end_time = time.time()
    duration = end_time - start_time

    print(f"Recording stopped for both cameras. Total Duration: {duration:.2f} seconds.")
    for serial_number, info in camera_info.items():
        print(f"Camera {serial_number} ({info['arm_type']}) video saved as '{output_paths[serial_number]['mp4_path']}' and timestamps as '{output_paths[serial_number]['npy_path']}'")

if __name__ == "__main__":
    args = parse_args()
    main(args)