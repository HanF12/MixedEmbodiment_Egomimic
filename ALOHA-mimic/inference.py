#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState # Explicitly import JointState
from ros_pub import * # Explicitly import function
from joint_lisener import joint_state_listener, get_current_slave_left_positions # Explicitly import functions
import numpy as np
import os
import time
import torch
import cv2
import pyrealsense2 as rs
import torch.nn.functional as F # For image resizing
import argparse
import warnings
from core import build
import collections # Import collections for deque
K = 80
class Args:
    def __init__(self):
        self.num_queries = K
        self.camera_names = ["cam0", "cam1"]
        self.hidden_dim = 512 # 256 before
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 3200
        self.enc_layers = 4
        self.dec_layers = 7
        self.pre_norm = False

        # Backbone/DETR args
        self.position_embedding = 'sine'
        self.backbone = 'resnet18'
        self.lr_backbone = 1e-5
        self.masks = False
        self.dilation = False

        # Custom for your use
        self.state_dim = 7


args = Args()
state_dim = 7
model = build(args)

# --- Global Configuration ---
DEBUG = False
CHUNKING = True # Change to true to do the real ACT.

# Defined by ACT Paper CONSTANTs:
K_PREDICTION_HORIZON = 80 # This is K from the paper (prediction horizon)
K_AGGREGATION_HORIZON = 80 # This is K from the paper (aggregation horizon), often same as prediction
m = 0.4 # Decay factor for chunking

INFERENCE_FPS = 30
target_interval_ms = (1.0 / INFERENCE_FPS) * 1000

resize_factor = 0.5 # If you want to resize images, change this. Set to 1.0 means no resizing for now.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Global buffer for ACT temporal chunking
# Stores (K_PREDICTION_HORIZON, 7) numpy arrays of predicted trajectories
past_predictions_buffer = collections.deque(maxlen=K_AGGREGATION_HORIZON)
model.to(device)

# --- Model Loading ---
try:
    # Ensure this path is correct relative to where you run the script, or use an absolute path
    state_dict = torch.load("single_arm_ACT_100.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # Set model to evaluation mode
    print("Model Load Success")
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
    init_publishers(topic_arm='/arm_joint_target_position', topic_gripper='/gripper_position_control_host')
    joint_state_listener() # Assuming this only sets up the subscriber, not blocks.
except rospy.ROSInitException as e:
    print(f"Failed to initialize ROS node: {e}")
    exit()

# --- GPU Information ---
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s):")
    for i in range(n_gpus):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs detected, falling back to CPU.")

# --- RealSense Camera Setup (Outside the loop) ---
pipeline = rs.pipeline()
config = rs.config()
color_width = 640
color_height = 480
color_fps = 30
config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
try:
    profile = pipeline.start(config)
except RuntimeError as e:
    print(f"Error starting RealSense pipeline: {e}")
    print("Please ensure the camera is connected and not in use by another application.")
    exit()

# --- Bird Camera Setup (Outside the loop) ---
# IMPORTANT: Move cv2.VideoCapture(6) OUTSIDE the loop!
bird_cap = cv2.VideoCapture(6) # Assume Bird Video is video6
if not bird_cap.isOpened():
    print(f"Error: Unable to open camera device {6} (Bird Camera).")
    pipeline.stop() # Clean up RealSense if bird cam fails
    exit()

action_number = 0 # Counter for the number of inference steps

try:
    while not rospy.is_shutdown(): # Use rospy.is_shutdown() for ROS node termination
        loop_start_time = time.perf_counter()

        # --- 1. Capture Frames Simultaneously ---
        # RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Skipping frame: No RealSense color frame.")
            continue
        color_image = np.asanyarray(color_frame.get_data()) # RealSense image (BGR)

        # Bird Camera
        ret, frame = bird_cap.read() # frame is the bird data (BGR)
        if not ret:
            warnings.warn("No frame from bird camera. Skipping this iteration.")
            continue # Skip loop iteration if bird cam fails

        # --- 2. Get Current Joint Data ---
        pos_vel_tuple = get_current_slave_left_positions()
        current_joint_data_np = np.array(list(pos_vel_tuple[0]))
        velocity = pos_vel_tuple[1]
        if current_joint_data_np is None:
            rospy.logwarn("No slave_left joint state message received yet. Waiting...")
            time.sleep(0.01) # Small sleep to avoid busy-waiting
            continue # Skip this loop iteration until data is available
        if DEBUG:
            print('joint_pos: ')
            print(current_joint_data_np)
            print('joint_vel: ')
            print(velocity)
        # Convert joint data to tensor and move to device
        # Assuming current_joint_data_np is a NumPy array
        current_joint_data_tensor = torch.from_numpy(current_joint_data_np).float().to(device)


        # --- 3. Preprocess and Stack Images ---
        # Resize if necessary (interpolation for images)
        if resize_factor != 1.0:
            # RealSense image
            h_rs, w_rs = color_image.shape[:2]
            color_image = cv2.resize(
                color_image,
                (int(w_rs * resize_factor), int(h_rs * resize_factor)),
                interpolation=cv2.INTER_AREA
            )
            # Bird image
            h_bird, w_bird = frame.shape[:2]
            frame = cv2.resize(
                frame,
                (int(w_bird * resize_factor), int(h_bird * resize_factor)),
                interpolation=cv2.INTER_AREA
            )

        # Convert BGR (OpenCV default) to RGB for consistency with PyTorch models
        # and normalize to [0, 1]
        left_arm_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        left_arm_img_tensor = torch.from_numpy(left_arm_img.transpose(2,0,1)).float().div_(255.0)

        bird_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bird_img_tensor = torch.from_numpy(bird_img_rgb.transpose(2,0,1)).float().div_(255.0)

        # Stack images (batch dimension added later)
        # Resulting shape: (2, C, H, W)
        stacked_images = torch.stack([left_arm_img_tensor, bird_img_tensor], dim=0)

        # Add batch dimension and move to device: (1, 2, C, H, W)
        stacked_images = stacked_images.unsqueeze(0).to(device)
        # Add batch dimension to joint data: (1, JOINT_DOF)
        current_joint_data_tensor = current_joint_data_tensor.unsqueeze(0).to(device)

        if DEBUG:
            print('Stacked Image Shape:', stacked_images.shape)
            print('Joint Read Data Shape:', current_joint_data_tensor.shape)

        # --- 4. Model Inference ---
        with torch.no_grad(): # Essential for evaluation
            # The model expects current_joint_data, stacked_images, and a third arg (e.g., target_positions)
            # Pass None for target_positions during inference if not used.
            predicted_trajectory = model(current_joint_data_tensor, stacked_images, None)

        # Remove batch dimension: predicted_trajectory shape: (K_PREDICTION_HORIZON, JOINT_DOF)
        predicted_trajectory = predicted_trajectory[0].squeeze(0)

        # Ensure the predicted trajectory is on CPU and is a numpy array for storage
        predicted_trajectory_np = predicted_trajectory.cpu().numpy()

        # Add the current full prediction to the buffer
        past_predictions_buffer.append(predicted_trajectory_np)

        # --- 5. Action Aggregation (Temporal Chunking) ---
        positions_to_publish = np.array([0,0,0,0,0,0,0], dtype=np.float32)

        if CHUNKING:
            # Iterate through the predictions stored in the buffer (most recent to oldest)
            # The 'i' here is the lookahead index from the current time step (0 for current, 1 for next, etc.)
            w = 0
            for i in range(len(past_predictions_buffer)):
                # `past_predictions_buffer[-(i+1)]` gets the prediction made `i` steps ago
                # `[i]` gets the `i`-th action *from that specific prediction*
                # Example:
                # i=0: use the most recent prediction's (0-th) action
                # i=1: use the 2nd most recent prediction's (1-st) action
                # i=2: use the 3rd most recent prediction's (2-nd) action
                
                # Check if the required index exists in the past prediction
                if i < past_predictions_buffer[-(i+1)].shape[0]:
                    w_c = np.exp(-m * i)
                    positions_to_publish += past_predictions_buffer[-(i+1)][i] * np.exp(-m * i)
                    w += w_c
                else:
                    # This case means an older prediction's horizon was too short
                    # for the 'i' required. Should generally not happen if K_PREDICTION_HORIZON is consistent.
                    # This would typically happen only if K_AGGREGATION_HORIZON > K_PREDICTION_HORIZON
                    print(123)
                    pass # Or log a warning
            if w < 1:
                w = 1
            positions_to_publish /= w
            print(positions_to_publish)
        else:
            # If not chunking, just use the first predicted action of the current prediction
            positions_to_publish = predicted_trajectory_np[0] # Already numpy on CPU

        # --- 6. Publish Trajectory ---
        pos = positions_to_publish
        pos[-1]*= 32 # 30 before
        if pos[-1] > 60:
            pos[-1] = 60
        # pos[-2] += 1.5
        if pos[-2] > 2.8:
            pos[-2] = 2.8
        if pos[-2] < -2.8:
            pos[-2] = -2.8
        traj = []
        traj.append(pos)
        traj = np.array(traj)
        publish_trajectory(traj) # TEST TEST

        # --- Display frames (Optional) ---
        # cv2.imshow('RealSense Feed', color_image)
        # cv2.imshow('Bird Camera Feed', frame)

        # --- Maintain Loop Frequency ---
        elapsed_time_ms = (time.perf_counter() - loop_start_time) * 1000
        sleep_time_ms = target_interval_ms - elapsed_time_ms

        if sleep_time_ms > 0:
            time.sleep(sleep_time_ms / 1000.0)
        else:
            pass
            # print(f"Warning: Loop took too long ({elapsed_time_ms:.2f}ms), couldn't maintain {INFERENCE_FPS}Hz.")
        
        action_number += 1 # Increment action counter
        # Exit condition (for OpenCV window)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Inference stopped by user (KeyboardInterrupt).")
except rospy.ROSInterruptException:
    print("ROS node interrupted (ROSInterruptException).")
finally:
    # --- Clean Up ---
    print("Cleaning up resources...")
    pipeline.stop() # Stop RealSense pipeline
    bird_cap.release() # Release Bird Camera
    cv2.destroyAllWindows() # Close OpenCV windows
    print("Cameras released and windows closed.")

# The script implicitly exits after the try-finally block or if `rospy.is_shutdown()` becomes true.