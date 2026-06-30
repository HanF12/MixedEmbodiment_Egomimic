#!/usr/bin/env python
# This line tells the OS to execute the script using the python interpreter

import rospy
# Import the JointState message type
from sensor_msgs.msg import JointState
import h5py # Import the h5py library for HDF5 file operations
import numpy as np # Import numpy for array handling
import os # Import os for path manipulation
import cv2
import threading
import time
from datetime import datetime
'''
class CameraRecorder:
    def __init__(self, camera_index, filename, fps, resolution):
        self.camera_index = camera_index
        self.filename = filename
        self.fps = fps
        self.resolution = resolution
        self.cap = None
        self.out = None
        self.recording = False
        self.thread = None
        self.frames_captured = 0
        self.frames_written = 0
        self.start_time = None
        self.queue = [] # To store frames before writing
        self.queue_lock = threading.Lock()
        self.frames_to_process = threading.Event() # Event to signal frames are ready

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False

        # Try to set resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Verify actual settings (may not be exactly what you asked for)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera {self.camera_index}: Actual Resolution {actual_width}x{actual_height}, Actual FPS {actual_fps}")
        if actual_width != self.resolution[0] or actual_height != self.resolution[1]:
            print(f"Warning: Camera {self.camera_index} could not set desired resolution {self.resolution}. Using {actual_width}x{actual_height}.")
        if actual_fps != self.fps:
             print(f"Warning: Camera {self.camera_index} could not set desired FPS {self.fps}. Using {actual_fps}.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
        self.out = cv2.VideoWriter(self.filename, fourcc, actual_fps, (actual_width, actual_height))
        if not self.out.isOpened():
            print(f"Error: Could not create video writer for {self.filename}")
            self.cap.release()
            return False
        return True

    def _record_frames(self):
        frame_interval = 1.0 / self.fps
        last_frame_time = time.perf_counter()

        while self.recording:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error: Failed to read frame from camera {self.camera_index}")
                break

            current_time = time.perf_counter()
            # Simple software-based timestamping for rough synchronization
            frame_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            with self.queue_lock:
                self.queue.append((frame_timestamp, frame))
            self.frames_captured += 1
            self.frames_to_process.set() # Signal that there are frames in the queue

            # Try to maintain desired FPS by sleeping if too fast
            elapsed = time.perf_counter() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.perf_counter()


    def _write_frames(self):
        while self.recording or not self.frames_to_process.is_set() or len(self.queue) > 0:
            self.frames_to_process.wait(timeout=0.1) # Wait for new frames or timeout

            frames_to_write = []
            with self.queue_lock:
                if len(self.queue) > 0:
                    frames_to_write = sorted(self.queue, key=lambda x: x[0]) # Sort by timestamp
                    self.queue = [] # Clear the queue after copying

            for timestamp, frame in frames_to_write:
                if self.out.isOpened():
                    self.out.write(frame)
                    self.frames_written += 1
                # You can also save timestamped frames as images here for analysis if needed
                # cv2.imwrite(f"frames/camera_{self.camera_index}/{timestamp}.jpg", frame)

            self.frames_to_process.clear() # Reset the event


    def start_recording(self):
        if not self._init_camera():
            return False

        self.recording = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._record_frames)
        self.thread.start()

        self.write_thread = threading.Thread(target=self._write_frames)
        self.write_thread.start()

        print(f"Camera {self.camera_index} started recording to {self.filename}")
        return True

    def stop_recording(self):
        self.recording = False
        if self.thread:
            self.thread.join() # Wait for the recording thread to finish
        if self.write_thread:
            self.frames_to_process.set() # Signal one last time to drain queue
            self.write_thread.join() # Wait for the writing thread to finish

        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        print(f"Camera {self.camera_index} stopped recording. Captured: {self.frames_captured}, Written: {self.frames_written} frames.")

def main():
    output_dir = "recorded_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configuration for cameras
    camera_configs = [
        {"index": 2, "filename": os.path.join(output_dir, "camera1_video.mp4"), "resolution": (640, 480)},
        {"index": 4, "filename": os.path.join(output_dir, "camera2_video.mp4"), "resolution": (640, 480)},
    ]
    target_fps = 30 # Target recording frequency

    recorders = []
    for config in camera_configs:
        recorder = CameraRecorder(
            camera_index=config["index"],
            filename=config["filename"],
            fps=target_fps,
            resolution=config["resolution"]
        )
        recorders.append(recorder)

    # Start all cameras
    all_started = True
    for recorder in recorders:
        if not recorder.start_recording():
            all_started = False
            break

    if not all_started:
        print("Failed to start all cameras. Exiting.")
        for recorder in recorders:
            recorder.stop_recording() # Clean up any started cameras
        return

    record_duration_seconds = 10 # Record for 10 seconds
    print(f"Recording for {record_duration_seconds} seconds...")
    time.sleep(record_duration_seconds)

    # Stop all cameras
    print("Stopping recording...")
    for recorder in recorders:
        recorder.stop_recording()

    print("Recording complete.")

'''
# --- Global Variables ---
# HDF5 file handle
hdf5_file = None
# HDF5 dataset handles
positions_dset = None
velocities_dset = None
timestamps_dset = None
joint_names_dset = None
time_arr = []
position_arr = []
# Sampling parameters
sampling_frequency = 30.0 # Desired sampling frequency in Hz
sampling_interval = 1.0 / sampling_frequency # Time interval between samples in seconds
last_sampled_time = 0.0 # To keep track of the last time a message was sampled

# File path for the HDF5 file
# You can change this path and filename
# Using a timestamp in the filename is a good practice to avoid overwriting
# hdf5_file_path = os.path.join(os.path.expanduser("~"), "ros_data", f"slave_left_joint_states_{rospy.Time.now().secs}.hdf5")
id_str = datetime.now().strftime("%Y%m%d%H%M%S")
hdf5_file_path = os.path.join(os.path.expanduser("~"), "ros_data", f"host_left_joint_states_{id_str}.hdf5")

# Flag to indicate if joint names have been stored
joint_names_stored = False
num_joints = 0 # To store the number of joints once known


# --- Callback Function ---
# This function is called every time a new message is received on the subscribed topic

def slave_left_joint_states_callback(msg):
    """
    Callback function for receiving robot arm joint state feedback for slave_left.
    Samples the data and stores it in an HDF5 file.
    """
    global hdf5_file, positions_dset, velocities_dset, timestamps_dset, \
           last_sampled_time, joint_names_stored, joint_names_dset, num_joints, time_arr, position_arr

    current_time = msg.header.stamp.to_sec() # Get the timestamp of the message in seconds

    # Check if enough time has passed since the last sampled message
    if current_time - last_sampled_time >= sampling_interval:
        last_sampled_time = current_time # Update the last sampled time

        # Ensure the HDF5 file and datasets are open
        if hdf5_file is None:
            rospy.logerr("HDF5 file is not open. Cannot save data.")
            return

        # Ensure consistent data length
        if len(msg.position) != len(msg.velocity) or (num_joints > 0 and len(msg.position) != num_joints):
             rospy.logwarn(f"Received JointState message with inconsistent data lengths ({len(msg.position)} pos, {len(msg.velocity)} vel). Skipping.")
             return

        # Store the number of joints if it's the first sampled message
        if num_joints == 0:
            num_joints = len(msg.position)
            rospy.loginfo(f"Detected {num_joints} joints. Initializing HDF5 datasets.")
            try:
                 # Create resizable datasets once the number of joints is known
                 # maxshape=(None, num_joints) allows appending rows (time steps)
                 positions_dset = hdf5_file.create_dataset('positions', shape=(0, num_joints), maxshape=(None, num_joints), dtype='float32', chunks=True)
                 velocities_dset = hdf5_file.create_dataset('velocities', shape=(0, num_joints), maxshape=(None, num_joints), dtype='float32', chunks=True)
                 timestamps_dset = hdf5_file.create_dataset('timestamps', shape=(0,), maxshape=(None,), dtype='float64', chunks=True)
                 # Store joint names once
                 joint_names_dset = hdf5_file.create_dataset('joint_names', data=np.array(msg.name, dtype='S')) # Store as fixed-length strings (bytes)
                 joint_names_stored = True
                 rospy.loginfo("HDF5 datasets created.")
            except Exception as e:
                 rospy.logerr(f"Error creating HDF5 datasets: {e}")
                 # Close the file if dataset creation failed
                 close_hdf5_file()
                 return


        # Append data to the datasets
        try:
            # Resize datasets to accommodate new data
            current_size = timestamps_dset.shape[0]
            positions_dset.resize((current_size + 1, num_joints))
            velocities_dset.resize((current_size + 1, num_joints))
            timestamps_dset.resize((current_size + 1,))

            # Write the new data
            positions_dset[current_size, :] = msg.position
            velocities_dset[current_size, :] = msg.velocity
            timestamps_dset[current_size] = current_time
            time_arr.append(current_time)
            position_arr.append(msg.position)
            # rospy.loginfo(f"Sampled and stored data at time: {current_time:.4f}")

        except Exception as e:
            rospy.logerr(f"Error appending data to HDF5 datasets: {e}")
            # Attempt to close the file on error
            close_hdf5_file()


# --- HDF5 File Handling Functions ---

def open_hdf5_file():
    """
    Opens or creates the HDF5 file and prepares datasets.
    """
    global hdf5_file, positions_dset, velocities_dset, timestamps_dset, joint_names_dset

    # Create directory if it doesn't exist
    data_dir = os.path.dirname(hdf5_file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        rospy.loginfo(f"Created data directory: {data_dir}")

    try:
        # Open the HDF5 file in write mode ('w').
        # Use 'a' mode if you want to append to an existing file instead of overwriting.
        hdf5_file = h5py.File(hdf5_file_path, 'w')
        rospy.loginfo(f"HDF5 file opened: {hdf5_file_path}")

        # Datasets will be created in the callback once the number of joints is known.

    except Exception as e:
        rospy.logerr(f"Error opening HDF5 file: {e}")
        # Set file handle to None if opening failed
        hdf5_file = None


def close_hdf5_file():
    """
    Closes the HDF5 file if it is open.
    This function is called when the node is shutting down.
    """
    global hdf5_file, time_arr, position_arr
    if hdf5_file is not None:
        try:
            hdf5_file.close()
            rospy.loginfo(f"HDF5 file closed: {hdf5_file_path}")
        except Exception as e:
            rospy.logerr(f"Error closing HDF5 file: {e}")
        finally:
            np.save(f'joint_timestamp_{id_str}', time_arr)
            np.save(f'joint_position_{id_str}', position_arr)
            hdf5_file = None # Ensure the handle is None after trying to close


# --- Main Listener Function ---
def joint_state_listener():
    """
    Initializes the ROS node, sets up the subscriber, opens the HDF5 file,
    and keeps the node running to listen for messages.
    """
    # Initialize the ROS node.
    rospy.init_node('slave_left_joint_state_recorder', anonymous=True)
    rospy.loginfo("ROS Slave Left Joint State Recorder Node Initialized.")

    # --- Open HDF5 File ---
    open_hdf5_file()
    # If file opening failed, exit the node
    if hdf5_file is None:
        rospy.logerr("Failed to open HDF5 file. Exiting node.")
        return


    # --- Set up Shutdown Hook ---
    # Register the close_hdf5_file function to be called when the node is shutting down.
    rospy.on_shutdown(close_hdf5_file)
    rospy.loginfo("Shutdown hook registered for HDF5 file closure.")


    # --- Subscriber ---
    # Create a subscriber to the /joint_states_slave_left topic.
    rospy.Subscriber('/joint_states_host_left', JointState, slave_left_joint_states_callback)
    rospy.loginfo("Subscriber created for topic: /joint_states_host_left")

    # --- Keep Node Alive ---
    # rospy.spin() enters a loop that keeps the node alive and allows the
    # callback functions to be executed when messages are received.
    rospy.loginfo("Node is spinning and listening for messages...")
    rospy.spin()

    # This part is reached when rospy.spin() exits (e.g., node shutdown)
    rospy.loginfo("Node finished spinning.")


# --- Script Entry Point ---
if __name__ == '__main__':
    # This block is executed when the script is run directly.
    try:
        # Call the main listener function
        joint_state_listener()
    except rospy.ROSInterruptException:
        # Catch an exception that is raised when the node receives a shutdown signal (like Ctrl+C).
        rospy.loginfo("ROS Interrupt Exception caught. Node shutting down.")
    except Exception as e:
        # Catch any other unexpected exceptions.
        rospy.logerr(f"An unexpected error occurred: {e}")

    rospy.loginfo("Script finished.")
