import os
import h5py

# mps_sample_path = "/coc/flash9/skareer6/Projects/EgoPlay/aria/aria_demo/simar/"

from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze,
    get_nearest_pose,
)
from projectaria_tools.core.stream_id import StreamId
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional

from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from aria_utils import (
    build_camera_matrix,
    undistort_to_linear,
    split_train_val_from_hdf5,
    slam_to_rgb,
)

import argparse

import json

from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    WIDE_LENS_HAND_LEFT_K,
    interpolate_keys,
    interpolate_arr
)
from egomimic.scripts.masking.utils import *

HORIZON = 10
STEP = 3.0


"""
Example usage
python aria_to_robomimic.py --dataset /coc/flash7/datasets/egoplay/oboo_aria_apr16/oboo_aria_apr16/ --out /coc/flash7/datasets/egoplay/oboo_aria_apr16/converted/oboo_aria_apr16_rightMimicplay.hdf5 --hand right
"""

# Load the VRS file


def single_file_conversion(dataset, mps_sample_path, filename, hand):
    """
    dataste: path to the dataset
    mps_sample_path: path to the mps sample
    filename: name of the vrs file
    hand: left, right, bimanual

    Returns: actions, front_img_1, ee_pose
    """
    vrsfile = os.path.join(dataset, filename)

    # Hand tracking
    wrist_and_palm_poses_path = os.path.join(
        mps_sample_path, "hand_tracking", "wrist_and_palm_poses.csv"
    )

    # Create data provider and get T_device_rgb
    provider = data_provider.create_vrs_data_provider(vrsfile)

    ## Load hand tracking
    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(
        wrist_and_palm_poses_path
    )

    # Get device calibration and transform from device to sensor
    device_calibration = provider.get_device_calibration()

    time_domain: TimeDomain = TimeDomain.DEVICE_TIME
    time_query_closest: TimeQueryOptions = TimeQueryOptions.CLOSEST

    # Get stream ids, stream labels, stream timestamps, and camera calibrations for RGB and SLAM cameras
    stream_ids: Dict[str, StreamId] = {
        "rgb": StreamId("214-1"),
        "slam-left": StreamId("1201-1"),
        "slam-right": StreamId("1201-2"),
    }
    stream_labels: Dict[str, str] = {
        key: provider.get_label_from_stream_id(stream_id)
        for key, stream_id in stream_ids.items()
    }
    stream_timestamps_ns: Dict[str, List[int]] = {
        key: provider.get_timestamps_ns(stream_id, time_domain)
        for key, stream_id in stream_ids.items()
    }
    cam_calibrations = {
        key: device_calibration.get_camera_calib(stream_label)
        for key, stream_label in stream_labels.items()
    }
    for key, cam_calibration in cam_calibrations.items():
        assert cam_calibration is not None, f"no camera calibration for {key}"

    mps_data_paths_provider = mps.MpsDataPathsProvider(mps_sample_path)
    mps_data_paths = mps_data_paths_provider.get_data_paths()
    mps_data_provider = mps.MpsDataProvider(mps_data_paths)

    frame_length = len(stream_timestamps_ns["rgb"]) - 1
    print(frame_length)
    actions = []
    front_img_1 = []
    ee_pose = []
    ## get camera matrices
    vrs_data_provider = data_provider.create_vrs_data_provider(vrsfile)

    transform = slam_to_rgb(vrs_data_provider)

    for t in range(frame_length + 1):
        # if t >= 2000:
        #     break
        if t + HORIZON * STEP < frame_length + 1:
            if (t % 1000) == 0:
                print(f"{t} frames ingested")
            ## sampled image and camera pose at time t
            sample_timestamp_ns_t: int = stream_timestamps_ns["rgb"][t]
            sample_frames = {
                key: provider.get_image_data_by_time_ns(
                    stream_id,
                    sample_timestamp_ns_t,
                    time_domain,
                    time_query_closest,
                )[0]
                for key, stream_id in stream_ids.items()
            }
            front_img_1_t = undistort_to_linear(
                provider,
                stream_ids,
                raw_image=sample_frames["rgb"].to_numpy_array(),
            )
            ## obs ee_pose
            wrist_and_palm_pose_t = mps_data_provider.get_wrist_and_palm_pose(
                sample_timestamp_ns_t, time_query_closest
            )

            rotation_matrix = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

            if hand == "right":
                ee_pose_obs_t_rot = (
                    transform
                    @ wrist_and_palm_pose_t.right_hand.palm_position_device
                ).T
                ee_pose_obs_t = np.dot(ee_pose_obs_t_rot, rotation_matrix.T)
                actions_t = np.zeros((10, 3))
            elif hand == "left":
                ee_pose_obs_t_rot = (
                    transform @ wrist_and_palm_pose_t.left_hand.palm_position_device
                ).T
                ee_pose_obs_t = np.dot(ee_pose_obs_t_rot, rotation_matrix.T)
                actions_t = np.zeros((10, 3))
            elif hand == "bimanual":
                pose_l = (
                    transform @ wrist_and_palm_pose_t.left_hand.palm_position_device
                ).T
                pose_r = (
                    transform
                    @ wrist_and_palm_pose_t.right_hand.palm_position_device
                ).T

                pose_l_rot = np.dot(pose_l, rotation_matrix.T)
                pose_r_rot = np.dot(pose_r, rotation_matrix.T)

                ee_pose_obs_t = np.concatenate(
                    (pose_l_rot, pose_r_rot), axis=None
                )  ## left, right -> [x, y, z, x, y, z]
                actions_t = np.zeros((10, 6))
            ## t to t + 9

            # breakpoint()

            pose_t = mps_data_provider.get_closed_loop_pose(
                sample_timestamp_ns_t, time_query_closest
            )

            camera_matrix = build_camera_matrix(vrs_data_provider, pose_t)
            camera_t_inv = np.linalg.inv(camera_matrix)

            for offset in range(HORIZON):
                sample_timestamp_ns = stream_timestamps_ns["rgb"][
                    int(t + offset * STEP)
                ]
                wrist_and_palm_pose = mps_data_provider.get_wrist_and_palm_pose(
                    sample_timestamp_ns, time_query_closest
                )
                ## LR pose at time t + offset in camera t + offset frame
                right_palm = (
                    transform @ wrist_and_palm_pose.right_hand.palm_position_device
                ).T
                left_palm = (
                    transform @ wrist_and_palm_pose.left_hand.palm_position_device
                ).T

                pose_offset = mps_data_provider.get_closed_loop_pose(
                    sample_timestamp_ns, time_query_closest
                )
                camera_matrix_offset = build_camera_matrix(
                    vrs_data_provider, pose_offset
                )

                if hand == "right":
                    if np.any(right_palm) or not np.all(right_palm == 0):
                        right_palm_hom = np.append(right_palm, 1)
                        hand_world = np.dot(
                            camera_matrix_offset, right_palm_hom
                        )  ## hand_pose[t + offset] in world frame
                        hand_in_camera_t_frame = np.dot(
                            camera_t_inv, hand_world
                        )  ## hand_pose[t + offset] in camera t frame
                        actions_t[offset] = hand_in_camera_t_frame[:3]
                elif hand == "left":
                    if np.any(left_palm) or not np.all(left_palm == 0):
                        left_palm_hom = np.append(left_palm, 1)
                        hand_world = np.dot(
                            camera_matrix_offset, left_palm_hom
                        )  ## hand_pose[t + offset] in world frame
                        hand_in_camera_t_frame = np.dot(
                            camera_t_inv, hand_world
                        )  ## hand_pose[t + offset] in camera t frame
                        actions_t[offset] = hand_in_camera_t_frame[:3]
                elif hand == "bimanual":
                    if (np.any(right_palm) or not np.all(right_palm == 0)) or (
                        np.any(left_palm) or not np.all(left_palm == 0)
                    ):
                        ## right palm
                        right_palm_hom = np.append(right_palm, 1)
                        hand_world_r = np.dot(
                            camera_matrix_offset, right_palm_hom
                        )  ## right hand_pose[t + offset] in world frame
                        hand_in_camera_t_frame_r = np.dot(
                            camera_t_inv, hand_world_r
                        )  ## right hand_pose[t + offset] in world frame

                        ## left palm
                        left_palm_hom = np.append(left_palm, 1)
                        hand_world_l = np.dot(
                            camera_matrix_offset, left_palm_hom
                        )  ## left hand_pose[t + offset] in world frame
                        hand_in_camera_t_frame_l = np.dot(
                            camera_t_inv, hand_world_l
                        )  ## left hand_pose[t + offset] in camera t frame
                        actions_t[offset] = np.concatenate(
                            (
                                hand_in_camera_t_frame_l[:3],
                                hand_in_camera_t_frame_r[:3],
                            ),
                            axis=None,
                        )

            if actions_t.shape == (10, 6):
                # [x1 y1 z1 x2 y2 z2] -> [[x1 y1 z1], [x2 y2 z2]]
                # actions_t_reshaped = actions_t.reshape(-1, 3)

                # # apply rotation
                # rotated_actions_t_reshaped = np.dot(actions_t_reshaped, rotation_matrix.T)

                # # Reshape back to [x1 y1 z1 x2 y2 z2]
                # rotated_actions_t = rotated_actions_t_reshaped.reshape(10, 6)
                actions_t_l = actions_t[:, :3]
                actions_t_r = actions_t[:, 3:]
                actions_t_rot_l = np.dot(actions_t_l, rotation_matrix.T)
                actions_t_rot_r = np.dot(actions_t_r, rotation_matrix.T)
                rotated_actions_t = np.concatenate(
                    (actions_t_rot_l, actions_t_rot_r), axis=1
                )

            else:
                rotated_actions_t = np.dot(actions_t, rotation_matrix.T)

            if np.any(rotated_actions_t) or not np.all(rotated_actions_t == 0):
                actions.append(rotated_actions_t.flatten())
                front_img_1.append(front_img_1_t)
                ee_pose.append(np.ravel(ee_pose_obs_t))


    actions, front_img_1, ee_pose = (
        np.array(actions),
        np.array(front_img_1),
        np.array(ee_pose),
    )

    ac_dim = actions_t.shape[-1]
    actions_flat = actions.copy().reshape((-1, 3))
    px = cam_frame_to_cam_pixels(
        transform_actions(actions_flat), WIDE_LENS_HAND_LEFT_K
    )
    px = px.reshape((-1, HORIZON, ac_dim))
    if ac_dim == 3:
        bad_data_mask = (
            (px[:, :, 0] < 0)
            | (px[:, :, 0] > 640)
            | (px[:, :, 1] < 0)
            | (px[:, :, 1] > 480)
        )
    elif ac_dim == 6:
        BUFFER = 0
        bad_data_mask = (
            (px[:, :, 0] < 0 - BUFFER)
            | (px[:, :, 0] > 640 + BUFFER)
            | (px[:, :, 1] < 0)
            # | (px[:, :, 1] > 480 + BUFFER)
            | (px[:, :, 3] < 0 - BUFFER)
            | (px[:, :, 3] > 640 + BUFFER)
            | (px[:, :, 4] < 0)
            # | (px[:, :, 4] > 480 + BUFFER)
        )

        px_diff = np.diff(px, axis=1)
        px_diff = np.concatenate((
            px_diff, 
            np.zeros((px_diff.shape[0], 1, px_diff.shape[-1]))
        ), axis=1)
        px_diff = np.abs(px_diff)
        bad_data_mask = bad_data_mask | np.any(px_diff > 100, axis=2)

    bad_data_mask = np.any(bad_data_mask, axis=1)

    actions = actions[~bad_data_mask]
    front_img_1 = front_img_1[~bad_data_mask]
    ee_pose = ee_pose[~bad_data_mask]

    return np.array(actions), np.array(front_img_1), np.array(ee_pose)

def transform_ee_pose(ee_pose):
    if ee_pose.shape[1] == 3:
        ee_pose[:, 0] *= -1  # Multiply x by -1
        ee_pose[:, 1] *= -1  # Multiply y by -1
    elif ee_pose.shape[1] == 6:
        ee_pose[:, 0] *= -1  # Multiply x by -1 for first set
        ee_pose[:, 1] *= -1  # Multiply y by -1 for first set
        ee_pose[:, 3] *= -1  # Multiply x by -1 for second set
        ee_pose[:, 4] *= -1  # Multiply y by -1 for second set

    return ee_pose


def transform_actions(actions):
    print("Transforming coordinates for actions and ee_pose")

    if actions.shape[1] == 3:
        actions[:, 0] *= -1  # Multiply x by -1
        actions[:, 1] *= -1  # Multiply y by -1
    elif actions.shape[1] == 6:
        actions[:, 0] *= -1  # Multiply x by -1 for first set
        actions[:, 1] *= -1  # Multiply y by -1 for first set
        actions[:, 3] *= -1  # Multiply x by -1 for second set
        actions[:, 4] *= -1  # Multiply y by -1 for second set
    elif actions.shape[1] == 30:
        for i in range(10):
            actions[:, 3 * i] *= -1  # Multiply x by -1 for each set
            actions[:, 3 * i + 1] *= -1  # Multiply y by -1 for each set
    elif actions.shape[1] == 60:
        for i in range(20):
            actions[:, 3 * i] *= -1  # Multiply x by -1 for each set
            actions[:, 3 * i + 1] *= -1  # Multiply y by -1 for each set

    return actions


def get_bounds(binary_image):
    """
    Get the bounding box of the hand mask
    binary_image: np.array of shape (h, w)

    Returns: min_x, max_x, min_y, max_y
    """
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold the grayscale image to create a binary image
    # _,binary_image = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store max and min x and y values
    max_x = max_y = 0
    min_x = min_y = float('inf')

    if len(contours) == 0:
        return None, None, None, None

    # Loop through all contours to find max and min x and y values
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    return min_x, max_x, min_y, max_y

def line_on_hand(images, masks, arm):
    """
    Draw a line on the hand
    images: np.array of shape (n, h, w, c)
    masks: np.array of shape (n, h, w)
    arm: str, "left" or "right"
    """
    overlayed_imgs = np.zeros_like(images)
    for k, (image, mask) in enumerate(zip(images, masks)):
        min_x, max_x, min_y, max_y = get_bounds(mask.astype(np.uint8))
        if min_x is None:
            overlayed_imgs[k] = image
            continue

        gamma = 0.8
        alpha = 0.2
        scale = max_y - min_y
        min_x = int(max_x + gamma * (min_x - max_x))
        min_y = int(max_y + gamma * (min_y - max_y))
        max_x = int(max_x - scale * alpha)

        if arm == "right":
            line_image = cv2.line(image.copy(), (min_x,min_y),(max_x,max_y),color=(255,0,0), thickness=25)
        elif arm == "left":
            line_image = cv2.line(image.copy(), (min_x,max_y),(max_x,min_y),color=(255,0,0), thickness=25)
        else:
            raise ValueError(f"Invalid arm: {arm}")
        overlayed_imgs[k] = line_image
    
    return overlayed_imgs


def sam_processing(dataset, debug=False):
    """
    Applying masking to all images in the dataset

    dataset: path to the hdf5 file
    """
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = SAM()

    with h5py.File(dataset, "r+") as data:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in tqdm(range(len(data["data"].keys()))):
                demo = data[f"data/demo_{i}"]
                imgs = demo["obs/front_img_1"]
                ee_poses = demo["obs/ee_pose"]

                overlayed_imgs, masked_imgs, raw_masks = sam.get_hand_mask_line_batched(imgs, ee_poses, ARIA_INTRINSICS, debug=debug)
                
                if "front_img_1_masked" in demo["obs"]:
                    print("Deleting existing masked images")
                    del demo["obs/front_img_1_masked"]
                if "front_img_1_line" in demo["obs"]:
                    print("Deleting existing line images")
                    del demo["obs/front_img_1_line"]
                if "front_img_1_mask" in demo["obs"]:
                    print("Deleting existing masks")
                    del demo["obs/front_img_1_mask"]

                demo["obs"].create_dataset("front_img_1_masked", data=masked_imgs, chunks=(1, 480, 640, 3))
                demo["obs"].create_dataset("front_img_1_mask", data=raw_masks, chunks=(1, 480, 640), dtype=bool)
                demo["obs"].create_dataset("front_img_1_line", data=overlayed_imgs, chunks=(1, 480, 640, 3))


def main(args):
    filenames = [f for f in os.listdir(args.dataset) if f.endswith(".vrs")]
    mps_paths = [
        os.path.join(args.dataset, "mps_" + filename.split(".")[0] + "_vrs")
        for filename in filenames
    ]

    if args.debug:
        filenames = filenames[0:2]
        mps_paths = mps_paths[0:2]
    
    with h5py.File(args.out, "w") as f:
        if args.hand == "left" or args.hand == "right":
            ac_dim = 3
        elif args.hand == "bimanual":
            ac_dim = 6

        demo_index = 0
        data = f.create_group(f"data")
        data.attrs["env_args"] = json.dumps({})
        print(f"Using {args.hand} data")
        for j, filename in enumerate(filenames):
            print(f"Adding {filename} to hdf5 file")
            actions, front_img_1, ee_pose = single_file_conversion(
                args.dataset, mps_paths[j], filename, args.hand
            )
            actions, ee_pose = transform_actions(actions), transform_ee_pose(ee_pose)
            N = actions.shape[0]
            print(f"{N} frames in vrs file")
            chunk_size = 300  # Define chunk size
            for i in range(0, N, chunk_size):
                # print(i)
                group = data.create_group(f"demo_{demo_index}")
                # group.create_dataset("label", data=np.array([1]))
                # if args.prestack:
                ac_reshape = actions[i : i + chunk_size].reshape(-1, HORIZON, ac_dim)
                group.create_dataset("actions_xyz", data=ac_reshape)

                ac_reshape_interp = interpolate_arr(ac_reshape, 100)
                group.create_dataset("actions_xyz_act", data=ac_reshape_interp)
                    
                # else:
                #     group.create_dataset("actions", data=actions[i : i + chunk_size])
                group.attrs["num_samples"] = group["actions_xyz"].shape[0]
                group.create_dataset(
                    "obs/front_img_1", data=front_img_1[i : i + chunk_size]
                )
                group.create_dataset("obs/ee_pose", data=ee_pose[i : i + chunk_size])
                demo_index += 1
            print(f"Completed adding {filename}")
            # break

    split_train_val_from_hdf5(hdf5_path=args.out, val_ratio=0.2)
    

    ## Apply masking
    if args.mask:
        print("Starting Masking")
        sam_processing(args.out, args.debug)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to folder containing vrs and mps",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="output file path",
    )
    parser.add_argument(
        "--hand",
        type=str,
        help="left; right; bimanual",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="if true, debug runs for two files only. Defaults to False",
    )
    parser.add_argument(
        "--mask", action="store_true"
    )
    # parser.add_argument(
    #     "--prestack", action="store_true", help="if true, stacks actions in Tx3"
    # )

    args = parser.parse_args()

    assert (
        args.hand is not None or not "left" or not "right" or not "bimanual"
    ), "Must provide the correct key (left, right, bimanual)"
    assert args.dataset is not None, "Must provide correct dataset folder"
    assert args.out is not None, "Must provide output file path"

    dataset_path = args.dataset

    main(args)
    