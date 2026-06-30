import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm
from egomimic.utils.egomimicUtils import (
    nds,
    ee_pose_to_cam_frame,
    EXTRINSICS,
    AlohaFK
)
import pytorch_kinematics as pk
import torch

# from modern_robotics import FKinSpace
from robomimic.scripts.split_train_val import split_train_val_from_hdf5
import json

from external.robomimic.robomimic.utils.dataset import interpolate_arr
from egomimic.scripts.masking.utils import *

"""
aloha_hdf5 has the following format
dict with keys:  <KeysViewHDF5 ['action', 'observations']>
action: (500, 14)
observations: dict with keys:  <KeysViewHDF5 ['effort', 'images', 'qpos', 'qvel']>
        effort: (500, 14)
        images: dict with keys:  <KeysViewHDF5 ['cam_high', 'cam_right_wrist']>
                cam_high: (500, 480, 640, 3)
                cam_right_wrist: (500, 480, 640, 3)
        qpos: (500, 14)
        qvel: (500, 14)
"""


def get_future_points(arr, POINT_GAP=15, FUTURE_POINTS_COUNT=10):
    """
    arr: (T, ACTION_DIM)
    POINT_GAP: how many timesteps to skip
    FUTURE_POINTS_COUNT: how many future points to collect
    given an array arr, prepack the future points into each timestep.  return an array of size (T, FUTURE_POINTS_COUNT, ACTION_DIM).  If there are not enough future points, pad with the last point.
    do it purely vectorized
    """
    T, ACTION_DIM = arr.shape
    result = np.zeros((T, FUTURE_POINTS_COUNT, ACTION_DIM))
    
    for t in range(T):
        future_indices = np.arange(t, t + POINT_GAP * (FUTURE_POINTS_COUNT), POINT_GAP)
        future_indices = np.clip(future_indices, 0, T - 1)
        result[t] = arr[future_indices]
    return result


def sample_interval_points(arr, POINT_GAP=15, FUTURE_POINTS_COUNT=10):
    """
    arr: (T, ACTION_DIM)
    POINT_GAP: how many timesteps to skip between points
    FUTURE_POINTS_COUNT: how many future points to collect
    Returns an array of points sampled at intervals of POINT_GAP * FUTURE_POINTS_COUNT.
    """
    num_samples, T, ACTION_DIM = arr.shape
    interval = T / 10
    indices = np.arange(0, T, interval).astype(int)
    sampled_points = arr[:, indices, :]
    return sampled_points


def is_valid_path(path):
    return not os.path.isdir(path) and "episode" in path and ".hdf5" in path


def apply_masking(hdf5_file, arm, extrinsics):
    """
    hdf5_file: path to the hdf5 file to iterate over
    arm: arm to mask - left, right, or both
    extrinsics: which extrinsics to use
    Apply SAM-based masking and overlayed line to images in the hdf5 file.
    """
    print(".........Starting Masking........")
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = SAM()

    with h5py.File(hdf5_file, 'r+') as aloha_hdf5, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        keys_list = list(aloha_hdf5['data'].keys())
        keys_list = [k.split('_')[1] for k in keys_list]
        for j in tqdm(keys_list):
            print(f"Processing episode {j}")
            
            px_dict = sam.project_joint_positions_to_image(torch.from_numpy(aloha_hdf5[f'data/demo_{j}/obs/joint_positions'][:, :]), extrinsics, ARIA_INTRINSICS, arm=arm)

            mask_images, line_images = sam.get_robot_mask_line_batched(
                aloha_hdf5[f'data/demo_{j}/obs/front_img_1'], px_dict, arm=arm)

            if "front_img_1_line" in aloha_hdf5[f'data/demo_{j}/obs']:
                del aloha_hdf5[f'data/demo_{j}/obs/front_img_1_line']
            
            if "front_img_1_masked" in aloha_hdf5[f'data/demo_{j}/obs']:
                del aloha_hdf5[f'data/demo_{j}/obs/front_img_1_masked']

            aloha_hdf5[f'data/demo_{j}/obs'].create_dataset('front_img_1_line', data=line_images, chunks=(1, 480, 640, 3))
            aloha_hdf5[f'data/demo_{j}/obs'].create_dataset('front_img_1_masked', data=mask_images, chunks=(1, 480, 640, 3))


def add_image_obs(demo_hdf5, demo_obs_group, cam_name):
    """
    demo_hdf5: the demo hdf5 file
    demo_obs_group: the demo obs object
    cam_name: the name of the camera to add
    Add an image to the demo hdf5 file.
    """
    if cam_name == "cam_high":
        demo_obs_group.create_dataset(
            "front_img_1",
            data=demo_hdf5["observations"]["images"]["cam_high"],
            dtype="uint8",
            chunks=(1, 480, 640, 3),
        )
    elif cam_name == "cam_left_wrist":
        demo_obs_group.create_dataset(
            "left_wrist_img",
            data=demo_hdf5["observations"]["images"]["cam_left_wrist"],
            dtype="uint8",
            chunks=(1, 480, 640, 3),
        )
    elif cam_name == "cam_right_wrist":
        demo_obs_group.create_dataset(
            "right_wrist_img",
            data=demo_hdf5["observations"]["images"]["cam_right_wrist"],
            dtype="uint8",
            chunks=(1, 480, 640, 3),
        )    

def add_joint_actions(demo_hdf5, demo_i_group, joint_start, joint_end, prestack=False, POINT_GAP=2, FUTURE_POINTS_COUNT=100):
    """
    demo_hdf5: the demo hdf5 file
    demo_i_group: the demo group to write the data to
    joint_start: the start index of the joint actions
    joint_end: the end index of the joint actions
    prestack: whether to prestack the future points
    POINT_GAP: how many timesteps to skip
    FUTURE_POINTS_COUNT: how many future points to collect

    Add joint actions to the demo hdf5 file.
    """
    joint_actions = demo_hdf5["action"][:,  joint_start:joint_end]
    if prestack:
        joint_actions = get_future_points(joint_actions, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
        joint_actions_sampled =  sample_interval_points(joint_actions, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
    demo_i_group.create_dataset(
        "actions_joints", data=joint_actions_sampled
    )
    demo_i_group.create_dataset(
        "actions_joints_act", data=joint_actions
    )
    

def add_xyz_actions(demo_hdf5, demo_i_group, arm, left_extrinsics=None, right_extrinsics=None, prestack=False, POINT_GAP=2, FUTURE_POINTS_COUNT=100):
    """
    demo_hdf5: the demo hdf5 file
    demo_i_group: the demo group to write the data to
    arm: the arm to process
    left_extrinsics: the left camera extrinsics
    right_extrinsics: the right camera extrinsics
    prestack: whether to prestack the future points
    POINT_GAP: how many timesteps to skip
    FUTURE_POINTS_COUNT: how many future points to collect

    Add xyz actions to the demo hdf5 file.
    """
    aloha_fk = AlohaFK()

    if arm == "both":
        joint_start = 0
        joint_end = 14

        #Needed for forward kinematics
        joint_left_start = 0
        joint_left_end = 7
        joint_right_start = 7
        joint_right_end = 14
        
        fk_left_positions = aloha_fk.fk(demo_hdf5["action"][:, joint_left_start:joint_left_end - 1])
        fk_right_positions = aloha_fk.fk(demo_hdf5["action"][:, joint_right_start:joint_right_end - 1])
    else:
        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14
        fk_positions = aloha_fk.fk(demo_hdf5["action"][:, joint_start:joint_end - 1])
    
    if arm == "both":
        fk_left_positions = ee_pose_to_cam_frame(
            fk_left_positions, left_extrinsics
        )[:, :3]
        fk_right_positions = ee_pose_to_cam_frame(
            fk_right_positions, right_extrinsics
        )[:, :3]
        fk_positions = np.concatenate([fk_left_positions, fk_right_positions], axis=1)
    else:
        extrinsics = left_extrinsics if arm == "left" else right_extrinsics         
        fk_positions = ee_pose_to_cam_frame(
            fk_positions, extrinsics
        )[:, :3]

    if prestack:
        print("prestacking", fk_positions.shape)
        fk_positions = get_future_points(fk_positions, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
        print("AFTER prestacking", fk_positions.shape)
        fk_positions_sampled = sample_interval_points(fk_positions, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)

    demo_i_group.create_dataset("actions_xyz_act", data=fk_positions)
    demo_i_group.create_dataset("actions_xyz", data=fk_positions_sampled)

def add_ee_pose_obs(demo_hdf5, demo_i_obs_group, arm, left_extrinsics=None, right_extrinsics=None): 
    """
    demo_hdf5: the demo hdf5 file
    demo_i_obs_group: the demo obs group to write the data to
    arm: the arm to process
    left_extrinsics: the left camera extrinsics
    right_extrinsics: the right camera extrinsics

    Add ee pose obs to the demo hdf5 file.
    """
    aloha_fk = AlohaFK()

    if arm == "both":
        joint_start = 0
        joint_end = 14
        #Needed for forward kinematics
        joint_left_start = 0
        joint_left_end = 7
        joint_right_start = 7
        joint_right_end = 14
        fk_left_positions = aloha_fk.fk(demo_hdf5["observations"]["qpos"][:, joint_left_start:joint_left_end - 1])
        fk_right_positions = aloha_fk.fk(demo_hdf5["observations"]["qpos"][:, joint_right_start:joint_right_end - 1])
    else:
        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14    
        fk_positions = aloha_fk.fk(demo_hdf5["observations"]["qpos"][:, joint_start:joint_end - 1])
    
    if arm == "both":
        fk_left_positions = ee_pose_to_cam_frame(
            fk_left_positions, left_extrinsics
        )[:, :3]
        fk_right_positions = ee_pose_to_cam_frame(
            fk_right_positions, right_extrinsics
        )[:, :3]
        fk_positions = np.concatenate([fk_left_positions, fk_right_positions], axis=1)
    else:
        extrinsics = left_extrinsics if arm == "left" else right_extrinsics   
        fk_positions = ee_pose_to_cam_frame(
            fk_positions, extrinsics
        )[:, :3]

    demo_i_obs_group.create_dataset("ee_pose", data=fk_positions)

def process_demo(demo_path, data_group, arm, extrinsics, prestack=False):
    """
    demo_path: path to the demo hdf5 file
    data_group: the group in the output hdf5 file to write the data to
    arm: arm to process - left, right, or both
    extrinsics: camera extrinsics. It is a tuple of (left_extrinsics, right_extrinsics) if arm is both
    prestack: whether to prestack the future points
    Process a single demo hdf5 file and write the data to the output hdf5 file.
    """

    left_extrinsics = None
    right_extrinsics = None
    if arm == "both":
        if not isinstance(extrinsics, dict):
            print("Error: Both arms selected. Expected extrinsics for both arms.")
        left_extrinsics = extrinsics["left"]
        right_extrinsics = extrinsics["right"]
    elif args.arm == "left":
        extrinsics = extrinsics["left"]
        left_extrinsics = extrinsics
    elif args.arm == "right":
        extrinsics = extrinsics["right"]
        right_extrinsics = extrinsics
    with h5py.File(demo_path, "r") as demo_hdf5:
        demo_number = demo_path.split("_")[-1].split(".")[0]
        demo_i_group = data_group.create_group(f"demo_{demo_number}")
        demo_i_group.attrs["num_samples"] = demo_hdf5["action"].shape[0]
        demo_i_obs_group = demo_i_group.create_group("obs")

        # Extract the data from the aloha hdf5 file
        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14
        elif arm == "both":
            joint_start = 0
            joint_end = 14

            #Needed for forward kinematics
            joint_left_start = 0
            joint_left_end = 7
            joint_right_start = 7
            joint_right_end = 14

        # obs
        ## adding images
        add_image_obs(demo_hdf5, demo_i_obs_group, "cam_high")
        if arm in ["left", "both"]:
            add_image_obs(demo_hdf5, demo_i_obs_group, "cam_left_wrist")
        if arm in ["right", "both"]:
            add_image_obs(demo_hdf5, demo_i_obs_group, "cam_right_wrist")
        
        ## add joint obs
        demo_i_obs_group.create_dataset(
            "joint_positions", data=demo_hdf5["observations"]["qpos"][:, joint_start:joint_end]
        )

        # add ee_pose
        add_ee_pose_obs(demo_hdf5, demo_i_obs_group, arm, left_extrinsics=left_extrinsics, right_extrinsics=right_extrinsics)

        POINT_GAP = 2
        FUTURE_POINTS_COUNT = 100

        # add joint actions
        add_joint_actions(demo_hdf5, demo_i_group, joint_start, joint_end, prestack=prestack, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)

        # actions_xyz
        add_xyz_actions(demo_hdf5, demo_i_group, arm, left_extrinsics, right_extrinsics, prestack=prestack, POINT_GAP=POINT_GAP, FUTURE_POINTS_COUNT=FUTURE_POINTS_COUNT)
   
def  main(args):
    # before converting everything, check it all at least opens
    for file in tqdm(os.listdir(args.dataset)):
        #  if os.path.isfile(os.path.join(args.dataset, file)):
        #     print(file.split("_")[1].split(".")[0])
        #     if int(file.split("_")[1].split(".")[0]) <= 5:
        print("Trying to open " + file)
        to_open = os.path.join(args.dataset, file)
        print(to_open)
        if is_valid_path(to_open):
            with h5py.File(to_open, "r") as f:
                pass

    with h5py.File(args.out, "w", rdcc_nbytes=1024**2 * 2) as dataset:
        data_group = dataset.create_group("data")
        data_group.attrs["env_args"] = json.dumps({})  # if no normalize obs

        for i, aloha_demo in enumerate(tqdm(os.listdir(args.dataset))):
            if not is_valid_path(os.path.join(args.dataset, aloha_demo)):
                continue

            aloha_demo_path = os.path.join(args.dataset, aloha_demo)

            process_demo(aloha_demo_path, data_group, args.arm, EXTRINSICS[args.extrinsics], args.prestack)

    split_train_val_from_hdf5(hdf5_path=args.out, val_ratio=0.2, filter_key=None)

    ## Masking
    if args.mask:
        print("Starting Masking")
        apply_masking(args.out, args.arm, EXTRINSICS[args.extrinsics])
    print("Successful Conversion!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to rawAloha folder",
    )
    parser.add_argument("--arm", type=str, help="which arm to convert data for")
    parser.add_argument("--extrinsics", type=str, help="which arm to convert data for")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        help="path to output dataset: /coc/flash7/datasets/oboov2/<ds_name>.hdf5",
    )

    parser.add_argument(
        "--prestack",
        action="store_true"
    )

    args = parser.parse_args()

    main(args)