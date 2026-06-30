import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2
from egomimic.utils.egomimicUtils import nds, ee_pose_to_cam_frame, EXTRINSICS, ARIA_INTRINSICS, cam_frame_to_cam_pixels
# Example call: in aloha_process folder: python aloha_to_robomimic.py --dataset /coc/flash7/skareer6/calibrate_samples/ --arm left --out /coc/flash7/skareer6/calibrate_samples/ --task-name robomimic
import pytorch_kinematics as pk
import torch
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import *

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
    future_traj = []

    for i in range(POINT_GAP, (FUTURE_POINTS_COUNT + 1) * POINT_GAP, POINT_GAP):
        # Identify the indices for the current and prior points
        index_current = min(len(arr) - 1, i)

        current_point = arr[index_current]
        future_traj.extend(current_point)

    return future_traj

def is_valid_path(path):
    return not os.path.isdir(path) and "episode" in path

def plot_points_on_image(image, points, save_path):
    """
    Plots an array of points on an image and saves the result.

    Parameters:
    image (ndarray): The image on which to plot the points.
    points (ndarray): An array of points with shape (n, 2), where n is the number of points.
    save_path (str): The file path to save the resulting image.
    """
    plt.imshow(image)  # Display the image

    # Plot each point in the array
    for point in points:
        plt.plot(point[0], point[1], 'ro')  # 'ro' means red circle marker

    plt.axis('off')  # Optionally turn off the axis
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save the image
    plt.close()  # Close the plot to free memory


def main(args):
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = SAM()

    arm = args.arm
    chain = pk.build_serial_chain_from_urdf(open("/coc/flash9/skareer6/Projects/EgoPlay/EgoPlay/egomimic/resources/model.urdf").read(), "vx300s/ee_gripper_link")

    with h5py.File(args.dataset, 'r+') as aloha_hdf5, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        keys_list = list(aloha_hdf5['data'].keys())
        keys_list = [k.split('_')[1] for k in keys_list]
        for j in tqdm(keys_list):
        # for j in tqdm(range(0, 1)):
            print(f"Processing episode {j}")
            print(torch.from_numpy(aloha_hdf5[f'data/demo_{j}/obs/joint_positions'][:,:-1]).shape)
            
            px_dict = sam.project_joint_positions_to_image(torch.from_numpy(aloha_hdf5[f'data/demo_{j}/obs/joint_positions'][:, :]), EXTRINSICS[args.extrinsics], ARIA_INTRINSICS, arm=arm)

            mask_images, line_images = sam.get_robot_mask_line_batched(
                aloha_hdf5[f'data/demo_{j}/obs/front_img_1'], px_dict, arm=arm)

            if "front_img_1_line" in aloha_hdf5[f'data/demo_{j}/obs']:
                del aloha_hdf5[f'data/demo_{j}/obs/front_img_1_line']
            
            if "front_img_1_masked" in aloha_hdf5[f'data/demo_{j}/obs']:
                del aloha_hdf5[f'data/demo_{j}/obs/front_img_1_masked']

            aloha_hdf5[f'data/demo_{j}/obs'].create_dataset('front_img_1_line', data=line_images, chunks=(1, 480, 640, 3))
            aloha_hdf5[f'data/demo_{j}/obs'].create_dataset('front_img_1_masked', data=mask_images, chunks=(1, 480, 640, 3))
    
    print("Successful Conversion!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to source folder",
    )
    parser.add_argument(
        "--arm",
        type=str,
        help="which arm to convert data for"
    )
    parser.add_argument(
        "--extrinsics",
        type=str,
        help="which arm to convert data for"
    )
    args = parser.parse_args()

    main(args)
