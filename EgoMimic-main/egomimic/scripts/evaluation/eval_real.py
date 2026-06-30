"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    bddl_file (str): if provided, the task's goal is specified as the symbolic goal in the bddl file (several symbolic predicates connected with AND / OR)

    video_prompt (str): if provided, a task video prompt is loaded and used in the evaluation rollouts

    debug (bool): set this flag to run a quick training run for debugging purposes
"""

import argparse
import numpy as np
import time
import os

import torch
import robomimic.utils.obs_utils as ObsUtils
from torchvision.utils import save_image
import cv2
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

from aloha.constants import DT, FOLLOWER_GRIPPER_JOINT_OPEN, START_ARM_POSE


from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    general_unnorm,
    miniviewer,
    nds,
    ARIA_INTRINSICS,
    EXTRINSICS,
    ee_pose_to_cam_frame,
    AlohaFK,
)
import torchvision


from egomimic.configs import config_factory
from egomimic.pl_utils.pl_model import ModelWrapper
import datetime

from aloha.robot_utils import move_grippers, move_arms  # requires aloha
from aloha.real_env import make_real_env  # requires aloha

from egomimic.scripts.evaluation.real_utils import *
import matplotlib.pyplot as plt
from egomimic.algo.act import ACT
from egomimic.scripts.masking.utils import SAM

import pickle


# For debugging
# sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


CURR_INTRINSICS = ARIA_INTRINSICS
CURR_EXTRINSICS = EXTRINSICS["ariaJul29R"]
# NORM_STATS = to_torch(NORM_STATS, torch.device("cuda"))
CAM_KEY = "front_img_1"
TEMPORAL_AGG = False


class TemporalAgg:
    def __init__(self):
        self.recent_actions = []
    
    def add_action(self, action):
        """
            actions: (100, 7) tensor
        """
        self.recent_actions.append(action)
        if len(self.recent_actions) > 4:
            del self.recent_actions[0]

    def smoothed_action(self):
        """
            returns smooth action (100, 7)
        """
        mask = []
        count = 0

        shifted_actions = []
        # breakpoint()

        for ac in self.recent_actions[::-1]:
            basic_mask = np.zeros(100)
            basic_mask[:100-count] = 1
            mask.append(basic_mask)
            shifted_ac = ac[count:]
            shifted_ac = np.concatenate([shifted_ac, np.zeros((count, 7))], axis=0)
            shifted_actions.append(shifted_ac)
            count += 25

        mask = mask[::-1]
        mask = ~(np.array(mask).astype(bool))
        recent_actions = shifted_actions[::-1]
        recent_actions = np.array(recent_actions)
        # breakpoint()
        mask = np.repeat(mask[:, :, None], 7, axis=2)
        smoothed_action = np.ma.array(recent_actions, mask=mask).mean(axis=0)

        # PLOT_JOINT = 0
        # for i in range(recent_actions.shape[0]):
        #     plt.plot(recent_actions[i, :, PLOT_JOINT], label=f"index{i}")
        # plt.plot(smoothed_action[:, PLOT_JOINT], label="smooth")
        # plt.legend()
        # plt.savefig("smoothing.png")
        # plt.close()
        # breakpoint()

        return smoothed_action

def eval_real(model, env, rollout_dir, norm_stats, arm="right"):
    device = torch.device("cuda")

    aloha_fk = AlohaFK()
    sam = SAM()

    query_frequency = 25


    # max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks
    qpos_t, actions_t = [], []
    num_rollouts = 50
    for rollout_id in range(num_rollouts):
        if TEMPORAL_AGG:
            TA = TemporalAgg()

        ts = env.reset()

        t0 = time.time()
        with torch.inference_mode():
            rollout_images = []
            for t in range(1000):
                time.sleep(max(0, DT*2 - (time.time() - t0)))
                # print(f"DT: {time.time() - t0}")
                t0 = time.time()

                obs = ts.observation
                # plt.imsave(os.path.join(rollout_dir, f"viz{t}.png"), obs["images"]["cam_high"])
                # plt.imsave(os.path.join(rollout_dir, f"wrist{t}.png"), obs["images"]["cam_right_wrist"])

                qpos = np.array(obs["qpos"])
                qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
                inference_t = time.time()


                ### query policy
                if t % query_frequency == 0:

                    # right wrist data
                    data = {
                        "obs": {
                            "right_wrist_img": (
                                torch.from_numpy(obs["images"]["cam_right_wrist"][None, None, :])
                            ).to(torch.uint8),
                            "pad_mask": torch.ones((1, 100, 1)).to(device).bool(),
                            "joint_positions": qpos[..., 7:].reshape((1, 1, -1)),
                        },
                        "type": torch.tensor([0]),
                    }

                    # add regular or line overlay top camera
                    if CAM_KEY == "front_img_1":
                        data["obs"][CAM_KEY] = torch.from_numpy(
                            obs["images"]["cam_high"][None, None, :]
                        ).to(torch.uint8)

                    if arm == "right":
                        data["obs"]["joint_positions"] =  qpos[..., 7:].reshape((1, 1, -1))
                        
                        if CAM_KEY == "front_img_1_line":
                            _, line_image = sam.get_robot_mask_line_batched_from_qpos(obs["images"]["cam_high"][None, :], qpos, EXTRINSICS["ariaJul29"], ARIA_INTRINSICS, arm=arm)
                            line_image = line_image[0]
                            data["obs"][CAM_KEY] = torch.from_numpy(
                                line_image[None, None, :]
                            ).to(torch.uint8)

                        # postprocess_batch
                        input_batch = model.process_batch_for_training(
                            data, "actions_joints_act"
                        )

                        input_batch["obs"]["right_wrist_img"] = input_batch["obs"]["right_wrist_img"].permute(0, 3, 1, 2)
                        input_batch["obs"]["right_wrist_img"] /= 255.0
                    
                    elif arm == "both":
                        data["obs"]["left_wrist_img"] = torch.from_numpy(obs["images"]["cam_left_wrist"][None, None, :]).to(torch.uint8)
                        data["obs"]["joint_positions"] = qpos[..., :].reshape((1, 1, -1))


                        if CAM_KEY == "front_img_1_line":
                            _, line_image = sam.get_robot_mask_line_batched_from_qpos(obs["images"]["cam_high"][None, :], qpos, EXTRINSICS["ariaJul29"], ARIA_INTRINSICS, arm=arm)
                            line_image = line_image[0]
                            data["obs"][CAM_KEY] = torch.from_numpy(
                                line_image[None, None, :]
                            ).to(torch.uint8)

                        # postprocess_batch
                        input_batch = model.process_batch_for_training(
                            data, "actions_joints_act"
                        )

                        # right
                        input_batch["obs"]["right_wrist_img"] = input_batch["obs"]["right_wrist_img"].permute(0, 3, 1, 2)/255.0

                        # left
                        input_batch["obs"]["left_wrist_img"] = input_batch["obs"]["left_wrist_img"].permute(0, 3, 1, 2)/255.0

                    # breakpoint()
                    input_batch["obs"][CAM_KEY] = input_batch["obs"][CAM_KEY].permute(0, 3, 1, 2)
                    input_batch["obs"][CAM_KEY] /= 255.0
                    input_batch = ObsUtils.normalize_batch(input_batch, normalization_stats=norm_stats, normalize_actions=False)
                    info = model.forward_eval(input_batch, unnorm_stats=norm_stats)

                    all_actions = info["actions_joints_act"].cpu().numpy()

                    if TEMPORAL_AGG:
                        TA.add_action(all_actions[0])
                        all_actions = TA.smoothed_action()[None, :]


                    if rollout_dir:
                        # Draw Actions
                        im = data["obs"][CAM_KEY][0, 0].cpu().numpy()
                        pred_values = info["actions_joints_act"][0].cpu().numpy()

                        if "joints" in model.ac_key:
                            pred_values_drawable = aloha_fk.fk(pred_values[:, :6])
                            pred_values_drawable = ee_pose_to_cam_frame(pred_values_drawable, CURR_EXTRINSICS)
                        else:
                            pred_values_drawable = pred_values


                        pred_values_drawable = cam_frame_to_cam_pixels(
                            pred_values_drawable, CURR_INTRINSICS
                        )

                        im = np.array(im, dtype="uint8")
                        frame = draw_dot_on_frame(
                            im, pred_values_drawable[[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]], show=False, palette="Greens"
                        )


                        # Draw ee_pose
                        ee_pose_input = aloha_fk.fk(qpos[:, 7:13]).to(device)
                        ee_pose_cam_frame = ee_pose_to_cam_frame(
                            ee_pose_input.cpu().numpy(), CURR_EXTRINSICS
                        )[:, None, :]
                        ee_pose_pixels = cam_frame_to_cam_pixels(
                            ee_pose_cam_frame[0], CURR_INTRINSICS
                        )
                        frame = draw_dot_on_frame(
                            frame, ee_pose_pixels, show=False, palette="Set1"
                        )


                        # Save images
                        rollout_images.append(frame)
                        plt.imsave(os.path.join(rollout_dir, f"viz{t}.png"), frame)
                        plt.imsave(os.path.join(rollout_dir, f"wrist_rgb{t}.png"), data["obs"]["right_wrist_img"][0, 0].cpu().numpy())
                    
                    print(f"Inference time: {time.time() - inference_t}")

                raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action[0]
                # action = post_process(raw_action)
                target_qpos = raw_action
                # target_qpos = action

                ### step the environment
                if arm == "right":
                    target_qpos = np.concatenate([np.zeros(7), target_qpos])
                
                ts = env.step(target_qpos)

                # debugging control loop
                qpos_t.append(ts.observation["qpos"])
                actions_t.append(target_qpos)

        if rollout_dir:
            qpos_t = np.array(qpos_t)
            actions_t = np.array(actions_t)
            for i in range(7, 14):
                plt.plot(qpos_t[:, i], label=f"qpos joint {i}")
                plt.plot(actions_t[:, i], label=f"ac joint {i}")
                plt.legend()

                plt.savefig(f"/home/rl2-bonjour/EgoPlay/EgoPlay/debug_ims/joint{i}_actions.png", dpi=300)
                plt.close()

                plt.plot(actions_t[:, i] - qpos_t[:, i], label="error joint{i}")
                plt.legend()

                plt.savefig(f"/home/rl2-bonjour/EgoPlay/EgoPlay/debug_ims/joint{i}_error.png", dpi=300)
                plt.close()


        # save_images(rollout_images, viz_dir)
        # write_vid(rollout_images, os.path.join(viz_dir, "video_0.mp4"))
        rollout_images = []

        print("moving robot")
        if arm == "right":
            move_grippers(
                [env.follower_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN], moving_time=0.5
            )  # open
            move_arms([env.follower_bot_right], [START_ARM_POSE[:6]], moving_time=1.0)
        elif arm == "both":
            move_grippers(
                [env.follower_bot_left, env.follower_bot_right], [FOLLOWER_GRIPPER_JOINT_OPEN]*2, moving_time=0.5
            )  # open
            move_arms([env.follower_bot_left, env.follower_bot_right], [START_ARM_POSE[:6]]*2, moving_time=1.0)

        time.sleep(12.0)
    return


def main(args):
    """
    Train a model using the algorithm.
    """
    # first set seeds
    np.random.seed(101)
    torch.manual_seed(101)

    # print("\n============= New Training Run with Config =============")
    # print(config)
    # print("")
    # log_dir, ckpt_dir, video_dir, uid = get_exp_dir(config)

    # breakpoint()
    model = ModelWrapper.load_from_checkpoint(args.eval_path, datamodule=None)
    norm_stats = os.path.join(os.path.dirname(os.path.dirname(args.eval_path)), "ds1_norm_stats.pkl")
    norm_stats = open(norm_stats, "rb")
    norm_stats = pickle.load(norm_stats)
    
    node = create_interbotix_global_node('aloha')
    arm = "right"
    if model.model.ac_dim == 14:
        arm = "both"
        env = make_real_env(node, active_arms="both", setup_robots=True)
    elif model.model.ac_dim == 7:
        arm = "right"
        env = make_real_env(node, active_arms="right", setup_robots=True)
    robot_startup(node)
    model.eval()
    rollout_dir = os.path.dirname(os.path.dirname(args.eval_path))
    rollout_dir = os.path.join(rollout_dir, "rollouts")
    if not os.path.exists(rollout_dir):
        os.mkdir(rollout_dir)

    if not args.debug:
        rollout_dir = None

    eval_real(model.model, env, rollout_dir, norm_stats, arm=arm)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="(optional) path to the model to be evaluated",
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    args = parser.parse_args()
    # if "DT" not in args.description:
    #     time_str = f"{args.description}_DT_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
    #     args.description = time_str
    main(args)
