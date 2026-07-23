"""Robot episode dataset for Combined ACT (4 cameras, 20D EEF pose + 14D joints)."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Combined.config import (
    DEFAULT_NUM_QUERIES,
    EMBODIMENT_ROBOT,
    POSE_DIM,
    ROBOT_EEF_GRIPPER_BINARIZE_THRESHOLD,
    ROBOT_JOINT_DIM,
    ROBOT_SYNC_INDEX_COLUMNS,
    ROBOT_TEMP_CUT_INDEX_COLUMNS,
    binarize_flat_pose_grippers,
    camera_mask_tensor,
    concat_bimanual_joints,
    flatten_bimanual_pose,
    stack_camera_tensors,
)
from Combined.data_synchronization import xyz_gripper_valid_mask
from Combined.dataloader_utils import (
    build_image_transform,
    demo_id_from_hash_filename,
    demo_id_from_joint_npy,
    demo_id_from_robot_eef_npz,
    index_paths_by_demo_id,
    load_video_frames,
    normalize_future_chunk,
    resolve_path,
    zero_rgb_like,
)


class RobotEpisodeDataset(Dataset):
    """
    One item = one robot episode (random start inside episode).

    Action-chunk convention:
      pose_actions[k] / joint_actions[k] = values at absolute timestep t+k,
      where t is the observation index. Thus actions[0] matches current state.

    Returns dict (common batch schema):
      embodiment, images, camera_mask,
      pose_state [8], pose_actions [K,8]  (xyz+gripper; rot dropped),
      joint_state [14], joint_actions [K,14],
      has_joint_target True, is_pad [K]
    """

    def __init__(
        self,
        *,
        bird_vids_dir,
        front_vids_dir,
        left_arm_vids_dir,
        right_arm_vids_dir,
        left_joint_data_dir,
        right_joint_data_dir,
        eef_pose_data_dir,
        sync_csv_dir,
        num_queries: int = DEFAULT_NUM_QUERIES,
        transform: str = "resnet_normalization",
        max_demos: int | None = None,
        temp_cut: int = 10,
        resize_factor: float = 1.0,
        max_sync_rows: int | None = None,
        require_valid_eef: bool = True,
        disable_front_camera: bool = False,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.temp_cut = int(temp_cut)
        self.resize_factor = float(resize_factor)
        self.max_sync_rows = int(max_sync_rows) if max_sync_rows is not None else None
        self.require_valid_eef = bool(require_valid_eef)
        self.disable_front_camera = bool(disable_front_camera)
        self.image_transform = build_image_transform(transform)

        bird_vids = sorted(resolve_path(bird_vids_dir).glob("*.mp4"))
        front_vids = sorted(resolve_path(front_vids_dir).glob("*.mp4"))
        left_vids = sorted(resolve_path(left_arm_vids_dir).glob("*.mp4"))
        right_vids = sorted(resolve_path(right_arm_vids_dir).glob("*.mp4"))
        left_joints = sorted(resolve_path(left_joint_data_dir).glob("*.npy"))
        right_joints = sorted(resolve_path(right_joint_data_dir).glob("*.npy"))
        eef_files = sorted(resolve_path(eef_pose_data_dir).glob("*.npz"))
        sync_csvs = sorted(resolve_path(sync_csv_dir).glob("*.csv"))
        if max_demos is not None and max_demos > 0:
            sync_csvs = sync_csvs[:max_demos]
        if not sync_csvs:
            raise FileNotFoundError(f"No robot sync CSVs in {sync_csv_dir}")

        bird_by = index_paths_by_demo_id(bird_vids, demo_id_from_hash_filename)
        front_by = index_paths_by_demo_id(front_vids, demo_id_from_hash_filename)
        left_by = index_paths_by_demo_id(left_vids, demo_id_from_hash_filename)
        right_by = index_paths_by_demo_id(right_vids, demo_id_from_hash_filename)
        left_j_by = index_paths_by_demo_id(left_joints, demo_id_from_joint_npy)
        right_j_by = index_paths_by_demo_id(right_joints, demo_id_from_joint_npy)
        eef_by = index_paths_by_demo_id(eef_files, demo_id_from_robot_eef_npz)
        if self.disable_front_camera:
            print("Robot dataset: --no_front_camera → front images zeroed + masked out")

        self.bird_frames: list[np.ndarray] = []
        self.front_frames: list[np.ndarray] = []
        self.left_frames: list[np.ndarray] = []
        self.right_frames: list[np.ndarray] = []
        self.joint_data: list[torch.Tensor] = []
        self.eef_pose_data: list[torch.Tensor] = []
        self.sample_demo_idx: List[int] = []
        self.demo_start_idx: List[int] = []
        self.demo_lengths: List[int] = []
        self.num_demos = 0
        self.num_samples = 0

        print("Loading robot Combined demos...")
        for csv_path in sync_csvs:
            rec_id = csv_path.stem
            needed = {
                "bird": bird_by,
                "left": left_by,
                "right": right_by,
                "left_joint": left_j_by,
                "right_joint": right_j_by,
                "eef_pose": eef_by,
            }
            if not self.disable_front_camera:
                needed["front"] = front_by
            missing = [k for k, m in needed.items() if rec_id not in m]
            if missing:
                print(f"WARNING: skip robot {rec_id} missing {missing}")
                continue

            df = pd.read_csv(csv_path)
            if not set(ROBOT_SYNC_INDEX_COLUMNS).issubset(df.columns):
                raise KeyError(f"{csv_path} missing robot sync columns {ROBOT_SYNC_INDEX_COLUMNS}")

            eef_npz = np.load(eef_by[rec_id])
            if "pose" not in eef_npz.files or "timestamps" not in eef_npz.files:
                raise KeyError(f"{eef_by[rec_id].name} must contain 'pose' and 'timestamps'")
            eef_arr = np.asarray(eef_npz["pose"], dtype=np.float32)
            if eef_arr.ndim != 3 or eef_arr.shape[1:] != (2, 10):
                raise ValueError(f"Expected EEF pose [T,2,10] for {rec_id}, got {eef_arr.shape}")

            valid_pos = np.asarray(eef_npz["valid_pos"], dtype=bool) if "valid_pos" in eef_npz.files else None
            valid_open = np.asarray(eef_npz["valid_open"], dtype=bool) if "valid_open" in eef_npz.files else None
            frame_ok = None
            if self.require_valid_eef:
                if valid_pos is None or valid_open is None:
                    missing_keys = [
                        k for k, v in (("valid_pos", valid_pos), ("valid_open", valid_open)) if v is None
                    ]
                    raise KeyError(f"{eef_by[rec_id].name} missing required validity keys: {missing_keys}")
                # Orientation / valid_rot ignored; pose uses xyz+gripper only.
                frame_ok = xyz_gripper_valid_mask(
                    valid_pos=valid_pos,
                    valid_open=valid_open,
                    n_frames=len(eef_arr),
                    required_slots=(0, 1),
                )

            # temp_cut only on video/joint indices — NOT on eef_pose_index (original NPZ timeline)
            mask = np.ones(len(df), dtype=bool)
            for col in ROBOT_TEMP_CUT_INDEX_COLUMNS:
                mask &= df[col].to_numpy() >= self.temp_cut
            df = df[mask].reset_index(drop=True)
            for col in ROBOT_TEMP_CUT_INDEX_COLUMNS:
                df[col] = df[col] - self.temp_cut
            if df.empty:
                continue

            if frame_ok is not None:
                keep = []
                for i in range(len(df)):
                    eidx = int(df.loc[i, "eef_pose_index"])
                    if eidx < 0 or eidx >= len(frame_ok):
                        keep.append(False)
                        continue
                    keep.append(bool(frame_ok[eidx]))
                df = df[np.asarray(keep, dtype=bool)].reset_index(drop=True)
            if df.empty:
                print(f"WARNING: skip robot {rec_id} - no valid EEF rows after filters")
                continue
            if self.max_sync_rows is not None and len(df) > self.max_sync_rows:
                df = df.iloc[: self.max_sync_rows].reset_index(drop=True)

            bird_f = load_video_frames(bird_by[rec_id], resize_factor=self.resize_factor, label=f"bird({rec_id})")[
                self.temp_cut :
            ]
            front_f = None
            if not self.disable_front_camera:
                front_f = load_video_frames(
                    front_by[rec_id], resize_factor=self.resize_factor, label=f"front({rec_id})"
                )[self.temp_cut :]
            left_f = load_video_frames(left_by[rec_id], resize_factor=self.resize_factor, label=f"left({rec_id})")[
                self.temp_cut :
            ]
            right_f = load_video_frames(right_by[rec_id], resize_factor=self.resize_factor, label=f"right({rec_id})")[
                self.temp_cut :
            ]
            left_j = np.load(left_j_by[rec_id]).astype(np.float32)[self.temp_cut :]
            right_j = np.load(right_j_by[rec_id]).astype(np.float32)[self.temp_cut :]

            demo_idx = self.num_demos
            self.demo_start_idx.append(len(self.joint_data))
            n_i = len(df)
            self.demo_lengths.append(n_i)

            for i in range(n_i):
                self.bird_frames.append(bird_f[int(df.loc[i, "bird_index"])])
                if front_f is not None:
                    self.front_frames.append(front_f[int(df.loc[i, "front_index"])])
                self.left_frames.append(left_f[int(df.loc[i, "left_index"])])
                self.right_frames.append(right_f[int(df.loc[i, "right_index"])])
                self.joint_data.append(
                    concat_bimanual_joints(
                        left_j[int(df.loc[i, "left_joint_index"])],
                        right_j[int(df.loc[i, "right_joint_index"])],
                        rec_id=rec_id,
                    )
                )
                eidx = int(df.loc[i, "eef_pose_index"])
                flat = flatten_bimanual_pose(eef_arr[eidx], rec_id=rec_id)
                # Binarize EEF gripper dims only (joint grippers untouched).
                flat = binarize_flat_pose_grippers(
                    flat, threshold=ROBOT_EEF_GRIPPER_BINARIZE_THRESHOLD
                )
                self.eef_pose_data.append(flat)
                self.sample_demo_idx.append(demo_idx)

            self.num_samples += n_i
            self.num_demos += 1
            print(f"    -> robot {rec_id}: {n_i} samples")

        if self.num_demos == 0:
            raise FileNotFoundError("No complete robot demos found for Combined.")

        all_q = torch.stack(self.joint_data, dim=0)  # [N, 14]
        all_e = torch.stack(self.eef_pose_data, dim=0)  # [N, 20]
        self.joint_mean = all_q.mean(dim=0)
        self.joint_std = all_q.std(dim=0).clamp(min=1e-2)
        self.eef_mean = all_e.mean(dim=0)
        self.eef_std = all_e.std(dim=0).clamp(min=1e-2)
        # Backward-compatible aliases
        self.state_mean = self.joint_mean
        self.state_std = self.joint_std
        print(f"Robot dataset ready: demos={self.num_demos} samples={self.num_samples}")

    def __len__(self) -> int:
        return self.num_demos

    def __getitem__(self, idx: int) -> dict:
        episode_idx = int(idx)
        ep_start = self.demo_start_idx[episode_idx]
        ep_len = self.demo_lengths[episode_idx]
        start_in_ep = int(np.random.randint(0, ep_len))
        sample_idx = ep_start + start_in_ep
        demo_end = ep_start + ep_len

        bird_t = self.image_transform(self.bird_frames[sample_idx])
        if self.disable_front_camera:
            front_t = self.image_transform(zero_rgb_like(self.bird_frames[sample_idx]))
        else:
            front_t = self.image_transform(self.front_frames[sample_idx])
        images = stack_camera_tensors(
            bird_t,
            front_t,
            self.image_transform(self.left_frames[sample_idx]),
            self.image_transform(self.right_frames[sample_idx]),
        )

        pose_raw = self.eef_pose_data[sample_idx]  # [20]
        joint_raw = self.joint_data[sample_idx]  # [14]
        pose_state = (pose_raw - self.eef_mean) / self.eef_std
        joint_state = (joint_raw - self.joint_mean) / self.joint_std

        # actions[0] at current timestep t; actions[k] at t+k
        slice_end = min(demo_end, sample_idx + self.num_queries)
        pose_future = list(self.eef_pose_data[sample_idx:slice_end])
        joint_future = list(self.joint_data[sample_idx:slice_end])
        if len(pose_future) != len(joint_future):
            raise RuntimeError("Pose/joint future chunk length mismatch")

        pose_actions, is_pad = normalize_future_chunk(
            pose_future, mean=self.eef_mean, std=self.eef_std, num_queries=self.num_queries
        )
        joint_actions, is_pad_j = normalize_future_chunk(
            joint_future, mean=self.joint_mean, std=self.joint_std, num_queries=self.num_queries
        )
        if not torch.equal(is_pad, is_pad_j):
            raise RuntimeError("Pose/joint pad masks diverged")

        return {
            "embodiment": EMBODIMENT_ROBOT,
            "images": images,
            "camera_mask": camera_mask_tensor(
                EMBODIMENT_ROBOT, disable_front=self.disable_front_camera
            ),
            "pose_state": pose_state,
            "pose_actions": pose_actions,
            "joint_state": joint_state,
            "joint_actions": joint_actions,
            "has_joint_target": True,
            "is_pad": is_pad,
        }
