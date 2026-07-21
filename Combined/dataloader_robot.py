"""Robot episode dataset for Combined ACT (4 cameras, 14D qpos)."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Combined.config import (
    DEFAULT_NUM_QUERIES,
    EMBODIMENT_ROBOT,
    ROBOT_CAMERA_MASK,
    ROBOT_STATE_DIM,
    ROBOT_SYNC_INDEX_COLUMNS,
    camera_mask_tensor,
    concat_bimanual_joints,
    stack_camera_tensors,
)
from Combined.dataloader_utils import (
    build_image_transform,
    demo_id_from_hash_filename,
    demo_id_from_joint_npy,
    index_paths_by_demo_id,
    load_video_frames,
    resolve_path,
)


class RobotEpisodeDataset(Dataset):
    """
    One item = one robot episode (random start inside episode).

    Returns dict:
      embodiment: int (0)
      images:     [4, 3, H, W]  order [bird, front, left_wrist, right_wrist]
      camera_mask:[4]
      state:      [14]
      actions:    [K, 14]
      is_pad:     [K]
      action_dim: 14
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
        sync_csv_dir,
        num_queries: int = DEFAULT_NUM_QUERIES,
        transform: str = "resnet_normalization",
        max_demos: int | None = None,
        temp_cut: int = 10,
        resize_factor: float = 1.0,
        max_sync_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.temp_cut = int(temp_cut)
        self.resize_factor = float(resize_factor)
        self.max_sync_rows = int(max_sync_rows) if max_sync_rows is not None else None
        self.image_transform = build_image_transform(transform)

        bird_vids = sorted(resolve_path(bird_vids_dir).glob("*.mp4"))
        front_vids = sorted(resolve_path(front_vids_dir).glob("*.mp4"))
        left_vids = sorted(resolve_path(left_arm_vids_dir).glob("*.mp4"))
        right_vids = sorted(resolve_path(right_arm_vids_dir).glob("*.mp4"))
        left_joints = sorted(resolve_path(left_joint_data_dir).glob("*.npy"))
        right_joints = sorted(resolve_path(right_joint_data_dir).glob("*.npy"))
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

        self.bird_frames: list[np.ndarray] = []
        self.front_frames: list[np.ndarray] = []
        self.left_frames: list[np.ndarray] = []
        self.right_frames: list[np.ndarray] = []
        self.joint_data: list[torch.Tensor] = []
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
                "front": front_by,
                "left": left_by,
                "right": right_by,
                "left_joint": left_j_by,
                "right_joint": right_j_by,
            }
            missing = [k for k, m in needed.items() if rec_id not in m]
            if missing:
                print(f"WARNING: skip robot {rec_id} missing {missing}")
                continue

            df = pd.read_csv(csv_path)
            if not set(ROBOT_SYNC_INDEX_COLUMNS).issubset(df.columns):
                raise KeyError(f"{csv_path} missing robot sync columns")
            mask = np.ones(len(df), dtype=bool)
            for col in ROBOT_SYNC_INDEX_COLUMNS:
                mask &= df[col].to_numpy() >= self.temp_cut
            df = df[mask].reset_index(drop=True)
            for col in ROBOT_SYNC_INDEX_COLUMNS:
                df[col] -= self.temp_cut
            if df.empty:
                continue
            if self.max_sync_rows is not None and len(df) > self.max_sync_rows:
                df = df.iloc[: self.max_sync_rows].reset_index(drop=True)

            bird_f = load_video_frames(bird_by[rec_id], resize_factor=self.resize_factor, label=f"bird({rec_id})")[
                self.temp_cut :
            ]
            front_f = load_video_frames(front_by[rec_id], resize_factor=self.resize_factor, label=f"front({rec_id})")[
                self.temp_cut :
            ]
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
                self.sample_demo_idx.append(demo_idx)

            self.num_samples += n_i
            self.num_demos += 1
            print(f"    -> robot {rec_id}: {n_i} samples")

        if self.num_demos == 0:
            raise FileNotFoundError("No complete robot demos found for Combined.")

        all_q = torch.stack(self.joint_data, dim=0)  # [N, 14]
        self.state_mean = all_q.mean(dim=0)
        self.state_std = all_q.std(dim=0).clamp(min=1e-2)
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

        # images: each transform -> [3,H,W]; stack -> [4,3,H,W]
        images = stack_camera_tensors(
            self.image_transform(self.bird_frames[sample_idx]),
            self.image_transform(self.front_frames[sample_idx]),
            self.image_transform(self.left_frames[sample_idx]),
            self.image_transform(self.right_frames[sample_idx]),
        )

        state_raw = self.joint_data[sample_idx]  # [14]
        state = (state_raw - self.state_mean) / self.state_std

        slice_end = min(demo_end, sample_idx - 1 + self.num_queries)
        future = list(self.joint_data[max(0, sample_idx - 1) : slice_end])
        raw_len = len(future)
        pad_len = self.num_queries - raw_len
        pad_tensor = torch.zeros(ROBOT_STATE_DIM, dtype=torch.float32)
        future.extend([pad_tensor] * pad_len)
        is_pad = torch.zeros(self.num_queries, dtype=torch.bool)
        if pad_len > 0:
            is_pad[-pad_len:] = True
        actions = torch.stack([((step - self.state_mean) / self.state_std) for step in future], dim=0)  # [K,14]

        return {
            "embodiment": EMBODIMENT_ROBOT,
            "images": images,
            "camera_mask": camera_mask_tensor(EMBODIMENT_ROBOT),
            "state": state,
            "actions": actions,
            "is_pad": is_pad,
            "action_dim": ROBOT_STATE_DIM,
        }
