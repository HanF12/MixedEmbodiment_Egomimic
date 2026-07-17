"""Human hand episode dataset for Combined ACT (2 cameras + 20D hand pose)."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from Combined.config import (
    DEFAULT_NUM_QUERIES,
    EMBODIMENT_HUMAN,
    HUMAN_STATE_DIM,
    HUMAN_SYNC_INDEX_COLUMNS,
    camera_mask_tensor,
    flatten_hand_pose,
    stack_camera_tensors,
)
from Combined.dataloader_utils import (
    build_image_transform,
    demo_id_from_hash_filename,
    demo_id_from_pose_npz,
    index_paths_by_demo_id,
    load_video_frames,
    resolve_path,
    zero_rgb_like,
)


class HumanEpisodeDataset(Dataset):
    """
    One item = one human episode (random start inside episode).

    Camera slots are always 4:
      [bird, front, left_wrist=zeros, right_wrist=zeros]
    camera_mask = [1,1,0,0]

    Hand pose from NPZ key `pose` with shape [T, 2, 10] -> flattened [20].
    """

    def __init__(
        self,
        *,
        bird_vids_dir,
        front_vids_dir,
        pose_npz_dir,
        sync_csv_dir,
        num_queries: int = DEFAULT_NUM_QUERIES,
        transform: str = "resnet_normalization",
        max_demos: int | None = None,
        temp_cut: int = 10,
        resize_factor: float = 1.0,
        max_sync_rows: int | None = None,
        require_valid_pos: bool = True,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.temp_cut = int(temp_cut)
        self.resize_factor = float(resize_factor)
        self.max_sync_rows = int(max_sync_rows) if max_sync_rows is not None else None
        self.require_valid_pos = bool(require_valid_pos)
        self.image_transform = build_image_transform(transform)

        bird_vids = sorted(resolve_path(bird_vids_dir).glob("*.mp4"))
        front_vids = sorted(resolve_path(front_vids_dir).glob("*.mp4"))
        pose_files = sorted(resolve_path(pose_npz_dir).glob("*.npz"))
        sync_csvs = sorted(resolve_path(sync_csv_dir).glob("*.csv"))
        if max_demos is not None and max_demos > 0:
            sync_csvs = sync_csvs[:max_demos]
        if not sync_csvs:
            raise FileNotFoundError(f"No human sync CSVs in {sync_csv_dir}")

        bird_by = index_paths_by_demo_id(bird_vids, demo_id_from_hash_filename)
        front_by = index_paths_by_demo_id(front_vids, demo_id_from_hash_filename)
        pose_by = index_paths_by_demo_id(pose_files, demo_id_from_pose_npz)

        self.bird_frames: list[np.ndarray] = []
        self.front_frames: list[np.ndarray] = []
        self.pose_data: list[torch.Tensor] = []
        self.sample_demo_idx: List[int] = []
        self.demo_start_idx: List[int] = []
        self.demo_lengths: List[int] = []
        self.num_demos = 0
        self.num_samples = 0

        print("Loading human Combined demos...")
        for csv_path in sync_csvs:
            rec_id = csv_path.stem
            missing = []
            if rec_id not in bird_by:
                missing.append("bird")
            if rec_id not in front_by:
                missing.append("front")
            if rec_id not in pose_by:
                missing.append("pose_npz")
            if missing:
                print(f"WARNING: skip human {rec_id} missing {missing}")
                continue

            df = pd.read_csv(csv_path)
            if not set(HUMAN_SYNC_INDEX_COLUMNS).issubset(df.columns):
                raise KeyError(f"{csv_path} missing human sync columns")

            pose_npz = np.load(pose_by[rec_id])
            pose_arr = np.asarray(pose_npz["pose"], dtype=np.float32)  # [T,2,10]
            valid_pos = np.asarray(pose_npz["valid_pos"], dtype=bool) if "valid_pos" in pose_npz.files else None

            mask = (df["bird_index"].to_numpy() >= self.temp_cut) & (df["front_index"].to_numpy() >= self.temp_cut)
            # pose_index indexes the original NPZ; do not temp_cut pose timeline the same way as video
            # (pose NPZ is already aligned to bag/camera time). Still drop early video frames.
            df = df[mask].reset_index(drop=True)
            df["bird_index"] = df["bird_index"] - self.temp_cut
            df["front_index"] = df["front_index"] - self.temp_cut
            if df.empty:
                continue

            if self.require_valid_pos and valid_pos is not None:
                keep = []
                for i in range(len(df)):
                    pidx = int(df.loc[i, "pose_index"])
                    if pidx < 0 or pidx >= len(pose_arr):
                        keep.append(False)
                        continue
                    # require both hands valid when available
                    keep.append(bool(valid_pos[pidx].all()))
                df = df[np.asarray(keep, dtype=bool)].reset_index(drop=True)
            if df.empty:
                print(f"WARNING: skip human {rec_id} - no valid pose rows after filters")
                continue
            if self.max_sync_rows is not None and len(df) > self.max_sync_rows:
                df = df.iloc[: self.max_sync_rows].reset_index(drop=True)

            bird_f = load_video_frames(bird_by[rec_id], resize_factor=self.resize_factor, label=f"bird({rec_id})")[
                self.temp_cut :
            ]
            front_f = load_video_frames(front_by[rec_id], resize_factor=self.resize_factor, label=f"front({rec_id})")[
                self.temp_cut :
            ]

            demo_idx = self.num_demos
            self.demo_start_idx.append(len(self.pose_data))
            n_i = len(df)
            self.demo_lengths.append(n_i)

            for i in range(n_i):
                bidx = int(df.loc[i, "bird_index"])
                fidx = int(df.loc[i, "front_index"])
                pidx = int(df.loc[i, "pose_index"])
                self.bird_frames.append(bird_f[bidx])
                self.front_frames.append(front_f[fidx])
                self.pose_data.append(flatten_hand_pose(pose_arr[pidx], rec_id=rec_id))  # [20]
                self.sample_demo_idx.append(demo_idx)

            self.num_samples += n_i
            self.num_demos += 1
            print(f"    -> human {rec_id}: {n_i} samples")

        if self.num_demos == 0:
            raise FileNotFoundError("No complete human demos found for Combined.")

        all_p = torch.stack(self.pose_data, dim=0)  # [N, 20]
        self.state_mean = all_p.mean(dim=0)
        self.state_std = all_p.std(dim=0).clamp(min=1e-2)
        print(f"Human dataset ready: demos={self.num_demos} samples={self.num_samples}")

    def __len__(self) -> int:
        return self.num_demos

    def __getitem__(self, idx: int) -> dict:
        episode_idx = int(idx)
        ep_start = self.demo_start_idx[episode_idx]
        ep_len = self.demo_lengths[episode_idx]
        start_in_ep = int(np.random.randint(0, ep_len))
        sample_idx = ep_start + start_in_ep
        demo_end = ep_start + ep_len

        bird_t = self.image_transform(self.bird_frames[sample_idx])  # [3,H,W]
        front_t = self.image_transform(self.front_frames[sample_idx])
        # Wrist slots unused for human: zeros with same spatial size as bird.
        zero_np = zero_rgb_like(self.bird_frames[sample_idx])
        left_t = self.image_transform(zero_np)
        right_t = self.image_transform(zero_np)
        images = stack_camera_tensors(bird_t, front_t, left_t, right_t)  # [4,3,H,W]

        state_raw = self.pose_data[sample_idx]  # [20]
        state = (state_raw - self.state_mean) / self.state_std

        slice_end = min(demo_end, sample_idx - 1 + self.num_queries)
        future = list(self.pose_data[max(0, sample_idx - 1) : slice_end])
        raw_len = len(future)
        pad_len = self.num_queries - raw_len
        pad_tensor = torch.zeros(HUMAN_STATE_DIM, dtype=torch.float32)
        future.extend([pad_tensor] * pad_len)
        is_pad = torch.zeros(self.num_queries, dtype=torch.bool)
        if pad_len > 0:
            is_pad[-pad_len:] = True
        actions = torch.stack([((step - self.state_mean) / self.state_std) for step in future], dim=0)  # [K,20]

        return {
            "embodiment": EMBODIMENT_HUMAN,
            "images": images,
            "camera_mask": camera_mask_tensor(EMBODIMENT_HUMAN),
            "state": state,
            "actions": actions,
            "is_pad": is_pad,
            "action_dim": HUMAN_STATE_DIM,
        }


def collate_homogeneous(batch: list[dict]) -> dict:
    """Collate a batch that is all-robot or all-human (same action_dim)."""
    emb = int(batch[0]["embodiment"])
    action_dim = int(batch[0]["action_dim"])
    if any(int(b["embodiment"]) != emb for b in batch):
        raise ValueError("collate_homogeneous requires a single embodiment per batch")
    return {
        "embodiment": emb,
        "images": torch.stack([b["images"] for b in batch], dim=0),  # [B,4,3,H,W]
        "camera_mask": torch.stack([b["camera_mask"] for b in batch], dim=0),  # [B,4]
        "state": torch.stack([b["state"] for b in batch], dim=0),  # [B,D]
        "actions": torch.stack([b["actions"] for b in batch], dim=0),  # [B,K,D]
        "is_pad": torch.stack([b["is_pad"] for b in batch], dim=0),  # [B,K]
        "action_dim": action_dim,
    }
