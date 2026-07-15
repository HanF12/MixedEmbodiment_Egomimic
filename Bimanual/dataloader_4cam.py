from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from Bimanual.config import (
    DEFAULT_NUM_QUERIES,
    SYNC_INDEX_COLUMNS,
    concat_bimanual_joints,
    stack_camera_tensors,
)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def demo_id_from_hash_filename(path: str | Path) -> str:
    name = Path(path).name
    if "#" not in name:
        raise ValueError(f"Expected '#' in filename: {name}")
    return name.split("#", 1)[1].rsplit(".", 1)[0]


def demo_id_from_joint_npy(path: str | Path, prefix: str = "joint_position_") -> str:
    name = Path(path).name
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected joint file name: {name}")
    return name[len(prefix) :].rsplit(".", 1)[0]


def index_paths_by_demo_id(paths: list[Path], id_fn) -> dict[str, Path]:
    return {id_fn(p): p for p in paths}


def _load_video_frames(video_path: Path, *, resize_factor: float = 1.0, label: str = "Video") -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    read_count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        read_count += 1
        if resize_factor != 1.0:
            h, w = frame_bgr.shape[:2]
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(w * resize_factor), int(h * resize_factor)),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"  - {label} '{video_path.name}' -> read {read_count}/{total} frames")
    return frames


class PreloadedBimanualEpisodeDataset(Dataset):
    """
    Preloads 4 RGB views plus 14D bimanual joint state.

    Camera order is fixed as:
      [left wrist, right wrist, bird, front]
    Joint order is fixed as:
      [left_7d, right_7d]
    """

    def __init__(
        self,
        *,
        bird_vids_dir: str | Path,
        front_vids_dir: str | Path,
        left_arm_vids_dir: str | Path,
        right_arm_vids_dir: str | Path,
        left_joint_data_dir: str | Path,
        right_joint_data_dir: str | Path,
        sync_csv_dir: str | Path,
        num_queries: int = DEFAULT_NUM_QUERIES,
        pad: bool = True,
        transform: str = "resnet_normalization",
        max_demos: int | None = None,
        temp_cut: int = 10,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.pad = bool(pad)
        self.transform = transform
        self.temp_cut = int(temp_cut)
        self.demo_lengths: list[int] = []

        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        base_normalization = transforms.Compose([transforms.ToTensor(), normalize_transform])
        self.image_transforms = [base_normalization] if "resnet_normalization" in transform else [transforms.ToTensor()]

        self.bird_vids_path = resolve_path(bird_vids_dir)
        self.front_vids_path = resolve_path(front_vids_dir)
        self.left_arm_vids_path = resolve_path(left_arm_vids_dir)
        self.right_arm_vids_path = resolve_path(right_arm_vids_dir)
        self.left_joint_data_path = resolve_path(left_joint_data_dir)
        self.right_joint_data_path = resolve_path(right_joint_data_dir)
        self.sync_csv_path = resolve_path(sync_csv_dir)

        bird_vids_all = sorted(self.bird_vids_path.glob("*.mp4"))
        front_vids_all = sorted(self.front_vids_path.glob("*.mp4"))
        left_arm_vids_all = sorted(self.left_arm_vids_path.glob("*.mp4"))
        right_arm_vids_all = sorted(self.right_arm_vids_path.glob("*.mp4"))
        left_joint_npy_all = sorted(self.left_joint_data_path.glob("*.npy"))
        right_joint_npy_all = sorted(self.right_joint_data_path.glob("*.npy"))
        sync_csv_all = sorted(self.sync_csv_path.glob("*.csv"))

        if max_demos is not None and max_demos > 0:
            sync_csv_all = sync_csv_all[:max_demos]

        if not sync_csv_all:
            raise FileNotFoundError(f"No sync CSV files found in {self.sync_csv_path}")

        bird_vids_by_id = index_paths_by_demo_id(bird_vids_all, demo_id_from_hash_filename)
        front_vids_by_id = index_paths_by_demo_id(front_vids_all, demo_id_from_hash_filename)
        left_arm_vids_by_id = index_paths_by_demo_id(left_arm_vids_all, demo_id_from_hash_filename)
        right_arm_vids_by_id = index_paths_by_demo_id(right_arm_vids_all, demo_id_from_hash_filename)
        left_joint_npy_by_id = index_paths_by_demo_id(left_joint_npy_all, demo_id_from_joint_npy)
        right_joint_npy_by_id = index_paths_by_demo_id(right_joint_npy_all, demo_id_from_joint_npy)

        self.left_frames_list: list[np.ndarray] = []
        self.right_frames_list: list[np.ndarray] = []
        self.bird_frames_list: list[np.ndarray] = []
        self.front_frames_list: list[np.ndarray] = []
        self.joint_data: list[torch.Tensor] = []
        self.num_demos = 0
        total_samples = 0

        print("Loading and synchronizing all bimanual recordings...")
        for csv_path in sync_csv_all:
            rec_id = csv_path.stem
            missing = []
            if rec_id not in bird_vids_by_id:
                missing.append("bird video")
            if rec_id not in front_vids_by_id:
                missing.append("front video")
            if rec_id not in left_arm_vids_by_id:
                missing.append("left wrist video")
            if rec_id not in right_arm_vids_by_id:
                missing.append("right wrist video")
            if rec_id not in left_joint_npy_by_id:
                missing.append("left joint position")
            if rec_id not in right_joint_npy_by_id:
                missing.append("right joint position")
            if missing:
                print(f"WARNING: skipping {rec_id} - missing {', '.join(missing)}")
                continue

            df_sync = pd.read_csv(csv_path)
            required_cols = set(SYNC_INDEX_COLUMNS)
            if not required_cols.issubset(df_sync.columns):
                raise KeyError(f"Sync CSV '{csv_path}' missing columns: {required_cols - set(df_sync.columns)}")
            if df_sync.empty:
                print(f"WARNING: skipping {rec_id} - sync CSV has 0 rows")
                continue

            mask = (
                (df_sync["left_joint_index"] >= self.temp_cut)
                & (df_sync["right_joint_index"] >= self.temp_cut)
                & (df_sync["left_index"] >= self.temp_cut)
                & (df_sync["right_index"] >= self.temp_cut)
                & (df_sync["bird_index"] >= self.temp_cut)
                & (df_sync["front_index"] >= self.temp_cut)
            )
            df_sync = df_sync[mask].reset_index(drop=True)
            for col in SYNC_INDEX_COLUMNS:
                df_sync[col] -= self.temp_cut
            if df_sync.empty:
                print(f"WARNING: skipping {rec_id} - no synced rows after temp_cut={self.temp_cut}")
                continue

            left_frames = _load_video_frames(left_arm_vids_by_id[rec_id], label=f"Left wrist ({rec_id})")[self.temp_cut :]
            right_frames = _load_video_frames(right_arm_vids_by_id[rec_id], label=f"Right wrist ({rec_id})")[self.temp_cut :]
            bird_frames = _load_video_frames(bird_vids_by_id[rec_id], label=f"Bird ({rec_id})")[self.temp_cut :]
            front_frames = _load_video_frames(front_vids_by_id[rec_id], label=f"Front ({rec_id})")[self.temp_cut :]

            left_joint_arr = np.load(left_joint_npy_by_id[rec_id]).astype(np.float32)[self.temp_cut :]
            right_joint_arr = np.load(right_joint_npy_by_id[rec_id]).astype(np.float32)[self.temp_cut :]
            if left_joint_arr.ndim != 2 or right_joint_arr.ndim != 2:
                raise ValueError(f"Expected 2D joint arrays for {rec_id}")
            left_joint_lookup = [torch.from_numpy(step) for step in left_joint_arr]
            right_joint_lookup = [torch.from_numpy(step) for step in right_joint_arr]

            left_joint_idxs = df_sync["left_joint_index"].to_numpy(dtype=np.int64)
            right_joint_idxs = df_sync["right_joint_index"].to_numpy(dtype=np.int64)
            left_idxs = df_sync["left_index"].to_numpy(dtype=np.int64)
            right_idxs = df_sync["right_index"].to_numpy(dtype=np.int64)
            bird_idxs = df_sync["bird_index"].to_numpy(dtype=np.int64)
            front_idxs = df_sync["front_index"].to_numpy(dtype=np.int64)

            n_i = len(df_sync)
            self.demo_lengths.append(n_i)
            for idx in range(n_i):
                if left_joint_idxs[idx] >= len(left_joint_lookup) or right_joint_idxs[idx] >= len(right_joint_lookup):
                    raise IndexError(
                        f"Sync CSV for {rec_id} references out-of-range joint indices "
                        f"(left={left_joint_idxs[idx]}, right={right_joint_idxs[idx]})"
                    )
                self.left_frames_list.append(left_frames[left_idxs[idx]])
                self.right_frames_list.append(right_frames[right_idxs[idx]])
                self.bird_frames_list.append(bird_frames[bird_idxs[idx]])
                self.front_frames_list.append(front_frames[front_idxs[idx]])
                self.joint_data.append(
                    concat_bimanual_joints(
                        left_joint_lookup[left_joint_idxs[idx]],
                        right_joint_lookup[right_joint_idxs[idx]],
                        rec_id=rec_id,
                    )
                )

            total_samples += n_i
            self.num_demos += 1
            print(f"    -> Added {n_i} synced samples for {rec_id} (total={total_samples})")

        if self.num_demos == 0:
            raise FileNotFoundError("No complete bimanual demonstrations found after matching sync CSVs to data files.")

        all_joints = torch.stack(self.joint_data, dim=0)
        self.joint_mean = all_joints.mean(dim=0)
        self.joint_std = all_joints.std(dim=0).clamp(min=1e-2)
        print(f"Finished initializing bimanual dataset with {self.num_demos} demos.")

    def __len__(self) -> int:
        return self.num_demos * len(self.image_transforms)

    def __getitem__(self, idx: int):
        view_type = idx // self.num_demos
        episode_idx = idx % self.num_demos

        ep_start = sum(self.demo_lengths[:episode_idx])
        ep_len = self.demo_lengths[episode_idx]
        if ep_len <= 0:
            raise RuntimeError(f"Episode {episode_idx} has no synced samples.")

        start_ts_in_ep = np.random.randint(0, ep_len)
        sample_idx = ep_start + start_ts_in_ep
        demo_end_idx = ep_start + ep_len

        transform_pipeline = self.image_transforms[view_type]
        left_frame = transform_pipeline(self.left_frames_list[sample_idx])
        right_frame = transform_pipeline(self.right_frames_list[sample_idx])
        bird_frame = transform_pipeline(self.bird_frames_list[sample_idx])
        front_frame = transform_pipeline(self.front_frames_list[sample_idx])

        joint_raw = self.joint_data[sample_idx]
        joint_data = (joint_raw - self.joint_mean) / self.joint_std

        slice_end = min(demo_end_idx, sample_idx - 1 + self.num_queries)
        future_list = self.joint_data[max(0, sample_idx - 1) : slice_end]
        raw_length = len(future_list)
        pad_length = self.num_queries - raw_length

        if not self.pad:
            raise RuntimeError("Padding must be enabled for ACT training.")
        pad_tensor = torch.zeros_like(self.joint_data[0])
        future_list.extend([pad_tensor] * pad_length)
        is_pad = torch.zeros(self.num_queries, dtype=torch.bool)
        if pad_length > 0:
            is_pad[-pad_length:] = True

        prediction_joint_data = torch.stack(
            [((torch.as_tensor(step, dtype=torch.float32) - self.joint_mean) / self.joint_std) for step in future_list],
            dim=0,
        )
        stacked_images = stack_camera_tensors(left_frame, right_frame, bird_frame, front_frame)
        return [joint_data, stacked_images, prediction_joint_data, is_pad]


def batch_sanity_check(train_loader: DataLoader) -> None:
    batch = next(iter(train_loader))
    qpos, image, actions, is_pad = batch
    print("--- Bimanual batch sanity check ---")
    print(f"qpos shape:    {tuple(qpos.shape)}")
    print(f"image shape:   {tuple(image.shape)}")
    print(f"actions shape: {tuple(actions.shape)}")
    print(f"is_pad shape:  {tuple(is_pad.shape)}")
