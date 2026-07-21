"""Shared path / video helpers for Combined dataloaders."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def demo_id_from_hash_filename(path: str | Path) -> str:
    """video_recording_...#demo_id.ext -> demo_id"""
    name = Path(path).name
    if "#" not in name:
        raise ValueError(f"Expected '#' in filename: {name}")
    return name.split("#", 1)[1].rsplit(".", 1)[0]


def demo_id_from_joint_npy(path: str | Path, prefix: str = "joint_position_") -> str:
    name = Path(path).name
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected joint file name: {name}")
    return name[len(prefix) :].rsplit(".", 1)[0]


def demo_id_from_pose_npz(path: str | Path) -> str:
    """
    ...#human_hands_bimanual_raw_YYYYMMDDHHMMSS_wilor_rgbd_pose.npz
      -> human_hands_bimanual_raw_YYYYMMDDHHMMSS
    """
    stem = demo_id_from_hash_filename(path)
    for suffix in ("_wilor_rgbd_pose", "_mediapipe_rgbd_pose", "_hand_pose"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def index_paths_by_demo_id(paths: list[Path], id_fn) -> dict[str, Path]:
    return {id_fn(p): p for p in paths}


def build_image_transform(transform: str = "resnet_normalization"):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if "resnet_normalization" in transform:
        return transforms.Compose([transforms.ToTensor(), normalize])
    return transforms.ToTensor()


def load_video_frames(
    video_path: Path,
    *,
    resize_factor: float = 1.0,
    label: str = "Video",
) -> list[np.ndarray]:
    """Load full RGB video as list of HxWx3 uint8 arrays."""
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


def zero_rgb_like(ref: np.ndarray) -> np.ndarray:
    return np.zeros_like(ref)


def pad_state(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad trailing zeros so robot [14] can sit in a [20] buffer if needed."""
    state = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
    if state.numel() == target_dim:
        return state
    if state.numel() > target_dim:
        raise ValueError(f"Cannot pad state of dim {state.numel()} down to {target_dim}")
    out = torch.zeros(target_dim, dtype=torch.float32)
    out[: state.numel()] = state
    return out
