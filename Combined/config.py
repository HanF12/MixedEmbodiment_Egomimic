"""
Combined (mixed human + robot) ACT constants and helpers.

Camera slot order is fixed as:
  [bird, front, left_wrist, right_wrist]  -> model cams cam0..cam3

Robot:
  state/action dim = 14  (left 7 + right 7)
  camera_mask = [1, 1, 1, 1]

Human:
  state/action dim = 20  (left hand 10 + right hand 10)
  hand 10D = xyz(3) + rot6d(6) + open/close(1)
  camera_mask = [1, 1, 0, 0]
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_NUM_QUERIES = 45

# --- Robot ---
ROBOT_STATE_DIM = 14
JOINT_DIM_PER_ARM = 7
LEFT_ARM_SLICE = slice(0, JOINT_DIM_PER_ARM)
RIGHT_ARM_SLICE = slice(JOINT_DIM_PER_ARM, ROBOT_STATE_DIM)
GRIPPER_INDICES = (JOINT_DIM_PER_ARM - 1, ROBOT_STATE_DIM - 1)

# --- Human hands ---
HAND_DIM_PER_HAND = 10
HUMAN_STATE_DIM = 20  # left 10 + right 10
LEFT_HAND_SLICE = slice(0, HAND_DIM_PER_HAND)
RIGHT_HAND_SLICE = slice(HAND_DIM_PER_HAND, HUMAN_STATE_DIM)

# --- Shared camera layout ---
CAMERA_ORDER = ("bird", "front", "left_wrist", "right_wrist")
MODEL_CAMERA_NAMES = ("cam0", "cam1", "cam2", "cam3")
NUM_CAMERAS = 4

ROBOT_CAMERA_MASK = (1, 1, 1, 1)
HUMAN_CAMERA_MASK = (1, 1, 0, 0)

EMBODIMENT_ROBOT = 0
EMBODIMENT_HUMAN = 1
EMBODIMENT_NAMES = ("robot", "human")

# Robot sync CSV columns (same semantics as Bimanual, independent copy)
ROBOT_SYNC_INDEX_COLUMNS = (
    "left_joint_index",
    "right_joint_index",
    "left_index",
    "right_index",
    "bird_index",
    "front_index",
)

# Human sync: bird + front cameras + hand-pose stream
HUMAN_SYNC_INDEX_COLUMNS = (
    "bird_index",
    "front_index",
    "pose_index",
)

MAX_STATE_DIM = HUMAN_STATE_DIM  # pad robot -> 20 when needed for mixed utilities


def default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_camera_names(camera_names: list[str] | tuple[str, ...]) -> None:
    if tuple(camera_names) != MODEL_CAMERA_NAMES:
        raise ValueError(
            f"Expected camera_names={MODEL_CAMERA_NAMES}, got {tuple(camera_names)}. "
            "Combined uses fixed slots [bird, front, left_wrist, right_wrist]."
        )


def concat_bimanual_joints(
    left_step: np.ndarray | torch.Tensor,
    right_step: np.ndarray | torch.Tensor,
    *,
    rec_id: str = "",
) -> torch.Tensor:
    """Robot qpos: [7 left, 7 right] -> [14]."""
    left_tensor = torch.as_tensor(left_step, dtype=torch.float32).reshape(-1)
    right_tensor = torch.as_tensor(right_step, dtype=torch.float32).reshape(-1)
    if left_tensor.numel() < JOINT_DIM_PER_ARM or right_tensor.numel() < JOINT_DIM_PER_ARM:
        raise ValueError(
            f"Expected >= {JOINT_DIM_PER_ARM} joints/arm for {rec_id or 'sample'}, "
            f"got left={left_tensor.numel()} right={right_tensor.numel()}"
        )
    return torch.cat([left_tensor[:JOINT_DIM_PER_ARM], right_tensor[:JOINT_DIM_PER_ARM]], dim=0)


def flatten_hand_pose(pose_t: np.ndarray | torch.Tensor, *, rec_id: str = "") -> torch.Tensor:
    """
    Flatten one timestep of hand pose.

    Expected input shape: [2, 10] with slot0=left, slot1=right.
    Output shape: [20]
    """
    pose = torch.as_tensor(pose_t, dtype=torch.float32)
    if pose.shape != (2, HAND_DIM_PER_HAND):
        raise ValueError(
            f"Expected pose timestep shape (2, {HAND_DIM_PER_HAND}) for {rec_id or 'sample'}, got {tuple(pose.shape)}"
        )
    return pose.reshape(HUMAN_STATE_DIM)


def stack_camera_tensors(
    bird_frame: torch.Tensor,
    front_frame: torch.Tensor,
    left_frame: torch.Tensor,
    right_frame: torch.Tensor,
) -> torch.Tensor:
    """Stack to [4, C, H, W] in CAMERA_ORDER."""
    # shapes: each [C,H,W] -> stacked [4,C,H,W]
    return torch.stack([bird_frame, front_frame, left_frame, right_frame], dim=0)


def camera_mask_tensor(embodiment: int) -> torch.Tensor:
    if int(embodiment) == EMBODIMENT_ROBOT:
        return torch.tensor(ROBOT_CAMERA_MASK, dtype=torch.float32)
    if int(embodiment) == EMBODIMENT_HUMAN:
        return torch.tensor(HUMAN_CAMERA_MASK, dtype=torch.float32)
    raise ValueError(f"Unknown embodiment id {embodiment}")


def build_run_metadata(
    *,
    robot_data_root: str | Path | None,
    human_data_root: str | Path | None,
    robot_sync_dir: str | Path | None,
    human_sync_dir: str | Path | None,
    num_queries: int,
    max_skew_s: float,
) -> dict[str, Any]:
    return {
        "variant": "combined-mixed-act",
        "camera_order": list(CAMERA_ORDER),
        "model_camera_names": list(MODEL_CAMERA_NAMES),
        "robot_state_dim": ROBOT_STATE_DIM,
        "human_state_dim": HUMAN_STATE_DIM,
        "robot_camera_mask": list(ROBOT_CAMERA_MASK),
        "human_camera_mask": list(HUMAN_CAMERA_MASK),
        "num_queries": int(num_queries),
        "robot_data_root": str(robot_data_root) if robot_data_root is not None else None,
        "human_data_root": str(human_data_root) if human_data_root is not None else None,
        "robot_sync_dir": str(robot_sync_dir) if robot_sync_dir is not None else None,
        "human_sync_dir": str(human_sync_dir) if human_sync_dir is not None else None,
        "max_skew_s": float(max_skew_s),
    }


def save_run_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata_path


def load_run_metadata(run_dir: str | Path) -> dict[str, Any] | None:
    metadata_path = Path(run_dir) / "run_metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def validate_run_metadata(metadata: dict[str, Any], *, num_queries: int | None = None) -> None:
    if tuple(metadata.get("camera_order", [])) != CAMERA_ORDER:
        raise ValueError(
            f"Saved camera_order {metadata.get('camera_order')} does not match expected {CAMERA_ORDER}"
        )
    if tuple(metadata.get("model_camera_names", [])) != MODEL_CAMERA_NAMES:
        raise ValueError(
            "Saved model camera names do not match the fixed Combined camera order"
        )
    if int(metadata.get("robot_state_dim", -1)) != ROBOT_STATE_DIM:
        raise ValueError(
            f"Saved robot_state_dim {metadata.get('robot_state_dim')} does not match {ROBOT_STATE_DIM}"
        )
    if num_queries is not None and int(metadata.get("num_queries", -1)) != int(num_queries):
        raise ValueError(
            f"Saved num_queries {metadata.get('num_queries')} does not match CLI {num_queries}"
        )
