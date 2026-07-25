"""
Combined-relative-xyz-3cam (mixed human + robot) ACT constants and helpers.

Sibling of Combined_relative, but shared pose is *xyz only* (no gripper/open
channel in the pose prediction pipeline).

Camera slot order is fixed as:
  [bird, front, left_wrist, right_wrist]  -> model cams cam0..cam3

True 3-camera architecture (no front model slot) + xyz-only pose pathway.

Camera slots: [bird, left_wrist, right_wrist] -> cam0..cam2
Front RGB is sync-only (not a network input).

Shared pose (human hands / robot EEF):
  6D = left 3 + right 3 (xyz only; rot+grip dropped at load)

Robot joints:
  14D = left 7 + right 7  (absolute; gripper joints treated as continuous)

Robot proprio for the model (EgoMimic-style): joints only [14]
Human proprio: absolute xyz pose [6]
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_NUM_QUERIES = 45  # keep Combined horizon (EgoMimic uses 100)

# EgoMimic-matched training defaults
DEFAULT_NUM_EPOCHS = 10000
# One epoch = one full pass over the longer modality's demo loader
# (max(len(robot_loader), len(human_loader))); shorter modality is recycled.
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_KL_WEIGHT = 10.0
DEFAULT_HAND_LAMBDA = 1.0
DEFAULT_RECON_LOSS = "l1"

# Temporal convention for action chunks:
# Chunk is built starting at the current observation index t.
# Pose actions are relative to that first observation:
#   pose_actions[k] = pose[t+k] - pose[t]
# Joint actions remain absolute: joint_actions[k] = joints[t+k].
ACTION_CHUNK_STARTS_AT_CURRENT = True
POSE_ACTION_RELATIVE_TO_CHUNK_ANCHOR = True

# --- Robot joints ---
ROBOT_JOINT_DIM = 14
JOINT_DIM_PER_ARM = 7
LEFT_ARM_SLICE = slice(0, JOINT_DIM_PER_ARM)
RIGHT_ARM_SLICE = slice(JOINT_DIM_PER_ARM, ROBOT_JOINT_DIM)
# Gripper joint indices in flattened [14]; values are continuous (not binarized).
GRIPPER_INDICES = (JOINT_DIM_PER_ARM - 1, ROBOT_JOINT_DIM - 1)
ROBOT_STATE_DIM = ROBOT_JOINT_DIM  # proprio = joints only (EgoMimic)

# --- Raw NPZ pose layout (before dropping rotation / gripper) ---
RAW_HAND_DIM = 10  # xyz(3) + rot6d(6) + grip(1)
RAW_XYZ_SLICE = slice(0, 3)
RAW_GRIP_INDEX = 9  # unused in this variant (dropped)

# --- Shared pose used in training (xyz only) ---
POSE_DIM_PER_SIDE = 3  # xyz(3)
POSE_DIM = 6  # left 3 + right 3
HUMAN_STATE_DIM = POSE_DIM
HUMAN_PROPRIO_DIM = POSE_DIM
ROBOT_PROPRIO_DIM = ROBOT_JOINT_DIM  # EgoMimic: joints only

# --- Shared camera layout (NO front slot in the model) ---
CAMERA_ORDER = ("bird", "left_wrist", "right_wrist")
MODEL_CAMERA_NAMES = ("cam0", "cam1", "cam2")
NUM_CAMERAS = 3

ROBOT_CAMERA_MASK = (1, 1, 1)  # bird + both wrists
HUMAN_CAMERA_MASK = (1, 0, 0)  # bird only; wrist slots zeroed

EMBODIMENT_ROBOT = 0
EMBODIMENT_HUMAN = 1
EMBODIMENT_NAMES = ("robot", "human")

# Sync CSVs still include front_index for timestamp alignment (not a model cam).
ROBOT_SYNC_INDEX_COLUMNS = (
    "left_joint_index",
    "right_joint_index",
    "left_index",
    "right_index",
    "bird_index",
    "front_index",
    "eef_pose_index",
)

ROBOT_TEMP_CUT_INDEX_COLUMNS = (
    "left_joint_index",
    "right_joint_index",
    "left_index",
    "right_index",
    "bird_index",
    "front_index",
)

HUMAN_SYNC_INDEX_COLUMNS = (
    "bird_index",
    "front_index",
    "pose_index",
)

ROBOT_EEF_RELDIR = Path("joint-data") / "combined_npz_commonframe"
ROBOT_EEF_COORD_FRAME = (
    "arm_fk_targetframe_commonframe: left/right arm-base FK poses "
    "transformed into a shared midline frame. "
    "Training uses xyz only (rotation and gripper dropped from pose pathway)."
)

# Default pose NPZ layouts under each modality root (0714 layout).
HUMAN_POSE_RELDIR = Path("bird-realsense-data") / "combined_npz_targetframe"
DEFAULT_ROBOT_DATA_ROOT = Path("sessions") / "teleop_bimanual" / "0714"
DEFAULT_HUMAN_DATA_ROOT = Path("sessions") / "human_hands_bimanual_raw" / "0714"

MAX_STATE_DIM = max(POSE_DIM, ROBOT_JOINT_DIM)


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
    """
    Robot qpos: [7 left, 7 right] -> [14].

    Gripper joint channels (indices 6 and 13) are left as continuous floats;
    no thresholding / binarization is applied.
    """
    left_tensor = torch.as_tensor(left_step, dtype=torch.float32).reshape(-1)
    right_tensor = torch.as_tensor(right_step, dtype=torch.float32).reshape(-1)
    if left_tensor.numel() < JOINT_DIM_PER_ARM or right_tensor.numel() < JOINT_DIM_PER_ARM:
        raise ValueError(
            f"Expected >= {JOINT_DIM_PER_ARM} joints/arm for {rec_id or 'sample'}, "
            f"got left={left_tensor.numel()} right={right_tensor.numel()}"
        )
    return torch.cat([left_tensor[:JOINT_DIM_PER_ARM], right_tensor[:JOINT_DIM_PER_ARM]], dim=0)


def flatten_bimanual_pose(pose_t: np.ndarray | torch.Tensor, *, rec_id: str = "") -> torch.Tensor:
    """
    Extract xyz-only from one bimanual pose timestep and flatten.

    Expected input shape: [2, 10] (raw NPZ layout).
    Output shape: [6] = [left_xyz(3), right_xyz(3)].
    Rotation (indices 3:9) and gripper (index 9) are discarded.
    """
    pose = torch.as_tensor(pose_t, dtype=torch.float32)
    if pose.shape != (2, RAW_HAND_DIM):
        raise ValueError(
            f"Expected pose timestep shape (2, {RAW_HAND_DIM}) for {rec_id or 'sample'}, "
            f"got {tuple(pose.shape)}"
        )
    flat = torch.cat([pose[0, RAW_XYZ_SLICE], pose[1, RAW_XYZ_SLICE]], dim=0)
    if flat.numel() != POSE_DIM:
        raise ValueError(f"Internal pose flatten error: got {flat.numel()} != {POSE_DIM}")
    if not torch.isfinite(flat).all():
        raise ValueError(f"Non-finite pose values for {rec_id or 'sample'}")
    return flat


# Backward-compatible alias
flatten_hand_pose = flatten_bimanual_pose


def stack_camera_tensors(
    bird_frame: torch.Tensor,
    left_frame: torch.Tensor,
    right_frame: torch.Tensor,
) -> torch.Tensor:
    """Stack to [3, C, H, W] in CAMERA_ORDER (bird, left_wrist, right_wrist)."""
    return torch.stack([bird_frame, left_frame, right_frame], dim=0)



def camera_mask_tensor(embodiment: int) -> torch.Tensor:
    if int(embodiment) == EMBODIMENT_ROBOT:
        mask = list(ROBOT_CAMERA_MASK)
    elif int(embodiment) == EMBODIMENT_HUMAN:
        mask = list(HUMAN_CAMERA_MASK)
    else:
        raise ValueError(f"Unknown embodiment id {embodiment}")
    return torch.tensor(mask, dtype=torch.float32)



def build_run_metadata(
    *,
    robot_data_root: str | Path | None,
    human_data_root: str | Path | None,
    robot_sync_dir: str | Path | None,
    human_sync_dir: str | Path | None,
    num_queries: int,
    max_skew_s: float,
    robot_eef_dir: str | Path | None = None,
    action_chunk_starts_at_current: bool = ACTION_CHUNK_STARTS_AT_CURRENT,
    pose_loss_weight: float = 1.0,
    joint_loss_weight: float = 1.0,
    kl_weight: float = DEFAULT_KL_WEIGHT,
    reconstruction_loss: str = DEFAULT_RECON_LOSS,
    joint_modality_update: bool = True,
    hand_lambda: float = DEFAULT_HAND_LAMBDA,
    steps_per_epoch: int | None = None,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
) -> dict[str, Any]:
    return {
        "variant": "combined-relative-xyz-3cam",
        "pose_layout": "xyz only (6D); rot6d and gripper/open dropped at load",
        "num_cameras": NUM_CAMERAS,
        "front_camera_in_model": False,
        "front_camera_for_sync_only": True,
        "pose_action_space": "relative_to_chunk_anchor",
        "embodiment_cue": "modality routing (separate projs/heads/cams), no embedding token",
        "camera_order": list(CAMERA_ORDER),
        "model_camera_names": list(MODEL_CAMERA_NAMES),
        "pose_dim": POSE_DIM,
        "robot_joint_dim": ROBOT_JOINT_DIM,
        "robot_proprio_dim": ROBOT_PROPRIO_DIM,
        "human_proprio_dim": HUMAN_PROPRIO_DIM,
        "robot_camera_mask": list(ROBOT_CAMERA_MASK),
        "human_camera_mask": list(HUMAN_CAMERA_MASK),
        "num_queries": int(num_queries),
        "action_chunk_starts_at_current": bool(action_chunk_starts_at_current),
        "pose_action_relative_to_chunk_anchor": bool(POSE_ACTION_RELATIVE_TO_CHUNK_ANCHOR),
        "action_chunk_convention": (
            "pose_actions[k] = pose[t+k] - pose[t] (xyz delta vs first observation in the chunk); "
            "pose_actions[0] is zeros; at inference absolute_pose[k] = anchor_pose[t] + denorm(pred[k]). "
            "joint_actions[k] remain absolute joints at t+k (gripper joints continuous)."
        ),
        "robot_data_root": str(robot_data_root) if robot_data_root is not None else None,
        "human_data_root": str(human_data_root) if human_data_root is not None else None,
        "robot_sync_dir": str(robot_sync_dir) if robot_sync_dir is not None else None,
        "human_sync_dir": str(human_sync_dir) if human_sync_dir is not None else None,
        "robot_eef_dir": str(robot_eef_dir) if robot_eef_dir is not None else None,
        "robot_eef_file_format": "npz",
        "robot_eef_pose_dim": POSE_DIM,
        "robot_eef_sync_column": "eef_pose_index",
        "robot_eef_coord_frame": ROBOT_EEF_COORD_FRAME,
        "robot_eef_reldir": str(ROBOT_EEF_RELDIR),
        "human_pose_reldir": str(HUMAN_POSE_RELDIR),
        "gripper_semantics": (
            "Pose pathway: no gripper/open channel (xyz only). "
            "Robot gripper control is via continuous joint channels at indices "
            f"{list(GRIPPER_INDICES)} (joint NPYs assumed continuous; never binarized). "
            "Human open/close flags are unused for pose targets (may still appear in NPZ)."
        ),
        "robot_joint_grippers_continuous": True,
        "pose_includes_gripper": False,
        "max_skew_s": float(max_skew_s),
        "pose_loss_weight": float(pose_loss_weight),
        "joint_loss_weight": float(joint_loss_weight),
        "kl_weight": float(kl_weight),
        "hand_lambda": float(hand_lambda),
        "reconstruction_loss": str(reconstruction_loss),
        "joint_modality_update": bool(joint_modality_update),
        "num_epochs": int(num_epochs),
        "epoch_length": "max(len(robot_loader), len(human_loader)); shorter modality recycled",
        "steps_per_epoch": int(steps_per_epoch) if steps_per_epoch is not None else None,
        "batch_size": int(batch_size),
        "lr": float(lr),
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
    saved_joint_dim = metadata.get("robot_joint_dim", metadata.get("robot_state_dim", -1))
    if int(saved_joint_dim) != ROBOT_JOINT_DIM:
        raise ValueError(
            f"Saved robot_joint_dim {saved_joint_dim} does not match {ROBOT_JOINT_DIM}"
        )
    saved_pose_dim = metadata.get("pose_dim", -1)
    if int(saved_pose_dim) != POSE_DIM:
        raise ValueError(
            f"Saved pose_dim {saved_pose_dim} does not match {POSE_DIM} (xyz-only variant)"
        )
    if num_queries is not None and int(metadata.get("num_queries", -1)) != int(num_queries):
        raise ValueError(
            f"Saved num_queries {metadata.get('num_queries')} does not match CLI {num_queries}"
        )
