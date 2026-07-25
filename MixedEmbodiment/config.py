"""
MixedEmbodiment ACT constants and helpers (fork of Combined_relative).

Same relative-pose layout as Combined_relative, plus a third modality:
  mixed = one human hand + one robot arm.

Camera slot order is fixed as:
  [bird, front, left_wrist, right_wrist]  -> model cams cam0..cam3

Shared pose (human hands / robot EEF / mixed hand+EEF), EgoMimic-style shared head:
  8D = left 4 + right 4
  each side: xyz(3) + gripper/open(1)
  Source NPZ layout is still [2, 10] = xyz(3)+rot6d(6)+grip(1); rot is dropped at load.

Robot joints:
  14D = left 7 + right 7  (still absolute; mixed zeros the inactive arm)

Robot / mixed proprio: joints only [14] (robot_input_proj + robot CVAE)
Human proprio: absolute pose [8]
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
GRIPPER_INDICES = (JOINT_DIM_PER_ARM - 1, ROBOT_JOINT_DIM - 1)
ROBOT_STATE_DIM = ROBOT_JOINT_DIM  # proprio = joints only (EgoMimic)

# --- Raw NPZ pose layout (before dropping rotation) ---
RAW_HAND_DIM = 10  # xyz(3) + rot6d(6) + grip(1)
RAW_XYZ_SLICE = slice(0, 3)
RAW_GRIP_INDEX = 9

# --- Shared pose used in training (xyz + gripper only) ---
POSE_DIM_PER_SIDE = 4  # xyz(3) + grip(1)
POSE_DIM = 8  # left 4 + right 4
# Gripper dims in flattened [8] pose: left grip, right grip
POSE_GRIP_INDICES = (POSE_DIM_PER_SIDE - 1, POSE_DIM - 1)  # (3, 7)
# Robot EEF NPZ grippers only (not joint-state grippers; not human pose)
ROBOT_EEF_GRIPPER_BINARIZE_THRESHOLD = 0.8
HUMAN_STATE_DIM = POSE_DIM
HUMAN_PROPRIO_DIM = POSE_DIM
ROBOT_PROPRIO_DIM = ROBOT_JOINT_DIM  # EgoMimic: joints only

# --- Shared camera layout ---
CAMERA_ORDER = ("bird", "front", "left_wrist", "right_wrist")
MODEL_CAMERA_NAMES = ("cam0", "cam1", "cam2", "cam3")
NUM_CAMERAS = 4
FRONT_CAMERA_INDEX = CAMERA_ORDER.index("front")  # 1

ROBOT_CAMERA_MASK = (1, 1, 1, 1)
HUMAN_CAMERA_MASK = (1, 1, 0, 0)
# Mixed: bird+front+active wrist; inactive wrist masked in camera_mask_tensor.
MIXED_CAMERA_MASK_LEFT_ROBOT = (1, 1, 1, 0)   # left wrist live
MIXED_CAMERA_MASK_RIGHT_ROBOT = (1, 1, 0, 1)  # right wrist live

EMBODIMENT_ROBOT = 0
EMBODIMENT_HUMAN = 1
EMBODIMENT_MIXED = 2
EMBODIMENT_NAMES = ("robot", "human", "mixed")

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

MIXED_SYNC_INDEX_COLUMNS = (
    "bird_index",
    "front_index",
    "wrist_index",
    "joint_index",
    "hand_pose_index",
    "eef_pose_index",
)

MIXED_TEMP_CUT_INDEX_COLUMNS = (
    "bird_index",
    "front_index",
    "wrist_index",
    "joint_index",
)

ROBOT_EEF_RELDIR = Path("joint-data") / "combined_npz_commonframe"
ROBOT_EEF_COORD_FRAME = (
    "arm_fk_targetframe_commonframe: left/right arm-base FK poses "
    "transformed into a shared midline frame. "
    "Training uses xyz+gripper only (rotation dropped)."
)

# Default pose NPZ layouts under each modality root (0714 layout).
HUMAN_POSE_RELDIR = Path("bird-realsense-data") / "combined_npz_targetframe"
DEFAULT_ROBOT_DATA_ROOT = Path("Combined") / "teleop_bimanual" / "0714"
DEFAULT_HUMAN_DATA_ROOT = Path("Combined") / "human_hands_bimanual_raw" / "0714"

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
    """Robot qpos: [7 left, 7 right] -> [14]."""
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
    Extract xyz+gripper from one bimanual pose timestep and flatten.

    Expected input shape: [2, 10] (raw NPZ layout).
    Output shape: [8] = [left_xyz(3), left_grip(1), right_xyz(3), right_grip(1)].
    Rotation (indices 3:9) is discarded.
    """
    pose = torch.as_tensor(pose_t, dtype=torch.float32)
    if pose.shape != (2, RAW_HAND_DIM):
        raise ValueError(
            f"Expected pose timestep shape (2, {RAW_HAND_DIM}) for {rec_id or 'sample'}, "
            f"got {tuple(pose.shape)}"
        )
    left = torch.cat([pose[0, RAW_XYZ_SLICE], pose[0, RAW_GRIP_INDEX : RAW_GRIP_INDEX + 1]], dim=0)
    right = torch.cat([pose[1, RAW_XYZ_SLICE], pose[1, RAW_GRIP_INDEX : RAW_GRIP_INDEX + 1]], dim=0)
    flat = torch.cat([left, right], dim=0)
    if flat.numel() != POSE_DIM:
        raise ValueError(f"Internal pose flatten error: got {flat.numel()} != {POSE_DIM}")
    if not torch.isfinite(flat).all():
        raise ValueError(f"Non-finite pose values for {rec_id or 'sample'}")
    return flat


def binarize_flat_pose_grippers(
    flat_pose: torch.Tensor | np.ndarray,
    *,
    threshold: float = ROBOT_EEF_GRIPPER_BINARIZE_THRESHOLD,
) -> torch.Tensor:
    """
    Binarize left/right gripper dims in a flattened [8] pose.

    grip -> 1.0 if grip >= threshold else 0.0. XYZ dims unchanged.
    """
    flat = torch.as_tensor(flat_pose, dtype=torch.float32).clone()
    if flat.numel() != POSE_DIM:
        raise ValueError(f"Expected flat pose dim {POSE_DIM}, got {flat.numel()}")
    thr = float(threshold)
    for idx in POSE_GRIP_INDICES:
        flat[idx] = 1.0 if float(flat[idx]) >= thr else 0.0
    return flat


# Backward-compatible alias
flatten_hand_pose = flatten_bimanual_pose


def stack_camera_tensors(
    bird_frame: torch.Tensor,
    front_frame: torch.Tensor,
    left_frame: torch.Tensor,
    right_frame: torch.Tensor,
) -> torch.Tensor:
    """Stack to [4, C, H, W] in CAMERA_ORDER."""
    return torch.stack([bird_frame, front_frame, left_frame, right_frame], dim=0)


def extract_side_pose4(pose_t: np.ndarray | torch.Tensor, slot: int, *, rec_id: str = "") -> torch.Tensor:
    """Extract xyz+gripper [4] for one side from raw [2, 10] pose."""
    pose = torch.as_tensor(pose_t, dtype=torch.float32)
    if pose.shape != (2, RAW_HAND_DIM):
        raise ValueError(
            f"Expected pose timestep shape (2, {RAW_HAND_DIM}) for {rec_id or 'sample'}, "
            f"got {tuple(pose.shape)}"
        )
    if int(slot) not in (0, 1):
        raise ValueError(f"slot must be 0 or 1, got {slot}")
    out = torch.cat(
        [pose[int(slot), RAW_XYZ_SLICE], pose[int(slot), RAW_GRIP_INDEX : RAW_GRIP_INDEX + 1]],
        dim=0,
    )
    if not torch.isfinite(out).all():
        raise ValueError(f"Non-finite side pose (slot={slot}) for {rec_id or 'sample'}")
    return out


def compose_mixed_flat_pose(
    hand_t: np.ndarray | torch.Tensor,
    eef_t: np.ndarray | torch.Tensor,
    *,
    hand_slot: int,
    robot_slot: int,
    binarize_eef_gripper: bool = True,
    eef_gripper_threshold: float = ROBOT_EEF_GRIPPER_BINARIZE_THRESHOLD,
    rec_id: str = "",
) -> torch.Tensor:
    """
    Build [8] pose = hand xyz+open in hand_slot + robot EEF xyz+grip in robot_slot.
    """
    if int(hand_slot) == int(robot_slot):
        raise ValueError(f"hand_slot and robot_slot must differ, got {hand_slot}")
    hand4 = extract_side_pose4(hand_t, int(hand_slot), rec_id=rec_id)
    eef4 = extract_side_pose4(eef_t, int(robot_slot), rec_id=rec_id).clone()
    if binarize_eef_gripper:
        eef4[3] = 1.0 if float(eef4[3]) >= float(eef_gripper_threshold) else 0.0
    flat = torch.zeros(POSE_DIM, dtype=torch.float32)
    hs = int(hand_slot) * POSE_DIM_PER_SIDE
    rs = int(robot_slot) * POSE_DIM_PER_SIDE
    flat[hs : hs + POSE_DIM_PER_SIDE] = hand4
    flat[rs : rs + POSE_DIM_PER_SIDE] = eef4
    return flat


def single_arm_joint_vector(
    arm_qpos: np.ndarray | torch.Tensor,
    *,
    robot_side: str,
    rec_id: str = "",
) -> torch.Tensor:
    """Pack one arm's 7 joints into a [14] vector; inactive arm is zeros."""
    arm = torch.as_tensor(arm_qpos, dtype=torch.float32).reshape(-1)
    if arm.numel() < JOINT_DIM_PER_ARM:
        raise ValueError(
            f"Expected >= {JOINT_DIM_PER_ARM} joints for {rec_id or 'sample'}, got {arm.numel()}"
        )
    out = torch.zeros(ROBOT_JOINT_DIM, dtype=torch.float32)
    if robot_side == "left":
        out[LEFT_ARM_SLICE] = arm[:JOINT_DIM_PER_ARM]
    elif robot_side == "right":
        out[RIGHT_ARM_SLICE] = arm[:JOINT_DIM_PER_ARM]
    else:
        raise ValueError(f"robot_side must be 'left' or 'right', got {robot_side!r}")
    return out


def joint_loss_mask_for_side(robot_side: str | None = None, *, full: bool = False) -> torch.Tensor:
    """Float mask [14]; 1 = supervise that joint dim."""
    if full or robot_side is None:
        return torch.ones(ROBOT_JOINT_DIM, dtype=torch.float32)
    mask = torch.zeros(ROBOT_JOINT_DIM, dtype=torch.float32)
    if robot_side == "left":
        mask[LEFT_ARM_SLICE] = 1.0
    elif robot_side == "right":
        mask[RIGHT_ARM_SLICE] = 1.0
    else:
        raise ValueError(f"robot_side must be 'left' or 'right', got {robot_side!r}")
    return mask


def camera_mask_tensor(
    embodiment: int,
    *,
    disable_front: bool = False,
    robot_side: str | None = None,
) -> torch.Tensor:
    if int(embodiment) == EMBODIMENT_ROBOT:
        mask = list(ROBOT_CAMERA_MASK)
    elif int(embodiment) == EMBODIMENT_HUMAN:
        mask = list(HUMAN_CAMERA_MASK)
    elif int(embodiment) == EMBODIMENT_MIXED:
        if robot_side == "left":
            mask = list(MIXED_CAMERA_MASK_LEFT_ROBOT)
        elif robot_side == "right":
            mask = list(MIXED_CAMERA_MASK_RIGHT_ROBOT)
        else:
            raise ValueError("EMBODIMENT_MIXED requires robot_side='left' or 'right'")
    else:
        raise ValueError(f"Unknown embodiment id {embodiment}")
    if disable_front:
        mask[FRONT_CAMERA_INDEX] = 0
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
    mixed_data_root: str | Path | None = None,
    mixed_sync_dir: str | Path | None = None,
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
    disable_front_camera: bool = False,
) -> dict[str, Any]:
    robot_mask = list(ROBOT_CAMERA_MASK)
    human_mask = list(HUMAN_CAMERA_MASK)
    mixed_left = list(MIXED_CAMERA_MASK_LEFT_ROBOT)
    mixed_right = list(MIXED_CAMERA_MASK_RIGHT_ROBOT)
    if disable_front_camera:
        robot_mask[FRONT_CAMERA_INDEX] = 0
        human_mask[FRONT_CAMERA_INDEX] = 0
        mixed_left[FRONT_CAMERA_INDEX] = 0
        mixed_right[FRONT_CAMERA_INDEX] = 0
    return {
        "variant": "mixed-embodiment-relative-egomimic-style",
        "pose_layout": "xyz+gripper only (8D); rot6d dropped at load",
        "pose_action_space": "relative_to_chunk_anchor",
        "embodiment_cue": "modality routing (separate projs/heads/cams), no embedding token",
        "modalities": ["robot", "human", "mixed"],
        "mixed_policy": (
            "third modality; proprio/CVAE use robot joint path; "
            "pose [8]=hand slot + robot EEF slot; joint loss masked to active arm; "
            "camera_mask keeps bird/front/active wrist"
        ),
        "camera_order": list(CAMERA_ORDER),
        "model_camera_names": list(MODEL_CAMERA_NAMES),
        "pose_dim": POSE_DIM,
        "robot_joint_dim": ROBOT_JOINT_DIM,
        "robot_proprio_dim": ROBOT_PROPRIO_DIM,
        "human_proprio_dim": HUMAN_PROPRIO_DIM,
        "disable_front_camera": bool(disable_front_camera),
        "robot_camera_mask": robot_mask,
        "human_camera_mask": human_mask,
        "mixed_camera_mask_left_robot": mixed_left,
        "mixed_camera_mask_right_robot": mixed_right,
        "num_queries": int(num_queries),
        "action_chunk_starts_at_current": bool(action_chunk_starts_at_current),
        "pose_action_relative_to_chunk_anchor": bool(POSE_ACTION_RELATIVE_TO_CHUNK_ANCHOR),
        "action_chunk_convention": (
            "pose_actions[k] = pose[t+k] - pose[t] (delta vs first observation in the chunk); "
            "pose_actions[0] is zeros; at inference absolute_pose[k] = anchor_pose[t] + denorm(pred[k]). "
            "joint_actions[k] remain absolute joints at t+k."
        ),
        "robot_data_root": str(robot_data_root) if robot_data_root is not None else None,
        "human_data_root": str(human_data_root) if human_data_root is not None else None,
        "mixed_data_root": str(mixed_data_root) if mixed_data_root is not None else None,
        "robot_sync_dir": str(robot_sync_dir) if robot_sync_dir is not None else None,
        "human_sync_dir": str(human_sync_dir) if human_sync_dir is not None else None,
        "mixed_sync_dir": str(mixed_sync_dir) if mixed_sync_dir is not None else None,
        "robot_eef_dir": str(robot_eef_dir) if robot_eef_dir is not None else None,
        "robot_eef_file_format": "npz",
        "robot_eef_pose_dim": POSE_DIM,
        "robot_eef_sync_column": "eef_pose_index",
        "robot_eef_coord_frame": ROBOT_EEF_COORD_FRAME,
        "robot_eef_reldir": str(ROBOT_EEF_RELDIR),
        "human_pose_reldir": str(HUMAN_POSE_RELDIR),
        "gripper_semantics": (
            "shared last per-side dim: human open/close (typically binary); "
            f"robot EEF NPZ gripper binarized at load with threshold "
            f"{ROBOT_EEF_GRIPPER_BINARIZE_THRESHOLD} (joint-state grippers unchanged); "
            "relative pose deltas include gripper dims"
        ),
        "robot_eef_gripper_binarize_threshold": float(ROBOT_EEF_GRIPPER_BINARIZE_THRESHOLD),
        "max_skew_s": float(max_skew_s),
        "pose_loss_weight": float(pose_loss_weight),
        "joint_loss_weight": float(joint_loss_weight),
        "kl_weight": float(kl_weight),
        "hand_lambda": float(hand_lambda),
        "reconstruction_loss": str(reconstruction_loss),
        "joint_modality_update": bool(joint_modality_update),
        "num_epochs": int(num_epochs),
        "epoch_length": (
            "max(len(robot_loader), len(human_loader), len(mixed_loader)); "
            "shorter modalities recycled; loss = mean over available modalities"
        ),
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
    if num_queries is not None and int(metadata.get("num_queries", -1)) != int(num_queries):
        raise ValueError(
            f"Saved num_queries {metadata.get('num_queries')} does not match CLI {num_queries}"
        )
