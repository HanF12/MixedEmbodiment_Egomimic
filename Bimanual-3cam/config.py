from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_NUM_QUERIES = 45
DEFAULT_NUM_EPOCHS = 40000
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 1e-4  # AdamW L2 regularization strength
DEFAULT_SAVE_EVERY_EPOCHS = 10000
# Do not write best / periodic epoch checkpoints until this epoch (1-indexed).
DEFAULT_SAVE_AFTER_EPOCHS = 30000

JOINT_DIM_PER_ARM = 7
STATE_DIM = 14

CAMERA_ORDER = ("left_wrist", "right_wrist", "bird")
MODEL_CAMERA_NAMES = ("cam0", "cam1", "cam2")
SYNC_INDEX_COLUMNS = (
    "left_joint_index",
    "right_joint_index",
    "left_index",
    "right_index",
    "bird_index",
)
JOINT_ORDER = ("left", "right")
LEFT_ARM_SLICE = slice(0, JOINT_DIM_PER_ARM)
RIGHT_ARM_SLICE = slice(JOINT_DIM_PER_ARM, STATE_DIM)
GRIPPER_INDICES = (JOINT_DIM_PER_ARM - 1, STATE_DIM - 1)
# Match Combined_relative_3cam: binarize joint grippers at load.
ROBOT_JOINT_GRIPPER_BINARIZE_THRESHOLD = 0.5


def default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_camera_names(camera_names: list[str] | tuple[str, ...]) -> None:
    if tuple(camera_names) != MODEL_CAMERA_NAMES:
        raise ValueError(
            f"Expected camera_names={MODEL_CAMERA_NAMES}, got {tuple(camera_names)}. "
            "The bimanual-3cam pipeline relies on a fixed camera order."
        )


def validate_state_dim(state_dim: int) -> None:
    if int(state_dim) != STATE_DIM:
        raise ValueError(f"Expected state_dim={STATE_DIM}, got {state_dim}")


def concat_bimanual_joints(
    left_step: np.ndarray | torch.Tensor,
    right_step: np.ndarray | torch.Tensor,
    *,
    rec_id: str = "",
    binarize_grippers: bool = True,
    gripper_threshold: float = ROBOT_JOINT_GRIPPER_BINARIZE_THRESHOLD,
) -> torch.Tensor:
    """
    Robot qpos: [7 left, 7 right] -> [14].

    By default, gripper joints (last channel per arm; flattened indices 6 and 13)
    are binarized identically to Combined_relative_3cam:
      grip -> 1.0 if grip >= threshold else 0.0
    """
    left_tensor = torch.as_tensor(left_step, dtype=torch.float32).reshape(-1)
    right_tensor = torch.as_tensor(right_step, dtype=torch.float32).reshape(-1)
    if left_tensor.numel() < JOINT_DIM_PER_ARM or right_tensor.numel() < JOINT_DIM_PER_ARM:
        raise ValueError(
            f"Expected at least {JOINT_DIM_PER_ARM} joints per arm for {rec_id or 'sample'}, "
            f"got left={left_tensor.numel()} right={right_tensor.numel()}"
        )
    out = torch.cat(
        [left_tensor[LEFT_ARM_SLICE], right_tensor[LEFT_ARM_SLICE]],
        dim=0,
    )
    if binarize_grippers:
        thr = float(gripper_threshold)
        for idx in GRIPPER_INDICES:
            out[idx] = 1.0 if float(out[idx]) >= thr else 0.0
    return out


def stack_camera_tensors(
    left_frame: torch.Tensor,
    right_frame: torch.Tensor,
    bird_frame: torch.Tensor,
) -> torch.Tensor:
    return torch.stack([left_frame, right_frame, bird_frame], dim=0)


def build_run_metadata(
    *,
    data_root: str | Path,
    sync_dir: str | Path,
    num_queries: int,
    max_skew_s: float,
) -> dict[str, Any]:
    return {
        "camera_order": list(CAMERA_ORDER),
        "model_camera_names": list(MODEL_CAMERA_NAMES),
        "joint_order": list(JOINT_ORDER),
        "joint_dim_per_arm": JOINT_DIM_PER_ARM,
        "state_dim": STATE_DIM,
        "num_queries": int(num_queries),
        "data_root": str(Path(data_root)),
        "sync_dir": str(Path(sync_dir)),
        "max_skew_s": float(max_skew_s),
        "variant": "bimanual-3cam",
        "robot_joint_gripper_binarize_threshold": float(ROBOT_JOINT_GRIPPER_BINARIZE_THRESHOLD),
        "gripper_semantics": (
            f"joint-state grippers (indices {GRIPPER_INDICES}) binarized at load with "
            f"threshold {ROBOT_JOINT_GRIPPER_BINARIZE_THRESHOLD}"
        ),
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
            "Saved model camera names do not match the fixed bimanual-3cam camera order"
        )
    if tuple(metadata.get("joint_order", [])) != JOINT_ORDER:
        raise ValueError(
            f"Saved joint_order {metadata.get('joint_order')} does not match expected {JOINT_ORDER}"
        )
    validate_state_dim(int(metadata.get("state_dim", STATE_DIM)))
    if num_queries is not None and int(metadata.get("num_queries", num_queries)) != int(num_queries):
        raise ValueError(
            f"Saved num_queries={metadata.get('num_queries')} does not match requested {num_queries}"
        )
