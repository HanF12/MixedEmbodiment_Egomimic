"""
Sync for mixed single-hand + single-robot-arm sessions.

Required streams (per demo):
  - bird camera timestamps
  - front camera timestamps
  - one wrist camera timestamps (robot side)
  - one arm joint timestamps (robot side)
  - hand-pose NPZ timestamps (human side slot)
  - robot EEF NPZ timestamps (robot side slot)

Frame validity uses xyz + gripper/open only (orientation / valid_rot ignored).
Only the active hand slot and active robot-arm slot must be valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from Combined.data_synchronization import (
    _assert_finite_1d,
    _sync_streams,
    _validate_index_column,
    make_unique_increasing_timeline,
    xyz_gripper_valid_mask,
)

Side = Literal["left", "right"]

# Session-name presets used under recording/sessions/
EMBODIMENT_PRESETS: dict[str, dict[str, Side]] = {
    "left_robot_right_hand": {"robot_side": "left", "hand_side": "right"},
    "right_robot_left_hand": {"robot_side": "right", "hand_side": "left"},
}

SLOT = {"left": 0, "right": 1}

MIXED_SYNC_INDEX_COLUMNS = (
    "bird_index",
    "front_index",
    "wrist_index",
    "joint_index",
    "hand_pose_index",
    "eef_pose_index",
)


def side_to_slot(side: Side) -> int:
    if side not in SLOT:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return SLOT[side]


def synchronize_mixed_hand_robot(
    bird_ts: np.ndarray,
    front_ts: np.ndarray,
    wrist_ts: np.ndarray,
    joint_ts: np.ndarray,
    hand_pose_ts: np.ndarray,
    eef_ts: np.ndarray,
    out_csv: str | Path,
    *,
    robot_side: Side,
    hand_side: Side,
    hand_valid_pos: np.ndarray,
    hand_valid_open: np.ndarray,
    eef_valid_pos: np.ndarray,
    eef_valid_open: np.ndarray,
    max_skew_s: float = 0.050,
    debug: bool = False,
    require_valid_active_slots: bool = True,
) -> pd.DataFrame:
    """
    Synchronize mixed one-hand + one-arm demos.

    Index columns:
      bird_index, front_index, wrist_index, joint_index,
      hand_pose_index, eef_pose_index

    hand_pose_index / eef_pose_index index the *original* NPZ timelines
    (duplicate timestamps are collapsed then remapped).
    """
    robot_slot = side_to_slot(robot_side)
    hand_slot = side_to_slot(hand_side)

    bird = _assert_finite_1d("bird_cam", bird_ts)
    front = _assert_finite_1d("front_cam", front_ts)
    wrist = _assert_finite_1d("wrist_cam", wrist_ts)
    joint = _assert_finite_1d("joint", joint_ts)
    hand_raw = _assert_finite_1d("hand_pose", hand_pose_ts)
    eef_raw = _assert_finite_1d("eef_pose", eef_ts)

    hand_unique, hand_src = make_unique_increasing_timeline(hand_raw)
    eef_unique, eef_src = make_unique_increasing_timeline(eef_raw)

    tmp_csv = Path(out_csv).with_suffix(".tmp.csv")
    df = _sync_streams(
        ["bird_cam", "front_cam", "wrist_cam", "joint", "hand_pose", "eef_pose"],
        [bird, front, wrist, joint, hand_unique, eef_unique],
        tmp_csv,
        index_columns=[
            "bird_index",
            "front_index",
            "wrist_index",
            "joint_index",
            "hand_pose_unique_index",
            "eef_pose_unique_index",
        ],
        max_skew_s=max_skew_s,
        debug=True,
        label=f"mixed({robot_side}-robot/{hand_side}-hand)",
        write_csv=True,
    )
    try:
        tmp_csv.unlink(missing_ok=True)
    except TypeError:
        if tmp_csv.exists():
            tmp_csv.unlink()

    n_before = len(df)
    if len(df) > 0:
        df["hand_pose_index"] = hand_src[df["hand_pose_unique_index"].to_numpy(dtype=np.int64)]
        df["eef_pose_index"] = eef_src[df["eef_pose_unique_index"].to_numpy(dtype=np.int64)]
    else:
        df["hand_pose_index"] = []
        df["eef_pose_index"] = []

    if require_valid_active_slots and len(df) > 0:
        hand_ok = xyz_gripper_valid_mask(
            valid_pos=hand_valid_pos,
            valid_open=hand_valid_open,
            n_frames=len(hand_raw),
            required_slots=(hand_slot,),
        )
        eef_ok = xyz_gripper_valid_mask(
            valid_pos=eef_valid_pos,
            valid_open=eef_valid_open,
            n_frames=len(eef_raw),
            required_slots=(robot_slot,),
        )
        h_idx = df["hand_pose_index"].to_numpy(dtype=np.int64)
        e_idx = df["eef_pose_index"].to_numpy(dtype=np.int64)
        row_ok = (
            (h_idx >= 0)
            & (h_idx < len(hand_ok))
            & (e_idx >= 0)
            & (e_idx < len(eef_ok))
        )
        keep = np.zeros(len(df), dtype=bool)
        keep[row_ok] = hand_ok[h_idx[row_ok]] & eef_ok[e_idx[row_ok]]
        df = df[keep].reset_index(drop=True)
        if len(df) > 0:
            df["master_index"] = np.arange(len(df), dtype=np.int64)
        print(
            f"  filtered inactive/invalid slots: kept {len(df)}/{n_before} "
            f"(hand_slot={hand_slot}, robot_slot={robot_slot}; orient ignored)"
        )

    _validate_index_column(df, "bird_index", len(bird), label="mixed sync")
    _validate_index_column(df, "front_index", len(front), label="mixed sync")
    _validate_index_column(df, "wrist_index", len(wrist), label="mixed sync")
    _validate_index_column(df, "joint_index", len(joint), label="mixed sync")
    _validate_index_column(df, "hand_pose_index", len(hand_raw), label="mixed sync")
    _validate_index_column(df, "eef_pose_index", len(eef_raw), label="mixed sync")

    keep_cols = ["master_index", *MIXED_SYNC_INDEX_COLUMNS]
    if debug:
        keep_cols = keep_cols + [
            c
            for c in df.columns
            if c.endswith("_time")
            or c == "time_diff"
            or c.endswith("_unique_index")
        ]
    # Ensure master_index exists even if filter emptied via unique path
    if "master_index" not in df.columns and len(df) > 0:
        df["master_index"] = np.arange(len(df), dtype=np.int64)
    elif "master_index" not in df.columns:
        df["master_index"] = []

    df = df[[c for c in keep_cols if c in df.columns]]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(
        f"Wrote mixed sync CSV -> {out_path} "
        f"(rows={len(df)}, robot={robot_side}, hand={hand_side}, debug={debug})"
    )
    return df
