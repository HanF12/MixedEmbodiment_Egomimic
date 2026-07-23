"""
Timestamp synchronization for Combined training.

1) synchronize_robot_bimanual  — joints + 4 cameras + EEF pose timeline
   (by default drops rows lacking xyz + gripper/open for both arms)
2) synchronize_human_hands    — bird + front cameras + hand-pose NPZ timeline
   (by default drops rows lacking xyz + gripper/open for both hands)

Orientation / valid_rot is never used for frame validity (training drops rot6d).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _assert_sorted(name: str, values: np.ndarray, *, strict: bool = True) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {values.shape}")
    if len(values) <= 1:
        return
    diffs = np.diff(values)
    ok = np.all(diffs > 0) if strict else np.all(diffs >= 0)
    if not ok:
        raise AssertionError(
            f"{name} timestamps are not {'strictly ' if strict else ''}non-decreasing "
            f"(min_diff={float(diffs.min())})"
        )


def _assert_finite_1d(name: str, values: np.ndarray) -> np.ndarray:
    ts = np.asarray(values, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        raise ValueError(f"{name} timestamps are empty")
    if not np.isfinite(ts).all():
        bad = int((~np.isfinite(ts)).sum())
        raise ValueError(f"{name} has {bad} non-finite timestamps")
    return ts


def make_unique_increasing_timeline(timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse duplicate timestamps (common in bag-rate pose NPZs).

    Returns:
      unique_ts: [M] strictly increasing timestamps
      src_index: [M] original index into the full pose array for each unique_ts
    """
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        return ts, np.zeros((0,), dtype=np.int64)
    unique_ts, first_idx = np.unique(ts, return_index=True)
    # np.unique sorts by value; for time keys that is chronological order.
    _assert_sorted("unique_pose_ts", unique_ts, strict=True)
    return unique_ts.astype(np.float64), first_idx.astype(np.int64)


def _sync_streams(
    stream_names: list[str],
    stream_ts: list[np.ndarray],
    out_csv: str | Path,
    *,
    index_columns: list[str],
    max_skew_s: float,
    debug: bool,
    label: str,
    write_csv: bool = True,
) -> pd.DataFrame:
    """Generic multi-stream nearest-window synchronizer (same algorithm as Bimanual)."""
    n = len(stream_ts)
    if n != len(stream_names) or n != len(index_columns):
        raise ValueError("stream_names, stream_ts, and index_columns must have the same length")

    for name, ts in zip(stream_names, stream_ts):
        _assert_sorted(name, ts, strict=True)

    idxs = [0] * n
    lengths = [len(ts) for ts in stream_ts]
    rows = []
    master = 0

    while all(i < L for i, L in zip(idxs, lengths)):
        pivot = max(float(stream_ts[s][idxs[s]]) for s in range(n))

        for s in range(n):
            while idxs[s] < lengths[s] and pivot - float(stream_ts[s][idxs[s]]) > max_skew_s:
                idxs[s] += 1

        if not all(i < L for i, L in zip(idxs, lengths)):
            break

        values = [float(stream_ts[s][idxs[s]]) for s in range(n)]
        t_min = min(values)
        t_max = max(values)

        if t_max - t_min <= max_skew_s:
            rows.append((master, *idxs, *values, t_max - t_min))
            master += 1
            for s in range(n):
                idxs[s] += 1
        else:
            earliest = int(np.argmin(values))
            idxs[earliest] += 1

    time_cols = [f"{c.replace('_index', '')}_time" for c in index_columns]
    all_cols = ["master_index", *index_columns, *time_cols, "time_diff"]
    df = pd.DataFrame(rows, columns=all_cols)
    if not debug:
        df = df[["master_index", *index_columns]]

    if write_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Synced {len(df)} {label} -> {out_path} (debug={debug})")
    return df


def xyz_gripper_valid_mask(
    *,
    valid_pos: np.ndarray | None = None,
    valid_open: np.ndarray | None = None,
    n_frames: int | None = None,
    required_slots: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Per-frame mask: True iff required slots have valid xyz + open/gripper.

    - ``required_slots`` defaults to both hands/arms ``(0, 1)``.
      Pass ``(0,)`` or ``(1,)`` for single-hand / single-arm demos.
    - Orientation / ``valid_rot`` is never consulted.
    - Each validity array is expected as [T, 2] (left=0, right=1).
    """
    arrays = [a for a in (valid_pos, valid_open) if a is not None]
    if not arrays:
        if n_frames is None:
            raise ValueError("Need at least one validity array or n_frames")
        return np.ones((int(n_frames),), dtype=bool)

    t = int(arrays[0].shape[0])
    if n_frames is not None and int(n_frames) != t:
        raise ValueError(f"n_frames={n_frames} != validity length {t}")

    slots = tuple(int(s) for s in (required_slots if required_slots is not None else (0, 1)))
    if not slots:
        raise ValueError("required_slots must be non-empty")
    for s in slots:
        if s not in (0, 1):
            raise ValueError(f"required_slots entries must be 0 or 1, got {slots}")

    ok = np.ones((t,), dtype=bool)
    for name, arr in (("valid_pos", valid_pos), ("valid_open", valid_open)):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=bool)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError(f"{name} must have shape [T, 2], got {a.shape}")
        if a.shape[0] != t:
            raise ValueError(f"{name} length {a.shape[0]} != {t}")
        ok &= a[:, list(slots)].all(axis=1)
    return ok


# Backward-compatible name used by older call sites
def full_hand_pose_valid_mask(
    *,
    valid_pos: np.ndarray | None = None,
    valid_rot: np.ndarray | None = None,
    valid_open: np.ndarray | None = None,
    n_frames: int | None = None,
    required_slots: Sequence[int] | None = None,
) -> np.ndarray:
    """Deprecated alias: ignores valid_rot; requires xyz + gripper only."""
    del valid_rot  # unused; orientation never gates validity
    return xyz_gripper_valid_mask(
        valid_pos=valid_pos,
        valid_open=valid_open,
        n_frames=n_frames,
        required_slots=required_slots,
    )


def _validate_index_column(df: pd.DataFrame, col: str, length: int, *, label: str) -> None:
    if col not in df.columns or len(df) == 0:
        return
    idx = df[col].to_numpy(dtype=np.int64)
    if (idx < 0).any() or (idx >= length).any():
        raise ValueError(
            f"{label}: {col} out of range for length={length} "
            f"(min={int(idx.min())}, max={int(idx.max())})"
        )


def synchronize_robot_bimanual(
    left_joint_ts: np.ndarray,
    right_joint_ts: np.ndarray,
    left_cam_ts: np.ndarray,
    right_cam_ts: np.ndarray,
    bird_ts: np.ndarray,
    front_ts: np.ndarray,
    out_csv: str | Path,
    *,
    eef_ts: np.ndarray | None = None,
    max_skew_s: float = 0.050,
    debug: bool = False,
    valid_pos: np.ndarray | None = None,
    valid_rot: np.ndarray | None = None,
    valid_open: np.ndarray | None = None,
    require_full_eef_pose: bool = True,
) -> pd.DataFrame:
    """
    Synchronize robot streams into one CSV (7 streams when EEF is provided).

    Index columns:
      left_joint_index, right_joint_index, left_index, right_index,
      bird_index, front_index, eef_pose_index

    When require_full_eef_pose is True, keep rows whose EEF index has valid
    xyz + gripper for both arms. Orientation / valid_rot is ignored.
    """
    del valid_rot  # orientation never gates sync validity
    left_j = _assert_finite_1d("left_joint", left_joint_ts)
    right_j = _assert_finite_1d("right_joint", right_joint_ts)
    left_c = _assert_finite_1d("left_cam", left_cam_ts)
    right_c = _assert_finite_1d("right_cam", right_cam_ts)
    bird = _assert_finite_1d("bird_cam", bird_ts)
    front = _assert_finite_1d("front_cam", front_ts)

    if eef_ts is None:
        raise ValueError(
            "eef_ts is required for Combined robot sync. "
            "Pass timestamps from joint-data/combined_npz_commonframe."
        )

    eef_raw = _assert_finite_1d("eef_pose", eef_ts)
    eef_unique, eef_src_index = make_unique_increasing_timeline(eef_raw)

    tmp_csv = Path(out_csv).with_suffix(".tmp.csv")
    df = _sync_streams(
        [
            "left_joint",
            "right_joint",
            "left_cam",
            "right_cam",
            "bird_cam",
            "front_cam",
            "eef_pose",
        ],
        [left_j, right_j, left_c, right_c, bird, front, eef_unique],
        tmp_csv,
        index_columns=[
            "left_joint_index",
            "right_joint_index",
            "left_index",
            "right_index",
            "bird_index",
            "front_index",
            "eef_pose_unique_index",
        ],
        max_skew_s=max_skew_s,
        debug=True,
        label="robot septuplets",
        write_csv=True,
    )
    try:
        tmp_csv.unlink(missing_ok=True)
    except TypeError:
        if tmp_csv.exists():
            tmp_csv.unlink()

    n_before = len(df)
    if len(df) > 0:
        uniq_idx = df["eef_pose_unique_index"].to_numpy(dtype=np.int64)
        df["eef_pose_index"] = eef_src_index[uniq_idx]
    else:
        df["eef_pose_index"] = []

    if require_full_eef_pose and len(df) > 0:
        missing = [name for name, arr in (("valid_pos", valid_pos), ("valid_open", valid_open)) if arr is None]
        if missing:
            raise ValueError(
                f"require_full_eef_pose=True but missing NPZ masks: {missing}. "
                "Pass valid_pos and valid_open from the EEF NPZ."
            )
        frame_ok = xyz_gripper_valid_mask(
            valid_pos=valid_pos,
            valid_open=valid_open,
            n_frames=len(eef_raw),
            required_slots=(0, 1),
        )
        eef_idx = df["eef_pose_index"].to_numpy(dtype=np.int64)
        in_range = (eef_idx >= 0) & (eef_idx < len(frame_ok))
        row_ok = np.zeros(len(df), dtype=bool)
        row_ok[in_range] = frame_ok[eef_idx[in_range]]
        df = df[row_ok].reset_index(drop=True)
        if len(df) > 0:
            df["master_index"] = np.arange(len(df), dtype=np.int64)
        print(
            f"  filtered incomplete EEF poses: kept {len(df)}/{n_before} "
            f"(dropped {n_before - len(df)}, require xyz+gripper for both arms; orient ignored)"
        )

    # Validate index ranges against original stream lengths.
    _validate_index_column(df, "left_joint_index", len(left_j), label="robot sync")
    _validate_index_column(df, "right_joint_index", len(right_j), label="robot sync")
    _validate_index_column(df, "left_index", len(left_c), label="robot sync")
    _validate_index_column(df, "right_index", len(right_c), label="robot sync")
    _validate_index_column(df, "bird_index", len(bird), label="robot sync")
    _validate_index_column(df, "front_index", len(front), label="robot sync")
    _validate_index_column(df, "eef_pose_index", len(eef_raw), label="robot sync")

    keep = [
        "master_index",
        "left_joint_index",
        "right_joint_index",
        "left_index",
        "right_index",
        "bird_index",
        "front_index",
        "eef_pose_index",
    ]
    if debug:
        keep = keep + [
            c
            for c in df.columns
            if c.endswith("_time") or c == "time_diff" or c == "eef_pose_unique_index"
        ]
    df = df[keep]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote robot sync CSV with eef_pose_index -> {out_path} (rows={len(df)}, debug={debug})")
    return df


def synchronize_human_hands(
    bird_ts: np.ndarray,
    front_ts: np.ndarray,
    pose_ts: np.ndarray,
    out_csv: str | Path,
    *,
    max_skew_s: float = 0.050,
    debug: bool = False,
    valid_pos: np.ndarray | None = None,
    valid_rot: np.ndarray | None = None,
    valid_open: np.ndarray | None = None,
    require_full_pose: bool = True,
) -> pd.DataFrame:
    """
    Synchronize human bird/front cameras with hand-pose NPZ timestamps.

    pose_ts may contain duplicates (bag-rate NPZs). We collapse to a unique
    increasing timeline and write pose_index into the *original* NPZ array.

    When require_full_pose is True (default), rows whose pose_index lacks
    valid xyz + open/close for both hands are dropped. Orientation / valid_rot
    is ignored. Pass valid_pos / valid_open from the NPZ.

    Index columns:
      bird_index, front_index, pose_index
    """
    del valid_rot  # orientation never gates sync validity
    bird = _assert_finite_1d("bird_cam", bird_ts)
    front = _assert_finite_1d("front_cam", front_ts)
    pose_raw = _assert_finite_1d("pose", pose_ts)
    pose_unique, pose_src_index = make_unique_increasing_timeline(pose_raw)

    tmp_csv = Path(out_csv).with_suffix(".tmp.csv")
    df = _sync_streams(
        ["bird_cam", "front_cam", "pose"],
        [bird, front, pose_unique],
        tmp_csv,
        index_columns=["bird_index", "front_index", "pose_unique_index"],
        max_skew_s=max_skew_s,
        debug=True,
        label="human triplets",
    )
    try:
        tmp_csv.unlink(missing_ok=True)
    except TypeError:
        if tmp_csv.exists():
            tmp_csv.unlink()

    n_before = len(df)
    if len(df) > 0:
        uniq_idx = df["pose_unique_index"].to_numpy(dtype=np.int64)
        df["pose_index"] = pose_src_index[uniq_idx]
    else:
        df["pose_index"] = []

    if require_full_pose and len(df) > 0:
        missing = [name for name, arr in (("valid_pos", valid_pos), ("valid_open", valid_open)) if arr is None]
        if missing:
            raise ValueError(
                f"require_full_pose=True but missing NPZ masks: {missing}. "
                "Pass valid_pos and valid_open, or set require_full_pose=False."
            )
        frame_ok = xyz_gripper_valid_mask(
            valid_pos=valid_pos,
            valid_open=valid_open,
            n_frames=len(pose_raw),
            required_slots=(0, 1),
        )
        pose_idx = df["pose_index"].to_numpy(dtype=np.int64)
        in_range = (pose_idx >= 0) & (pose_idx < len(frame_ok))
        row_ok = np.zeros(len(df), dtype=bool)
        row_ok[in_range] = frame_ok[pose_idx[in_range]]
        df = df[row_ok].reset_index(drop=True)
        if len(df) > 0:
            df["master_index"] = np.arange(len(df), dtype=np.int64)
        n_dropped = n_before - len(df)
        print(
            f"  filtered incomplete poses: kept {len(df)}/{n_before} "
            f"(dropped {n_dropped}, require xyz+open for both hands; orient ignored)"
        )

    _validate_index_column(df, "bird_index", len(bird), label="human sync")
    _validate_index_column(df, "front_index", len(front), label="human sync")
    _validate_index_column(df, "pose_index", len(pose_raw), label="human sync")

    keep = ["master_index", "bird_index", "front_index", "pose_index"]
    if debug:
        keep = keep + [
            c
            for c in df.columns
            if c.endswith("_time") or c == "time_diff" or c == "pose_unique_index"
        ]
    df = df[keep]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote human sync CSV with pose_index remap -> {out_path} (rows={len(df)}, debug={debug})")
    return df
