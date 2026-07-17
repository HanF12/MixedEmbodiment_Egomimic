"""
Timestamp synchronization for Combined training.

1) synchronize_robot_bimanual  — robot joints + 4 cameras (same idea as Bimanual)
2) synchronize_human_hands    — bird + front cameras + hand-pose NPZ timeline
"""

from __future__ import annotations

from pathlib import Path

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

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Synced {len(df)} {label} -> {out_path} (debug={debug})")
    return df


def synchronize_robot_bimanual(
    left_joint_ts: np.ndarray,
    right_joint_ts: np.ndarray,
    left_cam_ts: np.ndarray,
    right_cam_ts: np.ndarray,
    bird_ts: np.ndarray,
    front_ts: np.ndarray,
    out_csv: str | Path,
    *,
    max_skew_s: float = 0.050,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Synchronize robot streams into one CSV.

    Index columns:
      left_joint_index, right_joint_index, left_index, right_index, bird_index, front_index
    """
    return _sync_streams(
        [
            "left_joint",
            "right_joint",
            "left_cam",
            "right_cam",
            "bird_cam",
            "front_cam",
        ],
        [
            np.asarray(left_joint_ts, dtype=np.float64).reshape(-1),
            np.asarray(right_joint_ts, dtype=np.float64).reshape(-1),
            np.asarray(left_cam_ts, dtype=np.float64).reshape(-1),
            np.asarray(right_cam_ts, dtype=np.float64).reshape(-1),
            np.asarray(bird_ts, dtype=np.float64).reshape(-1),
            np.asarray(front_ts, dtype=np.float64).reshape(-1),
        ],
        out_csv,
        index_columns=[
            "left_joint_index",
            "right_joint_index",
            "left_index",
            "right_index",
            "bird_index",
            "front_index",
        ],
        max_skew_s=max_skew_s,
        debug=debug,
        label="robot sextuplets",
    )


def synchronize_human_hands(
    bird_ts: np.ndarray,
    front_ts: np.ndarray,
    pose_ts: np.ndarray,
    out_csv: str | Path,
    *,
    max_skew_s: float = 0.050,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Synchronize human bird/front cameras with hand-pose NPZ timestamps.

    pose_ts may contain duplicates (bag-rate NPZs). We collapse to a unique
    increasing timeline and write pose_index into the *original* NPZ array.

    Index columns:
      bird_index, front_index, pose_index
    """
    bird = np.asarray(bird_ts, dtype=np.float64).reshape(-1)
    front = np.asarray(front_ts, dtype=np.float64).reshape(-1)
    pose_raw = np.asarray(pose_ts, dtype=np.float64).reshape(-1)
    pose_unique, pose_src_index = make_unique_increasing_timeline(pose_raw)

    # Write to a temp path inside _sync_streams, then remap pose indices and overwrite.
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
        # Python <3.8 compatibility (unlikely here)
        if tmp_csv.exists():
            tmp_csv.unlink()

    # Remap unique pose index -> original NPZ row index.
    if len(df) > 0:
        uniq_idx = df["pose_unique_index"].to_numpy(dtype=np.int64)
        df["pose_index"] = pose_src_index[uniq_idx]
    else:
        df["pose_index"] = []

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
