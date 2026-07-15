from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _assert_sorted(name: str, values: np.ndarray) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {values.shape}")
    if len(values) > 1 and not np.all(np.diff(values) > 0):
        raise AssertionError(f"{name} timestamps are not strictly increasing")


def synchronize_bimanual_with_front(
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
    """Synchronize two joint streams and four camera streams into one CSV."""
    _assert_sorted("left_joint", left_joint_ts)
    _assert_sorted("right_joint", right_joint_ts)
    _assert_sorted("left_cam", left_cam_ts)
    _assert_sorted("right_cam", right_cam_ts)
    _assert_sorted("bird_cam", bird_ts)
    _assert_sorted("front_cam", front_ts)

    rows = []
    li = ri = lc = rc = bc = fc = master = 0
    lengths = (
        len(left_joint_ts),
        len(right_joint_ts),
        len(left_cam_ts),
        len(right_cam_ts),
        len(bird_ts),
        len(front_ts),
    )

    while all(idx < size for idx, size in zip((li, ri, lc, rc, bc, fc), lengths)):
        pivot = max(
            left_joint_ts[li],
            right_joint_ts[ri],
            left_cam_ts[lc],
            right_cam_ts[rc],
            bird_ts[bc],
            front_ts[fc],
        )

        while li < lengths[0] and pivot - left_joint_ts[li] > max_skew_s:
            li += 1
        while ri < lengths[1] and pivot - right_joint_ts[ri] > max_skew_s:
            ri += 1
        while lc < lengths[2] and pivot - left_cam_ts[lc] > max_skew_s:
            lc += 1
        while rc < lengths[3] and pivot - right_cam_ts[rc] > max_skew_s:
            rc += 1
        while bc < lengths[4] and pivot - bird_ts[bc] > max_skew_s:
            bc += 1
        while fc < lengths[5] and pivot - front_ts[fc] > max_skew_s:
            fc += 1

        if not all(idx < size for idx, size in zip((li, ri, lc, rc, bc, fc), lengths)):
            break

        values = (
            left_joint_ts[li],
            right_joint_ts[ri],
            left_cam_ts[lc],
            right_cam_ts[rc],
            bird_ts[bc],
            front_ts[fc],
        )
        t_min = min(values)
        t_max = max(values)

        if t_max - t_min <= max_skew_s:
            rows.append(
                (
                    master,
                    li,
                    ri,
                    lc,
                    rc,
                    bc,
                    fc,
                    *values,
                    t_max - t_min,
                )
            )
            master += 1
            li += 1
            ri += 1
            lc += 1
            rc += 1
            bc += 1
            fc += 1
        else:
            earliest = int(np.argmin(values))
            if earliest == 0:
                li += 1
            elif earliest == 1:
                ri += 1
            elif earliest == 2:
                lc += 1
            elif earliest == 3:
                rc += 1
            elif earliest == 4:
                bc += 1
            else:
                fc += 1

    all_cols = [
        "master_index",
        "left_joint_index",
        "right_joint_index",
        "left_index",
        "right_index",
        "bird_index",
        "front_index",
        "left_joint_time",
        "right_joint_time",
        "left_time",
        "right_time",
        "bird_time",
        "front_time",
        "time_diff",
    ]
    df = pd.DataFrame(rows, columns=all_cols)
    if not debug:
        df = df[
            [
                "master_index",
                "left_joint_index",
                "right_joint_index",
                "left_index",
                "right_index",
                "bird_index",
                "front_index",
            ]
        ]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Synced {len(df)} sextuplets -> {out_path} (debug={debug})")
    return df
