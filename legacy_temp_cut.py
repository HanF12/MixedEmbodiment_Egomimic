"""Legacy leading-frame trim (temp_cut), disabled by default in dataloaders.

Previously dataloaders used temp_cut=10 to:
  1) drop sync rows where video/joint index columns are < temp_cut
  2) subtract temp_cut from those columns
  3) slice video/joint arrays with [temp_cut:]

Pose / EEF / hand indices were NEVER trimmed (absolute into full NPZs).

As of the no-trim default, dataloaders do not call this. Keep this file only
as a reference / opt-in helper if you intentionally want the old behavior:

    from legacy_temp_cut import apply_leading_trim, LEGACY_DEFAULT_TEMP_CUT
    df, arrays = apply_leading_trim(df, cut_columns, arrays, temp_cut=LEGACY_DEFAULT_TEMP_CUT)
"""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

LEGACY_DEFAULT_TEMP_CUT = 10

# Typical cut columns (pose/eef/hand excluded on purpose).
ROBOT_CUT_COLUMNS = (
    "left_joint_index",
    "right_joint_index",
    "left_index",
    "right_index",
    "bird_index",
    "front_index",
)
HUMAN_CUT_COLUMNS = (
    "bird_index",
    "front_index",
)
MIXED_CUT_COLUMNS = (
    "bird_index",
    "front_index",
    "wrist_index",
    "joint_index",
)
BIMANUAL_3CAM_CUT_COLUMNS = (
    "left_joint_index",
    "right_joint_index",
    "left_index",
    "right_index",
    "bird_index",
)


def apply_leading_trim(
    df: pd.DataFrame,
    cut_columns: Sequence[str],
    arrays: Mapping[str, np.ndarray] | None = None,
    temp_cut: int = LEGACY_DEFAULT_TEMP_CUT,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Apply the old leading trim to a sync dataframe and optional arrays.

    Args:
        df: sync CSV rows with absolute stream indices.
        cut_columns: video/joint index columns to filter + remap.
        arrays: optional name -> array to slice with [temp_cut:].
        temp_cut: leading frames to drop (legacy default was 10).

    Returns:
        (trimmed_df, sliced_arrays). Pose/EEF columns in df are left unchanged.
    """
    cut = int(temp_cut)
    out_arrays: dict[str, np.ndarray] = {}
    if cut <= 0:
        if arrays:
            out_arrays = {k: v for k, v in arrays.items()}
        return df.reset_index(drop=True), out_arrays

    use = [c for c in cut_columns if c in df.columns]
    mask = np.ones(len(df), dtype=bool)
    for col in use:
        mask &= df[col].to_numpy() >= cut
    out = df.loc[mask].copy()
    for col in use:
        out[col] = out[col] - cut
    out = out.reset_index(drop=True)

    if arrays:
        for name, arr in arrays.items():
            out_arrays[name] = arr[cut:]
    return out, out_arrays
