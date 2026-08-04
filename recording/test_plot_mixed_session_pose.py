#!/usr/bin/env python3
"""
Plot mixed embodiment demos: bird hand pose + robot FK in ONE figure.

Matches demos by stamp within a session folder pair, e.g.:
  left_robot_right_hand  -> keep RIGHT hand (bird) + LEFT robot (FK)
  right_robot_left_hand  -> keep LEFT hand (bird) + RIGHT robot (FK)

Timelines are aligned on absolute timestamps (overlap only), resampling both
sources onto a shared grid, then plotted with test_plot_combined_pose.plot_pose.

Example:
  python recording/test_plot_mixed_session_pose.py \\
    --session recording/sessions/left_robot_right_hand/0729

  python recording/test_plot_mixed_session_pose.py \\
    --session recording/sessions/left_robot_right_hand/0729 \\
    --session recording/sessions/right_robot_left_hand/0729
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plot_combined_pose import plot_pose  # noqa: E402

STAMP_RE = re.compile(r"(20\d{12})")


def _stamp_map(dir_path: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not dir_path.exists():
        return out
    for p in sorted(dir_path.glob("*.npz")):
        m = STAMP_RE.search(p.name)
        if m:
            out[m.group(1)] = p
    return out


def _infer_keep_slots(session_name: str) -> Tuple[int, int]:
    """
    Returns (hand_slot_from_bird, robot_slot_from_fk) to KEEP.
    """
    name = session_name.lower()
    if "left_robot_right_hand" in name:
        return 1, 0  # right hand, left robot
    if "right_robot_left_hand" in name:
        return 0, 1  # left hand, right robot
    raise ValueError(
        f"Cannot infer slots from session name {session_name!r}. "
        "Expected left_robot_right_hand or right_robot_left_hand."
    )


def _resample_slot(
    ts: np.ndarray,
    pose: np.ndarray,
    valid: np.ndarray,
    slot: int,
    t_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample one slot onto t_grid.
    Returns pose_slot (G,10), valid_slot (G,), open_slot (G,).
    Uses linear interp on xyz + rot6d where neighbors are valid; open is nearest.
    """
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    pose = np.asarray(pose, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    G = t_grid.shape[0]
    out_pose = np.full((G, 10), np.nan, dtype=np.float64)
    out_valid = np.zeros((G,), dtype=bool)

    m = valid[:, slot] & np.isfinite(pose[:, slot, 0:3]).all(axis=-1)
    if int(m.sum()) < 2:
        return out_pose, out_valid, out_pose[:, 9]

    t_src = ts[m]
    p_src = pose[m, slot]  # (N,10)

    # Only interpolate inside source time span.
    in_span = (t_grid >= t_src[0]) & (t_grid <= t_src[-1])
    if not np.any(in_span):
        return out_pose, out_valid, out_pose[:, 9]

    # xyz + rot6d continuous channels
    for c in range(9):
        out_pose[in_span, c] = np.interp(t_grid[in_span], t_src, p_src[:, c])

    # open flag: nearest neighbor
    idx = np.searchsorted(t_src, t_grid[in_span], side="left")
    idx = np.clip(idx, 0, t_src.size - 1)
    # pick closer of idx-1 / idx
    idx0 = np.clip(idx - 1, 0, t_src.size - 1)
    choose1 = np.abs(t_src[idx] - t_grid[in_span]) <= np.abs(t_src[idx0] - t_grid[in_span])
    nn = np.where(choose1, idx, idx0)
    out_pose[in_span, 9] = p_src[nn, 9]
    out_valid[in_span] = True
    return out_pose, out_valid, out_pose[:, 9]


def merge_hand_robot(
    bird_path: Path,
    robot_path: Path,
    *,
    hand_slot: int,
    robot_slot: int,
    dt: float = 0.033,
) -> Dict[str, np.ndarray]:
    """
    Merge into bimanual schema:
      slot 0 = left, slot 1 = right
    using hand_slot from bird and robot_slot from FK.
    """
    zb = np.load(bird_path, allow_pickle=True)
    zr = np.load(robot_path, allow_pickle=True)

    tb = np.asarray(zb["timestamps"], dtype=np.float64)
    tr = np.asarray(zr["timestamps"], dtype=np.float64)
    pb = np.asarray(zb["pose"], dtype=np.float64)
    pr = np.asarray(zr["pose"], dtype=np.float64)
    vb = np.asarray(zb["valid_pos"], dtype=bool) if "valid_pos" in zb.files else np.isfinite(pb[..., 0])
    vr = np.asarray(zr["valid_pos"], dtype=bool) if "valid_pos" in zr.files else np.isfinite(pr[..., 0])

    t0 = max(float(tb[0]), float(tr[0]))
    t1 = min(float(tb[-1]), float(tr[-1]))
    if t1 <= t0:
        raise ValueError(f"No time overlap between {bird_path.name} and {robot_path.name}")

    t_grid = np.arange(t0, t1 + 0.5 * dt, dt, dtype=np.float64)
    G = t_grid.shape[0]
    pose = np.full((G, 2, 10), np.nan, dtype=np.float64)
    valid = np.zeros((G, 2), dtype=bool)

    # Hand -> its natural slot (0 left / 1 right)
    hand_pose, hand_valid, _ = _resample_slot(tb, pb, vb, hand_slot, t_grid)
    pose[:, hand_slot] = hand_pose
    valid[:, hand_slot] = hand_valid

    # Robot -> its natural slot
    robot_pose, robot_valid, _ = _resample_slot(tr, pr, vr, robot_slot, t_grid)
    pose[:, robot_slot] = robot_pose
    valid[:, robot_slot] = robot_valid

    # Also fill R_raw-like fields for schema completeness if plotter only needs pose
    return {
        "timestamps": t_grid,
        "pose": pose,
        "valid_pos": valid,
        "valid_rot": valid.copy(),
        "valid_open": valid.copy(),
        "pose_xyz_raw": pose[:, :, 0:3].copy(),
        "valid_pos_raw": valid.copy(),
        "R_raw": np.full((G, 2, 3, 3), np.nan, dtype=np.float64),
        "valid_rot_raw": valid.copy(),
        "open_score_raw": pose[:, :, 9].copy(),
        "open_score_filled": pose[:, :, 9].copy(),
        "open_score_valid": valid.copy(),
        "open_threshold": np.array(0.0, dtype=np.float64),
        "pose_timeline": np.array("merged"),
    }


def plot_session(session_dir: Path, *, dt: float, show: bool) -> None:
    session_dir = Path(session_dir)
    bird_dir = session_dir / "bird-realsense-data" / "combined_npz_targetframe"
    robot_dir = session_dir / "joint-data" / "combined_npz_commonframe"
    if not bird_dir.exists() or not robot_dir.exists():
        raise FileNotFoundError(f"Missing bird/robot dirs under {session_dir}")

    hand_slot, robot_slot = _infer_keep_slots(session_dir.as_posix())
    bird = _stamp_map(bird_dir)
    robot = _stamp_map(robot_dir)
    stamps = sorted(set(bird) & set(robot))
    if not stamps:
        raise SystemExit(f"No matching stamps in {session_dir}")

    print(
        f"{session_dir.name}: {len(stamps)} matched demos "
        f"(keep hand slot {hand_slot}, robot slot {robot_slot})"
    )

    for stamp in stamps:
        bird_p = bird[stamp]
        robot_p = robot[stamp]
        print(f"=== {session_dir.parts[-2]}/{stamp} ===")
        print(f"  hand:  {bird_p.name}")
        print(f"  robot: {robot_p.name}")
        merged = merge_hand_robot(
            bird_p,
            robot_p,
            hand_slot=hand_slot,
            robot_slot=robot_slot,
            dt=dt,
        )
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / f"{stamp}_mixed_merged.npz"
            np.savez(tmp, **merged)
            # Rename displayed title via a symlink-like name is awkward; plot_pose uses path.name.
            # Write with a descriptive name instead.
            titled = Path(td) / f"{session_dir.parts[-2]}_{stamp}_hand+robot.npz"
            tmp.replace(titled)
            plot_pose(titled, show=show)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot mixed hand+robot demos matched by stamp.")
    ap.add_argument(
        "--session",
        action="append",
        required=True,
        help="Session root, e.g. recording/sessions/left_robot_right_hand/0729 (repeatable).",
    )
    ap.add_argument("--dt", type=float, default=0.033, help="Merged timeline step in seconds.")
    ap.add_argument("--no-show", action="store_true", help="Do not open interactive windows.")
    args = ap.parse_args()

    for s in args.session:
        plot_session(Path(s), dt=float(args.dt), show=not args.no_show)


if __name__ == "__main__":
    main()
