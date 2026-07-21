#!/usr/bin/env python3
"""
Transform arm-FK pose .npz files from per-arm base frames into a common frame.

Input format: produced by `recording/generate_arm_fk_pose_npz.py`
  - timestamps (T,)
  - pose (T,2,10) for bimanual, or (T,10) for single-arm:
      [x,y,z, rot6d(6), open_flag]
  - pose_xyz_raw (T,2,3) or (T,3)
  - R_raw (T,2,3,3) or (T,3,3)
  - plus validity / open_score fields (copied through unchanged)

Requested transforms (default offset = 0.48 m = 48 cm):

  Left arm (hand index 0):
    translation only in common frame:
      p' = p + [-offset, 0, 0]
      R' = R

  Right arm (hand index 1):
    keep z, flip x, and flip y so the rotation is proper (180 deg about +z):
      R_rb = diag(-1, -1, +1)
    then translate +offset along the *rotated* x axis of the common frame:
      p' = R_rb @ p + [+offset, 0, 0]
      R' = R_rb @ R

Input is a folder of .npz files (or a single .npz). Outputs keep the same keys/shapes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# 180 deg about +z: flip x and y, keep z (proper rotation, det=+1).
R_RIGHT_BASE_TO_COMMON = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
R_LEFT_BASE_TO_COMMON = np.eye(3, dtype=np.float64)


def _rot6d_from_R(R: np.ndarray) -> np.ndarray:
    """R (...,3,3) -> rot6d (...,6) = [R[:,0]; R[:,1]]."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def _apply_rigid(
    xyz: np.ndarray,
    R_src: np.ndarray,
    *,
    R_tc: np.ndarray,
    t_tc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    xyz_t = R_tc @ xyz_s + t_tc
    R_t   = R_tc @ R_s
    xyz: (...,3), R_src: (...,3,3)
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    R_src = np.asarray(R_src, dtype=np.float64)
    R_tc = np.asarray(R_tc, dtype=np.float64)
    t_tc = np.asarray(t_tc, dtype=np.float64).reshape(3)

    flat_xyz = xyz.reshape(-1, 3)
    flat_R = R_src.reshape(-1, 3, 3)
    out_xyz = (R_tc @ flat_xyz.T).T + t_tc[None, :]
    out_R = R_tc @ flat_R
    return out_xyz.reshape(xyz.shape), out_R.reshape(R_src.shape)


def _transforms_for_offset(offset_m: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns dict arm -> (R_tc, t_tc) mapping base-frame coords into common frame.
    """
    off = float(offset_m)
    return {
        "left": (R_LEFT_BASE_TO_COMMON, np.array([-off, 0.0, 0.0], dtype=np.float64)),
        "right": (R_RIGHT_BASE_TO_COMMON, np.array([+off, 0.0, 0.0], dtype=np.float64)),
    }


def _pack_pose(xyz: np.ndarray, R: np.ndarray, open_flag: np.ndarray) -> np.ndarray:
    """
    xyz: (...,3), R: (...,3,3), open_flag: (...)
    returns pose (...,10)
    """
    rot6d = _rot6d_from_R(R)
    pose = np.concatenate(
        [xyz, rot6d, open_flag[..., None].astype(np.float64)],
        axis=-1,
    )
    return pose


def transform_bimanual(
    pose: np.ndarray,
    pose_xyz_raw: np.ndarray,
    R_raw: np.ndarray,
    *,
    offset_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pose.ndim != 3 or pose.shape[-1] != 10:
        raise ValueError(f"Expected bimanual pose (T,2,10), got {pose.shape}")
    if pose_xyz_raw.shape != pose.shape[:2] + (3,):
        raise ValueError(f"pose_xyz_raw shape {pose_xyz_raw.shape} mismatch with pose {pose.shape}")
    if R_raw.shape != pose.shape[:2] + (3, 3):
        raise ValueError(f"R_raw shape {R_raw.shape} mismatch with pose {pose.shape}")

    tf = _transforms_for_offset(offset_m)
    out_xyz = np.empty_like(pose_xyz_raw, dtype=np.float64)
    out_R = np.empty_like(R_raw, dtype=np.float64)

    for hand_idx, arm in ((0, "left"), (1, "right")):
        R_tc, t_tc = tf[arm]
        out_xyz[:, hand_idx], out_R[:, hand_idx] = _apply_rigid(
            pose_xyz_raw[:, hand_idx],
            R_raw[:, hand_idx],
            R_tc=R_tc,
            t_tc=t_tc,
        )

    open_flag = pose[..., 9]
    out_pose = _pack_pose(out_xyz, out_R, open_flag)
    return out_pose, out_xyz, out_R


def transform_single(
    pose: np.ndarray,
    pose_xyz_raw: np.ndarray,
    R_raw: np.ndarray,
    *,
    arm: str,
    offset_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pose.ndim != 2 or pose.shape[-1] != 10:
        raise ValueError(f"Expected single-arm pose (T,10), got {pose.shape}")
    if pose_xyz_raw.shape != (pose.shape[0], 3):
        raise ValueError(f"pose_xyz_raw shape {pose_xyz_raw.shape} mismatch with pose {pose.shape}")
    if R_raw.shape != (pose.shape[0], 3, 3):
        raise ValueError(f"R_raw shape {R_raw.shape} mismatch with pose {pose.shape}")
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")

    R_tc, t_tc = _transforms_for_offset(offset_m)[arm]
    out_xyz, out_R = _apply_rigid(pose_xyz_raw, R_raw, R_tc=R_tc, t_tc=t_tc)
    out_pose = _pack_pose(out_xyz, out_R, pose[:, 9])
    return out_pose, out_xyz, out_R


def _infer_single_arm(path: Path, arm_arg: Optional[str]) -> str:
    if arm_arg is not None:
        return arm_arg
    name = path.name.lower()
    if "left" in name and "right" not in name:
        return "left"
    if "right" in name and "left" not in name:
        return "right"
    raise ValueError(
        f"Single-arm npz {path} needs --arm left|right "
        "(could not infer from filename)."
    )


def transform_npz(
    in_path: Path,
    out_path: Path,
    *,
    offset_m: float,
    arm: Optional[str] = None,
) -> None:
    d = np.load(in_path, allow_pickle=True)
    keys = list(d.files)
    required = {"timestamps", "pose", "pose_xyz_raw", "R_raw"}
    missing = required - set(keys)
    if missing:
        raise KeyError(f"{in_path} missing required keys: {sorted(missing)}")

    pose = np.asarray(d["pose"], dtype=np.float64)
    pose_xyz_raw = np.asarray(d["pose_xyz_raw"], dtype=np.float64)
    R_raw = np.asarray(d["R_raw"], dtype=np.float64)

    out: Dict[str, np.ndarray] = {k: d[k] for k in keys}

    if pose.ndim == 3 and pose.shape[1] == 2:
        out_pose, out_xyz, out_R = transform_bimanual(
            pose, pose_xyz_raw, R_raw, offset_m=offset_m
        )
    elif pose.ndim == 2:
        which = _infer_single_arm(in_path, arm)
        out_pose, out_xyz, out_R = transform_single(
            pose, pose_xyz_raw, R_raw, arm=which, offset_m=offset_m
        )
    else:
        raise ValueError(f"{in_path}: unexpected pose shape {pose.shape}")

    out["pose"] = out_pose
    out["pose_xyz_raw"] = out_xyz
    out["R_raw"] = out_R

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transform arm FK pose .npz (per-arm base) -> common frame."
    )
    ap.add_argument(
        "input",
        type=str,
        help="Input .npz file OR a directory of .npz files",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npz (if input is a file) OR output directory (if input is a dir). "
            "Default: <in>_commonframe next to input."
        ),
    )
    ap.add_argument(
        "--offset_m",
        type=float,
        default=0.48,
        help="Base offset along x in meters (default: 0.48 = 48 cm).",
    )
    ap.add_argument(
        "--arm",
        choices=["left", "right"],
        default=None,
        help="Required/used only for single-arm (T,10) npz files.",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="*.npz",
        help="Glob used when input is a directory (default: '*.npz').",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    offset_m = float(args.offset_m)

    if in_path.is_file():
        out_path = (
            Path(args.output)
            if args.output
            else in_path.with_name(in_path.stem + "_commonframe.npz")
        )
        transform_npz(in_path, out_path, offset_m=offset_m, arm=args.arm)
        print(f"wrote {out_path}")
        return

    if not in_path.is_dir():
        raise SystemExit(f"input is neither a file nor a directory: {in_path}")

    out_dir = Path(args.output) if args.output else in_path.with_name(in_path.name + "_commonframe")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_path.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matching {args.glob!r} in {in_path}")

    for p in files:
        if not p.is_file() or p.suffix != ".npz":
            continue
        out_path = out_dir / (p.stem + "_commonframe.npz")
        transform_npz(p, out_path, offset_m=offset_m, arm=args.arm)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
