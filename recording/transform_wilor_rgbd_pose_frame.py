#!/usr/bin/env python3
"""
Transform WiLoR+RGBD wrist pose .npz from camera frame -> target frame.

Input format: produced by `recording/wilor_rgbd_wrist_pose.py`:
  - timestamps (N,)
  - pose (N,2,10): [x,y,z, rot6d(6), open_flag]
      rot6d = first two columns of a rotation matrix (R[:,0], R[:,1])
  - pose_xyz_raw (N,2,3)
  - R_raw (N,2,3,3)
  - plus validity / open_score fields

Transform specification (as requested):
  - Input is in camera frame.
  - Camera origin in target frame is (-35, 184, 775) mm.
  - Camera x axis and z axis are flipped.
  - Placeholder extra rotation about x axis (deg), default 0.

Optional orientation-only post-process (does NOT touch xyz):
  - After the camera->target rigid transform, apply a per-hand body-frame
    correction: R' = R @ R_post[hand].
  - Useful for ~180deg wrist-frame convention mismatch vs teleop
    (roll/yaw ~180) without changing the position axes.
  - Default: left=ry180, right=none (right is already ~aligned).

Outputs:
  - Writes .npz files with the EXACT SAME keys/shapes as the input,
    but with pose/pose_xyz_raw/R_raw transformed into the target frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


def _rot_x(deg: float) -> np.ndarray:
    th = np.deg2rad(float(deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _camera_to_target_R(*, flip_xz: bool, extra_rx_deg: float) -> np.ndarray:
    # Flip x and z axes (det=+1): equivalent to 180deg about +y.
    R_flip = np.diag([-1.0, 1.0, -1.0]).astype(np.float64) if flip_xz else np.eye(3, dtype=np.float64)
    R_extra = _rot_x(extra_rx_deg)  # placeholder knob (default 0 => identity)
    return R_extra @ R_flip


def _camera_origin_target_t_m(camera_pos_mm: tuple[float, float, float]) -> np.ndarray:
    return (np.asarray(camera_pos_mm, dtype=np.float64).reshape(3) / 1000.0).astype(np.float64)


_ORIENT_POST_NAMED = {
    "none": np.eye(3, dtype=np.float64),
    "rx180": np.diag([1.0, -1.0, -1.0]).astype(np.float64),
    "ry180": np.diag([-1.0, 1.0, -1.0]).astype(np.float64),
    "rz180": np.diag([-1.0, -1.0, 1.0]).astype(np.float64),
}


def _parse_orient_post(name: str) -> np.ndarray:
    key = str(name).strip().lower()
    if key not in _ORIENT_POST_NAMED:
        raise ValueError(
            f"unknown orient-post {name!r}; expected one of {sorted(_ORIENT_POST_NAMED)}"
        )
    return _ORIENT_POST_NAMED[key].copy()


def _orient_post_pair(left: str, right: str) -> tuple[np.ndarray, np.ndarray]:
    return _parse_orient_post(left), _parse_orient_post(right)


def _transform_xyz(xyz_cam_m: np.ndarray, R_tc: np.ndarray, t_tc_m: np.ndarray) -> np.ndarray:
    """
    xyz_t = R_tc * xyz_c + t_tc
    Accepts shape (...,3).
    """
    x = np.asarray(xyz_cam_m, dtype=np.float64)
    out = x.copy()
    m = np.isfinite(x).all(axis=-1)
    if not np.any(m):
        return out
    flat = x.reshape(-1, 3)
    mf = m.reshape(-1)
    out_flat = out.reshape(-1, 3)
    out_flat[mf] = (R_tc @ flat[mf].T).T + t_tc_m[None, :]
    return out


def _gs_orthonormalize_6d(col0: np.ndarray, col1: np.ndarray) -> np.ndarray:
    """
    6D (two 3-vectors) -> proper rotation matrix via Gram-Schmidt.
    Returns (3,3) with columns [x,y,z].
    """
    a1 = np.asarray(col0, dtype=np.float64).reshape(3)
    a2 = np.asarray(col1, dtype=np.float64).reshape(3)
    if not (np.isfinite(a1).all() and np.isfinite(a2).all()):
        return np.full((3, 3), np.nan, dtype=np.float64)
    n1 = float(np.linalg.norm(a1))
    if n1 < 1e-12:
        return np.full((3, 3), np.nan, dtype=np.float64)
    b1 = a1 / n1
    a2p = a2 - float(np.dot(b1, a2)) * b1
    n2 = float(np.linalg.norm(a2p))
    if n2 < 1e-12:
        return np.full((3, 3), np.nan, dtype=np.float64)
    b2 = a2p / n2
    b3 = np.cross(b1, b2)
    n3 = float(np.linalg.norm(b3))
    if n3 < 1e-12:
        return np.full((3, 3), np.nan, dtype=np.float64)
    b3 = b3 / n3
    return np.column_stack([b1, b2, b3])


def _apply_orient_post_R(R: np.ndarray, R_post_per_hand: Sequence[np.ndarray]) -> np.ndarray:
    """
    Orientation-only body-frame correction: R' = R @ R_post[hand].
    xyz is untouched by this helper. Accepts (N,2,3,3).
    """
    out = np.asarray(R, dtype=np.float64).copy()
    if out.ndim != 4 or out.shape[-2:] != (3, 3) or out.shape[1] != 2:
        raise ValueError(f"R has unexpected shape {out.shape}, expected (N,2,3,3)")
    if len(R_post_per_hand) != 2:
        raise ValueError("R_post_per_hand must have length 2 (left, right)")

    for h, R_post in enumerate(R_post_per_hand):
        Rp = np.asarray(R_post, dtype=np.float64).reshape(3, 3)
        if np.allclose(Rp, np.eye(3)):
            continue
        Rh = out[:, h]
        m = np.isfinite(Rh).all(axis=(-2, -1))
        if not np.any(m):
            continue
        out[m, h] = Rh[m] @ Rp
    return out


def _transform_pose_inplace(
    pose: np.ndarray,
    R_tc: np.ndarray,
    t_tc_m: np.ndarray,
    R_post_per_hand: Sequence[np.ndarray],
) -> np.ndarray:
    """
    pose[...,0:3] (xyz, meters) and pose[...,3:9] (rot6d) transformed to target frame,
    then orientation-only body-frame post R' = R @ R_post[hand].
    Returns a new array (does not mutate input).
    """
    p = np.asarray(pose, dtype=np.float64).copy()
    if p.ndim != 3 or p.shape[-1] != 10:
        raise ValueError(f"pose has unexpected shape {p.shape}, expected (N,2,10)")

    # XYZ
    p[:, :, 0:3] = _transform_xyz(p[:, :, 0:3], R_tc, t_tc_m)

    # Rot6D -> R_tgt = (R_tc @ R_cam) @ R_post[hand]
    for i in range(p.shape[0]):
        for h in range(p.shape[1]):
            c0 = p[i, h, 3:6]
            c1 = p[i, h, 6:9]
            R_cam = _gs_orthonormalize_6d(c0, c1)
            if not np.isfinite(R_cam).all():
                continue
            R_tgt = R_tc @ R_cam
            Rp = np.asarray(R_post_per_hand[h], dtype=np.float64).reshape(3, 3)
            if not np.allclose(Rp, np.eye(3)):
                R_tgt = R_tgt @ Rp
            p[i, h, 3:6] = R_tgt[:, 0]
            p[i, h, 6:9] = R_tgt[:, 1]
    return p


def _transform_R_raw(
    R_raw: np.ndarray,
    R_tc: np.ndarray,
    R_post_per_hand: Sequence[np.ndarray],
) -> np.ndarray:
    R = np.asarray(R_raw, dtype=np.float64).copy()
    if R.ndim != 4 or R.shape[-2:] != (3, 3):
        raise ValueError(f"R_raw has unexpected shape {R.shape}, expected (N,2,3,3)")
    # R_t = R_tc @ R_c
    out = R.copy()
    m = np.isfinite(R).all(axis=(-2, -1))
    if np.any(m):
        flat = R.reshape(-1, 3, 3)
        mf = m.reshape(-1)
        out_flat = out.reshape(-1, 3, 3)
        out_flat[mf] = R_tc @ flat[mf]
    # then orientation-only post (xyz unaffected)
    return _apply_orient_post_R(out, R_post_per_hand)


def transform_npz(
    in_path: Path,
    out_path: Path,
    *,
    camera_pos_mm: tuple[float, float, float],
    flip_xz: bool,
    extra_rx_deg: float,
    orient_post_left: str = "ry180",
    orient_post_right: str = "none",
):
    d = np.load(in_path, allow_pickle=False)
    keys = list(d.files)

    # Required keys for this format.
    required = {"timestamps", "pose", "pose_xyz_raw", "R_raw"}
    missing = required - set(keys)
    if missing:
        raise KeyError(f"{in_path} missing required keys: {sorted(missing)}")

    R_tc = _camera_to_target_R(flip_xz=flip_xz, extra_rx_deg=extra_rx_deg)
    t_tc_m = _camera_origin_target_t_m(camera_pos_mm)
    R_post = _orient_post_pair(orient_post_left, orient_post_right)

    out = {}
    for k in keys:
        out[k] = d[k]

    out["pose"] = _transform_pose_inplace(out["pose"], R_tc, t_tc_m, R_post)
    out["pose_xyz_raw"] = _transform_xyz(out["pose_xyz_raw"], R_tc, t_tc_m)
    out["R_raw"] = _transform_R_raw(out["R_raw"], R_tc, R_post)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)


def _parse_xyz(s: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 3:
        raise ValueError("expected 'x,y,z' (mm)")
    return float(parts[0]), float(parts[1]), float(parts[2])


def main():
    ap = argparse.ArgumentParser(description="Transform *_wilor_rgbd_pose.npz camera frame -> target frame.")
    ap.add_argument(
        "input",
        type=str,
        help="Input .npz file OR a directory containing *_wilor_rgbd_pose.npz files",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npz (if input is a file) OR output directory (if input is a dir). "
            "Default: <in>_targetframe next to input."
        ),
    )
    ap.add_argument(
        "--camera-pos-mm",
        type=str,
        default="-35,184,775",
        help="Camera origin in the target frame, in mm: 'x,y,z' (default: -35,184,775)",
    )
    ap.add_argument(
        "--no-flip-xz",
        action="store_true",
        help="Disable the requested x/z axis flips (for debugging).",
    )
    ap.add_argument(
        "--extra-rx-deg",
        type=float,
        default=-18.46,
        help="Placeholder extra rotation about +x in degrees (default 0).",
    )
    ap.add_argument(
        "--orient-post-left",
        type=str,
        default="ry180",
        choices=sorted(_ORIENT_POST_NAMED),
        help=(
            "Orientation-only body-frame post for left hand: R'=R@R_post. "
            "Does not change xyz. Default: ry180 (fixes ~180 roll/yaw vs teleop)."
        ),
    )
    ap.add_argument(
        "--orient-post-right",
        type=str,
        default="none",
        choices=sorted(_ORIENT_POST_NAMED),
        help=(
            "Orientation-only body-frame post for right hand: R'=R@R_post. "
            "Does not change xyz. Default: none (right already ~aligned)."
        ),
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    cam_pos = _parse_xyz(args.camera_pos_mm)
    flip_xz = not bool(args.no_flip_xz)
    extra_rx_deg = float(args.extra_rx_deg)
    orient_kwargs = dict(
        orient_post_left=str(args.orient_post_left),
        orient_post_right=str(args.orient_post_right),
    )

    if in_path.is_file():
        out_path = Path(args.output) if args.output else (in_path.with_name(in_path.stem + "_targetframe.npz"))
        transform_npz(
            in_path,
            out_path,
            camera_pos_mm=cam_pos,
            flip_xz=flip_xz,
            extra_rx_deg=extra_rx_deg,
            **orient_kwargs,
        )
        print(f"wrote {out_path}")
        return

    if not in_path.is_dir():
        raise SystemExit(f"input is neither a file nor a directory: {in_path}")

    out_dir = Path(args.output) if args.output else in_path.with_name(in_path.name + "_targetframe")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_path.glob("*_wilor_rgbd_pose.npz"))
    if not files:
        raise SystemExit(f"no '*_wilor_rgbd_pose.npz' files found in {in_path}")

    for p in files:
        out_path = out_dir / p.name
        transform_npz(
            p,
            out_path,
            camera_pos_mm=cam_pos,
            flip_xz=flip_xz,
            extra_rx_deg=extra_rx_deg,
            **orient_kwargs,
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

