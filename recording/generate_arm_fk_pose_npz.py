#!/usr/bin/env python3
"""
Generate an EgoMimic-style pose .npz from joint position logs by computing forward kinematics.

This script is designed to match the key / array schema of:
`*wilor_rgbd_pose_targetframe.npz`, specifically (for bimanual):
  - timestamps:        (T,)
  - pose:              (T, 2, 10)
  - valid_pos:         (T, 2)
  - valid_rot:         (T, 2)
  - valid_open:        (T, 2)
  - pose_xyz_raw:      (T, 2, 3)
  - valid_pos_raw:     (T, 2)
  - R_raw:             (T, 2, 3, 3)
  - valid_rot_raw:     (T, 2)
  - open_score_raw:    (T, 2)
  - open_score_filled: (T, 2)
  - open_score_valid:  (T, 2)
  - open_threshold:    scalar
  - pose_timeline:     scalar string (e.g. "bag")

For single-arm mode (left OR right), the same keys are emitted but without the arm axis:
  - pose:              (T, 10)
  - valid_pos:         (T,)
  - pose_xyz_raw:      (T, 3)
  - R_raw:             (T, 3, 3)
  - open_score_*:      (T,)
etc.

Pose vector convention used here (length 10), matching `recording/test_plot_combined_pose.py`:
  [x, y, z, rot6d(6), open_flag]

Where rot6d is the first two columns of the 3x3 rotation matrix (Zhou et al. 6D).

Notes:
  - Default FK model is Galaxea A1: URDF/A1/urdf/A1_URDF_0607_0028.urdf
    (EEF position = midpoint of gripper1/gripper2, orientation from arm_seg6).
  - If your joint vector contains an extra gripper joint (common: 7 values),
    FK uses the first 6 arm joints, and column 6 is kept as the gripper/open signal.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """
    R: (..., 3, 3) -> rot6d (..., 6) as [R[:,0], R[:,1]] flattened.
    """
    R = np.asarray(R, dtype=np.float64)
    assert R.shape[-2:] == (3, 3)
    b1 = R[..., 0]  # (...,3)
    b2 = R[..., 1]  # (...,3)
    return np.concatenate([b1, b2], axis=-1)


def _apply_T(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Apply homogeneous transform T (4x4) to xyz (...,3).
    """
    T = np.asarray(T, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    assert T.shape == (4, 4)
    flat = xyz.reshape(-1, 3)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    homog = np.concatenate([flat, ones], axis=1)  # (N,4)
    out = (T @ homog.T).T[:, :3]
    return out.reshape(xyz.shape)


def _load_optional_T(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transform file not found: {p}")
    if p.suffix == ".npy":
        T = np.load(p)
    else:
        # simple text format: whitespace-separated 4x4
        T = np.loadtxt(p)
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {T.shape} from {p}")
    return T


def _default_urdf_path() -> Path:
    """Default to Galaxea A1 URDF; fall back to EgoMimic vx300s only if A1 is missing."""
    here = Path(__file__).resolve()
    repo_root = here.parents[1]

    candidates = [
        Path("URDF/A1/urdf/A1_URDF_0607_0028.urdf"),
        repo_root / "URDF" / "A1" / "urdf" / "A1_URDF_0607_0028.urdf",
        Path("EgoMimic/egomimic/resources/model.urdf"),
        repo_root / "EgoMimic" / "egomimic" / "resources" / "model.urdf",
    ]
    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find A1 URDF. Pass --urdf explicitly, e.g. "
        "--urdf URDF/A1/urdf/A1_URDF_0607_0028.urdf"
    )


def _is_a1_urdf(urdf_path: Path) -> bool:
    s = str(urdf_path)
    return ("URDF/A1" in s) or ("A1_URDF" in urdf_path.name)


def _build_pk_chain(urdf_path: Path):
    try:
        import pytorch_kinematics as pk
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `pytorch_kinematics`. In `gaze-aloha`, install with:\n"
            "  python -m pip install pytorch-kinematics"
        ) from e

    # Use bytes to support URDFs with an XML encoding declaration.
    urdf_bytes = urdf_path.read_bytes()
    chain = pk.build_chain_from_urdf(urdf_bytes)
    joint_names = list(chain.get_joint_parameter_names())
    return chain, joint_names


def _compute_fk(
    qpos: np.ndarray,
    chain,
    joint_names: List[str],
    *,
    eef_links: List[str],
    ori_link: str,
    fk_dofs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - xyz: (T,3) float64
      - R:   (T,3,3) float64
      - rot6d: (T,6) float64
      - open_signal: (T,) float64 (best-effort; for 7-DoF logs, uses qpos[:, 6] exactly)
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency `torch` in your current Python env.") from e

    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos shape (T,D), got {qpos.shape}")
    n_chain = len(joint_names)
    if qpos.shape[1] < n_chain:
        raise ValueError(f"Joint vector has {qpos.shape[1]} values but FK chain requires {n_chain}. Joints: {joint_names}")

    # Preserve gripper/open exactly for common (T,7) joint logs.
    if qpos.shape[1] >= 7:
        open_signal = qpos[:, 6]
    elif qpos.shape[1] > n_chain:
        open_signal = qpos[:, -1]
    else:
        open_signal = np.ones((qpos.shape[0],), dtype=np.float64)

    # By default, avoid letting gripper affect FK even if the URDF chain includes it.
    if fk_dofs is None:
        if qpos.shape[1] >= 7:
            fk_dofs = min(n_chain, 6)
        else:
            fk_dofs = n_chain
    if fk_dofs < 1 or fk_dofs > min(qpos.shape[1], n_chain):
        raise ValueError(f"Invalid fk_dofs={fk_dofs}. Must be in [1, {min(qpos.shape[1], n_chain)}].")

    q_fk = qpos[:, :fk_dofs]

    q_fk_t = torch.as_tensor(q_fk, dtype=torch.float32)
    fk_all = chain.forward_kinematics(q_fk_t)  # dict: link_name -> Transform

    missing = [ln for ln in (eef_links + [ori_link]) if ln not in fk_all]
    if missing:
        raise ValueError(
            f"Requested link(s) not found in URDF: {missing}. "
            f"Available sample: {list(fk_all.keys())[:20]}"
        )

    # Position: either single link or mean of multiple links (e.g., gripper midpoint).
    xyzs = []
    for ln in eef_links:
        Tln = fk_all[ln].get_matrix().detach().cpu().numpy().astype(np.float64)  # (T,4,4)
        xyzs.append(Tln[:, :3, 3])
    xyz = np.mean(np.stack(xyzs, axis=0), axis=0)

    # Orientation: take from a designated link (defaults chosen by args).
    Tori = fk_all[ori_link].get_matrix().detach().cpu().numpy().astype(np.float64)
    R = Tori[:, :3, :3]
    rot6d = _rotmat_to_rot6d(R)
    return xyz, R, rot6d, open_signal.astype(np.float64)


def _build_single_arm_npz(
    timestamps: np.ndarray,
    xyz: np.ndarray,
    R: np.ndarray,
    rot6d: np.ndarray,
    open_signal: np.ndarray,
    open_threshold: float,
    pose_timeline: str,
    T_target_base: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    T = timestamps.shape[0]

    xyz = np.asarray(xyz, dtype=np.float64).reshape(T, 3)
    R = np.asarray(R, dtype=np.float64).reshape(T, 3, 3)
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(T, 6)
    open_signal = np.asarray(open_signal, dtype=np.float64).reshape(T)

    if T_target_base is not None:
        xyz = _apply_T(T_target_base, xyz)

    valid_pos = np.ones((T,), dtype=bool)
    valid_rot = np.ones((T,), dtype=bool)
    valid_open = np.ones((T,), dtype=bool)

    open_score_raw = open_signal.astype(np.float64)
    open_score_filled = open_score_raw.copy()
    open_score_valid = np.ones((T,), dtype=bool)
    open_flag = (open_score_filled > float(open_threshold)) & open_score_valid

    pose = np.zeros((T, 10), dtype=np.float64)
    pose[:, 0:3] = xyz
    pose[:, 3:9] = rot6d
    pose[:, 9] = open_flag.astype(np.float64)

    return {
        "timestamps": timestamps,
        "pose": pose,
        "valid_pos": valid_pos,
        "valid_rot": valid_rot,
        "valid_open": valid_open,
        "pose_xyz_raw": xyz,
        "valid_pos_raw": valid_pos.copy(),
        "R_raw": R,
        "valid_rot_raw": valid_rot.copy(),
        "open_score_raw": open_score_raw,
        "open_score_filled": open_score_filled,
        "open_score_valid": open_score_valid,
        "open_threshold": np.array(open_threshold, dtype=np.float64),
        "pose_timeline": np.array(pose_timeline),
    }


def _build_bimanual_npz(
    timestamps: np.ndarray,
    left: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    open_threshold: float,
    pose_timeline: str,
    T_target_base_left: Optional[np.ndarray],
    T_target_base_right: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    lt_xyz, lt_R, lt_rot6d, lt_open = left
    rt_xyz, rt_R, rt_rot6d, rt_open = right

    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    T = timestamps.shape[0]

    lt_xyz = np.asarray(lt_xyz, dtype=np.float64).reshape(T, 3)
    rt_xyz = np.asarray(rt_xyz, dtype=np.float64).reshape(T, 3)
    lt_R = np.asarray(lt_R, dtype=np.float64).reshape(T, 3, 3)
    rt_R = np.asarray(rt_R, dtype=np.float64).reshape(T, 3, 3)
    lt_rot6d = np.asarray(lt_rot6d, dtype=np.float64).reshape(T, 6)
    rt_rot6d = np.asarray(rt_rot6d, dtype=np.float64).reshape(T, 6)
    lt_open = np.asarray(lt_open, dtype=np.float64).reshape(T)
    rt_open = np.asarray(rt_open, dtype=np.float64).reshape(T)

    if T_target_base_left is not None:
        lt_xyz = _apply_T(T_target_base_left, lt_xyz)
    if T_target_base_right is not None:
        rt_xyz = _apply_T(T_target_base_right, rt_xyz)

    pose_xyz_raw = np.stack([lt_xyz, rt_xyz], axis=1)  # (T,2,3)
    R_raw = np.stack([lt_R, rt_R], axis=1)  # (T,2,3,3)
    rot6d = np.stack([lt_rot6d, rt_rot6d], axis=1)  # (T,2,6)

    open_score_raw = np.stack([lt_open, rt_open], axis=1).astype(np.float64)  # (T,2)
    open_score_filled = open_score_raw.copy()
    open_score_valid = np.ones((T, 2), dtype=bool)
    open_flag = (open_score_filled > float(open_threshold)) & open_score_valid

    pose = np.zeros((T, 2, 10), dtype=np.float64)
    pose[:, :, 0:3] = pose_xyz_raw
    pose[:, :, 3:9] = rot6d
    pose[:, :, 9] = open_flag.astype(np.float64)

    valid_pos = np.ones((T, 2), dtype=bool)
    valid_rot = np.ones((T, 2), dtype=bool)
    valid_open = np.ones((T, 2), dtype=bool)

    return {
        "timestamps": timestamps,
        "pose": pose,
        "valid_pos": valid_pos,
        "valid_rot": valid_rot,
        "valid_open": valid_open,
        "pose_xyz_raw": pose_xyz_raw,
        "valid_pos_raw": valid_pos.copy(),
        "R_raw": R_raw,
        "valid_rot_raw": valid_rot.copy(),
        "open_score_raw": open_score_raw,
        "open_score_filled": open_score_filled,
        "open_score_valid": open_score_valid,
        "open_threshold": np.array(open_threshold, dtype=np.float64),
        "pose_timeline": np.array(pose_timeline),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["left", "right", "both"],
        required=True,
        help="Which arm(s) to include in the output npz.",
    )
    ap.add_argument("--left_qpos", type=str, help="Left arm joint positions .npy (T,D).")
    ap.add_argument("--left_time", type=str, help="Left arm timestamps .npy (T,).")
    ap.add_argument("--right_qpos", type=str, help="Right arm joint positions .npy (T,D).")
    ap.add_argument("--right_time", type=str, help="Right arm timestamps .npy (T,).")
    ap.add_argument(
        "--joint_data_dir",
        type=str,
        default=None,
        help=(
            "Path to a joint-data folder that contains:\n"
            "  left/position/joint_position_*.npy\n"
            "  left/time/joint_timestamp_*.npy\n"
            "  right/position/joint_position_*.npy\n"
            "  right/time/joint_timestamp_*.npy\n"
            "If set, the script auto-parses available stamps and writes one output per stamp."
        ),
    )
    ap.add_argument(
        "--stamp",
        type=str,
        default=None,
        help="Optional stamp string to process a single set (e.g. 20260714175708). Only used with --joint_data_dir.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory when using --joint_data_dir (default: <joint_data_dir>/combined_npz).",
    )
    ap.add_argument(
        "--out_prefix",
        type=str,
        default="teleop_bimanual",
        help="Output filename prefix when using --joint_data_dir.",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="When using --joint_data_dir, print what would be processed without writing outputs.",
    )
    ap.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to robot URDF. Default: URDF/A1/urdf/A1_URDF_0607_0028.urdf",
    )
    ap.add_argument(
        "--end_link",
        type=str,
        default=None,
        help="End-effector link name used for FK position (ignored if --eef_links is set).",
    )
    ap.add_argument(
        "--eef_links",
        type=str,
        default=None,
        help="Comma-separated link names to average for EEF position. "
        "A1 default: 'gripper1,gripper2' (gripper midpoint).",
    )
    ap.add_argument(
        "--ori_link",
        type=str,
        default=None,
        help="Which link to use for orientation/rot6d. A1 default: arm_seg6.",
    )
    ap.add_argument(
        "--fk_dofs",
        type=int,
        default=None,
        help="How many leading joint values to use for FK (default: 6 when qpos has 7 cols; otherwise URDF chain joint count).",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=False,
        help="Output .npz path (required unless --joint_data_dir is used).",
    )
    ap.add_argument(
        "--open_threshold",
        type=float,
        default=1.1,
        help="Stored in output for schema-compatibility (default matches reference files).",
    )
    ap.add_argument(
        "--pose_timeline",
        type=str,
        default="bag",
        help="Stored in output for schema-compatibility (default matches reference files).",
    )
    ap.add_argument(
        "--T_target_base_left",
        type=str,
        default=None,
        help="Optional 4x4 transform (npy or whitespace txt). Applied to left XYZ: xyz_target = T * [xyz_base;1].",
    )
    ap.add_argument(
        "--T_target_base_right",
        type=str,
        default=None,
        help="Optional 4x4 transform (npy or whitespace txt). Applied to right XYZ: xyz_target = T * [xyz_base;1].",
    )
    args = ap.parse_args()
    if args.joint_data_dir is None and not args.out:
        ap.error("--out is required unless --joint_data_dir is provided.")

    def _iter_stamps(pos_dir: Path) -> Iterable[str]:
        for p in sorted(pos_dir.glob("joint_position_*.npy")):
            # expect: joint_position_<prefix>_<STAMP>.npy (we keep everything after the last underscore)
            stem = p.stem  # no .npy
            if "_" not in stem:
                continue
            yield stem.split("_")[-1]

    def _resolve_files(joint_data_dir: Path, stamp: str):
        left_pos = joint_data_dir / "left" / "position"
        left_time = joint_data_dir / "left" / "time"
        right_pos = joint_data_dir / "right" / "position"
        right_time = joint_data_dir / "right" / "time"

        # Be flexible on the middle token (teleop_bimanual vs other), but require the stamp suffix.
        lp = next(iter(left_pos.glob(f"joint_position_*_{stamp}.npy")), None)
        lt = next(iter(left_time.glob(f"joint_timestamp_*_{stamp}.npy")), None)
        rp = next(iter(right_pos.glob(f"joint_position_*_{stamp}.npy")), None)
        rt = next(iter(right_time.glob(f"joint_timestamp_*_{stamp}.npy")), None)
        return lp, lt, rp, rt

    urdf_path = Path(args.urdf) if args.urdf is not None else _default_urdf_path()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    chain, joint_names = _build_pk_chain(urdf_path)
    print(f"Using URDF: {urdf_path}")
    print(f"URDF joints ({len(joint_names)}): {joint_names}")

    # Defaults: A1 uses gripper midpoint + wrist orientation; other URDFs use --end_link.
    if args.eef_links is not None:
        eef_links = [s.strip() for s in args.eef_links.split(",") if s.strip()]
        if len(eef_links) == 0:
            raise ValueError("--eef_links was provided but empty")
    elif _is_a1_urdf(urdf_path):
        eef_links = ["gripper1", "gripper2"]
    else:
        if args.end_link is None:
            args.end_link = "vx300s/ee_gripper_link"
        eef_links = [args.end_link]

    if args.ori_link is None:
        args.ori_link = "arm_seg6" if _is_a1_urdf(urdf_path) else eef_links[0]

    print(f"EEF position links: {eef_links}")
    print(f"Orientation link: {args.ori_link}")

    T_left = _load_optional_T(args.T_target_base_left)
    T_right = _load_optional_T(args.T_target_base_right)

    # Folder mode: auto-parse joint-data folder and generate one output per stamp.
    if args.joint_data_dir is not None:
        joint_data_dir = Path(args.joint_data_dir)
        if not joint_data_dir.exists():
            raise FileNotFoundError(f"--joint_data_dir not found: {joint_data_dir}")

        out_dir = Path(args.out_dir) if args.out_dir is not None else (joint_data_dir / "combined_npz")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        # Gather candidate stamps from left positions by default.
        stamps = sorted(set(_iter_stamps(joint_data_dir / "left" / "position")))
        if args.stamp is not None:
            stamps = [args.stamp]
        if len(stamps) == 0:
            raise FileNotFoundError(f"No joint_position_*.npy found under {joint_data_dir/'left/position'}")

        for stamp in stamps:
            lp, lt, rp, rt = _resolve_files(joint_data_dir, stamp)
            need_left = args.mode in ("left", "both")
            need_right = args.mode in ("right", "both")
            if need_left and (lp is None or lt is None):
                print(f"Skipping {stamp} (missing left files)")
                continue
            if need_right and (rp is None or rt is None):
                print(f"Skipping {stamp} (missing right files)")
                continue

            out_path = out_dir / f"{args.out_prefix}_{stamp}_arm_fk_pose_targetframe.npz"
            print(f"[{args.mode}] {stamp} -> {out_path}")
            if args.dry_run:
                continue

            if args.mode == "left":
                left_qpos = np.load(str(lp))
                left_time = np.load(str(lt))
                xyz, R, rot6d, open_signal = _compute_fk(
                    left_qpos, chain, joint_names, eef_links=eef_links, ori_link=str(args.ori_link), fk_dofs=args.fk_dofs
                )
                payload = _build_single_arm_npz(
                    timestamps=left_time,
                    xyz=xyz,
                    R=R,
                    rot6d=rot6d,
                    open_signal=open_signal,
                    open_threshold=args.open_threshold,
                    pose_timeline=args.pose_timeline,
                    T_target_base=T_left,
                )
                np.savez(out_path, **payload)
                continue

            if args.mode == "right":
                right_qpos = np.load(str(rp))
                right_time = np.load(str(rt))
                xyz, R, rot6d, open_signal = _compute_fk(
                    right_qpos, chain, joint_names, eef_links=eef_links, ori_link=str(args.ori_link), fk_dofs=args.fk_dofs
                )
                payload = _build_single_arm_npz(
                    timestamps=right_time,
                    xyz=xyz,
                    R=R,
                    rot6d=rot6d,
                    open_signal=open_signal,
                    open_threshold=args.open_threshold,
                    pose_timeline=args.pose_timeline,
                    T_target_base=T_right,
                )
                np.savez(out_path, **payload)
                continue

            # both
            left_qpos = np.load(str(lp))
            left_time = np.load(str(lt))
            right_qpos = np.load(str(rp))
            right_time = np.load(str(rt))

            Tn = min(left_qpos.shape[0], right_qpos.shape[0], left_time.shape[0], right_time.shape[0])
            left_qpos = left_qpos[:Tn]
            right_qpos = right_qpos[:Tn]
            left_time = left_time[:Tn]
            right_time = right_time[:Tn]
            timestamps = left_time.astype(np.float64)

            left_fk = _compute_fk(
                left_qpos, chain, joint_names, eef_links=eef_links, ori_link=str(args.ori_link), fk_dofs=args.fk_dofs
            )
            right_fk = _compute_fk(
                right_qpos, chain, joint_names, eef_links=eef_links, ori_link=str(args.ori_link), fk_dofs=args.fk_dofs
            )
            payload = _build_bimanual_npz(
                timestamps=timestamps,
                left=left_fk,
                right=right_fk,
                open_threshold=args.open_threshold,
                pose_timeline=args.pose_timeline,
                T_target_base_left=T_left,
                T_target_base_right=T_right,
            )
            np.savez(out_path, **payload)

        return

    # Single-file mode (explicit file args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in ("left", "both"):
        if args.left_qpos is None or args.left_time is None:
            raise ValueError("--left_qpos and --left_time are required for mode left/both")
        left_qpos = np.load(args.left_qpos)
        left_time = np.load(args.left_time)
        if left_qpos.shape[0] != left_time.shape[0]:
            raise ValueError(f"Left qpos/time length mismatch: {left_qpos.shape[0]} vs {left_time.shape[0]}")

    if args.mode in ("right", "both"):
        if args.right_qpos is None or args.right_time is None:
            raise ValueError("--right_qpos and --right_time are required for mode right/both")
        right_qpos = np.load(args.right_qpos)
        right_time = np.load(args.right_time)
        if right_qpos.shape[0] != right_time.shape[0]:
            raise ValueError(f"Right qpos/time length mismatch: {right_qpos.shape[0]} vs {right_time.shape[0]}")

    if args.mode == "left":
        xyz, R, rot6d, open_signal = _compute_fk(
            left_qpos, chain, joint_names, eef_links=eef_links, ori_link=str(args.ori_link), fk_dofs=args.fk_dofs
        )
        payload = _build_single_arm_npz(
            timestamps=left_time,
            xyz=xyz,
            R=R,
            rot6d=rot6d,
            open_signal=open_signal,
            open_threshold=args.open_threshold,
            pose_timeline=args.pose_timeline,
            T_target_base=T_left,
        )
        np.savez(out_path, **payload)
        return

    if args.mode == "right":
        xyz, R, rot6d, open_signal = _compute_fk(
            right_qpos, chain, joint_names, eef_links=eef_links, ori_link=str(args.ori_link), fk_dofs=args.fk_dofs
        )
        payload = _build_single_arm_npz(
            timestamps=right_time,
            xyz=xyz,
            R=R,
            rot6d=rot6d,
            open_signal=open_signal,
            open_threshold=args.open_threshold,
            pose_timeline=args.pose_timeline,
            T_target_base=T_right,
        )
        np.savez(out_path, **payload)
        return

    # both
    # align by truncating to shared length and using a single timeline
    T = min(left_qpos.shape[0], right_qpos.shape[0], left_time.shape[0], right_time.shape[0])
    left_qpos = left_qpos[:T]
    right_qpos = right_qpos[:T]
    left_time = left_time[:T]
    right_time = right_time[:T]
    timestamps = left_time.astype(np.float64)

    left_fk = _compute_fk(left_qpos, chain, joint_names, eef_links=eef_links, ori_link=str(args.ori_link), fk_dofs=args.fk_dofs)
    right_fk = _compute_fk(right_qpos, chain, joint_names, eef_links=eef_links, ori_link=str(args.ori_link), fk_dofs=args.fk_dofs)
    payload = _build_bimanual_npz(
        timestamps=timestamps,
        left=left_fk,
        right=right_fk,
        open_threshold=args.open_threshold,
        pose_timeline=args.pose_timeline,
        T_target_base_left=T_left,
        T_target_base_right=T_right,
    )
    np.savez(out_path, **payload)


if __name__ == "__main__":
    main()

