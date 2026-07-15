#!/usr/bin/env python3
"""
Plot WiLoR-mini predictions saved by `recording/wilor.py`.

Input format: a `*_wilor_mini_pred.npy` file created with np.save(..., allow_pickle=True)
containing a dict with:
  - fps, stride, frames=[{frame_idx,time_s,pred=[{hand_bbox,is_right,wilor_preds}, ...]}, ...]

This script plots:
  - Wrist (or chosen keypoint) XYZ vs time
  - Δrotation (rotation-vector components) vs time from consecutive global orientations
  - 3D trajectory with blue->red time gradient + short orientation segments
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection


HAND_NAMES = {0: "Left", 1: "Right"}
AXIS_LABELS = ["X", "Y", "Z"]
AXIS_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]


def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / n, v)


def _axis_unit(axis: str) -> np.ndarray:
    axis = axis.lower().strip()
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError(f"Unknown axis: {axis!r} (expected x/y/z)")


def contiguous_segments(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.insert(idx[splits + 1], 0, idx[0])
    ends = np.append(idx[splits], idx[-1])
    return list(zip(starts, ends))


def _rotvec_to_rotmat(rv: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rotation-vector (axis-angle) -> 3x3 rotation matrix."""
    rv = np.asarray(rv, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < eps:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    kx, ky, kz = k.tolist()
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _so3_log(R: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """SO(3) log map: rotation matrix -> rotation vector (3,)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        # small-angle approximation: rv ≈ vee(R - R^T)/2
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    sin_theta = float(np.sin(theta))
    if abs(sin_theta) < eps:
        # near pi; fall back to numeric-safe axis extraction
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + eps)
        return axis * theta
    axis = (1.0 / (2.0 * sin_theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    )
    return axis * theta


def _add_gradient_trajectory_3d(ax3d, pts_xyz: np.ndarray, t_norm: np.ndarray, *, cmap, linewidth=2.0, alpha=0.95):
    if pts_xyz.shape[0] < 2:
        return None
    segs = np.stack([pts_xyz[:-1], pts_xyz[1:]], axis=1)
    colors = cmap(t_norm[:-1])
    colors[:, 3] *= alpha
    lc = Line3DCollection(segs, colors=colors, linewidths=linewidth)
    ax3d.add_collection3d(lc)
    return lc


def _add_orientation_segments_3d(
    ax3d,
    pos_xyz: np.ndarray,
    rotmats: np.ndarray,
    t_norm: np.ndarray,
    *,
    axis: str,
    scale: float,
    stride: int,
    cmap,
    alpha: float = 0.85,
    linewidth: float = 1.0,
):
    if pos_xyz.shape[0] == 0:
        return None
    stride = max(int(stride), 1)
    p = pos_xyz[::stride]
    Rm = rotmats[::stride]
    tn = t_norm[::stride]
    base = _axis_unit(axis)
    base = _normalize_rows(base)
    dirs = (Rm @ base.reshape(3, 1)).reshape(-1, 3)
    dirs = _normalize_rows(dirs)
    end = p + float(scale) * dirs
    segs = np.stack([p, end], axis=1)
    colors = cmap(tn)
    colors[:, 3] *= alpha
    lc = Line3DCollection(segs, colors=colors, linewidths=linewidth)
    ax3d.add_collection3d(lc)
    return lc


def load_wilor_mini_pred(pred_path: str | Path):
    pred_path = Path(pred_path)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    d = np.load(pred_path, allow_pickle=True).item()
    required = ("frames", "fps", "stride")
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"Missing keys in prediction file: {missing}")
    return d, pred_path


def extract_series(
    d: dict,
    *,
    keypoint_index: int = 0,
    apply_cam_t_full: bool = True,
    orientation_source: str = "joints",
):
    """
    Returns:
      t: (T,) seconds
      pos: (T,2,3) float with NaNs where missing
      rotmat: (T,2,3,3) rotation matrix with NaNs where missing
      valid: (T,2) bool
    """
    frames = d["frames"]
    t = np.asarray([f.get("time_s", np.nan) for f in frames], dtype=np.float64)
    T = len(frames)
    pos = np.full((T, 2, 3), np.nan, dtype=np.float64)
    rotmat = np.full((T, 2, 3, 3), np.nan, dtype=np.float64)
    valid = np.zeros((T, 2), dtype=bool)

    orientation_source = str(orientation_source).lower().strip()
    if orientation_source not in {"global", "joints"}:
        raise ValueError("orientation_source must be 'global' or 'joints'")

    for i, f in enumerate(frames):
        pred = f.get("pred", [])
        if not isinstance(pred, list) or len(pred) == 0:
            continue

        # pick at most one left and one right from this frame
        chosen = {0: None, 1: None}  # slot -> dict
        for det in pred:
            if not isinstance(det, dict):
                continue
            is_right = det.get("is_right", None)
            if is_right is None:
                continue
            slot = 1 if float(is_right) > 0.5 else 0
            if chosen[slot] is None:
                chosen[slot] = det

        for slot, det in chosen.items():
            if det is None:
                continue

            # Prefer precomputed camera-space wrist pose if present (added by recording/wilor.py)
            if (
                isinstance(det.get("wrist_pos_cam", None), (list, tuple, np.ndarray))
                and isinstance(det.get("wrist_rotvec_cam", None), (list, tuple, np.ndarray))
                and int(keypoint_index) == 0
                and apply_cam_t_full
                and orientation_source == "global"
            ):
                p = np.asarray(det["wrist_pos_cam"], dtype=np.float64).reshape(-1)[:3]
                rv = np.asarray(det["wrist_rotvec_cam"], dtype=np.float64).reshape(-1)[:3]
                if np.all(np.isfinite(p)) and np.all(np.isfinite(rv)):
                    pos[i, slot] = p
                    rotmat[i, slot] = _rotvec_to_rotmat(rv)
                    valid[i, slot] = True
                    continue

            if (
                isinstance(det.get("wrist_pos_cam", None), (list, tuple, np.ndarray))
                and isinstance(det.get("wrist_rotmat_cam_from_joints", None), (list, tuple, np.ndarray))
                and int(keypoint_index) == 0
                and apply_cam_t_full
                and orientation_source == "joints"
            ):
                p = np.asarray(det["wrist_pos_cam"], dtype=np.float64).reshape(-1)[:3]
                R = np.asarray(det["wrist_rotmat_cam_from_joints"], dtype=np.float64).reshape(3, 3)
                if np.all(np.isfinite(p)) and np.all(np.isfinite(R)):
                    pos[i, slot] = p
                    rotmat[i, slot] = R
                    valid[i, slot] = True
                    continue

            wp = det.get("wilor_preds", None)
            if not isinstance(wp, dict):
                continue

            k3d = wp.get("pred_keypoints_3d", None)
            go = wp.get("global_orient", None)
            if k3d is None or go is None:
                continue
            k3d = np.asarray(k3d)
            go = np.asarray(go)
            if k3d.ndim != 3 or k3d.shape[-1] != 3:
                continue
            if go.ndim != 3 or go.shape[-1] != 3:
                continue

            kp = int(keypoint_index)
            if not (0 <= kp < k3d.shape[1]):
                raise ValueError(f"keypoint_index={kp} out of range for pred_keypoints_3d with {k3d.shape[1]} points")

            p = k3d[0, kp].astype(np.float64)
            if apply_cam_t_full and ("pred_cam_t_full" in wp):
                cam_t = np.asarray(wp["pred_cam_t_full"]).reshape(-1)[:3].astype(np.float64)
                if np.all(np.isfinite(cam_t)):
                    p = p + cam_t
            pos[i, slot] = p

            if orientation_source == "global":
                rotmat[i, slot] = _rotvec_to_rotmat(go[0, 0].astype(np.float64))
            else:
                # Derive a palm frame from OpenPose-style joints: wrist=0, index_mcp=5, pinky_mcp=17
                if k3d.shape[1] > 17:
                    wrist = k3d[0, 0].astype(np.float64)
                    index_mcp = k3d[0, 5].astype(np.float64)
                    pinky_mcp = k3d[0, 17].astype(np.float64)
                    if apply_cam_t_full and ("pred_cam_t_full" in wp) and np.all(np.isfinite(cam_t)):
                        wrist = wrist + cam_t
                        index_mcp = index_mcp + cam_t
                        pinky_mcp = pinky_mcp + cam_t
                    x_axis = index_mcp - wrist
                    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-12)
                    palm_span = pinky_mcp - wrist
                    z_axis = np.cross(x_axis, palm_span)
                    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
                    rotmat[i, slot] = np.column_stack([x_axis, y_axis, z_axis])

            valid[i, slot] = True

    return t, pos, rotmat, valid


def plot_wilor_mini_pose(
    pred_path: str | Path,
    *,
    keypoint_index: int = 0,
    apply_cam_t_full: bool = True,
    orientation_source: str = "joints",
    orientation_axis: str = "z",
    orientation_scale: float = 0.05,
    orientation_stride: int = 2,
    output_path: str | Path | None = None,
    show: bool = True,
):
    d, src = load_wilor_mini_pred(pred_path)
    t, pos, rotmat, valid = extract_series(
        d,
        keypoint_index=keypoint_index,
        apply_cam_t_full=apply_cam_t_full,
        orientation_source=orientation_source,
    )

    cmap = LinearSegmentedColormap.from_list("history_blue_red", ["#1f77b4", "#d62728"])

    fig = plt.figure(figsize=(18, 10))
    cam_note = "+ cam_t_full" if apply_cam_t_full else "(root coords)"
    ori_note = f"ori={orientation_source}"
    fig.suptitle(
        f"WiLoR-mini pose (kp={keypoint_index}) {cam_note} {ori_note} — position + orientation\n{src.name}",
        fontsize=12,
    )

    for hand_idx in range(2):
        hand_name = HAND_NAMES.get(hand_idx, f"Hand {hand_idx}")
        base = hand_idx * 3

        ax_pos = fig.add_subplot(2, 3, base + 1)
        hand_valid = valid[:, hand_idx]
        n_valid = int(hand_valid.sum())

        for axis_idx in range(3):
            first = True
            for start, end in contiguous_segments(hand_valid):
                ax_pos.plot(
                    t[start : end + 1],
                    pos[start : end + 1, hand_idx, axis_idx],
                    color=AXIS_COLORS[axis_idx],
                    linewidth=1.4,
                    label=AXIS_LABELS[axis_idx] if first else None,
                )
                first = False

        ax_pos.set_title(f"{hand_name} — keypoint XYZ vs time ({n_valid}/{len(t)})")
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("Position (WiLoR units)")
        ax_pos.grid(True, alpha=0.3)
        if n_valid:
            ax_pos.legend(loc="upper right")

        ax_rot = fig.add_subplot(2, 3, base + 2)
        ax_rot.set_title(f"{hand_name} — Δrotation vs time")
        ax_rot.set_xlabel("Time (s)")
        ax_rot.set_ylabel("Δrotation (rad)")
        ax_rot.grid(True, alpha=0.3)

        for axis_idx in range(3):
            first = True
            for start, end in contiguous_segments(hand_valid):
                if end - start < 1:
                    continue
                idx = np.arange(start, end + 1, dtype=int)
                Rs = [rotmat[j, hand_idx] for j in idx]
                drot = []
                for j in range(1, len(Rs)):
                    R_rel = Rs[j] @ Rs[j - 1].T
                    drot.append(_so3_log(R_rel))
                drot = np.asarray(drot, dtype=np.float64)
                ax_rot.plot(
                    t[idx[1:]],
                    drot[:, axis_idx],
                    color=AXIS_COLORS[axis_idx],
                    linewidth=1.1,
                    label=AXIS_LABELS[axis_idx] if first else None,
                )
                first = False
        if n_valid:
            ax_rot.legend(loc="upper right")

        ax3d = fig.add_subplot(2, 3, base + 3, projection="3d")
        segments = contiguous_segments(hand_valid)
        if n_valid:
            t_valid = t[hand_valid]
            t0 = float(np.min(t_valid))
            t1 = float(np.max(t_valid))
            denom = (t1 - t0) if (t1 - t0) > 1e-12 else 1.0

            for start, end in segments:
                idx = np.arange(start, end + 1, dtype=int)
                pts = pos[idx, hand_idx]
                tn = (t[idx] - t0) / denom
                _add_gradient_trajectory_3d(ax3d, pts, tn, cmap=cmap)
                ax3d.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="#2ecc71", s=45, depthshade=False,
                             label="start" if start == segments[0][0] else None)
                ax3d.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="#e74c3c", s=45, depthshade=False,
                             label="end" if start == segments[0][0] else None)
                _add_orientation_segments_3d(
                    ax3d,
                    pts,
                    rotmat[idx, hand_idx],
                    tn,
                    axis=orientation_axis,
                    scale=orientation_scale,
                    stride=orientation_stride,
                    cmap=cmap,
                )

        ax3d.set_title(f"{hand_name} — 3D trajectory + orientation")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        if n_valid:
            ax3d.legend(loc="upper right")

    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot WiLoR-mini pose predictions saved by recording/wilor.py")
    parser.add_argument("pred_npy", type=str, help="Path to *_wilor_mini_pred.npy")
    parser.add_argument("--kp", type=int, default=0, help="Keypoint index to plot (0 is usually wrist)")
    parser.add_argument(
        "--no-cam-t",
        action="store_true",
        help="Do not add pred_cam_t_full to keypoints (plot MANO/root coordinates instead of camera translation)",
    )
    parser.add_argument("--ori-axis", type=str, default="z", choices=["x", "y", "z"], help="Local axis to visualize")
    parser.add_argument("--ori-scale", type=float, default=0.05, help="Orientation segment length")
    parser.add_argument("--ori-stride", type=int, default=2, help="Plot every Nth orientation segment")
    parser.add_argument(
        "--ori-source",
        type=str,
        default="joints",
        choices=["joints", "global"],
        help="Orientation source: 'joints' derives a palm frame from keypoints; 'global' uses wilor_preds.global_orient",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Save plot image to this path")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    args = parser.parse_args()

    plot_wilor_mini_pose(
        args.pred_npy,
        keypoint_index=args.kp,
        apply_cam_t_full=not args.no_cam_t,
        orientation_source=args.ori_source,
        orientation_axis=args.ori_axis,
        orientation_scale=args.ori_scale,
        orientation_stride=args.ori_stride,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

