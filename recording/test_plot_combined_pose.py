#!/usr/bin/env python3
"""Plot pose/combined .npz with left+right shown together in the 3D plot (default).

Supports:
- `pose` format (from `recording/wilor_rgbd_wrist_pose.py`):
    pose (N,2,10): [x,y,z, rot6d(6), open_flag]
    timestamps (N,)
- Legacy/full combined format:
    positions_xyz (N,2,3), orientation_rotvec (N,2,3), palm_dir (N,2,3), palm_normal (N,2,3), timestamps (N,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection


HAND_NAMES = {0: "Left", 1: "Right"}
# Used for the per-hand orientation rays (trajectory is time-colored).
HAND_COLORS = {0: "#1f77b4", 1: "#ff7f0e"}  # blue/orange
AXIS_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
AXIS_LABELS = ["X", "Y", "Z"]


def _find_npz_by_stamp(folder: Path, stamp14: str, *, prefer_contains: str | None = None) -> Path:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")
    cand = sorted([p for p in folder.glob("*.npz") if stamp14 in p.name])
    if not cand:
        raise FileNotFoundError(f"No .npz containing stamp={stamp14} under {folder}")
    if prefer_contains is not None:
        pref = [p for p in cand if prefer_contains in p.name]
        if pref:
            return pref[0]
    return cand[0]


def contiguous_segments(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.insert(idx[splits + 1], 0, idx[0])
    ends = np.append(idx[splits], idx[-1])
    return list(zip(starts, ends))


def _rot6d_to_axes(rot6d: np.ndarray):
    """
    rot6d: (N,2,6) -> (b1,b2,b3) each (N,2,3)
    Uses Gram-Schmidt (Zhou et al. 6D).
    """
    r = np.asarray(rot6d, dtype=np.float64)
    a1 = r[:, :, 0:3]
    a2 = r[:, :, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    a2_ortho = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2_ortho / (np.linalg.norm(a2_ortho, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return b1, b2, b3


def _maybe_rotvec_from_axes(b1: np.ndarray, b2: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Try to compute rotvec (for plotting) using scipy if available.
    Returns (N,2,3) or all-NaN if scipy missing.
    """
    try:
        from scipy.spatial.transform import Rotation as SciRotation  # type: ignore
    except Exception:
        return np.full((b1.shape[0], 2, 3), np.nan, dtype=np.float64)

    rotvec = np.full((b1.shape[0], 2, 3), np.nan, dtype=np.float64)
    for h in (0, 1):
        R_prev = None
        rv_prev = None
        for i in range(b1.shape[0]):
            if not (np.isfinite(b1[i, h]).all() and np.isfinite(b2[i, h]).all() and np.isfinite(b3[i, h]).all()):
                R_prev = None
                rv_prev = None
                continue
            R = np.column_stack([b1[i, h], b2[i, h], b3[i, h]])
            if R_prev is None:
                rv = SciRotation.from_matrix(R).as_rotvec()
            else:
                rv_rel = SciRotation.from_matrix(R_prev.T @ R).as_rotvec()
                rv = rv_prev + rv_rel
            rotvec[i, h] = rv
            R_prev = R
            rv_prev = rv
    return rotvec


def _load_pose_like(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)

    if "pose" in d.files:
        pose = np.asarray(d["pose"], dtype=np.float64)  # (N,2,10) typically
        pos = pose[:, :, 0:3]
        rot6d = pose[:, :, 3:9]
        hand_open = pose[:, :, 9] if pose.shape[-1] >= 10 else None

        if "timestamps" in d.files:
            ts = np.asarray(d["timestamps"], dtype=np.float64)
            t = ts - ts[0]
        else:
            t = np.arange(pose.shape[0], dtype=np.float64)
            ts = None

        b1, b2, b3 = _rot6d_to_axes(rot6d)
        palm_dir = b2
        palm_normal = b3
        rotvec = _maybe_rotvec_from_axes(b1, b2, b3)

        valid = np.isfinite(pos).all(axis=-1) & np.isfinite(rot6d).all(axis=-1)
        return t, pos, rotvec, palm_dir, palm_normal, valid, hand_open, ts

    # Legacy/full combined format
    ts = np.asarray(d["timestamps"], dtype=np.float64)
    t = ts - ts[0]
    pos = np.asarray(d["positions_xyz"], dtype=np.float64)
    rotvec = np.asarray(d["orientation_rotvec"], dtype=np.float64)
    palm_dir = np.asarray(d["palm_dir"], dtype=np.float64)
    palm_normal = np.asarray(d["palm_normal"], dtype=np.float64)
    valid_pos = np.asarray(d["valid_position"], dtype=bool) if "valid_position" in d.files else np.isfinite(pos).all(axis=-1)
    valid_ori = np.asarray(d["valid_orientation"], dtype=bool) if "valid_orientation" in d.files else np.isfinite(palm_dir).all(axis=-1)
    valid = (
        valid_pos
        & valid_ori
        & np.isfinite(pos).all(axis=-1)
        & np.isfinite(rotvec).all(axis=-1)
        & np.isfinite(palm_dir).all(axis=-1)
        & np.isfinite(palm_normal).all(axis=-1)
    )
    return t, pos, rotvec, palm_dir, palm_normal, valid, None, ts


def _add_ticks_3d(ax3d, pts: np.ndarray, vecs: np.ndarray, *, scale_m: float, stride: int, color: str, alpha: float = 0.75):
    stride = max(int(stride), 1)
    p = np.asarray(pts, dtype=np.float64)[::stride]
    v = np.asarray(vecs, dtype=np.float64)[::stride]
    if p.shape[0] == 0:
        return
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(n > 1e-12, v / n, v)
    end = p + float(scale_m) * v
    segs = np.stack([p, end], axis=1)  # (M,2,3)
    lc = Line3DCollection(segs, colors=[color], linewidths=1.0, alpha=alpha)
    ax3d.add_collection3d(lc)


def _add_gradient_trajectory_3d(ax3d, pts_xyz: np.ndarray, t_norm: np.ndarray, *, cmap, linewidth=2.2, alpha=0.95):
    if pts_xyz.shape[0] < 2:
        return None
    segs = np.stack([pts_xyz[:-1], pts_xyz[1:]], axis=1)
    colors = cmap(np.asarray(t_norm, dtype=np.float64)[:-1])
    colors[:, 3] *= float(alpha)
    lc = Line3DCollection(segs, colors=colors, linewidths=linewidth)
    ax3d.add_collection3d(lc)
    return lc


def _add_ticks_3d_gradient(
    ax3d,
    pts: np.ndarray,
    vecs: np.ndarray,
    t_norm: np.ndarray,
    *,
    scale_m: float,
    stride: int,
    cmap,
    alpha: float = 0.70,
    linewidth: float = 1.0,
):
    stride = max(int(stride), 1)
    p = np.asarray(pts, dtype=np.float64)[::stride]
    v = np.asarray(vecs, dtype=np.float64)[::stride]
    tn = np.asarray(t_norm, dtype=np.float64)[::stride]
    if p.shape[0] == 0:
        return None
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(n > 1e-12, v / n, v)
    end = p + float(scale_m) * v
    segs = np.stack([p, end], axis=1)  # (M,2,3)
    colors = cmap(np.clip(tn, 0.0, 1.0))
    colors[:, 3] *= float(alpha)
    lc = Line3DCollection(segs, colors=colors, linewidths=float(linewidth))
    ax3d.add_collection3d(lc)
    return lc


def plot_pose_arrays(
    *,
    title: str,
    t: np.ndarray,
    pos: np.ndarray,
    rotvec: np.ndarray,
    palm_dir: np.ndarray,
    palm_normal: np.ndarray,
    valid: np.ndarray,
    hand_open: np.ndarray | None,
    ori_scale_m: float = 0.015,
    ori_stride: int = 2,
    output: str | None = None,
    show: bool = True,
):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    pos = np.asarray(pos, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    palm_dir = np.asarray(palm_dir, dtype=np.float64)
    palm_normal = np.asarray(palm_normal, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(title, fontsize=12)

    # Layout (2 rows x 4 cols):
    #  - Row 0: left pos | right pos | combined 3D (spans 2 cols)
    #  - Row 1: left rotvec | right rotvec | left rays | right rays
    gs = fig.add_gridspec(2, 4)
    ax_pos_l = fig.add_subplot(gs[0, 0])
    ax_pos_r = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[0, 2:4], projection="3d")
    ax_ori_l = fig.add_subplot(gs[1, 0])
    ax_ori_r = fig.add_subplot(gs[1, 1])
    ax_rays_l = fig.add_subplot(gs[1, 2], projection="3d")
    ax_rays_r = fig.add_subplot(gs[1, 3], projection="3d")

    # Shared time colormap (start=blue, end=red) across BOTH hands.
    cmap = LinearSegmentedColormap.from_list("blue_red", ["#1f77b4", "#d62728"])
    any_valid = np.asarray(valid, dtype=bool).any(axis=1)
    if np.any(any_valid):
        tv = np.asarray(t, dtype=np.float64)[any_valid]
        t0 = float(np.min(tv))
        t1 = float(np.max(tv))
    else:
        t0, t1 = float(np.min(t)), float(np.max(t)) if t.size else (0.0, 1.0)
    denom = (t1 - t0) if (t1 - t0) > 1e-12 else 1.0
    t_norm_all = np.clip((np.asarray(t, dtype=np.float64) - t0) / denom, 0.0, 1.0)

    # Time series (per hand)
    for hand_idx, (ax_pos, ax_ori) in enumerate([(ax_pos_l, ax_ori_l), (ax_pos_r, ax_ori_r)]):
        hand_name = HAND_NAMES.get(hand_idx, f"Hand {hand_idx}")
        m = valid[:, hand_idx]
        n_valid = int(m.sum())

        for axis_idx in range(3):
            first = True
            for s, e in contiguous_segments(m):
                ax_pos.plot(
                    t[s : e + 1],
                    pos[s : e + 1, hand_idx, axis_idx],
                    color=AXIS_COLORS[axis_idx],
                    linewidth=1.4,
                    label=AXIS_LABELS[axis_idx] if first else None,
                )
                first = False
        ax_pos.set_title(f"{hand_name} — position vs time ({n_valid}/{len(t)})")
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("Position (m)")
        ax_pos.grid(True, alpha=0.3)
        if n_valid:
            ax_pos.legend(loc="upper right")

        # rotvec components if available (otherwise NaNs -> empty plotjs)
        for axis_idx, lbl in enumerate(["rx", "ry", "rz"]):
            first = True
            for s, e in contiguous_segments(m):
                ax_ori.plot(t[s : e + 1], rotvec[s : e + 1, hand_idx, axis_idx], linewidth=1.2, label=lbl if first else None)
                first = False
        ax_ori.set_title(f"{hand_name} — orientation rotvec vs time")
        ax_ori.set_xlabel("Time (s)")
        ax_ori.set_ylabel("rotvec (rad)")
        ax_ori.grid(True, alpha=0.3)
        if n_valid:
            ax_ori.legend(loc="upper right")

        if hand_open is not None:
            ax_open = ax_ori.twinx()
            ax_open.step(t, hand_open[:, hand_idx], where="post", color="#34495e", alpha=0.35, linewidth=1.1)
            ax_open.set_ylim(-0.1, 1.1)
            ax_open.set_ylabel("open (0/1)")

    # Combined 3D trajectory
    traj_pts = []
    legend_handles = []
    legend_labels = []
    for hand_idx in (0, 1):
        hand_name = HAND_NAMES.get(hand_idx, f"Hand {hand_idx}")
        m = valid[:, hand_idx]
        segments = contiguous_segments(m)
        for s, e in segments:
            pts = pos[s : e + 1, hand_idx]
            tn = t_norm_all[s : e + 1]
            traj_pts.append(pts)
            _add_gradient_trajectory_3d(ax_traj, pts, tn, cmap=cmap)
            _add_ticks_3d_gradient(
                ax_traj,
                pts,
                palm_dir[s : e + 1, hand_idx],
                tn,
                scale_m=float(ori_scale_m),
                stride=int(ori_stride),
                cmap=cmap,
                alpha=0.65,
                linewidth=1.0,
            )
        # Proxy legend entry only (empty ax.plot would force ~[0,1] 3D limits).
        legend_handles.append(plt.Line2D([0], [0], color="black", linewidth=2.0))
        legend_labels.append(hand_name)

    if traj_pts:
        all_pts = np.concatenate(traj_pts, axis=0)
        mins = np.nanmin(all_pts, axis=0)
        maxs = np.nanmax(all_pts, axis=0)
        center = 0.5 * (mins + maxs)
        # Equal-aspect cube with a little padding so orientation ticks stay visible.
        half = 0.5 * float(np.max(maxs - mins))
        half = max(half * 1.15, 0.05)  # at least ±5 cm
        ax_traj.set_xlim(center[0] - half, center[0] + half)
        ax_traj.set_ylim(center[1] - half, center[1] + half)
        ax_traj.set_zlim(center[2] - half, center[2] + half)
        try:
            ax_traj.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    ax_traj.set_title("3D trajectory — Left + Right")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_zlabel("Z (m)")
    ax_traj.legend(legend_handles, legend_labels, loc="upper right")

    # Separate orientation rays (one subplot per hand)
    for hand_idx, ax in [(0, ax_rays_l), (1, ax_rays_r)]:
        hand_name = HAND_NAMES.get(hand_idx, f"Hand {hand_idx}")
        c = HAND_COLORS[hand_idx]
        ax.set_title(f"{hand_name} rays")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)

        m = valid[:, hand_idx]
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        v = palm_dir[idx, hand_idx]
        n = palm_normal[idx, hand_idx]
        _add_ticks_3d(ax, np.zeros_like(v), v, scale_m=1.0, stride=int(ori_stride), color=c, alpha=0.60)
        _add_ticks_3d(ax, np.zeros_like(n), n, scale_m=1.0, stride=int(ori_stride), color=c, alpha=0.30)

    fig.tight_layout()
    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_overlay_left_from_one_right_from_other(
    *,
    left_npz: Path,
    right_npz: Path,
    ori_scale_m: float = 0.015,
    ori_stride: int = 2,
    output: str | None = None,
    show: bool = True,
):
    """
    Overlay two independent timelines in one figure:
      - LEFT hand (slot 0) is taken from left_npz
      - RIGHT hand (slot 1) is taken from right_npz

    No timestamp alignment/resampling is performed. Each subplot uses its own
    time axis for that stream; the 3D panel simply overlays both trajectories.
    """
    left_npz = Path(left_npz)
    right_npz = Path(right_npz)

    tL, posL, rotvecL, palm_dirL, palm_normL, validL, openL, _ = _load_pose_like(left_npz)
    tR, posR, rotvecR, palm_dirR, palm_normR, validR, openR, _ = _load_pose_like(right_npz)

    # Extract only the desired slots.
    t_left = np.asarray(tL, dtype=np.float64).reshape(-1)
    t_right = np.asarray(tR, dtype=np.float64).reshape(-1)
    left_pos = np.asarray(posL[:, 0], dtype=np.float64)
    right_pos = np.asarray(posR[:, 1], dtype=np.float64)
    left_rot = np.asarray(rotvecL[:, 0], dtype=np.float64)
    right_rot = np.asarray(rotvecR[:, 1], dtype=np.float64)
    left_dir = np.asarray(palm_dirL[:, 0], dtype=np.float64)
    right_dir = np.asarray(palm_dirR[:, 1], dtype=np.float64)
    left_norm = np.asarray(palm_normL[:, 0], dtype=np.float64)
    right_norm = np.asarray(palm_normR[:, 1], dtype=np.float64)
    left_valid = np.asarray(validL[:, 0], dtype=bool)
    right_valid = np.asarray(validR[:, 1], dtype=bool)

    left_open = None if openL is None else np.asarray(openL[:, 0], dtype=np.float64).reshape(-1)
    right_open = None if openR is None else np.asarray(openR[:, 1], dtype=np.float64).reshape(-1)

    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(
        "Overlay (no alignment): LEFT from hand NPZ, RIGHT from robot NPZ\n"
        f"left={left_npz.name}\nright={right_npz.name}",
        fontsize=11,
    )

    gs = fig.add_gridspec(2, 4)
    ax_pos_l = fig.add_subplot(gs[0, 0])
    ax_pos_r = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[0, 2:4], projection="3d")
    ax_ori_l = fig.add_subplot(gs[1, 0])
    ax_ori_r = fig.add_subplot(gs[1, 1])
    ax_rays_l = fig.add_subplot(gs[1, 2], projection="3d")
    ax_rays_r = fig.add_subplot(gs[1, 3], projection="3d")

    cmap = LinearSegmentedColormap.from_list("blue_red", ["#1f77b4", "#d62728"])

    def t_norm(t: np.ndarray, m: np.ndarray) -> np.ndarray:
        if t.size == 0:
            return t
        if np.any(m):
            tv = t[m]
            t0 = float(np.min(tv))
            t1 = float(np.max(tv))
        else:
            t0 = float(np.min(t))
            t1 = float(np.max(t))
        denom = (t1 - t0) if (t1 - t0) > 1e-12 else 1.0
        return np.clip((t - t0) / denom, 0.0, 1.0)

    tn_left = t_norm(t_left, left_valid)
    tn_right = t_norm(t_right, right_valid)

    # Left time series
    for axis_idx in range(3):
        first = True
        for s, e in contiguous_segments(left_valid):
            ax_pos_l.plot(
                t_left[s : e + 1],
                left_pos[s : e + 1, axis_idx],
                color=AXIS_COLORS[axis_idx],
                linewidth=1.4,
                label=AXIS_LABELS[axis_idx] if first else None,
            )
            first = False
    ax_pos_l.set_title(f"Left — position vs time ({int(left_valid.sum())}/{len(t_left)})")
    ax_pos_l.set_xlabel("Time (s)")
    ax_pos_l.set_ylabel("Position (m)")
    ax_pos_l.grid(True, alpha=0.3)
    if int(left_valid.sum()):
        ax_pos_l.legend(loc="upper right")

    for axis_idx, lbl in enumerate(["rx", "ry", "rz"]):
        first = True
        for s, e in contiguous_segments(left_valid):
            ax_ori_l.plot(t_left[s : e + 1], left_rot[s : e + 1, axis_idx], linewidth=1.2, label=lbl if first else None)
            first = False
    ax_ori_l.set_title("Left — orientation rotvec vs time")
    ax_ori_l.set_xlabel("Time (s)")
    ax_ori_l.set_ylabel("rotvec (rad)")
    ax_ori_l.grid(True, alpha=0.3)
    if int(left_valid.sum()):
        ax_ori_l.legend(loc="upper right")
    if left_open is not None:
        ax_open = ax_ori_l.twinx()
        ax_open.step(t_left, left_open, where="post", color="#34495e", alpha=0.35, linewidth=1.1)
        ax_open.set_ylim(-0.1, 1.1)
        ax_open.set_ylabel("open (0/1)")

    # Right time series
    for axis_idx in range(3):
        first = True
        for s, e in contiguous_segments(right_valid):
            ax_pos_r.plot(
                t_right[s : e + 1],
                right_pos[s : e + 1, axis_idx],
                color=AXIS_COLORS[axis_idx],
                linewidth=1.4,
                label=AXIS_LABELS[axis_idx] if first else None,
            )
            first = False
    ax_pos_r.set_title(f"Right — position vs time ({int(right_valid.sum())}/{len(t_right)})")
    ax_pos_r.set_xlabel("Time (s)")
    ax_pos_r.set_ylabel("Position (m)")
    ax_pos_r.grid(True, alpha=0.3)
    if int(right_valid.sum()):
        ax_pos_r.legend(loc="upper right")

    for axis_idx, lbl in enumerate(["rx", "ry", "rz"]):
        first = True
        for s, e in contiguous_segments(right_valid):
            ax_ori_r.plot(t_right[s : e + 1], right_rot[s : e + 1, axis_idx], linewidth=1.2, label=lbl if first else None)
            first = False
    ax_ori_r.set_title("Right — orientation rotvec vs time")
    ax_ori_r.set_xlabel("Time (s)")
    ax_ori_r.set_ylabel("rotvec (rad)")
    ax_ori_r.grid(True, alpha=0.3)
    if int(right_valid.sum()):
        ax_ori_r.legend(loc="upper right")
    if right_open is not None:
        ax_open = ax_ori_r.twinx()
        ax_open.step(t_right, right_open, where="post", color="#34495e", alpha=0.35, linewidth=1.1)
        ax_open.set_ylim(-0.1, 1.1)
        ax_open.set_ylabel("open (0/1)")

    # Combined 3D overlay
    traj_pts = []
    legend_handles = []
    legend_labels = []

    for s, e in contiguous_segments(left_valid):
        pts = left_pos[s : e + 1]
        tn = tn_left[s : e + 1]
        traj_pts.append(pts)
        _add_gradient_trajectory_3d(ax_traj, pts, tn, cmap=cmap)
        _add_ticks_3d_gradient(
            ax_traj,
            pts,
            left_dir[s : e + 1],
            tn,
            scale_m=float(ori_scale_m),
            stride=int(ori_stride),
            cmap=cmap,
            alpha=0.65,
            linewidth=1.0,
        )
    legend_handles.append(plt.Line2D([0], [0], color="black", linewidth=2.0))
    legend_labels.append("Left (hand)")

    for s, e in contiguous_segments(right_valid):
        pts = right_pos[s : e + 1]
        tn = tn_right[s : e + 1]
        traj_pts.append(pts)
        _add_gradient_trajectory_3d(ax_traj, pts, tn, cmap=cmap)
        _add_ticks_3d_gradient(
            ax_traj,
            pts,
            right_dir[s : e + 1],
            tn,
            scale_m=float(ori_scale_m),
            stride=int(ori_stride),
            cmap=cmap,
            alpha=0.65,
            linewidth=1.0,
        )
    legend_handles.append(plt.Line2D([0], [0], color="black", linewidth=2.0))
    legend_labels.append("Right (robot)")

    if traj_pts:
        all_pts = np.concatenate(traj_pts, axis=0)
        mins = np.nanmin(all_pts, axis=0)
        maxs = np.nanmax(all_pts, axis=0)
        center = 0.5 * (mins + maxs)
        half = 0.5 * float(np.max(maxs - mins))
        half = max(half * 1.15, 0.05)
        ax_traj.set_xlim(center[0] - half, center[0] + half)
        ax_traj.set_ylim(center[1] - half, center[1] + half)
        ax_traj.set_zlim(center[2] - half, center[2] + half)
        try:
            ax_traj.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    ax_traj.set_title("3D trajectory — Left(hand) + Right(robot)")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_zlabel("Z (m)")
    ax_traj.legend(legend_handles, legend_labels, loc="upper right")

    # Rays subplots
    for ax, v, n, m, title in (
        (ax_rays_l, left_dir, left_norm, left_valid, "Left rays"),
        (ax_rays_r, right_dir, right_norm, right_valid, "Right rays"),
    ):
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        _add_ticks_3d(ax, np.zeros_like(v[idx]), v[idx], scale_m=1.0, stride=int(ori_stride), color="#2c3e50", alpha=0.60)
        _add_ticks_3d(ax, np.zeros_like(n[idx]), n[idx], scale_m=1.0, stride=int(ori_stride), color="#2c3e50", alpha=0.30)

    fig.tight_layout()
    if output is not None:
        out = Path(output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pose(
    npz_path: str | Path,
    *,
    ori_scale_m: float = 0.015,
    ori_stride: int = 2,
    output: str | None = None,
    show: bool = True,
):
    npz_path = Path(npz_path)
    t, pos, rotvec, palm_dir, palm_normal, valid, hand_open, _ = _load_pose_like(npz_path)
    plot_pose_arrays(
        title=f"Hand pose — left+right together in 3D\n{npz_path.name}",
        t=t,
        pos=pos,
        rotvec=rotvec,
        palm_dir=palm_dir,
        palm_normal=palm_normal,
        valid=valid,
        hand_open=hand_open,
        ori_scale_m=float(ori_scale_m),
        ori_stride=int(ori_stride),
        output=output,
        show=show,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot pose .npz (left+right combined in 3D by default).")
    parser.add_argument(
        "npz",
        type=str,
        help="Path to a .npz file, or a directory of .npz files",
    )
    parser.add_argument("--ori-scale", type=float, default=0.015, help="Palm_dir tick length on the 3D trajectory (m)")
    parser.add_argument("--ori-stride", type=int, default=2, help="Plot every Nth orientation tick")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save figure path (file mode), or output directory (dir mode)",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    parser.add_argument(
        "--glob",
        type=str,
        default="*.npz",
        help="When npz is a directory, only plot files matching this glob (default: *.npz)",
    )
    parser.add_argument(
        "--combine-left-right",
        action="store_true",
        help=(
            "Combine LEFT slot from a hand-pose NPZ with RIGHT slot from a robot EEF NPZ "
            "and plot them together. Use --left-dir/--right-dir/--stamp."
        ),
    )
    parser.add_argument("--left-dir", type=str, default=None, help="Directory containing left-hand pose .npz files")
    parser.add_argument("--right-dir", type=str, default=None, help="Directory containing right-hand pose .npz files")
    parser.add_argument(
        "--stamp",
        type=str,
        default=None,
        help=(
            "14-digit stamp (e.g. 20260815144549). If omitted with --combine-left-right, "
            "plots ALL stamps that exist in both dirs."
        ),
    )
    args = parser.parse_args()

    if bool(args.combine_left_right):
        if not (args.left_dir and args.right_dir):
            raise SystemExit("--combine-left-right requires --left-dir and --right-dir")

        left_dir = Path(args.left_dir).expanduser().resolve()
        right_dir = Path(args.right_dir).expanduser().resolve()
        if not left_dir.is_dir():
            raise SystemExit(f"--left-dir is not a directory: {left_dir}")
        if not right_dir.is_dir():
            raise SystemExit(f"--right-dir is not a directory: {right_dir}")

        # Single-stamp mode
        if args.stamp is not None:
            stamp = str(args.stamp).strip()
            if len(stamp) != 14 or (not stamp.isdigit()):
                raise SystemExit("--stamp must be a 14-digit string like 20260815144549")
            left_npz = _find_npz_by_stamp(left_dir, stamp, prefer_contains="_wilor_rgbd_pose")
            right_npz = _find_npz_by_stamp(right_dir, stamp, prefer_contains="commonframe")
            plot_overlay_left_from_one_right_from_other(
                left_npz=left_npz,
                right_npz=right_npz,
                ori_scale_m=float(args.ori_scale),
                ori_stride=int(args.ori_stride),
                output=args.output,
                show=not args.no_show,
            )
            return

        # Batch mode: find all 14-digit stamps common to both dirs.
        import re

        stamp_re = re.compile(r"(\d{14})")

        def stamps_in_dir(d: Path) -> dict[str, Path]:
            out: dict[str, Path] = {}
            for p in sorted(d.glob("*.npz")):
                ms = stamp_re.findall(p.name)
                if not ms:
                    continue
                # Prefer the *last* 14-digit group in the filename (often the demo stamp).
                out[str(ms[-1])] = p
            return out

        left_map = stamps_in_dir(left_dir)
        right_map = stamps_in_dir(right_dir)
        stamps = sorted(set(left_map) & set(right_map))
        if not stamps:
            raise SystemExit(f"No shared 14-digit stamps between {left_dir} and {right_dir}")

        # Output handling: for batch, output must be a directory (or omitted -> default).
        out_dir = None
        if args.output:
            out_path = Path(args.output).expanduser().resolve()
            if out_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf"):
                raise SystemExit("For batch combine, -o/--output must be a directory (not a single file path).")
            out_dir = out_path
        else:
            out_dir = left_dir / "overlay_plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        show = not args.no_show
        print(f"[combine-left-right] Found {len(stamps)} matched stamps.")
        print(f"[combine-left-right] Saving plots under: {out_dir}")
        if show:
            print("Interactive: close each window to advance to the next plot.")

        for i, stamp in enumerate(stamps, 1):
            left_npz = _find_npz_by_stamp(left_dir, stamp, prefer_contains="_wilor_rgbd_pose")
            right_npz = _find_npz_by_stamp(right_dir, stamp, prefer_contains="commonframe")
            out = str(out_dir / f"overlay_{stamp}.png")
            print(f"[{i}/{len(stamps)}] {stamp} -> {Path(out).name}")
            plot_overlay_left_from_one_right_from_other(
                left_npz=left_npz,
                right_npz=right_npz,
                ori_scale_m=float(args.ori_scale),
                ori_stride=int(args.ori_stride),
                output=out,
                show=show,
            )
        print(f"Done. Wrote {len(stamps)} plots -> {out_dir}")
        return

    path = Path(args.npz).expanduser().resolve()
    if path.is_dir():
        files = sorted(path.glob(args.glob))
        if not files:
            raise SystemExit(f"No files matching {args.glob!r} in {path}")
        show = not args.no_show
        out_dir = None
        if args.output:
            out_dir = Path(args.output).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Plotting {len(files)} files from {path}")
        if out_dir is not None:
            print(f"Saving figures under {out_dir}")
        if show:
            print("Interactive: close each window to advance to the next file.")
        for i, f in enumerate(files, 1):
            out = str(out_dir / f"{f.stem}.png") if out_dir is not None else None
            print(f"[{i}/{len(files)}] {f.name}")
            plot_pose(
                f,
                ori_scale_m=float(args.ori_scale),
                ori_stride=int(args.ori_stride),
                output=out,
                show=show,
            )
        print(f"Done. {len(files)} plots.")
        return

    if not path.is_file():
        raise SystemExit(f"Not a file or directory: {path}")

    plot_pose(
        path,
        ori_scale_m=float(args.ori_scale),
        ori_stride=int(args.ori_stride),
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

