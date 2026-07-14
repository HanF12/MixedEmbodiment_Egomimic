#!/usr/bin/env python3
"""Plot combined pose .npz produced by combine_positions_wilor_orientation.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection


HAND_NAMES = {0: "Left", 1: "Right"}
AXIS_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
AXIS_LABELS = ["X", "Y", "Z"]


def contiguous_segments(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.insert(idx[splits + 1], 0, idx[0])
    ends = np.append(idx[splits], idx[-1])
    return list(zip(starts, ends))


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
    palm_dir: np.ndarray,
    t_norm: np.ndarray,
    *,
    scale_m: float,
    stride: int,
    cmap,
    alpha: float = 0.85,
    linewidth: float = 1.0,
):
    stride = max(int(stride), 1)
    p = pos_xyz[::stride]
    v = palm_dir[::stride]
    tn = t_norm[::stride]
    # normalize (in case)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(n > 1e-12, v / n, v)
    end = p + float(scale_m) * v
    segs = np.stack([p, end], axis=1)
    colors = cmap(tn)
    colors[:, 3] *= alpha
    lc = Line3DCollection(segs, colors=colors, linewidths=linewidth)
    ax3d.add_collection3d(lc)
    return lc


def _add_orientation_endpoints_3d(
    ax3d,
    vecs: np.ndarray,
    t_norm: np.ndarray,
    *,
    cmap,
    linewidth: float = 2.0,
    alpha: float = 0.95,
):
    """
    Plot a direction vector time-series as a colored 3D line in orientation space.
    vecs: (N,3) unit-ish vectors.
    """
    if vecs.shape[0] < 2:
        return None
    v = np.asarray(vecs, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(n > 1e-12, v / n, v)
    segs = np.stack([v[:-1], v[1:]], axis=1)
    colors = cmap(t_norm[:-1])
    colors[:, 3] *= alpha
    lc = Line3DCollection(segs, colors=colors, linewidths=linewidth)
    ax3d.add_collection3d(lc)
    return lc


def _add_orientation_rays_3d(
    ax3d,
    vecs: np.ndarray,
    t_norm: np.ndarray,
    *,
    cmap,
    length: float = 1.0,
    stride: int = 1,
    linewidth: float = 1.2,
    alpha: float = 0.85,
):
    """
    Plot per-frame orientation vectors as rays from origin (no endpoint-connecting curve).
    vecs: (N,3) unit-ish vectors.
    """
    stride = max(int(stride), 1)
    v = np.asarray(vecs, dtype=np.float64)[::stride]
    tn = np.asarray(t_norm, dtype=np.float64)[::stride]
    if v.shape[0] == 0:
        return None
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = np.where(n > 1e-12, v / n, v)
    v = float(length) * v
    origin = np.zeros((v.shape[0], 3), dtype=np.float64)
    segs = np.stack([origin, v], axis=1)  # (N,2,3)
    colors = cmap(tn)
    colors[:, 3] *= alpha
    lc = Line3DCollection(segs, colors=colors, linewidths=linewidth)
    ax3d.add_collection3d(lc)
    return lc


def plot_combined(npz_path: str | Path, *, ori_scale_m: float = 0.05, ori_stride: int = 2, output: str | None = None, show: bool = True):
    npz_path = Path(npz_path)
    d = np.load(npz_path, allow_pickle=True)

    if "pose" in d.files:
        pose = np.asarray(d["pose"], dtype=np.float64)  # (N,2,9)
        pos = pose[:, :, 0:3]
        if pose.shape[-1] == 10:
            rot6d = pose[:, :, 3:9]
            hand_open = pose[:, :, 9]
        else:
            rot6d = pose[:, :, 3:9]
            hand_open = None
        if "timestamps" in d.files:
            ts = np.asarray(d["timestamps"], dtype=np.float64)
            t = ts - ts[0]
        else:
            # frame index as fallback
            t = np.arange(pose.shape[0], dtype=np.float64)

        # Reconstruct R from rot6d (Zhou et al. 6D)
        a1 = rot6d[:, :, 0:3]
        a2 = rot6d[:, :, 3:6]
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
        a2_ortho = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = a2_ortho / (np.linalg.norm(a2_ortho, axis=-1, keepdims=True) + 1e-12)
        b3 = np.cross(b1, b2)

        # Our convention in combiner: columns are [x=b1, y=b2, z=b3]
        palm_dir = b2
        palm_normal = b3

        # rotvec from R via trace formula
        rotvec = np.full((pose.shape[0], 2, 3), np.nan, dtype=np.float64)
        for i in range(pose.shape[0]):
            for h in (0, 1):
                if not np.isfinite(b1[i, h]).all() or not np.isfinite(b2[i, h]).all():
                    continue
                R = np.column_stack([b1[i, h], b2[i, h], b3[i, h]])
                tr = float(np.trace(R))
                c = float(np.clip((tr - 1.0) * 0.5, -1.0, 1.0))
                theta = float(np.arccos(c))
                if theta < 1e-8:
                    rv = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
                else:
                    s = float(np.sin(theta))
                    if abs(s) < 1e-12:
                        rv = np.zeros(3, dtype=np.float64)
                    else:
                        axis = (1.0 / (2.0 * s)) * np.array(
                            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
                        )
                        rv = axis * theta
                rotvec[i, h] = rv

        valid = np.isfinite(pos).all(axis=-1) & np.isfinite(rot6d).all(axis=-1)
    else:
        ts = np.asarray(d["timestamps"], dtype=np.float64)
        t = ts - ts[0]
        pos = np.asarray(d["positions_xyz"], dtype=np.float64)  # (N,2,3)
        rotvec = np.asarray(d["orientation_rotvec"], dtype=np.float64)  # (N,2,3)
        palm_dir = np.asarray(d["palm_dir"], dtype=np.float64)  # (N,2,3)
        palm_normal = np.asarray(d["palm_normal"], dtype=np.float64)  # (N,2,3)
        valid_pos = np.asarray(d["valid_position"], dtype=bool)
        valid_ori = np.asarray(d["valid_orientation"], dtype=bool)

        valid = (
            valid_pos
            & valid_ori
            & np.isfinite(pos).all(axis=-1)
            & np.isfinite(rotvec).all(axis=-1)
            & np.isfinite(palm_dir).all(axis=-1)
            & np.isfinite(palm_normal).all(axis=-1)
        )

    cmap = LinearSegmentedColormap.from_list("history_blue_red", ["#1f77b4", "#d62728"])

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f"Combined hand pose: [x,y,z,rot6d] (+derived vectors)\n{npz_path.name}", fontsize=12)

    for hand_idx in range(2):
        hand_name = HAND_NAMES.get(hand_idx, f"Hand {hand_idx}")
        base = hand_idx * 4
        hand_valid = valid[:, hand_idx]
        n_valid = int(hand_valid.sum())

        ax_pos = fig.add_subplot(2, 4, base + 1)
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
        ax_pos.set_title(f"{hand_name} — position vs time ({n_valid}/{len(t)})")
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("Position (m)")
        ax_pos.grid(True, alpha=0.3)
        if n_valid:
            ax_pos.legend(loc="upper right")

        ax_ori = fig.add_subplot(2, 4, base + 2)
        for axis_idx, lbl in enumerate(["rx", "ry", "rz"]):
            first = True
            for start, end in contiguous_segments(hand_valid):
                ax_ori.plot(
                    t[start : end + 1],
                    rotvec[start : end + 1, hand_idx, axis_idx],
                    linewidth=1.2,
                    label=lbl if first else None,
                )
                first = False
        ax_ori.set_title(f"{hand_name} — orientation rotvec components vs time")
        ax_ori.set_xlabel("Time (s)")
        ax_ori.set_ylabel("rotvec (rad)")
        ax_ori.grid(True, alpha=0.3)
        if n_valid:
            ax_ori.legend(loc="upper right")

        if hand_open is not None:
            ax_open = ax_ori.twinx()
            ax_open.step(t, hand_open[:, hand_idx], where="post", color="#34495e", alpha=0.45, linewidth=1.2)
            ax_open.set_ylim(-0.1, 1.1)
            ax_open.set_ylabel("open (0/1)")

        ax3d = fig.add_subplot(2, 4, base + 3, projection="3d")
        segments = contiguous_segments(hand_valid)
        if n_valid:
            t_valid = t[hand_valid]
            t0 = float(np.min(t_valid))
            t1 = float(np.max(t_valid))
            denom = (t1 - t0) if (t1 - t0) > 1e-12 else 1.0

            for start, end in segments:
                idx = np.arange(start, end + 1, dtype=int)
                pts = pos[idx, hand_idx]
                v = palm_dir[idx, hand_idx]
                nrm = palm_normal[idx, hand_idx]
                tn = (t[idx] - t0) / denom
                _add_gradient_trajectory_3d(ax3d, pts, tn, cmap=cmap)
                ax3d.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="#2ecc71", s=45, depthshade=False,
                             label="start" if start == segments[0][0] else None)
                ax3d.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="#e74c3c", s=45, depthshade=False,
                             label="end" if start == segments[0][0] else None)
                _add_orientation_segments_3d(
                    ax3d,
                    pts,
                    v,
                    tn,
                    scale_m=ori_scale_m,
                    stride=ori_stride,
                    cmap=cmap,
                )
                # Also overlay palm normal ticks (dashed by lower alpha via smaller scale)
                _add_orientation_segments_3d(
                    ax3d,
                    pts,
                    nrm,
                    tn,
                    scale_m=float(ori_scale_m) * 0.8,
                    stride=ori_stride,
                    cmap=cmap,
                    alpha=0.55,
                    linewidth=0.9,
                )

        ax3d.set_title(f"{hand_name} — 3D trajectory + palm_dir (solid) + palm_normal (faint)")
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")
        if n_valid:
            ax3d.legend(loc="upper right")

        # Orientation-only 3D plot (wrist fixed at origin)
        ax_o3d = fig.add_subplot(2, 4, base + 4, projection="3d")
        ax_o3d.set_title(f"{hand_name} — orientation-only (unit vectors)")
        ax_o3d.set_xlabel("X")
        ax_o3d.set_ylabel("Y")
        ax_o3d.set_zlabel("Z")
        ax_o3d.set_xlim(-1.05, 1.05)
        ax_o3d.set_ylim(-1.05, 1.05)
        ax_o3d.set_zlim(-1.05, 1.05)

        if n_valid:
            # Build a normalized 0..1 time for this hand across valid frames
            tv = t[hand_valid]
            t0 = float(np.min(tv))
            t1 = float(np.max(tv))
            denom = (t1 - t0) if (t1 - t0) > 1e-12 else 1.0

            idx_all = np.flatnonzero(hand_valid)
            tn_all = (t[idx_all] - t0) / denom
            v_all = palm_dir[idx_all, hand_idx]
            n_all = palm_normal[idx_all, hand_idx]

            # Per-frame rays from origin (what you expect visually)
            _add_orientation_rays_3d(ax_o3d, v_all, tn_all, cmap=cmap, length=1.0, stride=ori_stride, linewidth=1.6, alpha=0.9)
            _add_orientation_rays_3d(ax_o3d, n_all, tn_all, cmap=cmap, length=1.0, stride=ori_stride, linewidth=1.2, alpha=0.6)

            # start/end markers (dir and normal)
            v0, v1 = v_all[0], v_all[-1]
            n0, n1 = n_all[0], n_all[-1]
            ax_o3d.scatter(v0[0], v0[1], v0[2], color="#2ecc71", s=35, depthshade=False, label="palm_dir start")
            ax_o3d.scatter(v1[0], v1[1], v1[2], color="#e74c3c", s=35, depthshade=False, label="palm_dir end")
            ax_o3d.scatter(n0[0], n0[1], n0[2], color="#27ae60", s=25, depthshade=False, label="normal start")
            ax_o3d.scatter(n1[0], n1[1], n1[2], color="#c0392b", s=25, depthshade=False, label="normal end")

            ax_o3d.legend(loc="upper right")

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


def main():
    parser = argparse.ArgumentParser(description="Plot combined XYZ + WiLoR orientation .npz")
    parser.add_argument("combined_npz", type=str, help="Path to combined .npz")
    parser.add_argument("--ori-scale", type=float, default=0.05, help="Orientation tick length (m)")
    parser.add_argument("--ori-stride", type=int, default=2, help="Plot every Nth orientation tick")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save figure to this path")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    args = parser.parse_args()
    plot_combined(
        args.combined_npz,
        ori_scale_m=float(args.ori_scale),
        ori_stride=int(args.ori_stride),
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()

