#!/usr/bin/env python3
"""Plot processed hand EEF pose (position + orientation) from hand_pose_*_processed.npy."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

HAND_NAMES = {0: "Left", 1: "Right"}
AXIS_LABELS = ["X", "Y", "Z"]
AXIS_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]


def load_processed(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed file not found: {path}")

    data = np.load(path, allow_pickle=True).item()
    required = ("timestamps", "positions", "valid_processed")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in processed file: {missing}")
    return data, path


def time_axis(timestamps):
    t = np.asarray(timestamps, dtype=np.float64)
    return t - t[0]


def contiguous_segments(mask):
    """Return (start, end) index pairs for True runs in a 1D boolean mask."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.insert(idx[splits + 1], 0, idx[0])
    ends = np.append(idx[splits], idx[-1])
    return list(zip(starts, ends))


def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / n, v)


def _quat_normalize_wxyz(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n > eps, q / n, q)


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:4] *= -1.0
    return out


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, quaternions in (w,x,y,z)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_to_rotvec_wxyz(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert quaternion(s) (w,x,y,z) to rotation vector(s) (rx,ry,rz) in radians.
    Assumes q is near unit; will normalize defensively.
    """
    q = _quat_normalize_wxyz(q, eps=eps)
    w = np.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:4]
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(v_norm, np.abs(w)[..., None])
    axis = np.where(v_norm > eps, v / v_norm, v)
    rotvec = axis * angle
    # Preserve sign convention: if w < 0, flip to keep shortest representation
    flip = (w < 0)[..., None]
    return np.where(flip, -rotvec, rotvec)


def _quat_rotate_vec_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector(s) v by quaternion(s) q in (w, x, y, z) format.
    Supports broadcasting across leading dimensions.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., 0:1]
    qv = q[..., 1:4]
    # v' = v + 2 * cross(qv, cross(qv, v) + w*v)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _axis_unit(axis: str) -> np.ndarray:
    axis = axis.lower().strip()
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raise ValueError(f"Unknown axis: {axis!r} (expected x/y/z)")


def _add_gradient_trajectory_3d(
    ax3d,
    pts_xyz: np.ndarray,
    t_norm: np.ndarray,
    *,
    linewidth: float = 1.6,
    cmap=None,
    alpha: float = 1.0,
):
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
    quat_wxyz: np.ndarray,
    t_norm: np.ndarray,
    *,
    axis: str = "z",
    scale_m: float = 0.05,
    stride: int = 2,
    cmap=None,
    alpha: float = 0.9,
    linewidth: float = 1.0,
):
    if pos_xyz.shape[0] == 0:
        return None
    stride = max(int(stride), 1)
    p = pos_xyz[::stride]
    q = quat_wxyz[::stride]
    tn = t_norm[::stride]

    base_axis = _axis_unit(axis)
    base_axis = _normalize_rows(base_axis)
    dirs = _quat_rotate_vec_wxyz(q, base_axis)  # (N,3)
    dirs = _normalize_rows(dirs)
    end = p + float(scale_m) * dirs
    segs = np.stack([p, end], axis=1)  # (N,2,3)
    colors = cmap(tn)
    colors[:, 3] *= alpha
    lc = Line3DCollection(segs, colors=colors, linewidths=linewidth)
    ax3d.add_collection3d(lc)
    return lc


def plot_processed_eef(
    processed_path,
    output_path=None,
    show=True,
    *,
    orientation_axis: str = "z",
    orientation_scale_m: float = 0.05,
    orientation_stride: int = 2,
):
    data, src_path = load_processed(processed_path)
    timestamps = data["timestamps"]
    positions = data["positions"]
    quaternions = data.get("quaternions", None)
    valid = data["valid_processed"]
    t = time_axis(timestamps)

    n_frames, n_hands, _ = positions.shape
    n_valid_total = int(valid.sum())
    if n_valid_total == 0:
        print(
            "WARNING: no valid processed hand frames — plot will be empty. "
            "Check valid_processed in the .npy or re-run post-processing."
        )

    has_quat = quaternions is not None
    ncols = 3 if has_quat else 2

    fig = plt.figure(figsize=(18 if has_quat else 14, 10))
    fig.suptitle(f"Processed hand EEF pose (position + orientation)\n{src_path.name}", fontsize=12)

    cmap = LinearSegmentedColormap.from_list("history_blue_red", ["#1f77b4", "#d62728"])

    for hand_idx in range(n_hands):
        hand_name = HAND_NAMES.get(hand_idx, f"Hand {hand_idx}")
        base = hand_idx * ncols
        ax = fig.add_subplot(n_hands, ncols, base + 1)
        hand_valid = valid[:, hand_idx]
        n_valid = int(hand_valid.sum())

        for axis_idx in range(3):
            first = True
            for start, end in contiguous_segments(hand_valid):
                ax.plot(
                    t[start : end + 1],
                    positions[start : end + 1, hand_idx, axis_idx],
                    color=AXIS_COLORS[axis_idx],
                    linewidth=1.5,
                    label=AXIS_LABELS[axis_idx] if first else None,
                )
                first = False

        ax.set_title(f"{hand_name} — position vs time ({n_valid}/{n_frames} frames)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.grid(True, alpha=0.3)
        if n_valid > 0:
            ax.legend(loc="upper right")

        ax_rot = None
        if has_quat:
            ax_rot = fig.add_subplot(n_hands, ncols, base + 2)
            ax_rot.set_title(f"{hand_name} — rotation change vs time")
            ax_rot.set_xlabel("Time (s)")
            ax_rot.set_ylabel("Δrotation (rad)")
            ax_rot.grid(True, alpha=0.3)

            for axis_idx in range(3):
                first = True
                for start, end in contiguous_segments(hand_valid):
                    if end - start < 1:
                        continue
                    idx = np.arange(start, end + 1, dtype=int)
                    qs = np.asarray(quaternions[idx, hand_idx], dtype=np.float64)
                    qs = _quat_normalize_wxyz(qs)
                    # Enforce continuity inside segment (q and -q represent same rotation).
                    for i in range(1, qs.shape[0]):
                        if np.dot(qs[i], qs[i - 1]) < 0:
                            qs[i] = -qs[i]

                    dq = _quat_mul_wxyz(qs[1:], _quat_conj_wxyz(qs[:-1]))
                    drot = _quat_to_rotvec_wxyz(dq)  # (N-1,3)
                    tt = t[idx[1:]]
                    ax_rot.plot(
                        tt,
                        drot[:, axis_idx],
                        color=AXIS_COLORS[axis_idx],
                        linewidth=1.2,
                        label=AXIS_LABELS[axis_idx] if first else None,
                    )
                    first = False

            if n_valid > 0:
                ax_rot.legend(loc="upper right")

        ax3d = fig.add_subplot(n_hands, ncols, base + ncols, projection="3d")
        segments = contiguous_segments(hand_valid)
        if n_valid > 0:
            t_valid = t[hand_valid]
            t0 = float(np.min(t_valid))
            t1 = float(np.max(t_valid))
            denom = (t1 - t0) if (t1 - t0) > 1e-12 else 1.0

            for start, end in segments:
                idx = np.arange(start, end + 1, dtype=int)
                pts = positions[idx, hand_idx]
                tn = (t[idx] - t0) / denom
                _add_gradient_trajectory_3d(ax3d, pts, tn, linewidth=2.0, cmap=cmap, alpha=0.95)

                # Explicit markers for start/end of each contiguous run
                ax3d.scatter(
                    pts[0, 0], pts[0, 1], pts[0, 2],
                    color="#2ecc71", s=45, depthshade=False,
                    label="start" if start == segments[0][0] else None,
                )
                ax3d.scatter(
                    pts[-1, 0], pts[-1, 1], pts[-1, 2],
                    color="#e74c3c", s=45, depthshade=False,
                    label="end" if start == segments[0][0] else None,
                )

                # Orientation overlay (short line per time-step)
                if has_quat:
                    qs = quaternions[idx, hand_idx]
                    _add_orientation_segments_3d(
                        ax3d,
                        pts,
                        qs,
                        tn,
                        axis=orientation_axis,
                        scale_m=orientation_scale_m,
                        stride=orientation_stride,
                        cmap=cmap,
                        alpha=0.85,
                        linewidth=1.0,
                    )

        ax3d.set_title(f"{hand_name} — 3D trajectory")
        ax3d.set_xlabel("X (m)")
        ax3d.set_ylabel("Y (m)")
        ax3d.set_zlabel("Z (m)")
        if hand_valid.any():
            ax3d.legend(loc="upper right")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot processed hand EEF pose (position + orientation) from *_processed.npy"
    )
    parser.add_argument(
        "processed_npy",
        nargs="?",
        default="hand-pose-data/hand_pose_332522076706#20260630164257_processed.npy",
        help="Path to processed .npy file",
    )
    parser.add_argument(
        "--ori-axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Which local axis to visualize as a direction (x/y/z) from quaternion",
    )
    parser.add_argument(
        "--ori-scale",
        type=float,
        default=0.05,
        help="Orientation line length in meters",
    )
    parser.add_argument(
        "--ori-stride",
        type=int,
        default=2,
        help="Plot every Nth orientation segment to reduce clutter",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Save figure to this path instead of only showing interactively",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful with --output)",
    )
    args = parser.parse_args()

    plot_processed_eef(
        args.processed_npy,
        output_path=args.output,
        show=not args.no_show,
        orientation_axis=args.ori_axis,
        orientation_scale_m=args.ori_scale,
        orientation_stride=args.ori_stride,
    )


if __name__ == "__main__":
    main()
