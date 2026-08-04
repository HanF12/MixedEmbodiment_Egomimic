#!/usr/bin/env python3
"""Overlay mixed-embodiment poses on one figure.

For a demo pair:
  - robot EEF NPZ from joint-data/combined_npz_commonframe  (use robot arm slot only)
  - hand NPZ from bird-realsense-data/combined_npz_targetframe (use human hand slot only)

Default for right_robot_left_hand:
  robot_slot=1 (Right), hand_slot=0 (Left)

Files are matched by the embedded demo timestamp (YYYYMMDDHHMMSS).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from test_plot_combined_pose import (
    AXIS_COLORS,
    AXIS_LABELS,
    _add_gradient_trajectory_3d,
    _add_ticks_3d_gradient,
    _load_pose_like,
    contiguous_segments,
)

DEMO_TS_RE = re.compile(r"(20\d{12})")


def demo_timestamp(path: Path) -> str | None:
    m = DEMO_TS_RE.search(path.name)
    return m.group(1) if m else None


def index_by_demo(dir_path: Path, glob_pat: str = "*.npz") -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(dir_path.glob(glob_pat)):
        ts = demo_timestamp(p)
        if ts is None:
            continue
        out[ts] = p
    return out


def _slot_valid(valid: np.ndarray, slot: int) -> np.ndarray:
    return np.asarray(valid[:, int(slot)], dtype=bool)


def plot_mixed_pair(
    *,
    robot_npz: Path,
    hand_npz: Path,
    robot_slot: int = 1,
    hand_slot: int = 0,
    ori_scale_m: float = 0.015,
    ori_stride: int = 2,
    output: str | Path | None = None,
    show: bool = True,
) -> None:
    robot_npz = Path(robot_npz)
    hand_npz = Path(hand_npz)

    t_r, pos_r, _rv_r, palm_r, _n_r, valid_r, open_r = _load_pose_like(robot_npz)
    t_h, pos_h, _rv_h, palm_h, _n_h, valid_h, open_h = _load_pose_like(hand_npz)

    # Absolute timestamps for a shared blue→red time colormap across both streams.
    ts_r = np.asarray(np.load(robot_npz)["timestamps"], dtype=np.float64)
    ts_h = np.asarray(np.load(hand_npz)["timestamps"], dtype=np.float64)

    m_r = _slot_valid(valid_r, robot_slot)
    m_h = _slot_valid(valid_h, hand_slot)

    hand_name = "Left" if int(hand_slot) == 0 else "Right"
    robot_name = "Left" if int(robot_slot) == 0 else "Right"

    fig = plt.figure(figsize=(20, 9))
    demo = demo_timestamp(robot_npz) or robot_npz.stem
    fig.suptitle(
        f"Mixed overlay — hand {hand_name} (slot {hand_slot}) + robot {robot_name} (slot {robot_slot})\n"
        f"demo={demo}  |  3D color = shared wall-clock time (blue→red)\n"
        f"robot: {robot_npz.name}\n"
        f"hand:  {hand_npz.name}",
        fontsize=11,
    )

    gs = fig.add_gridspec(2, 3)
    ax_pos_h = fig.add_subplot(gs[0, 0])
    ax_pos_r = fig.add_subplot(gs[0, 1])
    ax_traj = fig.add_subplot(gs[0, 2], projection="3d")
    ax_open_h = fig.add_subplot(gs[1, 0])
    ax_open_r = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis("off")

    # Same colormap as test_plot_combined_pose: start=blue, end=red.
    cmap = LinearSegmentedColormap.from_list("blue_red", ["#1f77b4", "#d62728"])

    # Normalize BOTH streams onto one shared absolute-time axis so similar
    # timestamps get the same color.
    abs_chunks = []
    if np.any(m_h):
        abs_chunks.append(ts_h[m_h])
    if np.any(m_r):
        abs_chunks.append(ts_r[m_r])
    if abs_chunks:
        t0 = float(np.min(np.concatenate(abs_chunks)))
        t1 = float(np.max(np.concatenate(abs_chunks)))
    else:
        t0, t1 = 0.0, 1.0
    denom = (t1 - t0) if (t1 - t0) > 1e-12 else 1.0
    tn_h = np.clip((ts_h - t0) / denom, 0.0, 1.0)
    tn_r = np.clip((ts_r - t0) / denom, 0.0, 1.0)

    # Position timeseries (relative seconds within each file)
    for ax, t, pos, m, title in [
        (
            ax_pos_h,
            t_h,
            pos_h[:, hand_slot],
            m_h,
            f"Hand {hand_name} position ({int(m_h.sum())}/{len(t_h)})",
        ),
        (
            ax_pos_r,
            t_r,
            pos_r[:, robot_slot],
            m_r,
            f"Robot {robot_name} position ({int(m_r.sum())}/{len(t_r)})",
        ),
    ]:
        for axis_idx in range(3):
            first = True
            for s, e in contiguous_segments(m):
                ax.plot(
                    t[s : e + 1],
                    pos[s : e + 1, axis_idx],
                    color=AXIS_COLORS[axis_idx],
                    linewidth=1.4,
                    label=AXIS_LABELS[axis_idx] if first else None,
                )
                first = False
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.grid(True, alpha=0.3)
        if m.any():
            ax.legend(loc="upper right")

    # Gripper / open flags
    for ax, t, open_arr, title, slot in [
        (ax_open_h, t_h, open_h, f"Hand {hand_name} open", hand_slot),
        (ax_open_r, t_r, open_r, f"Robot {robot_name} gripper", robot_slot),
    ]:
        if open_arr is None:
            ax.set_title(title + " — n/a")
            continue
        ax.step(t, open_arr[:, slot], where="post", color="#34495e", linewidth=1.2)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("open/grip (0/1)")
        ax.grid(True, alpha=0.3)

    # Shared 3D trajectories — same blue→red cmap, shared absolute time.
    traj_pts = []
    legend_handles = []
    legend_labels = []

    for label, pos, palm, m, tn, slot in [
        (f"Hand {hand_name}", pos_h, palm_h, m_h, tn_h, hand_slot),
        (f"Robot {robot_name}", pos_r, palm_r, m_r, tn_r, robot_slot),
    ]:
        for s, e in contiguous_segments(m):
            pts = pos[s : e + 1, slot]
            traj_pts.append(pts)
            _add_gradient_trajectory_3d(ax_traj, pts, tn[s : e + 1], cmap=cmap, linewidth=2.4)
            _add_ticks_3d_gradient(
                ax_traj,
                pts,
                palm[s : e + 1, slot],
                tn[s : e + 1],
                scale_m=float(ori_scale_m),
                stride=int(ori_stride),
                cmap=cmap,
                alpha=0.65,
                linewidth=1.0,
            )
        legend_handles.append(plt.Line2D([0], [0], color="black", linewidth=2.4))
        legend_labels.append(label)

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

    ax_traj.set_title("3D overlay — shared time color (blue→red)")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_zlabel("Z (m)")
    ax_traj.legend(legend_handles, legend_labels, loc="upper right")

    ax_info.text(
        0.0,
        0.95,
        "Notes\n"
        f"- hand valid: {int(m_h.sum())}/{len(m_h)}\n"
        f"- robot valid: {int(m_r.sum())}/{len(m_r)}\n"
        "- 3D color uses shared absolute timestamps\n"
        "  (same blue→red scale for hand + robot)\n"
        "- Hand: bird combined_npz_targetframe\n"
        "- Robot: joint combined_npz_commonframe\n"
        "- Close the window to advance to the next demo.",
        va="top",
        family="monospace",
        fontsize=10,
    )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay robot EEF (one slot) + hand pose (other slot) for mixed demos."
    )
    parser.add_argument(
        "--robot_dir",
        type=str,
        required=True,
        help="Directory of robot commonframe NPZs (joint-data/combined_npz_commonframe)",
    )
    parser.add_argument(
        "--hand_dir",
        type=str,
        required=True,
        help="Directory of hand targetframe NPZs (bird-realsense-data/combined_npz_targetframe)",
    )
    parser.add_argument("--robot_slot", type=int, default=1, choices=(0, 1), help="Robot arm slot (default 1=Right)")
    parser.add_argument("--hand_slot", type=int, default=0, choices=(0, 1), help="Hand slot (default 0=Left)")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="If set, save PNGs here (non-interactive). Omit for interactive one-by-one display.",
    )
    parser.add_argument("--ori-scale", type=float, default=0.015)
    parser.add_argument("--ori-stride", type=int, default=2)
    parser.add_argument("--demo", type=str, default=None, help="Optional single demo timestamp filter")
    parser.add_argument("--no-show", action="store_true", help="Never open windows (use with -o)")
    args = parser.parse_args()

    robot_dir = Path(args.robot_dir).expanduser().resolve()
    hand_dir = Path(args.hand_dir).expanduser().resolve()
    robots = index_by_demo(robot_dir)
    hands = index_by_demo(hand_dir)
    shared = sorted(set(robots) & set(hands))
    if args.demo:
        shared = [d for d in shared if d == args.demo]
    if not shared:
        raise SystemExit(f"No matched demos between\n  {robot_dir}\n  {hand_dir}")

    save = args.output_dir is not None
    if save:
        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Matched demos: {len(shared)}")
        print(f"Saving under: {out_dir}")
        show = False
    else:
        out_dir = None
        print(f"Matched demos: {len(shared)} (interactive; close each window to continue)")
        show = not args.no_show

    for i, demo in enumerate(shared, 1):
        out = None
        if out_dir is not None:
            out = out_dir / f"{demo}_mixed_hand{args.hand_slot}_robot{args.robot_slot}.png"
        print(f"[{i}/{len(shared)}] {demo}")
        plot_mixed_pair(
            robot_npz=robots[demo],
            hand_npz=hands[demo],
            robot_slot=int(args.robot_slot),
            hand_slot=int(args.hand_slot),
            ori_scale_m=float(args.ori_scale),
            ori_stride=int(args.ori_stride),
            output=out,
            show=show,
        )
    print(f"Done. {len(shared)} demos.")


if __name__ == "__main__":
    main()
