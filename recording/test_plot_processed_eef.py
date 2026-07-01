#!/usr/bin/env python3
"""Plot processed hand EEF positions from hand_pose_*_processed.npy."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def plot_processed_eef(processed_path, output_path=None, show=True):
    data, src_path = load_processed(processed_path)
    timestamps = data["timestamps"]
    positions = data["positions"]
    valid = data["valid_processed"]
    t = time_axis(timestamps)

    n_frames, n_hands, _ = positions.shape
    n_valid_total = int(valid.sum())
    if n_valid_total == 0:
        print(
            "WARNING: no valid processed hand frames — plot will be empty. "
            "Check valid_processed in the .npy or re-run post-processing."
        )

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Processed hand EEF positions\n{src_path.name}", fontsize=12)

    for hand_idx in range(n_hands):
        hand_name = HAND_NAMES.get(hand_idx, f"Hand {hand_idx}")
        ax = fig.add_subplot(n_hands, 2, hand_idx * 2 + 1)
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

        ax3d = fig.add_subplot(n_hands, 2, hand_idx * 2 + 2, projection="3d")
        segments = contiguous_segments(hand_valid)
        for start, end in segments:
            seg = positions[start : end + 1, hand_idx]
            ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#8e44ad", linewidth=1.2)
            ax3d.scatter(
                seg[0, 0], seg[0, 1], seg[0, 2],
                color="#27ae60", s=30,
                label="start" if start == segments[0][0] else None,
            )
            ax3d.scatter(
                seg[-1, 0], seg[-1, 1], seg[-1, 2],
                color="#c0392b", s=30,
                label="end" if start == segments[0][0] else None,
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
        description="Plot processed hand EEF positions from *_processed.npy"
    )
    parser.add_argument(
        "processed_npy",
        nargs="?",
        default="hand-pose-data/hand_pose_332522076706#20260630164257_processed.npy",
        help="Path to processed .npy file",
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
    )


if __name__ == "__main__":
    main()
