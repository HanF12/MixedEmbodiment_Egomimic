#!/usr/bin/env python3
"""Post-process a raw bird RealSense hand-pose .npz (offline re-run)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hand_pose_postprocess import postprocess_npz_file


def main():
    parser = argparse.ArgumentParser(
        description="Smooth/patch raw hand_pose_*.npz from bird RealSense recording."
    )
    parser.add_argument("input", type=str, help="Path to hand_pose_<serial>#<id>.npz")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output .npy path")
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--smooth-poly", type=int, default=3)
    args = parser.parse_args()

    postprocess_npz_file(
        args.input,
        output_path=args.output,
        max_gap_frames=args.max_gap,
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
    )


if __name__ == "__main__":
    main()
