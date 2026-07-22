#!/usr/bin/env python3
"""
Normalize pose-related keys across two folders of .npz files with a *shared*
mean/std computed over all files from both folders.

Normalized keys:
  - pose          (T,2,10) or (T,10): [x,y,z, rot6d(6), open_flag]
  - pose_xyz_raw  (T,2,3)  or (T,3)
  - R_raw         (T,2,3,3) or (T,3,3)

Gripper / open_flag (pose[..., 9]) is NOT normalized; it is copied through.

All other keys are copied unchanged. Outputs are written to two separate
output folders (one per input folder). Shared mean/std are also saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

NORM_KEYS = ("pose", "pose_xyz_raw", "R_raw")


def _list_npz(folder: Path, glob_pat: str) -> List[Path]:
    files = sorted(p for p in folder.glob(glob_pat) if p.is_file() and p.suffix == ".npz")
    if not files:
        raise SystemExit(f"no files matching {glob_pat!r} in {folder}")
    return files


def _accum_add(
    sums: Dict[str, np.ndarray],
    sumsq: Dict[str, np.ndarray],
    counts: Dict[str, np.ndarray],
    key: str,
    arr: np.ndarray,
) -> None:
    """Accumulate sum / sumsq / count over the leading time axis."""
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim < 1:
        raise ValueError(f"{key}: unexpected scalar array")
    # time is axis 0
    finite = np.isfinite(x)
    x0 = np.where(finite, x, 0.0)
    s = np.sum(x0, axis=0)
    ss = np.sum(x0 * x0, axis=0)
    c = np.sum(finite, axis=0).astype(np.float64)
    if key not in sums:
        sums[key] = s
        sumsq[key] = ss
        counts[key] = c
    else:
        if sums[key].shape != s.shape:
            raise ValueError(
                f"{key}: trailing shape mismatch {sums[key].shape} vs {s.shape}"
            )
        sums[key] = sums[key] + s
        sumsq[key] = sumsq[key] + ss
        counts[key] = counts[key] + c


def compute_shared_stats(files: Sequence[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns {key: {"mean": ..., "std": ..., "count": ...}} with trailing shapes
    (no time dim). For pose, mean/std of the open channel are unused later.
    """
    sums: Dict[str, np.ndarray] = {}
    sumsq: Dict[str, np.ndarray] = {}
    counts: Dict[str, np.ndarray] = {}

    for fp in files:
        with np.load(fp, allow_pickle=False) as z:
            missing = [k for k in NORM_KEYS if k not in z.files]
            if missing:
                raise KeyError(f"{fp} missing required keys: {missing}")
            for k in NORM_KEYS:
                _accum_add(sums, sumsq, counts, k, z[k])

    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for k in NORM_KEYS:
        c = np.maximum(counts[k], 1.0)
        mean = sums[k] / c
        var = np.maximum(sumsq[k] / c - mean * mean, 0.0)
        std = np.sqrt(var)
        # avoid divide-by-zero on constant channels
        std = np.where(std < 1e-12, 1.0, std)
        stats[k] = {"mean": mean.astype(np.float64), "std": std.astype(np.float64), "count": counts[k]}
    return stats


def _normalize_array(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    out = (x - mean) / std
    # preserve non-finite entries
    m = np.isfinite(x)
    out = np.where(m, out, x)
    return out


def normalize_pose(pose: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalize xyz + rot6d; leave open_flag (last channel) unchanged.
    pose: (T,2,10) or (T,10)
    """
    p = np.asarray(pose, dtype=np.float64).copy()
    if p.shape[-1] != 10:
        raise ValueError(f"pose last dim must be 10, got {p.shape}")
    if mean.shape != p.shape[1:] or std.shape != p.shape[1:]:
        raise ValueError(
            f"pose stats shape {mean.shape}/{std.shape} incompatible with {p.shape}"
        )
    normed = _normalize_array(p, mean, std)
    # restore gripper / open_flag
    normed[..., 9] = p[..., 9]
    return normed


def normalize_file(
    in_path: Path,
    out_path: Path,
    stats: Dict[str, Dict[str, np.ndarray]],
) -> None:
    with np.load(in_path, allow_pickle=False) as z:
        keys = list(z.files)
        out = {k: z[k] for k in keys}

    out["pose"] = normalize_pose(
        out["pose"], stats["pose"]["mean"], stats["pose"]["std"]
    )
    out["pose_xyz_raw"] = _normalize_array(
        out["pose_xyz_raw"], stats["pose_xyz_raw"]["mean"], stats["pose_xyz_raw"]["std"]
    )
    out["R_raw"] = _normalize_array(
        out["R_raw"], stats["R_raw"]["mean"], stats["R_raw"]["std"]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)


def _default_out_dir(in_dir: Path) -> Path:
    return in_dir.with_name(in_dir.name + "_normalized")


def _save_stats(stats: Dict[str, Dict[str, np.ndarray]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    meta = {}
    for k, st in stats.items():
        payload[f"{k}_mean"] = st["mean"]
        payload[f"{k}_std"] = st["std"]
        payload[f"{k}_count"] = st["count"]
        meta[k] = {
            "mean_shape": list(st["mean"].shape),
            "std_shape": list(st["std"].shape),
            "total_count": int(np.sum(st["count"])),
            "note": "pose[...,9] (open/gripper) is NOT applied during normalize",
        }
    np.savez(path, **payload)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Normalize pose/pose_xyz_raw/R_raw using shared mean/std over two "
            "folders. Gripper (pose[...,9]) is left unchanged. Writes two output folders."
        )
    )
    ap.add_argument("folder_a", type=str, help="First input folder of .npz files")
    ap.add_argument("folder_b", type=str, help="Second input folder of .npz files")
    ap.add_argument(
        "--out-a",
        type=str,
        default=None,
        help="Output folder for folder_a (default: <folder_a>_normalized)",
    )
    ap.add_argument(
        "--out-b",
        type=str,
        default=None,
        help="Output folder for folder_b (default: <folder_b>_normalized)",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="*.npz",
        help="Glob for input files (default: '*.npz')",
    )
    ap.add_argument(
        "--stats-out",
        type=str,
        default=None,
        help=(
            "Optional path for shared stats .npz (also writes .json). "
            "Default: <out-a>/../shared_normalize_stats.npz next to out-a, "
            "or <folder_a.parent>/shared_normalize_stats.npz"
        ),
    )
    args = ap.parse_args()

    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)
    if not folder_a.is_dir():
        raise SystemExit(f"folder_a is not a directory: {folder_a}")
    if not folder_b.is_dir():
        raise SystemExit(f"folder_b is not a directory: {folder_b}")

    out_a = Path(args.out_a) if args.out_a else _default_out_dir(folder_a)
    out_b = Path(args.out_b) if args.out_b else _default_out_dir(folder_b)
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    files_a = _list_npz(folder_a, args.glob)
    files_b = _list_npz(folder_b, args.glob)
    all_files = list(files_a) + list(files_b)

    print(
        f"computing shared mean/std over {len(files_a)} + {len(files_b)} "
        f"= {len(all_files)} files ..."
    )
    stats = compute_shared_stats(all_files)

    stats_path = (
        Path(args.stats_out)
        if args.stats_out
        else (out_a.parent / "shared_normalize_stats.npz")
    )
    _save_stats(stats, stats_path)
    print(f"wrote stats {stats_path} and {stats_path.with_suffix('.json')}")
    for k in NORM_KEYS:
        print(
            f"  {k}: mean shape={stats[k]['mean'].shape} "
            f"count={int(np.sum(stats[k]['count']))}"
        )

    for src, dst_dir in ((files_a, out_a), (files_b, out_b)):
        for fp in src:
            out_path = dst_dir / fp.name
            normalize_file(fp, out_path, stats)
            print(f"wrote {out_path}")

    print(f"done. out_a={out_a}")
    print(f"      out_b={out_b}")


if __name__ == "__main__":
    main()
