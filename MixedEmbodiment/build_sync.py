#!/usr/bin/env python3
"""
Build sync CSVs for mixed one-hand + one-robot-arm sessions.

Required per demo:
  bird + front camera npy timestamps
  one wrist camera npy (robot side)
  one arm joint timestamp npy (robot side)
  hand-pose NPZ (human side slot must be xyz+gripper valid)
  robot EEF NPZ in joint-data/combined_npz_commonframe (robot side slot valid)

Example:
  python -m MixedEmbodiment.build_sync \\
    --data_root recording/sessions/left_robot_right_hand/0720 \\
    --sync_dir recording/sessions/left_robot_right_hand/0720/sync_csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as `python MixedEmbodiment/build_sync.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Combined.config import HUMAN_POSE_RELDIR, ROBOT_EEF_RELDIR
from Combined.dataloader_utils import (
    demo_id_from_hash_filename,
    demo_id_from_joint_npy,
    demo_id_from_pose_npz,
    demo_id_from_robot_eef_npz,
)
from MixedEmbodiment.data_synchronization import (
    EMBODIMENT_PRESETS,
    Side,
    synchronize_mixed_hand_robot,
)


def _map_hash(dir_path: Path) -> dict[str, Path]:
    if not dir_path.is_dir():
        return {}
    out: dict[str, Path] = {}
    for p in sorted(dir_path.glob("*.npy")):
        out[demo_id_from_hash_filename(p)] = p
    return out


def _resolve_pose_dir(data_root: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    preferred = (data_root / HUMAN_POSE_RELDIR).resolve()
    if preferred.is_dir() and any(preferred.glob("*.npz")):
        return preferred
    fallback = (data_root / "bird-realsense-data" / "combined_npz").resolve()
    if fallback.is_dir() and any(fallback.glob("*.npz")):
        print(f"WARNING: falling back to hand pose dir {fallback}")
        return fallback
    raise FileNotFoundError(f"Hand pose NPZ dir not found under {data_root}")


def _resolve_eef_dir(data_root: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    preferred = (data_root / ROBOT_EEF_RELDIR).resolve()
    if preferred.is_dir() and any(preferred.glob("*.npz")):
        return preferred
    raise FileNotFoundError(f"Robot EEF NPZ dir not found: {preferred}")


def _infer_preset(data_root: Path) -> dict[str, Side] | None:
    parts = {p.lower() for p in data_root.parts}
    for name, preset in EMBODIMENT_PRESETS.items():
        if name in parts:
            return preset
    return None


def build_mixed_sync_csvs(
    data_root: Path,
    sync_dir: Path,
    *,
    robot_side: Side,
    hand_side: Side,
    pose_dir: Path,
    eef_dir: Path,
    max_skew_s: float,
    max_demos: int | None,
) -> list[str]:
    bird = _map_hash(data_root / "bird-realsense-data" / "npy")
    front = _map_hash(data_root / "front-realsense-data" / "npy")
    wrist = _map_hash(data_root / "aloha-data" / robot_side / "npy")

    joint_dir = data_root / "joint-data" / robot_side / "time"
    joints: dict[str, Path] = {}
    if joint_dir.is_dir():
        for p in sorted(joint_dir.glob("*.npy")):
            joints[demo_id_from_joint_npy(p, prefix="joint_timestamp_")] = p

    hands = {demo_id_from_pose_npz(p): p for p in sorted(pose_dir.glob("*.npz"))}
    eefs = {demo_id_from_robot_eef_npz(p): p for p in sorted(eef_dir.glob("*.npz"))}

    ids = sorted(set(bird) & set(front) & set(wrist) & set(joints) & set(hands) & set(eefs))
    if max_demos is not None and max_demos > 0:
        ids = ids[: int(max_demos)]

    print(
        f"Mixed sync discovery under {data_root}\n"
        f"  robot_side={robot_side} hand_side={hand_side}\n"
        f"  bird={len(bird)} front={len(front)} wrist({robot_side})={len(wrist)} "
        f"joint({robot_side})={len(joints)} hand_npz={len(hands)} eef_npz={len(eefs)}\n"
        f"  complete demos={len(ids)} -> {sync_dir}"
    )
    if not ids:
        raise FileNotFoundError(
            f"No complete mixed demos under {data_root}. "
            f"Need bird, front, {robot_side} wrist, {robot_side} joints, "
            f"hand NPZ in {pose_dir}, EEF NPZ in {eef_dir}."
        )

    sync_dir.mkdir(parents=True, exist_ok=True)
    wrote: list[str] = []
    for demo_id in ids:
        hand_npz = np.load(hands[demo_id])
        eef_npz = np.load(eefs[demo_id])
        for key in ("timestamps", "valid_pos", "valid_open"):
            if key not in hand_npz.files:
                raise KeyError(f"{hands[demo_id].name} missing '{key}'")
            if key not in eef_npz.files:
                raise KeyError(f"{eefs[demo_id].name} missing '{key}'")

        out_csv = sync_dir / f"{demo_id}.csv"
        synchronize_mixed_hand_robot(
            np.load(bird[demo_id]),
            np.load(front[demo_id]),
            np.load(wrist[demo_id]),
            np.load(joints[demo_id]),
            hand_npz["timestamps"],
            eef_npz["timestamps"],
            out_csv,
            robot_side=robot_side,
            hand_side=hand_side,
            hand_valid_pos=hand_npz["valid_pos"],
            hand_valid_open=hand_npz["valid_open"],
            eef_valid_pos=eef_npz["valid_pos"],
            eef_valid_open=eef_npz["valid_open"],
            max_skew_s=max_skew_s,
            debug=False,
            require_valid_active_slots=True,
        )
        wrote.append(demo_id)
    print(f"Done. Wrote {len(wrote)} mixed sync CSVs -> {sync_dir}")
    return wrote


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mixed one-hand + one-arm sync CSVs")
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Session date root, e.g. recording/sessions/left_robot_right_hand/0720",
    )
    p.add_argument(
        "--sync_dir",
        type=str,
        default=None,
        help="Output CSV directory (default: <data_root>/sync_csv)",
    )
    p.add_argument("--robot_side", choices=["left", "right"], default=None)
    p.add_argument("--hand_side", choices=["left", "right"], default=None)
    p.add_argument("--hand_pose_dir", type=str, default=None)
    p.add_argument("--robot_eef_dir", type=str, default=None)
    p.add_argument("--max_skew_s", type=float, default=0.050)
    p.add_argument("--max_demos", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    preset = _infer_preset(data_root)
    robot_side: Side = args.robot_side or (preset["robot_side"] if preset else None)  # type: ignore[assignment]
    hand_side: Side = args.hand_side or (preset["hand_side"] if preset else None)  # type: ignore[assignment]
    if robot_side is None or hand_side is None:
        raise SystemExit(
            "Could not infer robot_side/hand_side from path. "
            "Pass --robot_side and --hand_side explicitly "
            "(or use a session folder named left_robot_right_hand / right_robot_left_hand)."
        )
    if robot_side == hand_side:
        raise SystemExit("Mixed embodiment expects opposite robot_side and hand_side")

    sync_dir = (
        Path(args.sync_dir).expanduser().resolve()
        if args.sync_dir
        else (data_root / "sync_csv").resolve()
    )
    pose_dir = _resolve_pose_dir(data_root, args.hand_pose_dir)
    eef_dir = _resolve_eef_dir(data_root, args.robot_eef_dir)

    build_mixed_sync_csvs(
        data_root,
        sync_dir,
        robot_side=robot_side,
        hand_side=hand_side,
        pose_dir=pose_dir,
        eef_dir=eef_dir,
        max_skew_s=float(args.max_skew_s),
        max_demos=args.max_demos,
    )


if __name__ == "__main__":
    main()
