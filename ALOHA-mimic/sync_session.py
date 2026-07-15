#!/usr/bin/env python3
"""
Sync a recorded session folder into per-demo CSVs.

Supports:
  - 3-cam sync: left wrist + right wrist + bird + joints
  - 4-cam sync: left wrist + right wrist + bird + front + joints

Example:
  python3 ALOHA-mimic/sync_session.py recording/sessions/teleop_bimanual/0714 --with-front
"""

from __future__ import annotations

import argparse
from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd


def _load_sync_module() -> object:
    mod_path = (Path(__file__).resolve().parent / "data_synchronization.py").resolve()
    spec = importlib.util.spec_from_file_location("data_synchronization", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module at {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync a recording session into index CSVs.")
    p.add_argument("session_dir", type=str, help="Path like recording/sessions/<mode>/<run-subdir>")
    p.add_argument("--bird-serial", type=str, default="332522076706")
    p.add_argument("--max-skew-s", type=float, default=0.050)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--with-front", action="store_true", help="Include front camera timestamps.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: ALOHA-mimic/m-synced-csvs/<session_basename>[_front])",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ds = _load_sync_module()

    root = Path(args.session_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Session dir not found: {root}")

    bird_serial = str(args.bird_serial)

    bird_dir = root / "bird-realsense-data" / "npy"
    left_dir = root / "aloha-data" / "left" / "npy"
    right_dir = root / "aloha-data" / "right" / "npy"
    joint_dir = root / "joint-data" / "right" / "time"
    front_dir = root / "front-realsense-data" / "npy"

    joint_paths = sorted(joint_dir.glob("joint_timestamp_teleop_bimanual_*.npy"))
    demo_ids = [p.stem.replace("joint_timestamp_", "") for p in joint_paths]
    if not demo_ids:
        print(f"No right joint timestamp files under {joint_dir}")
        return 2

    tag = root.name + ("_front" if args.with_front else "")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path(__file__).resolve().parent / "m-synced-csvs" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for demo_id in demo_ids:
        bird_ts_path = bird_dir / f"video_recording_bird_realsense_{bird_serial}#{demo_id}.npy"
        left_ts_path = left_dir / f"video_recording_realsense_left#{demo_id}.npy"
        right_ts_path = right_dir / f"video_recording_realsense_right#{demo_id}.npy"
        joint_ts_path = joint_dir / f"joint_timestamp_{demo_id}.npy"
        front_ts_path = front_dir / f"video_recording_realsense_front#{demo_id}.npy"

        needed = [joint_ts_path, left_ts_path, right_ts_path, bird_ts_path]
        if args.with_front:
            needed.append(front_ts_path)

        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            rows.append({"demo_id": demo_id, "status": "missing_inputs", "missing": ";".join(missing)})
            continue

        joint_ts = np.load(joint_ts_path)
        left_ts = np.load(left_ts_path)
        right_ts = np.load(right_ts_path)
        bird_ts = np.load(bird_ts_path)
        front_ts = np.load(front_ts_path) if args.with_front else None

        lens = {
            "joint_n": int(len(joint_ts)),
            "left_n": int(len(left_ts)),
            "right_n": int(len(right_ts)),
            "bird_n": int(len(bird_ts)),
        }
        if args.with_front and front_ts is not None:
            lens["front_n"] = int(len(front_ts))

        out_csv = out_dir / f"{demo_id}.csv"
        try:
            if args.with_front and front_ts is not None:
                ds.synchronize_with_front(
                    joint_ts,
                    left_ts,
                    right_ts,
                    bird_ts,
                    front_ts,
                    str(out_csv),
                    max_skew_s=float(args.max_skew_s),
                    debug=bool(args.debug),
                )
            else:
                ds.synchronize(
                    joint_ts,
                    left_ts,
                    right_ts,
                    bird_ts,
                    str(out_csv),
                    max_skew_s=float(args.max_skew_s),
                    debug=bool(args.debug),
                )
            df = pd.read_csv(out_csv)
            synced = int(len(df))
            denom_min = max(1, min(v for k, v in lens.items() if k.endswith("_n")))
            rows.append(
                {
                    "demo_id": demo_id,
                    "status": "ok",
                    **lens,
                    "synced_rows": synced,
                    "coverage_vs_min_%": 100.0 * synced / denom_min,
                }
            )
        except Exception as e:
            rows.append({"demo_id": demo_id, "status": "error", **lens, "error": repr(e)})

    report = pd.DataFrame(rows).sort_values("demo_id")
    report_path = out_dir / "_sync_coverage_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

