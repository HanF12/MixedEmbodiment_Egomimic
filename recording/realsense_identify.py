#!/usr/bin/env python3
"""
Label RealSense cameras — no streaming, no freeze.

  python realsense_identify.py

Prints serial numbers. You label each one (know which USB cable is which,
or plug in one camera at a time).

Optional quick snapshot (5s max per camera, may fail if USB is busy):
  python realsense_identify.py --snap 0
"""

from __future__ import annotations

import argparse
import sys

import pyrealsense2 as rs


def get_devices():
    return list(rs.context().query_devices())


def list_serials(devices):
    rows = []
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        rows.append((i, serial, name))
    return rows


def print_map(roles):
    print("\n--- paste into hand_pose_track.py and realsense_double_record.py ---")
    print("CAMERA_MAP = {")
    for serial, role in sorted(roles.items()):
        print(f'    "{serial}": "{role}",')
    print("}")


def snap_one(serial: str, out_path: str) -> bool:
    """One frame, hard 5s cap. Returns False on any problem."""
    import time

    import cv2
    import numpy as np

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
    deadline = time.monotonic() + 5.0
    try:
        pipe.start(cfg)
        while time.monotonic() < deadline:
            try:
                frames = pipe.wait_for_frames(timeout_ms=500)
                color = frames.get_color_frame()
                if color:
                    img = np.asanyarray(color.get_data())
                    cv2.imwrite(out_path, img)
                    print(f"Saved {out_path}")
                    return True
            except RuntimeError:
                pass
        print("No frame in 5s (camera busy or unplugged?)")
        return False
    except Exception as exc:
        print(f"Could not open camera: {exc}")
        return False
    finally:
        try:
            pipe.stop()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap", type=int, default=None, metavar="INDEX",
                        help="Save one JPEG for camera index (optional)")
    args = parser.parse_args()

    devices = get_devices()
    rows = list_serials(devices)

    if not rows:
        print("No RealSense found.")
        return 1

    print(f"\n{len(rows)} camera(s):\n")
    for i, serial, name in rows:
        print(f"  [{i}]  {serial}  ({name})")
    print()

    if args.snap is not None:
        if args.snap < 0 or args.snap >= len(rows):
            print(f"Bad index {args.snap}")
            return 1
        serial = rows[args.snap][1]
        snap_one(serial, f"cam{args.snap}_{serial}.jpg")
        return 0

    print("Tip: plug in ONE camera at a time if unsure which is which.\n")
    roles = {}
    for i, serial, name in rows:
        while True:
            ans = input(f"[{i}] {serial}  role?  l=left  r=right  c=bird/center  s=skip  q=quit: ").strip().lower()
            if ans in ("q", "quit"):
                print_map(roles)
                return 0
            if ans in ("s", "skip", ""):
                break
            if ans in ("l", "left"):
                roles[serial] = "left"
                break
            if ans in ("r", "right"):
                roles[serial] = "right"
                break
            if ans in ("c", "center", "bird"):
                roles[serial] = "center"
                break
            print("  type l, r, c, s, or q")

    print_map(roles)
    return 0


if __name__ == "__main__":
    sys.exit(main())
