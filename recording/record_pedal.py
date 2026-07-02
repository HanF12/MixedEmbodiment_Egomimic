#!/usr/bin/env python3
"""
Orchestrate multi-stream recording with foot-pedal start/stop per demo.

Workflow:
  1. Press pedal  -> start recording (all streams share one timestamp id)
  2. Press pedal  -> stop recording and save files
  3. Repeat for the next demo
  4. Ctrl+C       -> stop current demo (if any) and exit
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime

from foot_pedal import cancel_pedal_wait, list_input_devices, resolve_pedal_device, wait_for_pedal
from hand_pose_track import CAMERA_MAP
from pathlib import Path
from realsense_utils import find_serial_for_role, list_connected_serials, serial_for_role

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

active_procs: list[subprocess.Popen] = []
shutting_down = False


@dataclass(frozen=True)
class RecordingMode:
    session_dir: str
    joint_arms: str | None  # left | right | both | None
    wrist_arms: str | None  # left | right | both | None
    webcam_bird: bool
    bird_realsense: bool
    hand_pose: bool
    track_hand: str  # left | right | both


RECORDING_MODES: dict[str, RecordingMode] = {
    "teleop_bimanual": RecordingMode(
        session_dir="teleop_bimanual",
        joint_arms="both",
        wrist_arms="both",
        webcam_bird=False,
        bird_realsense=True,
        hand_pose=False,
        track_hand="both",
    ),
    # Left robot arm + right human hand (left wrist cam, bird RS, right hand pose, left joints).
    "left_robot_right_hand": RecordingMode(
        session_dir="left_robot_right_hand",
        joint_arms="left",
        wrist_arms="left",
        webcam_bird=False,
        bird_realsense=True,
        hand_pose=True,
        track_hand="right",
    ),
    # Right robot arm + left human hand (opposite of above).
    "right_robot_left_hand": RecordingMode(
        session_dir="right_robot_left_hand",
        joint_arms="right",
        wrist_arms="right",
        webcam_bird=False,
        bird_realsense=True,
        hand_pose=True,
        track_hand="left",
    ),
    "human_hands_bimanual": RecordingMode(
        session_dir="human_hands_bimanual",
        joint_arms=None,
        wrist_arms=None,
        webcam_bird=False,
        bird_realsense=True,
        hand_pose=True,
        track_hand="both",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record demos with foot-pedal start/stop control.",
    )
    parser.add_argument(
        "--bird-camera",
        "-c",
        type=int,
        default=6,
        help="Webcam index for bird_record.py (default: 6)",
    )
    parser.add_argument(
        "--bird-realsense-serial",
        type=str,
        default=None,
        help="Serial for bird-view RealSense (default: auto via CAMERA_MAP center role)",
    )
    parser.add_argument(
        "--no-hand-pose",
        action="store_true",
        help="Disable RGBD hand pose on bird RealSense (video only).",
    )
    parser.add_argument(
        "--pedal-key",
        default="b",
        help="Pedal key (default: b — PCSensor keyboard on this machine). Use 'any' for any key.",
    )
    parser.add_argument(
        "--pedal-device",
        default="auto",
        help="evdev path or 'auto' for PCSensor keyboard interface (default: auto)",
    )
    parser.add_argument(
        "--stream-fps",
        type=int,
        default=15,
        help="RealSense color FPS for all 3 cameras (default: 15).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable bird RealSense preview window (arm/webcam previews stay on).",
    )
    parser.add_argument(
        "--list-pedal-devices",
        action="store_true",
        help="List evdev input devices and exit",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(RECORDING_MODES),
        default="teleop_bimanual",
        help=(
            "Recording preset (default: teleop_bimanual). "
            "Sets output folder under recording/sessions/<mode>/ and which streams run."
        ),
    )
    return parser.parse_args()


def resolve_mode(args: argparse.Namespace) -> RecordingMode:
    mode = RECORDING_MODES[args.mode]
    hand_pose = mode.hand_pose and not args.no_hand_pose
    return replace(mode, hand_pose=hand_pose)


def _spawn(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    return subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        env=proc_env,
        preexec_fn=os.setsid,
    )


def session_data_root(mode: RecordingMode) -> str:
    return os.path.join("sessions", mode.session_dir)


def build_bird_rs_cmd(
    datetime_id: str,
    bird_realsense_serial: str | None,
    hand_pose: bool,
    track_hand: str,
    display: bool,
    stream_fps: int,
) -> list[str]:
    python = sys.executable
    cmd = [
        python,
        os.path.join(SCRIPT_DIR, "realsense_bird_record.py"),
        "--datetime-id",
        datetime_id,
        "--fps",
        str(stream_fps),
        "--track-hand",
        track_hand,
    ]
    if bird_realsense_serial:
        cmd.extend(["--serial", bird_realsense_serial])
    if not hand_pose:
        cmd.append("--no-hand-pose")
    if not display:
        cmd.append("--no-display")
    return cmd


def wait_arms_ready(
    datetime_id: str,
    arm_proc: subprocess.Popen | None = None,
    timeout: float = 45.0,
) -> bool:
    ready_path = os.path.join(SCRIPT_DIR, ".recording", f"arms_ready_{datetime_id}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if shutting_down:
            return False
        if arm_proc is not None and arm_proc.poll() is not None:
            print("ERROR: arm RealSense recorder exited before becoming ready.", flush=True)
            return False
        if os.path.exists(ready_path):
            return True
        time.sleep(0.1)
    print(
        f"WARNING: arm RealSense not ready after {timeout:.0f}s; "
        "starting bird RealSense anyway.",
        flush=True,
    )
    return False


def resolve_bird_realsense_serial(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    return find_serial_for_role(CAMERA_MAP, "center")


def verify_demo_outputs(datetime_id: str, mode: RecordingMode) -> None:
    """Print whether each stream produced a non-empty file for this demo id."""
    root = session_data_root(mode)
    checks: list[tuple[str, Path | None]] = []

    if mode.wrist_arms in ("left", "both"):
        checks.append(
            (
                "LEFT wrist",
                Path(root) / "aloha-data/left/mp4" / f"video_recording_realsense_left#{datetime_id}.mp4",
            )
        )
    if mode.wrist_arms in ("right", "both"):
        checks.append(
            (
                "RIGHT wrist",
                Path(root) / "aloha-data/right/mp4" / f"video_recording_realsense_right#{datetime_id}.mp4",
            )
        )
    if mode.webcam_bird:
        checks.append(
            (
                "Webcam bird",
                Path(root) / "bird-data/mp4" / f"video_recording_bird#{datetime_id}.mp4",
            )
        )
    if mode.joint_arms in ("left", "both"):
        checks.append(
            (
                "LEFT joints",
                Path(root) / "joint-data/npy" / f"joint_position_{datetime_id}.npy",
            )
        )
    if mode.joint_arms in ("right", "both"):
        checks.append(
            (
                "RIGHT joints",
                Path(root) / "joint-data/npy" / f"joint_position_right_{datetime_id}.npy",
            )
        )

    print("Demo output check:", flush=True)
    for label, path in checks:
        if path is None or not path.exists():
            print(f"  MISSING  {label}: {path}", flush=True)
            continue
        size = path.stat().st_size
        if size < 1000:
            print(f"  EMPTY    {label}: {path} ({size} bytes)", flush=True)
        else:
            print(f"  OK       {label}: {path} ({size} bytes)", flush=True)

    if mode.bird_realsense:
        bird_rs_glob = list(Path(root, "bird-realsense-data/mp4").glob(f"*{datetime_id}.mp4"))
        bird_rs_path = bird_rs_glob[0] if bird_rs_glob else None
        if bird_rs_path is None:
            print(f"  MISSING  Bird RealSense: {root}/bird-realsense-data/mp4/*#{datetime_id}.mp4", flush=True)
        else:
            size = bird_rs_path.stat().st_size
            tag = "OK" if size >= 1000 else "EMPTY"
            print(f"  {tag:<8} Bird RealSense: {bird_rs_path} ({size} bytes)", flush=True)

    if mode.hand_pose:
        hand_glob = list(Path(root, "hand-pose-data").glob(f"hand_pose_*#{datetime_id}.npz"))
        hand_path = hand_glob[0] if hand_glob else None
        if hand_path is None:
            print(f"  MISSING  Hand pose: {root}/hand-pose-data/hand_pose_*#{datetime_id}.npz", flush=True)
        else:
            size = hand_path.stat().st_size
            tag = "OK" if size >= 1000 else "EMPTY"
            print(f"  {tag:<8} Hand pose: {hand_path} ({size} bytes)", flush=True)


def start_recorders(
    datetime_id: str,
    mode: RecordingMode,
    bird_camera: int,
    stream_fps: int,
) -> list[subprocess.Popen]:
    python = sys.executable
    common = ["--datetime-id", datetime_id]
    data_env = {"RECORDING_DATA_ROOT": session_data_root(mode)}

    procs: list[subprocess.Popen] = []

    if mode.joint_arms:
        procs.append(
            _spawn(
                [
                    python,
                    os.path.join(SCRIPT_DIR, "store_joint.py"),
                    *common,
                    "--arms",
                    mode.joint_arms,
                ],
                env=data_env,
            )
        )

    if mode.webcam_bird:
        procs.append(
            _spawn(
                [
                    python,
                    os.path.join(SCRIPT_DIR, "bird_record.py"),
                    "-c",
                    str(bird_camera),
                    *common,
                ],
                env=data_env,
            )
        )

    if mode.wrist_arms:
        if procs:
            time.sleep(0.5)
        procs.append(
            _spawn(
                [
                    python,
                    os.path.join(SCRIPT_DIR, "realsense_double_record.py"),
                    *common,
                    "--arms",
                    mode.wrist_arms,
                    "--color-fps",
                    str(stream_fps),
                ],
                env=data_env,
            )
        )
    return procs


def stop_recorders(procs: list[subprocess.Popen], timeout: float = 30.0) -> None:
    if not procs:
        return

    print("Stopping recorders (waiting for files to save)...", flush=True)
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def handle_sigint(signum, frame) -> None:
    global shutting_down
    if shutting_down:
        print("\nForce quit.", flush=True)
        cancel_pedal_wait()
        stop_recorders(active_procs)
        os._exit(130)
    shutting_down = True
    print("\nCtrl+C — stopping...", flush=True)
    cancel_pedal_wait()
    stop_recorders(active_procs)


def wait_pedal_or_quit(pedal_key: str, pedal_device: str) -> bool:
    """Wait for pedal. Returns False if interrupted by Ctrl+C."""
    try:
        wait_for_pedal(pedal_key, pedal_device, quiet=True)
        return not shutting_down
    except KeyboardInterrupt:
        return False


def required_wrist_role(mode: RecordingMode) -> str | None:
    if mode.wrist_arms in ("left", "right"):
        return mode.wrist_arms
    if mode.wrist_arms == "both":
        return "left"
    return None


def main() -> int:
    args = parse_args()

    if args.list_pedal_devices:
        list_input_devices()
        return 0

    mode = resolve_mode(args)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    pedal_path = resolve_pedal_device(args.pedal_device)
    if pedal_path is None:
        print("WARNING: No foot pedal found. Run: python3 test_pedal.py --list-devices")
    else:
        print(f"Pedal: {pedal_path}  (key={args.pedal_key!r})")

    print("=" * 60)
    print(f"Foot-pedal recording  [mode={mode.session_dir}]")
    print(f"  Output root: {session_data_root(mode)}/")
    print("  Pedal press  -> start demo")
    print("  Pedal press  -> stop demo and save")
    print("  Ctrl+C       -> quit")
    print("=" * 60)
    print()

    demo_num = 1
    global active_procs

    while not shutting_down:
        print(f"Press pedal to start demo {demo_num}...", flush=True)
        if not wait_pedal_or_quit(args.pedal_key, args.pedal_device):
            break

        raw_id = datetime.now().strftime("%Y%m%d%H%M%S")
        datetime_id = f"{mode.session_dir}_{raw_id}"
        print(f"\n>>> Demo {demo_num} START  (id={datetime_id})", flush=True)

        left_serial = find_serial_for_role(CAMERA_MAP, "left")
        left_expected = serial_for_role(CAMERA_MAP, "left") or "left"
        bird_expected = serial_for_role(CAMERA_MAP, "center") or "center"
        connected = list_connected_serials()
        print(f"RealSense on USB: {connected or '(none)'}", flush=True)
        print(
            f"  Map: left={left_expected}  right={serial_for_role(CAMERA_MAP, 'right')}  "
            f"bird={bird_expected}",
            flush=True,
        )

        required_role = required_wrist_role(mode)
        if required_role == "left" and not left_serial:
            print(
                f"ERROR: Left wrist camera ({left_expected}) not visible — skipping this demo. "
                "Reseat USB and press pedal again.",
                flush=True,
            )
            demo_num += 1
            continue
        if required_role == "right":
            right_serial = find_serial_for_role(CAMERA_MAP, "right")
            if not right_serial:
                right_expected = serial_for_role(CAMERA_MAP, "right") or "right"
                print(
                    f"ERROR: Right wrist camera ({right_expected}) not visible — skipping this demo. "
                    "Reseat USB and press pedal again.",
                    flush=True,
                )
                demo_num += 1
                continue

        active_procs = start_recorders(
            datetime_id,
            mode,
            args.bird_camera,
            args.stream_fps,
        )

        arm_proc = None
        if mode.wrist_arms:
            print("Waiting for wrist RealSense camera(s)...", flush=True)
            arm_proc = active_procs[-1]
            wait_arms_ready(datetime_id, arm_proc=arm_proc)

        if not shutting_down and mode.bird_realsense:
            bird_serial_now = resolve_bird_realsense_serial(args.bird_realsense_serial)
            if bird_serial_now:
                bird_cmd = build_bird_rs_cmd(
                    datetime_id,
                    bird_serial_now,
                    mode.hand_pose,
                    mode.track_hand,
                    not args.no_display,
                    args.stream_fps,
                )
                active_procs.append(
                    _spawn(bird_cmd, env={"RECORDING_DATA_ROOT": session_data_root(mode)})
                )
                print(f"Bird RealSense started ({bird_serial_now}).", flush=True)
            else:
                connected = list_connected_serials()
                print(
                    f"WARNING: bird RealSense (center role, serial {bird_expected}) not connected — "
                    "skipping bird-realsense-data/.",
                    flush=True,
                )
                print(f"  Connected RealSense: {connected or '(none)'}", flush=True)

        if shutting_down:
            break

        alive = [p for p in active_procs if p.poll() is None]
        if not alive:
            print("Warning: all recorders exited immediately.", flush=True)
            active_procs = []
            demo_num += 1
            continue

        print(f"    Recording... press pedal to STOP.", flush=True)
        if not wait_pedal_or_quit(args.pedal_key, args.pedal_device):
            stop_recorders(active_procs)
            active_procs = []
            break

        stop_recorders(active_procs)
        active_procs = []
        verify_demo_outputs(datetime_id, mode)
        print(f">>> Demo {demo_num} saved  (id={datetime_id})\n", flush=True)
        demo_num += 1

    stop_recorders(active_procs)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
