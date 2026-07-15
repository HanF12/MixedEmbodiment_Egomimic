"""Foot-pedal demo sync: recorders wait for a shared go signal before capturing."""

from __future__ import annotations

import time
from pathlib import Path

SIGNAL_DIR = Path(__file__).resolve().parent / ".recording"


def _signal_path(name: str, datetime_id: str) -> Path:
    return SIGNAL_DIR / f"{name}_{datetime_id}"


def recording_go_path(datetime_id: str) -> Path:
    return _signal_path("recording_go", datetime_id)


def recording_start_path(datetime_id: str) -> Path:
    return _signal_path("recording_start", datetime_id)


def wrist_ready_path(datetime_id: str, side: str) -> Path:
    return _signal_path(f"wrist_{side}_ready", datetime_id)


def bird_ready_path(datetime_id: str) -> Path:
    return _signal_path("bird_ready", datetime_id)


def wait_for_recording_go(
    datetime_id: str,
    label: str = "",
    poll: float = 0.05,
    should_stop=None,
) -> bool:
    path = recording_go_path(datetime_id)
    if label:
        print(f"[{label}] Booted — waiting for sync go...", flush=True)
    while not path.exists():
        if should_stop is not None and should_stop():
            return False
        time.sleep(poll)
    if label:
        print(f"[{label}] Go — draining buffers...", flush=True)
    return True


def read_recording_start(datetime_id: str) -> float | None:
    path = recording_start_path(datetime_id)
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return None


def signal_recording_go(datetime_id: str) -> float:
    """Publish shared wall-clock t0, then release all recorders."""
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recording_start_path(datetime_id).write_text(f"{t0:.9f}\n")
    recording_go_path(datetime_id).touch()
    return t0


def cleanup_sync_signals(datetime_id: str) -> None:
    for prefix in (
        "recording_go",
        "recording_start",
        "wrist_left_ready",
        "wrist_right_ready",
        "wrist_front_ready",
        "bird_ready",
    ):
        try:
            _signal_path(prefix, datetime_id).unlink(missing_ok=True)
        except OSError:
            pass
