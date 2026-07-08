"""Shared RealSense helpers for multi-camera recording."""

from __future__ import annotations

import threading
import time

import pyrealsense2 as rs


def list_connected_serials() -> list[str]:
    ctx = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]


def serial_for_role(camera_map: dict[str, str], role: str) -> str | None:
    for serial, mapped_role in camera_map.items():
        if mapped_role == role:
            return serial
    return None


def find_serial_for_role(camera_map: dict[str, str], role: str) -> str | None:
    for serial, mapped_role in camera_map.items():
        if mapped_role == role:
            for connected in list_connected_serials():
                if connected == serial or connected.endswith(serial) or serial in connected:
                    return connected
    return None


def poll_for_frames(pipeline, timeout_ms: int = 500, should_stop=None):
    """
    Return the next frame set if one arrives within timeout_ms, else None.

    Uses poll_for_frames() so callers never block for multi-second USB stalls.
    Pass should_stop as a zero-arg callable that returns True to abort early.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return None
        frames = pipeline.poll_for_frames()
        if frames:
            return frames
        time.sleep(0.002)
    return None


def warmup_pipeline(pipeline, frames_needed: int = 5, timeout_ms: int = 200, should_stop=None) -> bool:
    """Discard a few frames so exposure/auto-focus settle. Never blocks long."""
    got = 0
    deadline = time.monotonic() + 3.0
    while got < frames_needed and time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return False
        frames = poll_for_frames(pipeline, timeout_ms=timeout_ms, should_stop=should_stop)
        if frames is not None:
            got += 1
    return got > 0


def drain_pipeline(
    pipeline,
    max_ms: float = 400,
    idle_ms: float = 60,
    should_stop=None,
) -> int:
    """
    Discard queued frames after the go signal (clears countdown/pre-roll buffer).

    Stops when no frames arrive for idle_ms, or after max_ms.
    """
    discarded = 0
    idle_start = None
    deadline = time.monotonic() + max_ms / 1000.0
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            break
        frames = pipeline.poll_for_frames()
        if frames:
            discarded += 1
            idle_start = None
            continue
        if idle_start is None:
            idle_start = time.monotonic()
        elif time.monotonic() - idle_start >= idle_ms / 1000.0:
            break
        time.sleep(0.001)
    return discarded


def start_pipelines_parallel(entries, should_stop=None) -> dict:
    """
    Start multiple RealSense pipelines at the same instant (thread barrier).

    entries: iterable of (key, pipeline, config)
    Returns {key: profile} for each successfully started pipeline.
    """
    entries = list(entries)
    if not entries:
        return {}

    barrier = threading.Barrier(len(entries))
    profiles: dict = {}
    errors: dict = {}
    lock = threading.Lock()

    def _start_one(key, pipeline, config) -> None:
        try:
            barrier.wait()
            profile = pipeline.start(config)
            enable_global_time(profile.get_device())
            warmup_pipeline(pipeline, should_stop=should_stop)
            with lock:
                profiles[key] = profile
        except Exception as exc:
            with lock:
                errors[key] = exc

    threads = [
        threading.Thread(target=_start_one, args=(key, pipeline, config), daemon=True)
        for key, pipeline, config in entries
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        for _, pipeline, _ in entries:
            try:
                pipeline.stop()
            except Exception:
                pass
        detail = ", ".join(f"{key}: {exc}" for key, exc in errors.items())
        raise RuntimeError(f"Failed to start RealSense pipeline(s): {detail}")
    return profiles


def drain_pipelines(pipelines, label: str = "", should_stop=None) -> int:
    if not pipelines:
        return 0
    if len(pipelines) == 1:
        total = drain_pipeline(pipelines[0], should_stop=should_stop)
    else:
        totals: list[int] = []

        def _drain_one(pipe) -> None:
            totals.append(drain_pipeline(pipe, should_stop=should_stop))

        threads = [
            threading.Thread(target=_drain_one, args=(pipe,), daemon=True)
            for pipe in pipelines
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        total = sum(totals)

    if label:
        print(f"[{label}] Drained {total} stale frame(s) from pipeline buffer(s).", flush=True)
    return total


def enable_global_time(device) -> None:
    """Enable RealSense global timestamps when supported (cross-camera hardware clock)."""
    try:
        for sensor in device.sensors:
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)
    except Exception:
        pass


def poll_aligned_frame_sets(pipelines: dict, timeout_ms: int = 100, should_stop=None) -> dict | None:
    """
    Block until every pipeline has a color frame, or return None on timeout.

    pipelines: {role: rs.pipeline}
    Returns {role: frameset} stamped together, or None if incomplete.
    """
    if not pipelines:
        return None

    pending = dict(pipelines)
    got: dict = {}
    deadline = time.monotonic() + timeout_ms / 1000.0
    while pending and time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            return None
        for role in list(pending):
            frames = pending[role].poll_for_frames()
            if not frames:
                continue
            color_frame = frames.get_color_frame()
            if color_frame:
                got[role] = frames
                del pending[role]
        if not pending:
            return got
        time.sleep(0.001)
    return None
