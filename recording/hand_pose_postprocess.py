#!/usr/bin/env python3
"""Smooth and patch raw RGBD hand 6DOF trajectories before export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

HAND_SLOT = {"Left": 0, "Right": 1}


def canonicalize_hands(positions, quaternions, handedness, valid, max_hands=2):
    """Put Left in slot 0 and Right in slot 1 for consistent time series."""
    n = positions.shape[0]
    out_pos = np.full((n, max_hands, 3), np.nan, dtype=np.float64)
    out_quat = np.full((n, max_hands, 4), np.nan, dtype=np.float64)
    out_valid = np.zeros((n, max_hands), dtype=bool)
    out_labels = np.array([[""] * max_hands] * n, dtype=object)

    for t in range(n):
        used = set()
        for src in range(positions.shape[1]):
            if not valid[t, src]:
                continue
            label = str(handedness[t, src])
            if max_hands == 1:
                slot = 0
            else:
                slot = HAND_SLOT.get(label)
                if slot is None or slot in used or slot >= max_hands:
                    continue
            if slot in used:
                continue
            out_pos[t, slot] = positions[t, src]
            out_quat[t, slot] = quaternions[t, src]
            out_valid[t, slot] = True
            out_labels[t, slot] = label
            used.add(slot)
    return out_pos, out_quat, out_valid, out_labels


def reject_position_outliers(
    positions,
    valid,
    max_speed_m_s=3.0,
    max_jump_m=0.25,
    *,
    timestamps=None,
    default_fps=30.0,
):
    """
    Drop single-frame spikes using speed and per-frame jump limits.

    If timestamps are provided (seconds), compute dt from them; otherwise fall back
    to an index-based dt assuming default_fps.
    """
    cleaned = valid.copy()
    n = positions.shape[0]
    t = None
    if timestamps is not None:
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if t.shape[0] != n:
            t = None
    for h in range(positions.shape[1]):
        prev_idx = None
        prev_pos = None
        for idx in range(n):
            if not valid[idx, h]:
                continue
            pos = positions[idx, h]
            if prev_idx is not None:
                if t is not None:
                    dt = float(t[idx] - t[prev_idx])
                    if not np.isfinite(dt) or dt <= 1e-6:
                        dt = max(idx - prev_idx, 1) / float(default_fps)
                else:
                    dt = max(idx - prev_idx, 1) / float(default_fps)
                speed = np.linalg.norm(pos - prev_pos) / dt
                jump = np.linalg.norm(pos - prev_pos)
                if speed > max_speed_m_s or jump > max_jump_m:
                    cleaned[idx, h] = False
                    continue
            prev_idx = idx
            prev_pos = pos
    return cleaned


def _valid_runs(valid_1d):
    idx = np.flatnonzero(valid_1d)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.insert(idx[splits + 1], 0, idx[0])
    ends = np.append(idx[splits], idx[-1])
    return list(zip(starts, ends))


def interpolate_positions(positions, valid, max_gap_frames=15):
    """Linearly patch short gaps in xyz."""
    patched = positions.copy()
    patched_valid = valid.copy()
    n, n_hands, _ = positions.shape

    for h in range(n_hands):
        for start, end in _valid_runs(valid[:, h]):
            t0, t1 = start, end
            for axis in range(3):
                patched[t0 : t1 + 1, h, axis] = positions[t0 : t1 + 1, h, axis]

            # leading gap
            lead_start = max(0, t0 - max_gap_frames)
            if t0 > lead_start and not valid[lead_start : t0].any():
                gap_len = t0 - lead_start
                if gap_len <= max_gap_frames:
                    for t in range(lead_start, t0):
                        alpha = (t - lead_start + 1) / (gap_len + 1)
                        patched[t, h] = patched[t0, h]  # hold first valid (short leading gap)
                        patched_valid[t, h] = True

            # trailing gap
            trail_end = min(n - 1, t1 + max_gap_frames)
            if t1 < trail_end and not valid[t1 + 1 : trail_end + 1].any():
                gap_len = trail_end - t1
                if gap_len <= max_gap_frames:
                    for t in range(t1 + 1, trail_end + 1):
                        patched[t, h] = patched[t1, h]
                        patched_valid[t, h] = True

            # internal gaps
            t = t0
            while t < t1:
                if not valid[t, h]:
                    gap_start = t
                    while t <= t1 and not valid[t, h]:
                        t += 1
                    gap_end = t - 1
                    gap_len = gap_end - gap_start + 1
                    if gap_len <= max_gap_frames and gap_start > 0 and t <= t1:
                        left = patched[gap_start - 1, h]
                        right = patched[t, h] if t <= t1 else patched[gap_end + 1, h]
                        for g, tt in enumerate(range(gap_start, gap_end + 1)):
                            alpha = (g + 1) / (gap_len + 1)
                            patched[tt, h] = (1 - alpha) * left + alpha * right
                            patched_valid[tt, h] = True
                else:
                    t += 1

    # Re-scan internal gaps with valid anchor on both sides
    for h in range(n_hands):
        valid_idx = np.flatnonzero(valid[:, h])
        if valid_idx.size < 2:
            continue
        for i in range(len(valid_idx) - 1):
            a, b = valid_idx[i], valid_idx[i + 1]
            if b - a <= 1:
                continue
            gap_len = b - a - 1
            if gap_len > max_gap_frames:
                continue
            for t in range(a + 1, b):
                alpha = (t - a) / (b - a)
                patched[t, h] = (1 - alpha) * positions[a, h] + alpha * positions[b, h]
                patched_valid[t, h] = True

    return patched, patched_valid


def _normalize_quaternions(quats):
    out = quats.copy()
    for i in range(len(out)):
        if np.isnan(out[i]).any():
            continue
        n = np.linalg.norm(out[i])
        if n > 1e-8:
            out[i] /= n
    return out


def _fix_quaternion_sign(quats):
    out = quats.copy()
    for i in range(1, len(out)):
        if np.isnan(out[i]).any() or np.isnan(out[i - 1]).any():
            continue
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


def interpolate_quaternions(quaternions, valid, max_gap_frames=15):
    """SLERP patch short orientation gaps."""
    patched = quaternions.copy()
    patched_valid = valid.copy()
    n, n_hands, _ = quaternions.shape

    for h in range(n_hands):
        valid_idx = np.flatnonzero(valid[:, h])
        if valid_idx.size < 2:
            continue

        series = _normalize_quaternions(quaternions[:, h])
        series = _fix_quaternion_sign(series)

        for i in range(len(valid_idx) - 1):
            a, b = valid_idx[i], valid_idx[i + 1]
            if b - a <= 1:
                continue
            gap_len = b - a - 1
            if gap_len > max_gap_frames:
                continue

            q0 = series[a]
            q1 = series[b]
            if np.isnan(q0).any() or np.isnan(q1).any():
                continue
            if np.linalg.norm(q0) < 1e-8 or np.linalg.norm(q1) < 1e-8:
                continue
            if np.dot(q0, q1) < 0:
                q1 = -q1
            rots = Rotation.from_quat(np.vstack([q0, q1])[:, [1, 2, 3, 0]])
            slerp = Slerp([0.0, 1.0], rots)
            for t in range(a + 1, b):
                alpha = (t - a) / (b - a)
                q = slerp(alpha).as_quat()[[3, 0, 1, 2]]
                patched[t, h] = q
                patched_valid[t, h] = True

        for t in valid_idx:
            patched[t, h] = series[t]

    return patched, patched_valid


def smooth_positions(positions, valid, window=9, poly=3):
    """Savitzky-Golay smoothing on each axis for contiguous valid segments."""
    smoothed = positions.copy()
    n, n_hands, _ = positions.shape
    window = window if window % 2 == 1 else window + 1

    for h in range(n_hands):
        for start, end in _valid_runs(valid[:, h]):
            seg_len = end - start + 1
            if seg_len < window:
                continue
            for axis in range(3):
                seg = positions[start : end + 1, h, axis]
                smoothed[start : end + 1, h, axis] = savgol_filter(
                    seg, window_length=window, polyorder=min(poly, window - 1)
                )
    return smoothed


def smooth_quaternions(quaternions, valid, window=9, poly=3):
    """Smooth orientations via rotation-vector Savitzky-Golay."""
    smoothed = quaternions.copy()
    n, n_hands, _ = quaternions.shape
    window = window if window % 2 == 1 else window + 1

    for h in range(n_hands):
        for start, end in _valid_runs(valid[:, h]):
            seg_len = end - start + 1
            if seg_len < window:
                continue

            seg_q = _normalize_quaternions(quaternions[start : end + 1, h])
            seg_q = _fix_quaternion_sign(seg_q)
            if np.isnan(seg_q).any():
                continue
            seg_xyzw = seg_q[:, [1, 2, 3, 0]]
            rotvec = Rotation.from_quat(seg_xyzw).as_rotvec()

            for axis in range(3):
                rotvec[:, axis] = savgol_filter(
                    rotvec[:, axis],
                    window_length=window,
                    polyorder=min(poly, window - 1),
                )

            seg_out = Rotation.from_rotvec(rotvec).as_quat()[:, [3, 0, 1, 2]]
            smoothed[start : end + 1, h] = seg_out

    return smoothed


def postprocess_hand_pose(
    timestamps,
    positions,
    quaternions,
    valid,
    handedness,
    max_gap_frames=15,
    smooth_window=9,
    smooth_poly=3,
    max_speed_m_s=3.0,
    max_jump_m=0.25,
):
    """
    Full pipeline: canonicalize -> outlier reject -> interpolate gaps -> smooth.

    Returns dict with processed arrays and metadata.
    """
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    handedness = np.asarray(handedness, dtype=object)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    max_hands = positions.shape[1]
    pos, quat, valid, labels = canonicalize_hands(
        positions, quaternions, handedness, valid, max_hands=max_hands
    )

    valid_raw = valid.copy()
    valid = reject_position_outliers(
        pos,
        valid,
        max_speed_m_s=max_speed_m_s,
        max_jump_m=max_jump_m,
        timestamps=timestamps,
        default_fps=30.0,
    )

    pos_patched, valid_patched = interpolate_positions(pos, valid, max_gap_frames=max_gap_frames)
    quat_patched, valid_patched = interpolate_quaternions(
        quat, valid_patched, max_gap_frames=max_gap_frames
    )

    pos_smooth = smooth_positions(
        pos_patched, valid_patched, window=smooth_window, poly=smooth_poly
    )
    quat_smooth = smooth_quaternions(
        quat_patched, valid_patched, window=smooth_window, poly=smooth_poly
    )
    quat_smooth = _normalize_quaternions(quat_smooth)

    poses_6dof = np.concatenate([pos_smooth, quat_smooth], axis=-1)

    return {
        "timestamps": timestamps,
        "positions": pos_smooth,
        "quaternions": quat_smooth,
        "poses_6dof": poses_6dof,
        "handedness": labels,
        "valid_raw": valid_raw,
        "valid_after_outlier_reject": valid,
        "valid_processed": valid_patched,
        "representation": "xyz_m + quaternion_wxyz in RealSense optical frame; smoothed and gap-patched",
    }


def save_processed_npy(processed, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, processed, allow_pickle=True)
    n_valid = int(processed["valid_processed"].sum())
    n_total = processed["valid_processed"].size
    print(
        f"Saved processed hand pose -> {output_path} "
        f"({n_valid}/{n_total} hand-frames patched+smoothed, "
        f"{processed['positions'].shape[0]} time steps)"
    )
    return output_path


def postprocess_npz_file(input_path, output_path=None, **kwargs):
    input_path = Path(input_path)
    raw = np.load(input_path, allow_pickle=True)
    processed = postprocess_hand_pose(
        raw["timestamps"],
        raw["positions"],
        raw["quaternions"],
        raw["valid"],
        raw["handedness"],
        **kwargs,
    )
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_processed.npy")
    save_processed_npy(processed, output_path)
    return processed, output_path


def main():
    parser = argparse.ArgumentParser(description="Post-process raw hand pose .npz recordings.")
    parser.add_argument("input", type=str, help="Path to raw hand_pose_*.npz")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output .npy path")
    parser.add_argument("--max-gap", type=int, default=15, help="Max gap frames to interpolate")
    parser.add_argument("--smooth-window", type=int, default=9, help="Savitzky-Golay window (odd)")
    parser.add_argument("--smooth-poly", type=int, default=3, help="Savitzky-Golay polynomial order")
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
