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
    min_segment_frames: int = 3,
    min_segment_frac: float = 0.15,
):
    """
    Postprocess-only: clean a completed (N, hands, 3) track. Do not call
    this while iterating frames — collect all depth/deprojections first.

    1) Drop isolated spikes (jump in *and* out, while neighbors stay close).
    2) Split the remaining track at jumps that exceed ``max_jump_m`` /
       ``max_speed_m_s``.
    3) Drop short outlier islands vs the longest continuous segment
       (``min_segment_frames`` and ``min_segment_frac * longest``).

    Surviving large discontinuities stay as invalid gaps (not smoothed over);
    short gaps can be filled later by ``interpolate_positions``.
    """
    cleaned = valid.copy()
    n = positions.shape[0]
    t = None
    if timestamps is not None:
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if t.shape[0] != n:
            t = None

    def _dt(i0: int, i1: int) -> float:
        if t is not None:
            dt = float(t[i1] - t[i0])
            if np.isfinite(dt) and dt > 1e-6:
                return dt
        return max(i1 - i0, 1) / float(default_fps)

    min_seg = max(1, int(min_segment_frames))
    frac = float(min_segment_frac)
    max_jump = float(max_jump_m)
    max_speed = float(max_speed_m_s)

    for h in range(positions.shape[1]):
        # --- pass 1: isolated spikes (need both neighbors) ---
        idxs = np.flatnonzero(cleaned[:, h])
        for k in range(1, idxs.size - 1):
            i0, i1, i2 = int(idxs[k - 1]), int(idxs[k]), int(idxs[k + 1])
            p0, p1, p2 = positions[i0, h], positions[i1, h], positions[i2, h]
            j01 = float(np.linalg.norm(p1 - p0))
            j12 = float(np.linalg.norm(p2 - p1))
            j02 = float(np.linalg.norm(p2 - p0))
            s01 = j01 / _dt(i0, i1)
            s12 = j12 / _dt(i1, i2)
            spike_jump = j01 > max_jump and j12 > max_jump and j02 <= max_jump
            spike_speed = s01 > max_speed and s12 > max_speed and j02 <= max_jump
            if spike_jump or spike_speed:
                cleaned[i1, h] = False

        # --- pass 2: split at remaining large jumps; drop short islands ---
        idxs = np.flatnonzero(cleaned[:, h])
        if idxs.size == 0:
            continue
        segments: list[list[int]] = [[int(idxs[0])]]
        for k in range(1, idxs.size):
            i0 = int(idxs[k - 1])
            i1 = int(idxs[k])
            jump = float(np.linalg.norm(positions[i1, h] - positions[i0, h]))
            speed = jump / _dt(i0, i1)
            if jump > max_jump or speed > max_speed:
                segments.append([i1])
            else:
                segments[-1].append(i1)

        longest = max(len(seg) for seg in segments)
        min_keep = min_seg if longest < min_seg else max(min_seg, int(np.ceil(frac * longest)))
        keep = np.zeros(n, dtype=bool)
        for seg in segments:
            if len(seg) < min_keep:
                continue
            for idx in seg:
                keep[idx] = True
        cleaned[:, h] = keep
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


def smooth_binary_runs(flag, valid=None, *, min_run_frames: int = 5):
    """
    Remove short flickers in a binary open/close indicator.

    Within each contiguous valid segment, apply an odd-width median filter
    of size ``min_run_frames`` (rounded up to odd). That knocks out sudden
    open/close blips shorter than ~half the window while keeping real state
    changes. Runs after thresholding on the completed series.

    ``flag``: (N,) or (N, hands) with values in {0, 1} (or bool).
    ``valid``: optional mask; invalid frames are left unchanged and break segments.
    """
    from scipy.ndimage import median_filter

    out = np.asarray(flag, dtype=np.float64).copy()
    squeeze = False
    if out.ndim == 1:
        out = out.reshape(-1, 1)
        squeeze = True
    n, n_hands = out.shape[0], out.shape[1]
    vmask = None
    if valid is not None:
        vmask = np.asarray(valid, dtype=bool)
        if vmask.ndim == 1:
            vmask = vmask.reshape(-1, 1)
        if vmask.shape != out.shape:
            raise ValueError("valid shape must match flag shape")

    k = max(1, int(min_run_frames))
    if k % 2 == 0:
        k += 1

    for h in range(n_hands):
        if vmask is None:
            segments = [(0, n - 1)] if n > 0 else []
        else:
            segments = _valid_runs(vmask[:, h])
        for start, end in segments:
            seg = out[start : end + 1, h]
            binary = (seg > 0.5).astype(np.float64)
            if binary.size >= k:
                binary = median_filter(binary, size=k, mode="nearest")
            seg[:] = binary

    return out[:, 0] if squeeze else out


def _geodesic_angle_rad(R0: np.ndarray, R1: np.ndarray) -> float:
    """Geodesic angle (radians) between two rotation matrices."""
    if not (np.isfinite(R0).all() and np.isfinite(R1).all()):
        return float("inf")
    try:
        return float(Rotation.from_matrix(R0.T @ R1).magnitude())
    except Exception:
        return float("inf")


def enforce_rotation_sign_continuity(rotations, valid):
    """
    Remove discrete 180° palm-axis flips between consecutive valid frames.

    Palm frames from joints often flip the normal (or another axis) when the
    hand is near edge-on; the true motion is tiny but R jumps ~180°. For each
    frame we pick among {R, R@diag(1,-1,-1), R@diag(-1,1,-1), R@diag(-1,-1,1)}
    the candidate closest (geodesic) to the previous kept rotation.

    Operates on the completed (N, hands, 3, 3) track only.
    """
    out = np.asarray(rotations, dtype=np.float64).copy()
    v = np.asarray(valid, dtype=bool)
    # Proper 180° flips about principal axes (det = +1).
    candidates = (
        np.diag([1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    )
    for h in range(out.shape[1]):
        prev = None
        for i in range(out.shape[0]):
            if not v[i, h]:
                continue
            Ri = out[i, h]
            if not np.isfinite(Ri).all():
                continue
            if prev is not None:
                best = Ri
                best_ang = _geodesic_angle_rad(prev, Ri)
                for C in candidates[1:]:
                    Rc = Ri @ C
                    ang = _geodesic_angle_rad(prev, Rc)
                    if ang < best_ang:
                        best_ang = ang
                        best = Rc
                out[i, h] = best
                Ri = best
            prev = Ri
    return out


def reject_orientation_outliers(
    rotations,
    valid,
    max_angle_rad: float = 0.7,
    max_angle_rate_rad_s: float = 8.0,
    *,
    timestamps=None,
    default_fps=30.0,
    min_segment_frames: int = 3,
    min_segment_frac: float = 0.15,
):
    """
    Postprocess-only: clean a completed (N, hands, 3, 3) rotation track.

    Same structure as ``reject_position_outliers`` but uses SO(3) geodesic angle:
      1) drop isolated orientation spikes
      2) split at large angular jumps / rates
      3) drop short outlier islands vs the longest continuous segment

    Defaults (~40 deg jump, ~8 rad/s ≈ 460 deg/s) are loose enough for real
    hand motion but catch WiLoR palm-frame flips.
    """
    cleaned = valid.copy()
    n = rotations.shape[0]
    t = None
    if timestamps is not None:
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if t.shape[0] != n:
            t = None

    def _dt(i0: int, i1: int) -> float:
        if t is not None:
            dt = float(t[i1] - t[i0])
            if np.isfinite(dt) and dt > 1e-6:
                return dt
        return max(i1 - i0, 1) / float(default_fps)

    min_seg = max(1, int(min_segment_frames))
    frac = float(min_segment_frac)
    max_ang = float(max_angle_rad)
    max_rate = float(max_angle_rate_rad_s)

    for h in range(rotations.shape[1]):
        idxs = np.flatnonzero(cleaned[:, h])
        for k in range(1, idxs.size - 1):
            i0, i1, i2 = int(idxs[k - 1]), int(idxs[k]), int(idxs[k + 1])
            a01 = _geodesic_angle_rad(rotations[i0, h], rotations[i1, h])
            a12 = _geodesic_angle_rad(rotations[i1, h], rotations[i2, h])
            a02 = _geodesic_angle_rad(rotations[i0, h], rotations[i2, h])
            r01 = a01 / _dt(i0, i1)
            r12 = a12 / _dt(i1, i2)
            spike_ang = a01 > max_ang and a12 > max_ang and a02 <= max_ang
            spike_rate = r01 > max_rate and r12 > max_rate and a02 <= max_ang
            if spike_ang or spike_rate:
                cleaned[i1, h] = False

        idxs = np.flatnonzero(cleaned[:, h])
        if idxs.size == 0:
            continue
        segments: list[list[int]] = [[int(idxs[0])]]
        for k in range(1, idxs.size):
            i0 = int(idxs[k - 1])
            i1 = int(idxs[k])
            ang = _geodesic_angle_rad(rotations[i0, h], rotations[i1, h])
            rate = ang / _dt(i0, i1)
            if ang > max_ang or rate > max_rate:
                segments.append([i1])
            else:
                segments[-1].append(i1)

        longest = max(len(seg) for seg in segments)
        min_keep = min_seg if longest < min_seg else max(min_seg, int(np.ceil(frac * longest)))
        keep = np.zeros(n, dtype=bool)
        for seg in segments:
            if len(seg) < min_keep:
                continue
            for idx in seg:
                keep[idx] = True
        cleaned[:, h] = keep
    return cleaned


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
    """
    Smooth orientations by filtering quaternion components (not rotvec).

    Rotvec Savitzky-Golay is unsafe: ``as_rotvec`` has a branch cut near ±π, so
    tiny real motion can look like a ~2 rad jump in components and the filter
    invents huge orientation swings. Quaternions with hemisphere fixing stay
    continuous on the double cover.
    """
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
            for axis in range(4):
                seg_q[:, axis] = savgol_filter(
                    seg_q[:, axis],
                    window_length=window,
                    polyorder=min(poly, window - 1),
                )
            seg_q = _normalize_quaternions(seg_q)
            smoothed[start : end + 1, h] = seg_q

    return smoothed


def smooth_rotations_so3(rotations, valid, window=9, poly=3):
    """Smooth (N, hands, 3, 3) rotations via quaternion-component Savitzky-Golay."""
    R = np.asarray(rotations, dtype=np.float64)
    v = np.asarray(valid, dtype=bool)
    n, n_hands = R.shape[0], R.shape[1]
    # wxyz quaternions for smooth_quaternions
    quat = np.full((n, n_hands, 4), np.nan, dtype=np.float64)
    for h in range(n_hands):
        idx = np.flatnonzero(v[:, h])
        if idx.size == 0:
            continue
        q_xyzw = Rotation.from_matrix(R[idx, h]).as_quat()
        quat[idx, h, 0] = q_xyzw[:, 3]
        quat[idx, h, 1:4] = q_xyzw[:, 0:3]
    quat_s = smooth_quaternions(quat, v, window=window, poly=poly)
    out = R.copy()
    for h in range(n_hands):
        idx = np.flatnonzero(v[:, h])
        if idx.size == 0:
            continue
        q = quat_s[idx, h]
        q_xyzw = np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
        out[idx, h] = Rotation.from_quat(q_xyzw).as_matrix()
    return out


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
