#!/usr/bin/env python3
"""
WiLoR + RealSense depth -> wrist 6DOF dataset.

Inputs:
  - RealSense .bag (depth + color)
  - optional MP4 / timestamps .npy (only needed for --pose-timeline npy)
  - OR a bird-realsense-data directory containing bag/ (and optionally mp4/, npy/)

Timeline (--pose-timeline):
  - bag (default): ALL aligned bag color+depth frames, bag timestamps (~30 fps).
  - npy: subsample bag to mp4/npy timestamps (~15 fps).

Two-phase pipeline (postprocess never runs inside the frame loop):
  Phase 1 (per frame): WiLoR detect -> sample depth -> deproject XYZ; store rot/open raw
  Phase 2 (full arrays): outlier clean -> gap-fill -> smooth; pack pose

Output (.npz): timestamps, pose (N,2,10), valid_*, pose_xyz_raw, valid_pos_raw,
              open_score_*, pose_timeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def _so3_exp(rotvec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < eps:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    kx, ky, kz = k.tolist()
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _so3_log(R: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    sin_theta = float(np.sin(theta))
    if abs(sin_theta) < eps:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + eps)
        return axis * theta
    axis = (1.0 / (2.0 * sin_theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    )
    return axis * theta


def _fill_short_gaps_linear(t: np.ndarray, x: np.ndarray, valid: np.ndarray, *, max_gap_frames: int):
    out = np.asarray(x, dtype=np.float64).copy()
    v = np.asarray(valid, dtype=bool).copy()
    n = out.shape[0]
    if n == 0:
        return out, v
    idx = np.flatnonzero(v & np.isfinite(out))
    if idx.size == 0:
        return out, v

    first = int(idx[0])
    if first > 0 and first <= max_gap_frames:
        out[:first] = out[first]
        v[:first] = True
    last = int(idx[-1])
    if last < n - 1 and (n - 1 - last) <= max_gap_frames:
        out[last + 1 :] = out[last]
        v[last + 1 :] = True

    for a, b in zip(idx[:-1], idx[1:]):
        a = int(a)
        b = int(b)
        gap = b - a - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        ta, tb = float(t[a]), float(t[b])
        denom = tb - ta
        for i in range(a + 1, b):
            alpha = 0.0 if abs(denom) < 1e-12 else (float(t[i]) - ta) / denom
            out[i] = (1.0 - alpha) * out[a] + alpha * out[b]
            v[i] = True
    return out, v


def _fill_short_gaps_so3(t: np.ndarray, R: np.ndarray, valid: np.ndarray, *, max_gap_frames: int):
    out = np.asarray(R, dtype=np.float64).copy()
    v = np.asarray(valid, dtype=bool).copy()
    n = out.shape[0]
    if n == 0:
        return out, v
    idx = np.flatnonzero(v)
    if idx.size == 0:
        return out, v

    first = int(idx[0])
    if first > 0 and first <= max_gap_frames:
        out[:first] = out[first]
        v[:first] = True
    last = int(idx[-1])
    if last < n - 1 and (n - 1 - last) <= max_gap_frames:
        out[last + 1 :] = out[last]
        v[last + 1 :] = True

    for a, b in zip(idx[:-1], idx[1:]):
        a = int(a)
        b = int(b)
        gap = b - a - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        R0 = out[a]
        R1 = out[b]
        ta, tb = float(t[a]), float(t[b])
        denom = tb - ta
        R_rel = R0.T @ R1
        r_rel = _so3_log(R_rel)
        for i in range(a + 1, b):
            alpha = 0.0 if abs(denom) < 1e-12 else (float(t[i]) - ta) / denom
            out[i] = R0 @ _so3_exp(alpha * r_rel)
            v[i] = True
    return out, v


def _palm_center_openpose(j: np.ndarray) -> np.ndarray:
    return np.mean(j[[5, 9, 13, 17]], axis=0)


def _palm_frame_R_from_openpose_joints(j: np.ndarray) -> np.ndarray:
    """
    Palm orientation from WiLoR OpenPose joints (MediaPipe-style, translation-invariant).

    Columns of R:
      x: wrist -> index_mcp (5)
      z: palm normal = normalize(cross(x, wrist->pinky_mcp))
      y: normalize(cross(z, x))

    Using index×pinky is much more stable than projecting wrist->palm_center
    (that y-axis vanishes when the palm points along the index, causing ~180°
    normal flips that look like crazy rotation with almost no real motion).
    """
    wrist = j[0]
    index_mcp = j[5]
    pinky_mcp = j[17]
    x = _normalize(index_mcp - wrist)
    if float(np.linalg.norm(x)) < 1e-8:
        return np.full((3, 3), np.nan, dtype=np.float64)
    palm_span = pinky_mcp - wrist
    z = np.cross(x, palm_span)
    z_n = float(np.linalg.norm(z))
    if z_n < 1e-8:
        # Degenerate (index/pinky collinear): fall back to palm-center plane.
        palm_center = _palm_center_openpose(j)
        palm_dir = _normalize(palm_center - wrist)
        y = palm_dir - float(np.dot(palm_dir, x)) * x
        y = _normalize(y)
        if float(np.linalg.norm(y)) < 1e-8:
            return np.full((3, 3), np.nan, dtype=np.float64)
        z = _normalize(np.cross(x, y))
        y = _normalize(np.cross(z, x))
        return np.column_stack([x, y, z])
    z = z / z_n
    y = _normalize(np.cross(z, x))
    return np.column_stack([x, y, z])


def _open_score_from_openpose_joints(j: np.ndarray) -> float:
    palm_center = _palm_center_openpose(j)
    tips = j[[4, 8, 12, 16, 20]]
    tip_dist = float(np.mean(np.linalg.norm(tips - palm_center[None, :], axis=-1)))
    palm_w = float(np.linalg.norm(j[5] - j[17]))
    return tip_dist / (palm_w + 1e-12)


def _sample_depth_m(depth_frame, u: float, v: float, *, radius: int = 4):
    import numpy as _np

    depth_image = _np.asanyarray(depth_frame.get_data())
    h, w = depth_image.shape
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or vi < 0 or ui >= w or vi >= h:
        return None

    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    patch = depth_image[y0:y1, x0:x1].astype(_np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    depth_scale = depth_frame.get_units()
    return float(_np.median(valid) * depth_scale)


def _scan_bag_aligned_color_timestamps(bag_path: Path) -> np.ndarray:
    """Timestamps (s) for every aligned color+depth frame in the bag."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.color)
    ts = []
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            ts.append(float(color_frame.get_timestamp()) / 1000.0)
    finally:
        pipeline.stop()
    return np.asarray(ts, dtype=np.float64)


def _bag_indices_for_timestamps(bag_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """
    Map each MP4/npy timestamp to the nearest bag frame.

    Bags often store ~2x frames over the same timespan as the 15 FPS MP4. Taking the
    first N bag frames then writing at 15 FPS makes a slow-mo, clipped overlay/dataset.
    """
    bag_ts = np.asarray(bag_ts, dtype=np.float64)
    target_ts = np.asarray(target_ts, dtype=np.float64)
    if bag_ts.size == 0 or target_ts.size == 0:
        return np.zeros((0,), dtype=np.int64)
    idxs = np.empty((target_ts.shape[0],), dtype=np.int64)
    for i, t in enumerate(target_ts):
        idxs[i] = int(np.argmin(np.abs(bag_ts - float(t))))
    return idxs


def _det_bbox_center_x(det: dict) -> float | None:
    bb = det.get("hand_bbox", None)
    if not isinstance(bb, (list, tuple)) or len(bb) < 4:
        return None
    x0, y0, x1, y1 = [float(x) for x in bb[:4]]
    return 0.5 * (x0 + x1)


def _det_wrist_uv(det: dict) -> tuple[float, float] | None:
    wp = det.get("wilor_preds", None)
    if not isinstance(wp, dict):
        return None
    k2d = wp.get("pred_keypoints_2d", None)
    if k2d is None:
        return None
    k2d = np.asarray(k2d, dtype=np.float64)
    if k2d.ndim != 3 or k2d.shape[1] < 1:
        return None
    u, v = float(k2d[0, 0, 0]), float(k2d[0, 0, 1])
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return u, v


def _assign_detections_to_slots(
    dets: list[dict],
    prev_u: np.ndarray | None,
    *,
    slot_mode: str = "auto",
) -> dict[int, dict | None]:
    """
    Assign up to 2 detections to slots {0,1}.

    slot_mode:
      - "is_right": trust WiLoR's `is_right` field (right->slot1, left->slot0)
      - "xpos": assign by image x position (leftmost->slot0, rightmost->slot1)
      - "auto" (default): trust `is_right` so left/right stay in fixed slots even
        with a single hand. Fall back to xpos only if `is_right` is missing, or
        if two detections collapse onto the same handedness label.
    """
    chosen: dict[int, dict | None] = {0: None, 1: None}
    if not dets:
        return chosen

    # Filter to dict detections that have wrist uv
    cand = []
    for d in dets:
        if not isinstance(d, dict):
            continue
        uv = _det_wrist_uv(d)
        if uv is None:
            continue
        u, v = uv
        xc = _det_bbox_center_x(d)
        if xc is None:
            xc = u
        cand.append((d, float(u), float(xc)))
    if not cand:
        return chosen

    # keep at most two largest boxes (fallback to x-center if bbox missing)
    def area(d):
        bb = d.get("hand_bbox", None)
        if not isinstance(bb, (list, tuple)) or len(bb) < 4:
            return 0.0
        x0, y0, x1, y1 = [float(x) for x in bb[:4]]
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    cand_sorted = sorted(cand, key=lambda t: area(t[0]), reverse=True)[:2]

    # Path 1: assign by is_right (keeps left=slot0 / right=slot1)
    if slot_mode in ("is_right", "auto"):
        tmp = {0: None, 1: None}
        n_labeled = 0
        for d, u, xc in cand_sorted:
            try:
                is_r = float(d.get("is_right", np.nan))
            except Exception:
                is_r = np.nan
            if not np.isfinite(is_r):
                continue
            n_labeled += 1
            slot = 1 if is_r > 0.5 else 0
            if tmp[slot] is None:
                tmp[slot] = d
        n_filled = int(tmp[0] is not None) + int(tmp[1] is not None)
        if slot_mode == "is_right":
            return tmp
        # auto: keep labeled slots (incl. single right hand -> slot1).
        # Fall through only when labels are missing, or two dets share one label.
        collapsed = len(cand_sorted) >= 2 and n_labeled >= 2 and n_filled < 2
        if n_labeled > 0 and not collapsed:
            return tmp

    if prev_u is not None and np.isfinite(prev_u).any() and len(cand_sorted) == 2:
        # Match each detection to closest previous slot u
        u0 = cand_sorted[0][1]
        u1 = cand_sorted[1][1]
        # cost matrix
        c00 = abs(u0 - float(prev_u[0])) if np.isfinite(prev_u[0]) else 0.0
        c01 = abs(u0 - float(prev_u[1])) if np.isfinite(prev_u[1]) else 0.0
        c10 = abs(u1 - float(prev_u[0])) if np.isfinite(prev_u[0]) else 0.0
        c11 = abs(u1 - float(prev_u[1])) if np.isfinite(prev_u[1]) else 0.0
        if c00 + c11 <= c01 + c10:
            chosen[0], chosen[1] = cand_sorted[0][0], cand_sorted[1][0]
        else:
            chosen[0], chosen[1] = cand_sorted[1][0], cand_sorted[0][0]
        return chosen

    # Otherwise: assign by x center (leftmost -> slot 0, rightmost -> slot 1)
    cand_sorted = sorted(cand_sorted, key=lambda t: t[2])
    chosen[0] = cand_sorted[0][0]
    if len(cand_sorted) > 1:
        chosen[1] = cand_sorted[1][0]
    return chosen


def run(
    mp4_path: Path | None,
    bag_path: Path,
    *,
    out_path: Path,
    timestamps_npy: Path | None,
    pose_timeline: str = "bag",
    device: str = "auto",
    dtype: str = "auto",
    wilor_stride: int = 1,
    max_gap_frames: int = 3,
    open_threshold: float = 1.10,
    depth_radius: int = 4,
    match_mediapipe_xyz: bool = False,
    use_bag_color: bool = True,
    slot_mode: str = "auto",
    debug_overlay_path: Path | None = None,
    debug_max_frames: int = 10_000_000,
    debug_kp: bool = False,
    debug_kp_label: bool = True,
    smooth_window: int = 9,
    smooth_poly: int = 3,
    max_speed_m_s: float = 3.0,
    max_jump_m: float = 0.25,
    open_min_run_frames: int = 5,
):
    import cv2
    import pyrealsense2 as rs
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    bag_path = Path(bag_path)
    out_path = Path(out_path)
    mp4_path = Path(mp4_path) if mp4_path is not None else None
    pose_timeline = str(pose_timeline).lower().strip()
    if pose_timeline not in ("bag", "npy"):
        raise ValueError("pose_timeline must be 'bag' or 'npy'")

    cap = None
    if not use_bag_color:
        if mp4_path is None:
            raise RuntimeError("--use-mp4-color requires an mp4 path")
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open mp4: {mp4_path}")

    keep_bag_indices = None
    if pose_timeline == "bag":
        # Default: every aligned bag frame + bag timestamps.
        ts = _scan_bag_aligned_color_timestamps(bag_path)
        N = int(ts.shape[0])
        if N == 0:
            raise RuntimeError(f"No aligned color+depth frames in bag: {bag_path}")
        span = float(ts[-1] - ts[0]) if N > 1 else float("nan")
        fps_est = float((N - 1) / span) if span > 1e-6 else float("nan")
        print(
            f"[timeline=bag] using ALL bag frames N={N} span={span:.3f}s ~{fps_est:.2f} fps",
            flush=True,
        )
    else:
        # Subsample bag to mp4/npy timestamps.
        if timestamps_npy is None:
            raise RuntimeError("--pose-timeline npy requires --timestamps-npy or data-dir with npy/")
        ts = np.asarray(np.load(timestamps_npy), dtype=np.float64)
        N_target = int(ts.shape[0])
        if cap is not None:
            mp4_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if mp4_n > 0:
                N_target = min(N_target, mp4_n)
        ts = ts[: int(N_target)]
        N = int(ts.shape[0])
        bag_ts = _scan_bag_aligned_color_timestamps(bag_path)
        keep_bag_indices = _bag_indices_for_timestamps(bag_ts, ts)
        step = float(np.median(np.diff(keep_bag_indices))) if keep_bag_indices.size > 1 else 1.0
        print(
            f"[timeline=npy] bag_frames={bag_ts.size} npy_frames={N} "
            f"nearest-index playback (median step={step:.1f})",
            flush=True,
        )

    # Bag playback (main)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    # NOTE:
    # We align depth->color, so the aligned depth frame is in the color image geometry.
    # To avoid any confusion, we will deproject using the aligned depth frame's intrinsics
    # (which should match the color stream intrinsics after alignment).
    color_intr = color_profile.get_intrinsics()

    # WiLoR
    dev = device
    if dev == "auto":
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = dtype
    if dt == "auto":
        dt = "fp16" if dev == "cuda" else "fp32"
    import torch

    torch_dtype = torch.float16 if dt in ("fp16", "float16") else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device(dev), dtype=torch_dtype, verbose=False)

    pose = np.full((N, 2, 10), np.nan, dtype=np.float64)
    valid_pos = np.zeros((N, 2), dtype=bool)
    valid_rot = np.zeros((N, 2), dtype=bool)
    valid_open = np.zeros((N, 2), dtype=bool)
    R_raw = np.full((N, 2, 3, 3), np.nan, dtype=np.float64)
    open_score = np.full((N, 2), np.nan, dtype=np.float64)

    # Iterate bag frames (all of them, or the npy-aligned subset).
    i = 0
    bag_i = 0
    keep_ptr = 0
    keep_list = keep_bag_indices.tolist() if keep_bag_indices is not None else None
    prev_u = np.array([np.nan, np.nan], dtype=np.float64)
    oob_uv = np.zeros((2,), dtype=np.int64)
    have_uv = np.zeros((2,), dtype=np.int64)
    have_depth_at_uv = np.zeros((2,), dtype=np.int64)
    dbg_writer = None
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            take = True
            if keep_list is not None:
                if keep_ptr >= len(keep_list):
                    break
                if bag_i != int(keep_list[keep_ptr]):
                    take = False
                else:
                    keep_ptr += 1
            bag_i += 1
            if not take:
                continue

            if i >= N:
                break

            if use_bag_color:
                frame_bgr = np.asanyarray(color_frame.get_data())
            else:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

            H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
            if dbg_writer is None and debug_overlay_path is not None:
                debug_overlay_path = Path(debug_overlay_path)
                debug_overlay_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # Mean fps over the full timeline (avoids median-dt rounding that
                # stretches bag timelines when bags are ~2x denser than mp4).
                if ts is not None and ts.shape[0] > 2:
                    span = float(ts[-1] - ts[0])
                    fps_est = float((ts.shape[0] - 1) / span) if span > 1e-6 else float(color_profile.fps())
                    fps_out = float(round(fps_est)) if fps_est > 1e-6 else float(color_profile.fps())
                else:
                    fps_out = float(color_profile.fps())
                dbg_writer = cv2.VideoWriter(str(debug_overlay_path), fourcc, fps_out, (W, H))
                if not dbg_writer.isOpened():
                    dbg_writer = None

            # WiLoR stride
            pred = None
            if (i % max(int(wilor_stride), 1)) == 0:
                pred = pipe.predict(frame_bgr)
                if not isinstance(pred, list):
                    pred = []

            # Assign detections to slots
            chosen = _assign_detections_to_slots(
                pred if isinstance(pred, list) else [],
                prev_u=prev_u,
                slot_mode=str(slot_mode),
            )

            frame_dbg = frame_bgr.copy() if dbg_writer is not None and i < int(debug_max_frames) else None

            for hand in (0, 1):
                det = chosen[hand]
                if det is None:
                    continue
                wp = det.get("wilor_preds", None)
                if not isinstance(wp, dict):
                    continue

                # 2D wrist pixel from WiLoR
                uv = _det_wrist_uv(det)
                if uv is None:
                    continue
                u, v = uv
                prev_u[hand] = float(u)
                have_uv[hand] += 1

                # Optional: draw all 2D keypoints (if available) and label a few.
                if frame_dbg is not None and bool(debug_kp):
                    k2d = wp.get("pred_keypoints_2d", None)
                    if k2d is not None:
                        k2d = np.asarray(k2d, dtype=np.float64)
                        # expected shape: (1,21,2)
                        if k2d.ndim == 3 and k2d.shape[0] >= 1 and k2d.shape[1] >= 1 and k2d.shape[2] >= 2:
                            pts = k2d[0, :, 0:2]
                            col = (255, 0, 0) if hand == 0 else (0, 0, 255)
                            for ki in range(pts.shape[0]):
                                ku, kv = float(pts[ki, 0]), float(pts[ki, 1])
                                if not np.isfinite(ku) or not np.isfinite(kv):
                                    continue
                                if 0.0 <= ku < W and 0.0 <= kv < H:
                                    cv2.circle(frame_dbg, (int(round(ku)), int(round(kv))), 2, col, -1)
                            if bool(debug_kp_label):
                                # Label a small, informative subset to keep the overlay readable.
                                # 0=wrist, 4/8/12/16/20 fingertips, 5=index_mcp, 17=pinky_mcp
                                label_ids = [0, 5, 17, 4, 8, 12, 16, 20]
                                for li, kid in enumerate(label_ids):
                                    if kid >= pts.shape[0]:
                                        continue
                                    ku, kv = float(pts[kid, 0]), float(pts[kid, 1])
                                    if not np.isfinite(ku) or not np.isfinite(kv):
                                        continue
                                    txt = f"{kid}:{ku:.0f},{kv:.0f}"
                                    x = int(min(max(int(round(ku)) + 6, 0), W - 1))
                                    y = int(min(max(int(round(kv)) + 6, 0), H - 1))
                                    cv2.putText(
                                        frame_dbg,
                                        txt,
                                        (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        col,
                                        1,
                                        cv2.LINE_AA,
                                    )
                if not (0.0 <= u < W and 0.0 <= v < H):
                    oob_uv[hand] += 1
                    # don't even try depth if out of bounds
                    if frame_dbg is not None:
                        cv2.putText(
                            frame_dbg,
                            f"slot{hand} wrist OOB ({u:.1f},{v:.1f})",
                            (10, 30 + 30 * hand),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                    continue

                # Depth: try a couple radii (depth is often sparse at a single pixel)
                depth_m = None
                if match_mediapipe_xyz:
                    # Match hand_pose_track.py behavior as closely as possible:
                    # sample once at the configured radius with no fallback expansion.
                    depth_m = _sample_depth_m(depth_frame, u, v, radius=max(1, int(depth_radius)))
                else:
                    for r in (int(depth_radius), int(depth_radius) * 2, int(depth_radius) * 3):
                        depth_m = _sample_depth_m(depth_frame, u, v, radius=max(1, r))
                        if depth_m is not None:
                            break
                if depth_m is not None:
                    have_depth_at_uv[hand] += 1

                xyz_np = None
                if depth_m is not None:
                    # Deprojection:
                    # - For match_mediapipe_xyz: use the color intrinsics (like hand_pose_track.py),
                    #   because depth has already been aligned to the color image.
                    # - Otherwise: use the aligned depth intrinsics when available.
                    if match_mediapipe_xyz:
                        xyz = rs.rs2_deproject_pixel_to_point(color_intr, [u, v], depth_m)
                    else:
                        try:
                            depth_intr = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
                        except Exception:
                            depth_intr = color_intr
                        xyz = rs.rs2_deproject_pixel_to_point(depth_intr, [u, v], depth_m)
                    xyz = np.asarray(xyz, dtype=np.float64)
                    if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
                        xyz_np = xyz
                        pose[i, hand, 0:3] = xyz
                        valid_pos[i, hand] = True

                if frame_dbg is not None:
                    color = (255, 0, 0) if hand == 0 else (0, 0, 255)
                    cv2.circle(frame_dbg, (int(round(u)), int(round(v))), 6, color, 2)
                    bb = det.get("hand_bbox", None)
                    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                        x0, y0, x1, y1 = [int(round(float(x))) for x in bb[:4]]
                        cv2.rectangle(frame_dbg, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(
                        frame_dbg,
                        f"slot{hand} u,v=({u:.1f},{v:.1f}) depth={depth_m if depth_m is not None else -1:.3f}",
                        (10, 30 + 30 * hand),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                    if xyz_np is not None:
                        cv2.putText(
                            frame_dbg,
                            f"slot{hand} xyz=({xyz_np[0]:+.3f},{xyz_np[1]:+.3f},{xyz_np[2]:+.3f})m",
                            (10, 50 + 30 * hand),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                # Orientation (wrist frame) from 3D joints
                k3d = wp.get("pred_keypoints_3d", None)
                if k3d is not None:
                    k3d = np.asarray(k3d, dtype=np.float64)
                    if k3d.ndim == 3 and k3d.shape[1] > 20:
                        j = k3d[0]  # (21,3)
                        R = _palm_frame_R_from_openpose_joints(j)
                        if np.all(np.isfinite(R)):
                            R_raw[i, hand] = R
                            valid_rot[i, hand] = True
                            pose[i, hand, 3:6] = R[:, 0]
                            pose[i, hand, 6:9] = R[:, 1]

                        s = _open_score_from_openpose_joints(j)
                        if np.isfinite(s):
                            open_score[i, hand] = float(s)
                            valid_open[i, hand] = True

                            if frame_dbg is not None:
                                # show raw score and current threshold
                                col = (255, 0, 0) if hand == 0 else (0, 0, 255)
                                cv2.putText(
                                    frame_dbg,
                                    f"slot{hand} open_score={float(s):.3f}  thr={float(open_threshold):.3f}",
                                    (10, 70 + 30 * hand),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    col,
                                    2,
                                )

            if frame_dbg is not None and dbg_writer is not None and i < int(debug_max_frames):
                dbg_writer.write(frame_dbg)

            i += 1

    finally:
        if cap is not None:
            cap.release()
        pipeline.stop()
        if dbg_writer is not None:
            dbg_writer.release()

    processed = int(i)
    ts = np.asarray(ts, dtype=np.float64)[:N]
    if processed < N:
        # Keep trailing frames invalid so indices still match bag timestamps.
        pass
    if processed == N:
        pose = pose[:processed]
        valid_pos = valid_pos[:processed]
        valid_rot = valid_rot[:processed]
        valid_open = valid_open[:processed]
        R_raw = R_raw[:processed]
        open_score = open_score[:processed]
        ts = ts[:processed]

    # -------------------------------------------------------------------------
    # Phase 2 — postprocess the FULL arrays only (never during the frame loop).
    # Phase 1 above only: detect → sample depth → deproject → store raw values.
    # -------------------------------------------------------------------------
    if ts.shape[0] == processed:
        proc_slice = slice(None)
        ts_proc = ts
    else:
        proc_slice = slice(0, processed)
        ts_proc = ts[proc_slice]

    t_rel = ts_proc - ts_proc[0] if ts_proc.size > 0 else ts_proc

    from hand_pose_postprocess import (
        enforce_rotation_sign_continuity,
        interpolate_positions,
        reject_orientation_outliers,
        reject_position_outliers,
        smooth_binary_runs,
        smooth_positions,
        smooth_rotations_so3,
    )

    # Snapshot raw deprojected XYZ before any cleaning.
    pos_raw = pose[proc_slice, :, 0:3].copy()
    valid_pos_raw = valid_pos[proc_slice].copy() & np.isfinite(pos_raw).all(axis=-1)
    R_raw_snap = R_raw[proc_slice].copy()
    valid_rot_raw = valid_rot[proc_slice].copy()
    open_score_raw = open_score[proc_slice].copy()
    valid_open_raw = valid_open[proc_slice].copy() & np.isfinite(open_score_raw)

    # Position: reject outliers on the completed track → gap-fill → smooth.
    vpos_clean = reject_position_outliers(
        pos_raw,
        valid_pos_raw,
        max_speed_m_s=float(max_speed_m_s),
        max_jump_m=float(max_jump_m),
        timestamps=ts_proc,
    )
    pos_for_patch = pos_raw.copy()
    pos_for_patch[~vpos_clean] = np.nan
    pos_patched, vpos_patched = interpolate_positions(
        pos_for_patch, vpos_clean, max_gap_frames=int(max_gap_frames)
    )
    pos_f = smooth_positions(
        pos_patched, vpos_patched, window=int(smooth_window), poly=int(smooth_poly)
    )
    vpos = vpos_patched

    # Orientation: undo palm-axis sign flips, reject remaining jumps, gap-fill + smooth.
    R_f = enforce_rotation_sign_continuity(R_raw_snap, valid_rot_raw)
    vori = reject_orientation_outliers(
        R_f,
        valid_rot_raw,
        timestamps=ts_proc,
    )
    for h in (0, 1):
        R_f[:, h], vori[:, h] = _fill_short_gaps_so3(
            t_rel, R_f[:, h], vori[:, h], max_gap_frames=int(max_gap_frames)
        )

    # Quaternion-component smooth (NOT rotvec — rotvec Savitzky-Golay blows up
    # near the ±π branch cut and was inventing ~100°+ jumps on an otherwise
    # stable left-hand track).
    R_f = smooth_rotations_so3(
        R_f, vori, window=int(smooth_window), poly=int(smooth_poly)
    )

    # Open: fill score gaps on the full series, then threshold.
    s_raw = open_score_raw.copy()
    s_f = s_raw.copy()
    vs = valid_open_raw.copy()
    for h in (0, 1):
        s_f[:, h], vs[:, h] = _fill_short_gaps_linear(
            t_rel, s_f[:, h], vs[:, h], max_gap_frames=int(max_gap_frames)
        )
    open_flag = (s_f > float(open_threshold)).astype(np.float64)
    open_flag = smooth_binary_runs(
        open_flag, vs, min_run_frames=int(open_min_run_frames)
    )

    # Pack postprocessed pose (raw kept separately below).
    pose_out = np.full_like(pose[proc_slice], np.nan)
    pose_out[:, :, 0:3] = np.where(vpos[..., None], pos_f, np.nan)
    for k in range(pose_out.shape[0]):
        for h in (0, 1):
            if vori[k, h]:
                pose_out[k, h, 3:6] = R_f[k, h, :, 0]
                pose_out[k, h, 6:9] = R_f[k, h, :, 1]
            if vs[k, h]:
                pose_out[k, h, 9] = open_flag[k, h]

    if ts.shape[0] != processed:
        pose[proc_slice] = pose_out
        valid_pos[proc_slice] = vpos
        valid_rot[proc_slice] = vori
        valid_open[proc_slice] = vs
        # Pad raw snapshots to full timestamp length with NaNs / False.
        pos_raw_full = np.full((ts.shape[0], 2, 3), np.nan, dtype=np.float64)
        valid_pos_raw_full = np.zeros((ts.shape[0], 2), dtype=bool)
        R_raw_full = np.full((ts.shape[0], 2, 3, 3), np.nan, dtype=np.float64)
        valid_rot_raw_full = np.zeros((ts.shape[0], 2), dtype=bool)
        pos_raw_full[proc_slice] = pos_raw
        valid_pos_raw_full[proc_slice] = valid_pos_raw
        R_raw_full[proc_slice] = R_raw_snap
        valid_rot_raw_full[proc_slice] = valid_rot_raw
        pos_raw, valid_pos_raw = pos_raw_full, valid_pos_raw_full
        R_raw_snap, valid_rot_raw = R_raw_full, valid_rot_raw_full
    else:
        pose = pose_out
        valid_pos = vpos
        valid_rot = vori
        valid_open = vs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        timestamps=ts,
        pose=pose,
        valid_pos=valid_pos,
        valid_rot=valid_rot,
        valid_open=valid_open,
        # Phase-1 raw (before gap-fill / smooth).
        pose_xyz_raw=pos_raw,
        valid_pos_raw=valid_pos_raw,
        R_raw=R_raw_snap,
        valid_rot_raw=valid_rot_raw,
        open_score_raw=s_raw,
        open_score_filled=s_f,
        open_score_valid=vs,
        open_threshold=float(open_threshold),
        pose_timeline=np.asarray(pose_timeline),
    )
    if pose.shape[0] > 0:
        span = float(ts[-1] - ts[0]) if ts.shape[0] > 1 else float("nan")
        fps_est = float((ts.shape[0] - 1) / span) if span > 1e-6 else float("nan")
        print(
            f"Saved -> {out_path} (N={pose.shape[0]}, timeline={pose_timeline}, ~{fps_est:.2f} fps)  "
            f"raw_pos%={valid_pos_raw.mean(axis=0)} post_pos%={valid_pos.mean(axis=0)} "
            f"valid_rot%={valid_rot.mean(axis=0)} valid_open%={valid_open.mean(axis=0)}"
        )
        for h in (0, 1):
            if have_uv[h] > 0:
                print(
                    f"slot{h}: wrist_uv OOB%={(oob_uv[h]/have_uv[h]):.3f}  "
                    f"depth_at_uv%={(have_depth_at_uv[h]/have_uv[h]):.3f}  "
                    f"(have_uv={have_uv[h]})"
                )
    else:
        print(f"Saved -> {out_path} (N=0)")


def _contiguous_segments(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.insert(idx[splits + 1], 0, idx[0])
    ends = np.append(idx[splits], idx[-1])
    return list(zip(starts, ends))


def _parse_bird_mp4_name(mp4_name: str):
    """
    From realsense_bird_record.py:
      mp4: video_recording_bird_realsense_{serial}#{session}.mp4
      npy: video_recording_bird_realsense_{serial}#{session}.npy
      bag: bird_realsense_{serial}#{session}.bag
    """
    stem = Path(mp4_name).stem
    prefix = "video_recording_bird_realsense_"
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix) :]
    if "#" not in rest:
        return None
    serial, session = rest.split("#", 1)
    if not serial or not session:
        return None
    return serial, session


def _resolve_triplets(data_dir: Path, *, require_npy: bool = True):
    """
    Yield (mp4|None, bag, npy|None, out_base) for each recording.

    Prefer discovering from bags so --pose-timeline bag works without npy/mp4.
    """
    data_dir = Path(data_dir)
    mp4_dir, npy_dir, bag_dir = data_dir / "mp4", data_dir / "npy", data_dir / "bag"
    if not bag_dir.is_dir():
        raise FileNotFoundError(f"Expected bag folder: {bag_dir}")

    bags = sorted(bag_dir.glob("*.bag"))
    if bags:
        for bag in bags:
            stem = bag.stem  # bird_realsense_{serial}#{session}
            if not stem.startswith("bird_realsense_") or "#" not in stem:
                continue
            rest = stem[len("bird_realsense_") :]
            serial, session = rest.split("#", 1)
            out_base = f"video_recording_bird_realsense_{serial}#{session}"
            mp4 = mp4_dir / f"{out_base}.mp4"
            npy = npy_dir / f"{out_base}.npy"
            if require_npy and (not npy.is_file() or not mp4.is_file()):
                continue
            yield (mp4 if mp4.is_file() else None), bag, (npy if npy.is_file() else None), out_base
        return

    if not mp4_dir.is_dir():
        raise FileNotFoundError(f"Expected mp4 folder: {mp4_dir}")
    for mp4 in sorted(mp4_dir.glob("*.mp4")):
        parsed = _parse_bird_mp4_name(mp4.name)
        if parsed is None:
            continue
        serial, session = parsed
        npy = npy_dir / f"video_recording_bird_realsense_{serial}#{session}.npy"
        bag = bag_dir / f"bird_realsense_{serial}#{session}.bag"
        if not bag.is_file():
            continue
        if require_npy and not npy.is_file():
            continue
        yield mp4, bag, (npy if npy.is_file() else None), f"video_recording_bird_realsense_{serial}#{session}"


def main():
    parser = argparse.ArgumentParser(description="WiLoR + bag depth -> wrist XYZ + rot6d + open/close per frame.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Process a bird-realsense-data directory containing bag/ "
            "(and optionally mp4/, npy/). Discoveries are bag-first."
        ),
    )
    parser.add_argument("--mp4", type=str, default=None, help="Input RGB mp4 (required for --use-mp4-color / --pose-timeline npy)")
    parser.add_argument("--bag", type=str, default=None, help="Input RealSense .bag with depth")
    parser.add_argument("--timestamps-npy", type=str, default=None, help="Timestamps .npy (required for --pose-timeline npy)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output .npz path (single-file mode)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (data-dir batch mode). Writes one .npz per recording.",
    )
    parser.add_argument(
        "--pose-timeline",
        type=str,
        default="bag",
        choices=["bag", "npy"],
        help=(
            "bag (default): use ALL bag frames + bag timestamps (~30fps) in the npz "
            "(sync to mp4 later). npy: subsample bag to mp4/npy timestamps (~15fps)."
        ),
    )

    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|fp16|fp32")
    parser.add_argument("--wilor-stride", type=int, default=1, help="Run WiLoR every N frames")

    parser.add_argument("--max-gap-frames", type=int, default=3, help="Fill gaps up to this many frames per modality")
    parser.add_argument("--open-threshold", type=float, default=1.10, help="Fixed threshold for openness score")
    parser.add_argument(
        "--open-min-run-frames",
        type=int,
        default=5,
        help="Debounce binary open/close: flip runs shorter than this many frames (~5 @30fps ≈ 0.17s).",
    )
    parser.add_argument("--depth-radius", type=int, default=4, help="Depth patch radius (pixels)")
    parser.add_argument(
        "--match-mediapipe-xyz",
        action="store_true",
        help=(
            "Match hand_pose_track.py XYZ pipeline more closely: "
            "sample depth once at --depth-radius (no fallback radii), "
            "deproject with color intrinsics (depth is aligned to color). "
            "Postprocessing still runs (gap-fill + smoothing) similar to the MediaPipe pipeline."
        ),
    )
    parser.add_argument(
        "--use-bag-color",
        action="store_true",
        help="Run WiLoR on bag color frames (better depth alignment). This is already the default.",
    )
    parser.add_argument(
        "--use-mp4-color",
        action="store_true",
        help="Run WiLoR on mp4 frames instead of bag color frames.",
    )
    parser.add_argument(
        "--slot-mode",
        type=str,
        default="auto",
        choices=["auto", "is_right", "xpos"],
        help=(
            "How to assign detections to (left,right) slots. "
            "Default auto uses WiLoR is_right (left=slot0, right=slot1), "
            "including single-hand videos; falls back to xpos only on missing/"
            "conflicting labels."
        ),
    )
    parser.add_argument(
        "--debug-overlay-video",
        type=str,
        default=None,
        help="Optional path to save an mp4 overlay with wrist pixels + bboxes.",
    )
    parser.add_argument(
        "--debug-keypoints",
        action="store_true",
        help="If set, draw WiLoR 2D keypoints on the debug overlay video.",
    )
    parser.add_argument(
        "--debug-keypoints-no-labels",
        action="store_true",
        help="If set, do not print (id:u,v) text labels for keypoints on the overlay.",
    )
    parser.add_argument(
        "--debug-max-frames",
        type=int,
        default=10_000_000,
        help="Max frames to write into debug overlay video.",
    )

    parser.add_argument("--smooth-window", type=int, default=9, help="Savitzky-Golay window (odd)")
    parser.add_argument("--smooth-poly", type=int, default=3, help="Savitzky-Golay poly order")
    parser.add_argument("--max-speed", type=float, default=3.0, help="Position spike reject speed (m/s)")
    parser.add_argument("--max-jump", type=float, default=0.25, help="Position spike reject jump (m)")

    args = parser.parse_args()
    pose_timeline = str(args.pose_timeline)

    if args.data_dir:
        out_dir = Path(args.output_dir) if args.output_dir else (Path(args.data_dir) / "combined_npz")
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for mp4, bag, npy, out_base in _resolve_triplets(
            Path(args.data_dir), require_npy=(pose_timeline == "npy")
        ):
            out_path = out_dir / f"{out_base}_wilor_rgbd_pose.npz"
            run(
                mp4,
                bag,
                out_path=out_path,
                timestamps_npy=npy,
                pose_timeline=pose_timeline,
                device=args.device,
                dtype=args.dtype,
                wilor_stride=int(args.wilor_stride),
                max_gap_frames=int(args.max_gap_frames),
                open_threshold=float(args.open_threshold),
                depth_radius=int(args.depth_radius),
                match_mediapipe_xyz=bool(args.match_mediapipe_xyz),
                use_bag_color=not bool(args.use_mp4_color),
                slot_mode=str(args.slot_mode),
                debug_overlay_path=(out_dir / f"{out_base}_debug_overlay.mp4") if args.debug_overlay_video else None,
                debug_max_frames=int(args.debug_max_frames),
                debug_kp=bool(args.debug_keypoints),
                debug_kp_label=not bool(args.debug_keypoints_no_labels),
                smooth_window=int(args.smooth_window),
                smooth_poly=int(args.smooth_poly),
                max_speed_m_s=float(args.max_speed),
                max_jump_m=float(args.max_jump),
                open_min_run_frames=int(args.open_min_run_frames),
            )
            n += 1
        print(f"Done. Wrote {n} file(s) to {out_dir}")
        return

    if not (args.bag and args.output):
        raise SystemExit("Provide either --data-dir OR (--bag --output).")
    if pose_timeline == "npy" and not args.timestamps_npy:
        raise SystemExit("--pose-timeline npy requires --timestamps-npy.")
    if args.use_mp4_color and not args.mp4:
        raise SystemExit("--use-mp4-color requires --mp4.")

    run(
        Path(args.mp4) if args.mp4 else None,
        Path(args.bag),
        out_path=Path(args.output),
        timestamps_npy=Path(args.timestamps_npy) if args.timestamps_npy else None,
        pose_timeline=pose_timeline,
        device=args.device,
        dtype=args.dtype,
        wilor_stride=int(args.wilor_stride),
        max_gap_frames=int(args.max_gap_frames),
        open_threshold=float(args.open_threshold),
        depth_radius=int(args.depth_radius),
        match_mediapipe_xyz=bool(args.match_mediapipe_xyz),
        use_bag_color=not bool(args.use_mp4_color),
        slot_mode=str(args.slot_mode),
        debug_overlay_path=Path(args.debug_overlay_video) if args.debug_overlay_video else None,
        debug_max_frames=int(args.debug_max_frames),
        debug_kp=bool(args.debug_keypoints),
        debug_kp_label=not bool(args.debug_keypoints_no_labels),
        smooth_window=int(args.smooth_window),
        smooth_poly=int(args.smooth_poly),
        max_speed_m_s=float(args.max_speed),
        max_jump_m=float(args.max_jump),
        open_min_run_frames=int(args.open_min_run_frames),
    )


if __name__ == "__main__":
    main()

