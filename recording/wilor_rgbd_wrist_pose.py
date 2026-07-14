#!/usr/bin/env python3
"""
WiLoR + RealSense depth -> wrist 6DOF dataset aligned to recording frame indices.

Inputs:
  - RGB MP4 video (color)
  - RealSense .bag (depth + color, used to fetch aligned depth)
  - (optional) timestamps .npy written by realsense_bird_record.py (one per MP4 frame)
  - OR a single bird-realsense-data directory containing mp4/npy/bag subfolders

Pipeline per frame (frame index i):
  1) Run WiLoR-mini on the RGB frame -> per-hand outputs with:
       - pred_keypoints_2d (wrist pixel is joint 0)
       - pred_keypoints_3d (OpenPose hand joints in MANO/wrist frame)
  2) Sample depth around the wrist pixel from the bag's aligned depth frame
  3) Deproject (u,v,depth) -> XYZ in meters in the RealSense optical frame
  4) Orientation: derive a palm frame from WiLoR 3D joints in wrist frame and store rot6d (first two columns)
  5) Openness: compute a scale-invariant openness score from WiLoR joints and threshold it
  6) Postprocess each modality (pos, rot, open) independently:
       - fill short gaps (<= max_gap_frames)
       - smooth (pos: Savitzky-Golay)
       - invalidate long gaps (do not drop frames)

Output (.npz):
  - timestamps: (N,)
  - pose: (N,2,10) = [x,y,z, rot6d(6), hand_open] with NaNs where invalid
  - valid_pos, valid_rot, valid_open: (N,2) bool

Frame count:
  - If timestamps_npy is provided, N is taken from it and we align by frame index.
  - Otherwise N is min(n_mp4_frames, n_bag_frames_processed).
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
    Wrist-frame orientation from WiLoR MANO/OpenPose joints (translation irrelevant).
    Columns of R:
      x: wrist -> index_mcp (5)
      y: wrist -> palm_center (projected orthogonal to x)
      z: cross(x,y)
    """
    wrist = j[0]
    index_mcp = j[5]
    palm_center = _palm_center_openpose(j)
    palm_dir = _normalize(palm_center - wrist)
    x = _normalize(index_mcp - wrist)
    y = palm_dir - float(np.dot(palm_dir, x)) * x
    y = _normalize(y)
    if float(np.linalg.norm(y)) < 1e-8:
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(tmp, x))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y = _normalize(tmp - float(np.dot(tmp, x)) * x)
    z = _normalize(np.cross(x, y))
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


def run(
    mp4_path: Path,
    bag_path: Path,
    *,
    out_path: Path,
    timestamps_npy: Path | None,
    device: str = "auto",
    dtype: str = "auto",
    wilor_stride: int = 1,
    max_gap_frames: int = 3,
    open_threshold: float = 1.10,
    depth_radius: int = 4,
    smooth_window: int = 9,
    smooth_poly: int = 3,
    max_speed_m_s: float = 3.0,
    max_jump_m: float = 0.12,
):
    import cv2
    import pyrealsense2 as rs
    from scipy.signal import savgol_filter

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    mp4_path = Path(mp4_path)
    bag_path = Path(bag_path)
    out_path = Path(out_path)

    if timestamps_npy is not None:
        ts = np.asarray(np.load(timestamps_npy), dtype=np.float64)
        N_target = int(ts.shape[0])
    else:
        ts = None
        N_target = None

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open mp4: {mp4_path}")

    # Bag playback
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

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

    # Pre-allocate
    # pose = [x,y,z, rot6d, open]
    if N_target is None:
        # estimate from mp4 frame count
        N_target = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    else:
        # if timestamps are provided, drop extra timestamps that exceed the mp4 length
        mp4_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if mp4_n > 0:
            N_target = min(int(N_target), int(mp4_n))
    N = int(N_target)
    pose = np.full((N, 2, 10), np.nan, dtype=np.float64)
    valid_pos = np.zeros((N, 2), dtype=bool)
    valid_rot = np.zeros((N, 2), dtype=bool)
    valid_open = np.zeros((N, 2), dtype=bool)

    # Raw orientation as rotation matrix for gap filling
    R_raw = np.full((N, 2, 3, 3), np.nan, dtype=np.float64)
    open_score = np.full((N, 2), np.nan, dtype=np.float64)

    # Iterate frames by index using mp4, and depth using bag
    i = 0
    try:
        while i < N:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                i += 1
                continue

            # WiLoR stride
            pred = None
            if (i % max(int(wilor_stride), 1)) == 0:
                pred = pipe.predict(frame_bgr)
                if not isinstance(pred, list):
                    pred = []

            # Parse detections into left/right slots
            chosen = {0: None, 1: None}
            if isinstance(pred, list):
                for det in pred:
                    if not isinstance(det, dict):
                        continue
                    slot = 1 if float(det.get("is_right", 0.0)) > 0.5 else 0
                    if chosen[slot] is None:
                        chosen[slot] = det

            for hand in (0, 1):
                det = chosen[hand]
                if det is None:
                    continue
                wp = det.get("wilor_preds", None)
                if not isinstance(wp, dict):
                    continue

                # 2D wrist pixel from WiLoR
                k2d = wp.get("pred_keypoints_2d", None)
                if k2d is None:
                    continue
                k2d = np.asarray(k2d, dtype=np.float64)
                if k2d.ndim != 3 or k2d.shape[1] < 1:
                    continue
                u, v = float(k2d[0, 0, 0]), float(k2d[0, 0, 1])

                depth_m = _sample_depth_m(depth_frame, u, v, radius=int(depth_radius))
                if depth_m is not None:
                    xyz = rs.rs2_deproject_pixel_to_point(intr, [u, v], depth_m)
                    xyz = np.asarray(xyz, dtype=np.float64)
                    if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
                        pose[i, hand, 0:3] = xyz
                        valid_pos[i, hand] = True

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

            i += 1

    finally:
        cap.release()
        pipeline.stop()

    processed = int(i)

    # Build time axis for processing (keep full length when timestamps are provided)
    if ts is None:
        ts = np.arange(processed, dtype=np.float64)
        N_out = processed
    else:
        ts = np.asarray(ts, dtype=np.float64)[:N]
        N_out = int(ts.shape[0])

    if ts is None:
        # (unreachable, kept for clarity)
        pass
    elif processed < N_out:
        # We could not read all frames from mp4/bag; keep trailing frames invalid
        # so frame indices still match timestamps.
        pass

    if ts.shape[0] != pose.shape[0]:
        # If timestamps provided, pose was preallocated to same length.
        # If timestamps were not provided, we will truncate pose to processed below.
        pass

    if ts.shape[0] == processed:
        # normal case with no timestamps_npy: we operate on the full arrays and save them
        pose = pose[:processed]
        valid_pos = valid_pos[:processed]
        valid_rot = valid_rot[:processed]
        valid_open = valid_open[:processed]
        R_raw = R_raw[:processed]
        open_score = open_score[:processed]

    # --- Postprocess (no dropping frames) ---
    if ts.shape[0] == processed:
        proc_slice = slice(None)
        ts_proc = ts
    else:
        proc_slice = slice(0, processed)
        ts_proc = ts[proc_slice]

    t_rel = ts_proc - ts_proc[0] if ts_proc.size > 0 else ts_proc

    # Position: reject spikes -> fill short gaps -> smooth
    pos_f = pose[proc_slice, :, 0:3].copy()
    vpos = valid_pos[proc_slice].copy() & np.isfinite(pos_f).all(axis=-1)
    fps_est = 1.0 / float(np.median(np.diff(t_rel))) if t_rel.size > 2 else 15.0

    for h in (0, 1):
        # outlier reject based on speed/jump
        prev_idx = None
        prev_pos = None
        for k in range(pos_f.shape[0]):
            if not vpos[k, h]:
                continue
            p = pos_f[k, h]
            if prev_idx is not None:
                dt_s = max(k - prev_idx, 1) / float(fps_est)
                speed = float(np.linalg.norm(p - prev_pos) / max(dt_s, 1e-6))
                jump = float(np.linalg.norm(p - prev_pos))
                if speed > max_speed_m_s or jump > max_jump_m:
                    vpos[k, h] = False
                    pos_f[k, h] = np.nan
                    continue
            prev_idx = k
            prev_pos = p

        # fill short gaps per axis
        for ax in range(3):
            pos_f[:, h, ax], v_ax = _fill_short_gaps_linear(t_rel, pos_f[:, h, ax], vpos[:, h], max_gap_frames=int(max_gap_frames))
            vpos[:, h] &= v_ax

        # smooth valid runs
        win = int(smooth_window)
        if win % 2 == 0:
            win += 1
        for start, end in _contiguous_segments(vpos[:, h]):
            if end - start + 1 < win:
                continue
            for ax in range(3):
                seg = pos_f[start : end + 1, h, ax]
                pos_f[start : end + 1, h, ax] = savgol_filter(seg, window_length=win, polyorder=min(int(smooth_poly), win - 1))

    # Orientation: fill short gaps in SO(3)
    R_f = R_raw[proc_slice].copy()
    vori = valid_rot[proc_slice].copy()
    for h in (0, 1):
        R_f[:, h], vori[:, h] = _fill_short_gaps_so3(t_rel, R_f[:, h], vori[:, h], max_gap_frames=int(max_gap_frames))

    # Open: fill score short gaps and threshold
    s_f = open_score[proc_slice].copy()
    vs = valid_open[proc_slice].copy() & np.isfinite(s_f)
    for h in (0, 1):
        s_f[:, h], vs[:, h] = _fill_short_gaps_linear(t_rel, s_f[:, h], vs[:, h], max_gap_frames=int(max_gap_frames))
    open_flag = (s_f > float(open_threshold)).astype(np.float64)

    # Write back into pose with NaNs where invalid
    pose[proc_slice, :, 0:3] = np.where(vpos[..., None], pos_f, np.nan)
    for k in range(pose[proc_slice].shape[0]):
        for h in (0, 1):
            if vori[k, h]:
                pose[k, h, 3:6] = R_f[k, h, :, 0]
                pose[k, h, 6:9] = R_f[k, h, :, 1]
            else:
                pose[k, h, 3:9] = np.nan
            if vs[k, h]:
                pose[k, h, 9] = open_flag[k, h]
            else:
                pose[k, h, 9] = np.nan

    # If we processed a prefix only (timestamps_npy longer than readable frames),
    # write back validity masks for that prefix and keep the tail invalid.
    if ts.shape[0] != processed:
        valid_pos[proc_slice] = vpos
        valid_rot[proc_slice] = vori
        valid_open[proc_slice] = vs
    else:
        valid_pos = vpos
        valid_rot = vori
        valid_open = vs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, timestamps=ts, pose=pose, valid_pos=valid_pos, valid_rot=valid_rot, valid_open=valid_open)
    print(f"Saved -> {out_path} (N={pose.shape[0]})")


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


def _resolve_triplets(data_dir: Path):
    """
    Yield (mp4, bag, npy, out_name_base) for each recording found in data_dir/mp4.
    If bag/npy are missing, that recording is skipped.
    """
    data_dir = Path(data_dir)
    mp4_dir = data_dir / "mp4"
    npy_dir = data_dir / "npy"
    bag_dir = data_dir / "bag"
    if not mp4_dir.is_dir():
        raise FileNotFoundError(f"Expected mp4 folder: {mp4_dir}")
    if not npy_dir.is_dir():
        raise FileNotFoundError(f"Expected npy folder: {npy_dir}")
    if not bag_dir.is_dir():
        raise FileNotFoundError(f"Expected bag folder: {bag_dir}")

    mp4s = sorted(mp4_dir.glob("*.mp4"))
    for mp4 in mp4s:
        parsed = _parse_bird_mp4_name(mp4.name)
        if parsed is None:
            continue
        serial, session = parsed
        npy = npy_dir / f"video_recording_bird_realsense_{serial}#{session}.npy"
        bag = bag_dir / f"bird_realsense_{serial}#{session}.bag"
        if not npy.is_file() or not bag.is_file():
            # user requested: if trio isn't present, skip it
            continue
        out_base = f"video_recording_bird_realsense_{serial}#{session}"
        yield mp4, bag, npy, out_base


def main():
    parser = argparse.ArgumentParser(description="WiLoR + bag depth -> wrist XYZ + rot6d + open/close per frame.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Process a whole bird-realsense-data directory containing mp4/, npy/, bag/. "
            "Each mp4 is matched to its corresponding npy+bag by filename."
        ),
    )
    parser.add_argument("--mp4", type=str, default=None, help="Input RGB mp4")
    parser.add_argument("--bag", type=str, default=None, help="Input RealSense .bag with depth")
    parser.add_argument("--timestamps-npy", type=str, default=None, help="Optional timestamps .npy (frame-aligned)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output .npz path (single-file mode)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (data-dir batch mode). Writes one .npz per recording.",
    )

    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|fp16|fp32")
    parser.add_argument("--wilor-stride", type=int, default=1, help="Run WiLoR every N frames")

    parser.add_argument("--max-gap-frames", type=int, default=3, help="Fill gaps up to this many frames per modality")
    parser.add_argument("--open-threshold", type=float, default=1.10, help="Fixed threshold for openness score")
    parser.add_argument("--depth-radius", type=int, default=4, help="Depth patch radius (pixels)")

    parser.add_argument("--smooth-window", type=int, default=9, help="Savitzky-Golay window (odd)")
    parser.add_argument("--smooth-poly", type=int, default=3, help="Savitzky-Golay poly order")
    parser.add_argument("--max-speed", type=float, default=3.0, help="Position spike reject speed (m/s)")
    parser.add_argument("--max-jump", type=float, default=0.12, help="Position spike reject jump (m)")

    args = parser.parse_args()

    if args.data_dir:
        out_dir = Path(args.output_dir) if args.output_dir else (Path(args.data_dir) / "combined_npz")
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for mp4, bag, npy, out_base in _resolve_triplets(Path(args.data_dir)):
            out_path = out_dir / f"{out_base}_wilor_rgbd_pose.npz"
            run(
                mp4,
                bag,
                out_path=out_path,
                timestamps_npy=npy,
                device=args.device,
                dtype=args.dtype,
                wilor_stride=int(args.wilor_stride),
                max_gap_frames=int(args.max_gap_frames),
                open_threshold=float(args.open_threshold),
                depth_radius=int(args.depth_radius),
                smooth_window=int(args.smooth_window),
                smooth_poly=int(args.smooth_poly),
                max_speed_m_s=float(args.max_speed),
                max_jump_m=float(args.max_jump),
            )
            n += 1
        print(f"Done. Wrote {n} file(s) to {out_dir}")
        return

    if not (args.mp4 and args.bag and args.output):
        raise SystemExit("Provide either --data-dir OR (--mp4 --bag --output).")

    run(
        Path(args.mp4),
        Path(args.bag),
        out_path=Path(args.output),
        timestamps_npy=Path(args.timestamps_npy) if args.timestamps_npy else None,
        device=args.device,
        dtype=args.dtype,
        wilor_stride=int(args.wilor_stride),
        max_gap_frames=int(args.max_gap_frames),
        open_threshold=float(args.open_threshold),
        depth_radius=int(args.depth_radius),
        smooth_window=int(args.smooth_window),
        smooth_poly=int(args.smooth_poly),
        max_speed_m_s=float(args.max_speed),
        max_jump_m=float(args.max_jump),
    )


if __name__ == "__main__":
    main()

