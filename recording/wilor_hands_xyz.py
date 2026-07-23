#!/usr/bin/env python3
"""Extract 3D wrist xyz for both hands from video using WiLoR.

  mono  --video in.mp4 --intrinsics fx,fy,cx,cy      z from MANO hand size (rough)
  rgbd  --video in.mp4 --depth d.npy --intrinsics …  z from the depth sensor (real)
        --bag  rec.bag                                RealSense: depth + intrinsics

Output npz (T = frames, 2 = [left, right]):
  timestamps (T,)  xyz (T,2,3)m  valid (T,2)  R (T,2,3,3)
  kpts3d (T,2,21,3)  kpts2d (T,2,21,2)px  open_score (T,2)

Setup:
  pip install --no-deps "git+https://github.com/warmshao/WiLoR-mini"
  pip install ultralytics timm smplx roma scikit-image
  python fix_mano_chumpy.py
"""
import argparse
import os

import numpy as np
import cv2
import torch

# MANO/WiLoR 21-joint layout
WRIST = 0
TIPS = [4, 8, 12, 16, 20]
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
LEFT, RIGHT = 0, 1
COLORS = {LEFT: (0, 0, 255), RIGHT: (0, 255, 0)}   # BGR

_PIPE = None        # per-process model, built once by _init_worker


def palm_frame(k3d):
    """Right-handed palm frame. x = wrist->middle MCP, z = palm normal."""
    x = k3d[MIDDLE_MCP] - k3d[WRIST]
    nx = np.linalg.norm(x)
    if nx < 1e-8:
        return np.full((3, 3), np.nan)
    x = x / nx
    z = np.cross(x, k3d[INDEX_MCP] - k3d[PINKY_MCP])
    nz = np.linalg.norm(z)
    if nz < 1e-8:
        return np.full((3, 3), np.nan)
    z = z / nz
    return np.stack([x, np.cross(z, x), z], axis=1)     # columns = axes


def open_score(k3d):
    """Mean fingertip distance from wrist / palm length. Low = fist."""
    palm = np.linalg.norm(k3d[MIDDLE_MCP] - k3d[WRIST])
    if palm < 1e-8:
        return np.nan
    return float(np.mean([np.linalg.norm(k3d[t] - k3d[WRIST]) for t in TIPS]) / palm)


def deproject(u, v, z, fx, fy, cx, cy):
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def depth_at(depth, u, v, win=5):
    """Median of valid depth in a window - robust to holes at hand edges."""
    h, w = depth.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    r = win // 2
    patch = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    good = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(good)) if good.size else None


def pick_best(outputs):
    """One detection per hand: largest bbox wins if WiLoR fires twice on a hand."""
    best = {}
    for o in outputs:
        if "wilor_preds" not in o:
            continue
        side = RIGHT if int(o["is_right"]) == 1 else LEFT
        x1, y1, x2, y2 = o["hand_bbox"]
        area = (x2 - x1) * (y2 - y1)
        if side not in best or area > best[side][0]:
            best[side] = (area, o)
    return {s: o for s, (_, o) in best.items()}


def analyze(pipe, rgb, depth, K, depth_scale):
    """One frame -> per-hand arrays. Pure function of the inputs."""
    xyz = np.full((2, 3), np.nan)
    rot = np.full((2, 3, 3), np.nan)
    k3f = np.full((2, 21, 3), np.nan)
    k2f = np.full((2, 21, 2), np.nan)
    opn = np.full(2, np.nan)
    val = np.zeros(2, bool)

    for side, o in pick_best(pipe.predict(rgb)).items():
        p = o["wilor_preds"]
        k3 = np.asarray(p["pred_keypoints_3d"][0], np.float64)   # metric hand shape
        k2 = np.asarray(p["pred_keypoints_2d"][0], np.float64)   # full-image pixels
        cam_t = np.asarray(p["pred_cam_t_full"][0], np.float64)
        sfl = float(np.asarray(p["scaled_focal_length"]))

        k3_cam = k3 + cam_t          # camera frame, at WiLoR's assumed focal
        u, v = k2[WRIST]

        z = None
        if depth is not None and K is not None:
            z = depth_at(depth, u, v)                  # measured - the good path
            if z is not None:
                z *= depth_scale
        elif K is not None:
            # cam_crop_to_full makes tz proportional to the focal length it was
            # given, so undo WiLoR's assumed focal and apply the real one.
            z = cam_t[2] * K[0] / sfl

        if z is None and depth is not None:
            continue                                   # wanted depth, got a hole

        if z is not None and K is not None:
            wrist = deproject(u, v, z, *K)
            k3_cam = k3_cam + (wrist - k3_cam[WRIST])  # rigid shift onto real depth
        else:
            wrist = k3_cam[WRIST]                      # no intrinsics: arbitrary scale

        xyz[side] = wrist
        rot[side] = palm_frame(k3_cam)
        k3f[side] = k3_cam
        k2f[side] = k2
        opn[side] = open_score(k3_cam)
        val[side] = np.all(np.isfinite(wrist))

    return xyz, val, rot, k3f, k2f, opn


def build_pipe(device, threads):
    torch.set_num_threads(threads)
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )
    # Deliberately CPU: mps runs without error but returns silently wrong
    # keypoints (bboxes match, the WiLoR head diverges), so it is never default.
    return WiLorHandPose3dEstimationPipeline(
        device=torch.device(device), dtype=torch.float32, verbose=False)


def _init_worker(device, threads):
    global _PIPE
    _PIPE = build_pipe(device, threads)


def _run_chunk(job):
    """Decode + infer one contiguous slice of frames. Runs in a worker process."""
    video, start, count, K, depth_path, depth_scale = job
    cap = cv2.VideoCapture(video)
    for _ in range(start):
        cap.grab()                      # skip cheaply, no decode
    depth = np.load(depth_path, mmap_mode="r") if depth_path else None

    out = []
    for i in range(start, start + count):
        if not cap.grab():
            break
        ok, bgr = cap.retrieve()
        if not ok:
            break
        d = np.asarray(depth[i], np.float64) if depth is not None and i < len(depth) else None
        out.append(analyze(_PIPE, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), d, K, depth_scale))
    cap.release()
    return start, out


def label(frame, u, v, text, color):
    """Text next to the dot, on a dark pill so it stays readable on any background."""
    f, s, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, tht), _ = cv2.getTextSize(text, f, s, th)
    x = int(u) + 12
    if x + tw + 6 > frame.shape[1]:
        x = int(u) - tw - 12                    # flip to the left near the edge
    y = int(np.clip(v, tht + 6, frame.shape[0] - 6))
    cv2.rectangle(frame, (x - 4, y - tht - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), f, s, color, th, cv2.LINE_AA)


def draw_overlay(video, out_path, XYZ, VALID, K2D, fps):
    """Second pass: no model, just draws. Cheap to re-run."""
    cap = cv2.VideoCapture(video)
    writer = None
    for i in range(len(XYZ)):
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (frame.shape[1], frame.shape[0]))
        for side in (LEFT, RIGHT):
            if not VALID[i, side]:
                continue
            u, v = K2D[i, side, WRIST]
            x, y, z = XYZ[i, side]
            cv2.circle(frame, (int(u), int(v)), 7, COLORS[side], -1)
            cv2.circle(frame, (int(u), int(v)), 7, (255, 255, 255), 1)
            label(frame, u, v, f"{'LR'[side]} {x:+.2f},{y:+.2f},{z:.2f}", COLORS[side])
        writer.write(frame)
    cap.release()
    if writer:
        writer.release()


def bag_frames(path):
    """RealSense bag -> aligned RGB + depth in metres + real intrinsics."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, path, repeat_playback=False)
    profile = pipeline.start(cfg)
    profile.get_device().as_playback().set_real_time(False)
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    try:
        while True:
            try:
                ok, fs = pipeline.try_wait_for_frames(2000)
            except RuntimeError:
                break
            if not ok:
                break
            fs = align.process(fs)
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c or not d:
                continue
            it = c.get_profile().as_video_stream_profile().get_intrinsics()
            yield (fs.get_timestamp() / 1000.0,
                   np.asanyarray(c.get_data()),
                   np.asanyarray(d.get_data()).astype(np.float64) * scale,
                   (it.fx, it.fy, it.ppx, it.ppy))
    finally:
        pipeline.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="RGB video (mp4)")
    ap.add_argument("--bag", help="RealSense .bag (true metric depth)")
    ap.add_argument("--depth", help="npy of aligned depth (T,H,W) in metres, matches --video")
    ap.add_argument("--intrinsics", help="fx,fy,cx,cy of the RGB stream. Required with "
                                         "--depth; in mono mode it sets the metric scale")
    ap.add_argument("--depth-scale", type=float, default=1.0)
    ap.add_argument("-o", "--out", default="hands_xyz.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (0 = auto)")
    ap.add_argument("--viz", help="write an overlay video here")
    args = ap.parse_args()

    if not args.video and not args.bag:
        ap.error("need --video or --bag")
    if args.depth and not args.intrinsics:
        ap.error("--depth needs --intrinsics fx,fy,cx,cy")

    K = tuple(float(x) for x in args.intrinsics.split(",")) if args.intrinsics else None
    if K is None and not args.bag:
        print("[wilor] no --intrinsics: xyz will be in an arbitrary scale (~20x too far)")

    if args.bag:
        ts, res, K = run_bag(args)
        fps = 30.0
    else:
        ts, res, fps = run_video(args, K)

    XYZ, VALID, ROT, K3D, K2D, OPEN = (np.array([r[i] for r in res]) for i in range(6))
    np.savez(args.out, timestamps=np.array(ts), xyz=XYZ, valid=VALID, R=ROT,
             kpts3d=K3D, kpts2d=K2D, open_score=OPEN, hand_order="left,right",
             metric=("depth" if (args.bag or args.depth)
                     else "mono_scaled" if K is not None else "mono_arbitrary"))

    n = len(ts)
    print(f"\nwrote {args.out}  frames={n}")
    if n:
        print(f"  left  valid {VALID[:,0].sum()}/{n}   right valid {VALID[:,1].sum()}/{n}")
        if not (args.bag or args.depth):
            print("  NOTE: monocular z relies on MANO's average hand size, so it is only")
            print("        a rough guess. Use --bag/--depth for true metres.")
    if args.viz and args.video:
        draw_overlay(args.video, args.viz, XYZ, VALID, K2D, fps)
        print(f"  overlay -> {args.viz}")
    elif args.viz:
        print("  --viz is only supported with --video")


def run_video(args, K):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    cores = os.cpu_count() or 4
    # The model barely scales past a couple of threads, so spend cores on frames
    # instead. But EVERY worker loads its own copy of the ViT, so this is bounded
    # by RAM, not cores - too many workers will swap and wedge the machine.
    # Default stays deliberately timid; raise --workers only if RAM allows.
    workers = args.workers or max(1, min(2, cores // 4, total))
    threads = max(1, min(4, cores // max(1, workers)))
    print(f"[wilor] {total} frames, {workers} workers x {threads} threads on {args.device}")

    if workers == 1:
        _init_worker(args.device, threads)
        _, res = _run_chunk((args.video, 0, total, K, args.depth, args.depth_scale))
    else:
        import concurrent.futures as cf

        size = (total + workers - 1) // workers
        jobs = [(args.video, s, min(size, total - s), K, args.depth, args.depth_scale)
                for s in range(0, total, size)]
        # spawn: fork + torch in a worker is a deadlock risk on macOS
        ctx = __import__("multiprocessing").get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                    initializer=_init_worker,
                                    initargs=(args.device, threads)) as ex:
            chunks = sorted(ex.map(_run_chunk, jobs), key=lambda c: c[0])
        res = [r for _, part in chunks for r in part]

    return [i / fps for i in range(len(res))], res, fps


def run_bag(args):
    pipe = build_pipe(args.device, os.cpu_count() or 4)
    ts, res, K = [], [], None
    for t, rgb, depth, intr in bag_frames(args.bag):
        K = intr
        res.append(analyze(pipe, rgb, depth, K, args.depth_scale))
        ts.append(t)
        if len(res) % 25 == 0:
            print(f"  frame {len(res)}")
    return ts, res, K


if __name__ == "__main__":
    main()
