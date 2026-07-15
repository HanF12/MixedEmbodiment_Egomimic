#!/usr/bin/env python3
"""
Run `wilor-mini` (pip package) hand pose prediction on MP4 videos.

Your `wilor-mini` demo snippet did:

    pipe = WiLorHandPose3dEstimationPipeline(...)
    outputs = pipe.predict(image_rgb)

This script applies the same `pipe.predict()` call to frames extracted from the
bird RealSense MP4 recordings and saves the per-frame predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {what}: {path}")
    return path


def _rotvec_to_quat_wxyz(rv: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rotation-vector (axis-angle) -> quaternion (w,x,y,z)."""
    rv = np.asarray(rv, dtype=np.float64).reshape(-1)
    if rv.shape[0] != 3:
        raise ValueError(f"Expected rotvec shape (3,), got {rv.shape}")
    angle = float(np.linalg.norm(rv))
    if angle < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rv / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


def _rotvec_to_rotmat(rv: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Rotation-vector (axis-angle) -> 3x3 rotation matrix."""
    rv = np.asarray(rv, dtype=np.float64).reshape(-1)
    if rv.shape[0] != 3:
        raise ValueError(f"Expected rotvec shape (3,), got {rv.shape}")
    theta = float(np.linalg.norm(rv))
    if theta < eps:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    kx, ky, kz = k.tolist()
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    I = np.eye(3, dtype=np.float64)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _rotation_matrix_to_quaternion_wxyz(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> quaternion (w,x,y,z)."""
    m = np.asarray(rot, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"Expected rot shape (3,3), got {m.shape}")
    trace = float(np.trace(m))
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def _device_from_arg(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def _dtype_from_arg(dtype: str, device: torch.device) -> torch.dtype:
    dtype = dtype.lower().strip()
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype!r} (expected auto|fp16|fp32)")


def _to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _to_serializable(x: Any) -> Any:
    """Best-effort conversion of pipeline outputs for np.save(allow_pickle=True)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {k: _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]
    # Fallback: keep as-is (pickle may still handle it)
    return x


def _augment_pred_with_wrist_pose_cam(pred_ser: Any) -> Any:
    """
    Add camera-space 6DOF wrist pose to each detected hand entry if possible.

    Adds keys at the detection dict level:
      - wrist_pos_cam: (3,) float64
      - wrist_quat_cam_wxyz: (4,) float64   (from global_orient)
      - wrist_quat_cam_wxyz_from_joints: (4,) float64 (from palm frame joints, when possible)

    This keeps the original wilor-mini output intact and only adds derived fields.
    """
    if not isinstance(pred_ser, list):
        return pred_ser

    out_list: list[Any] = []
    for det in pred_ser:
        if not isinstance(det, dict):
            out_list.append(det)
            continue

        det2 = dict(det)
        wp = det2.get("wilor_preds", None)
        if not isinstance(wp, dict):
            out_list.append(det2)
            continue

        # Required for camera-space translation
        cam_t_full = wp.get("pred_cam_t_full", None)
        k3d = wp.get("pred_keypoints_3d", None)

        try:
            cam_t = np.asarray(cam_t_full, dtype=np.float64).reshape(-1)[:3] if cam_t_full is not None else None
            k3d_arr = np.asarray(k3d, dtype=np.float64) if k3d is not None else None
        except Exception:
            cam_t = None
            k3d_arr = None

        # Wrist position: keypoint 0 + camera translation (when both are present)
        if cam_t is not None and k3d_arr is not None and k3d_arr.ndim == 3 and k3d_arr.shape[1] >= 1:
            wrist = k3d_arr[0, 0]
            if wrist.shape == (3,) and np.all(np.isfinite(cam_t)) and np.all(np.isfinite(wrist)):
                det2["wrist_pos_cam"] = (wrist + cam_t).astype(np.float64)

        # Wrist orientation from global_orient (rotation vector)
        go = wp.get("global_orient", None)
        try:
            go_arr = np.asarray(go, dtype=np.float64)
            if go_arr.ndim == 3 and go_arr.shape[-1] == 3:
                rv = go_arr[0, 0]
                if rv.shape == (3,) and np.all(np.isfinite(rv)):
                    det2["wrist_rotvec_cam"] = rv.astype(np.float64)
                    det2["wrist_quat_cam_wxyz"] = _rotvec_to_quat_wxyz(rv).astype(np.float64)
        except Exception:
            pass

        # Optional: orientation derived from palm frame joints (wrist->index_mcp, wrist->pinky_mcp)
        # MANO joint indices typically: wrist=0, index_mcp=5, pinky_mcp=17
        if cam_t is not None and k3d_arr is not None and k3d_arr.ndim == 3 and k3d_arr.shape[1] > 17:
            try:
                wrist = k3d_arr[0, 0] + cam_t
                index_mcp = k3d_arr[0, 5] + cam_t
                pinky_mcp = k3d_arr[0, 17] + cam_t
                x_axis = _normalize(index_mcp - wrist)
                palm_span = pinky_mcp - wrist
                z_axis = _normalize(np.cross(x_axis, palm_span))
                y_axis = _normalize(np.cross(z_axis, x_axis))
                R = np.column_stack([x_axis, y_axis, z_axis])
                if np.all(np.isfinite(R)):
                    det2["wrist_quat_cam_wxyz_from_joints"] = _rotation_matrix_to_quaternion_wxyz(R)
                    det2["wrist_rotmat_cam_from_joints"] = R.astype(np.float64)
            except Exception:
                pass

        out_list.append(det2)

    return out_list


def run_wilor_mini_on_video(
    pipe,
    video_path: Path,
    *,
    out_path: Path,
    stride: int,
    max_frames: int | None,
    save_overlay_video: bool = False,
    overlay_out_path: Path | None = None,
) -> Path:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    stride = max(int(stride), 1)
    max_frames_eff = total_frames if max_frames is None else min(total_frames, int(max_frames))

    frames_out: list[dict[str, Any]] = []

    vout = None
    renderer = None
    overlay_path_final: Path | None = None
    if save_overlay_video:
        if overlay_out_path is None:
            overlay_path_final = out_path.with_suffix("").with_name(out_path.stem + "_overlay.mp4")
        else:
            overlay_path_final = overlay_out_path
        overlay_path_final.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Since we only write every `stride` frame, adjust fps so playback speed matches wall time.
        fps_out = float(fps) / float(stride) if fps > 1e-6 else 15.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vout = cv2.VideoWriter(str(overlay_path_final), fourcc, fps_out, (width, height))

    frame_idx = 0
    while frame_idx < max_frames_eff:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        img_rgb = _to_rgb(frame_bgr)
        pred = pipe.predict(img_rgb)
        pred_ser = _to_serializable(pred)
        pred_ser = _augment_pred_with_wrist_pose_cam(pred_ser)

        if vout is not None and isinstance(pred_ser, list):
            # Lazy-init renderer once we know the pipeline has loaded the MANO model.
            if renderer is None:
                try:
                    renderer = _MeshOverlayRenderer(faces=np.asarray(pipe.wilor_model.mano.faces))
                except Exception as e:
                    raise RuntimeError(
                        "Failed to initialize mesh renderer. If you don't need mesh video output, "
                        "re-run without `--save-overlay-video`."
                    ) from e
            overlay_bgr = renderer.overlay_on_bgr(frame_bgr, pred_ser)
            vout.write(overlay_bgr)

        frames_out.append(
            {
                "frame_idx": int(frame_idx),
                "time_s": float(frame_idx) / float(fps if fps > 1e-6 else 1.0),
                "pred": pred_ser,
            }
        )
        frame_idx += 1

    cap.release()
    if vout is not None:
        vout.release()
        print(f"Saved overlay video -> {overlay_path_final}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "stride": stride,
        "max_frames": max_frames,
        "frames": frames_out,
    }
    np.save(out_path, payload, allow_pickle=True)
    print(f"Saved wilor-mini predictions -> {out_path} ({len(frames_out)} frames)")
    return out_path


class _MeshOverlayRenderer:
    """
    Minimal pyrender-based mesh overlay renderer (adapted from WiLoR-mini tests).

    Imported only when `--save-overlay-video` is used.
    """

    def __init__(self, faces: np.ndarray):
        import pyrender
        import trimesh

        self.pyrender = pyrender
        self.trimesh = trimesh
        self.faces = np.asarray(faces, dtype=np.int64)
        self.faces_left = self.faces[:, [0, 2, 1]]

    def _vertices_to_trimesh(self, vertices: np.ndarray, *, is_right: bool, color_rgb=(0.25, 0.27, 0.66)):
        vertex_colors = np.array([(*color_rgb, 1.0)] * vertices.shape[0], dtype=np.float32)
        faces = self.faces if is_right else self.faces_left
        mesh = self.trimesh.Trimesh(vertices.copy(), faces.copy(), vertex_colors=vertex_colors, process=False)

        # Match WiLoR-mini test rendering conventions
        rot = self.trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def _render_rgba(
        self,
        vertices: np.ndarray,
        *,
        cam_t: np.ndarray,
        focal_length: float,
        render_res_wh: tuple[int, int],
        is_right: bool,
    ) -> np.ndarray:
        pyr = self.pyrender

        w, h = int(render_res_wh[0]), int(render_res_wh[1])
        renderer = pyr.OffscreenRenderer(viewport_width=w, viewport_height=h, point_size=1.0)

        # Camera translation convention used in WiLoR-mini tests
        camera_translation = np.asarray(cam_t, dtype=np.float64).copy()
        camera_translation[0] *= -1.0

        mesh_tm = self._vertices_to_trimesh(vertices, is_right=is_right)
        mesh = pyr.Mesh.from_trimesh(mesh_tm)

        scene = pyr.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, "mesh")

        camera_center = [w / 2.0, h / 2.0]
        camera = pyr.IntrinsicsCamera(
            fx=float(focal_length),
            fy=float(focal_length),
            cx=float(camera_center[0]),
            cy=float(camera_center[1]),
            zfar=1e12,
        )
        camera_pose = np.eye(4, dtype=np.float64)
        camera_pose[:3, 3] = camera_translation
        cam_node = pyr.Node(camera=camera, matrix=camera_pose)
        scene.add_node(cam_node)

        # Simple lighting: directional + point to get a usable overlay
        scene.add(pyr.DirectionalLight(color=np.ones(3), intensity=1.0), pose=np.eye(4))
        scene.add(pyr.PointLight(color=np.ones(3), intensity=0.8), pose=np.eye(4))

        color, _depth = renderer.render(scene, flags=pyr.RenderFlags.RGBA)
        renderer.delete()
        return color.astype(np.float32) / 255.0

    def overlay_on_bgr(self, frame_bgr: np.ndarray, pred_list: list[dict[str, Any]]) -> np.ndarray:
        import cv2

        h, w = frame_bgr.shape[:2]
        base_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out_rgb = base_rgb.copy()

        for det in pred_list:
            if not isinstance(det, dict):
                continue
            wp = det.get("wilor_preds", None)
            if not isinstance(wp, dict):
                continue

            verts = wp.get("pred_vertices", None)
            cam_t = wp.get("pred_cam_t_full", None)
            focal = wp.get("scaled_focal_length", None)
            if verts is None or cam_t is None or focal is None:
                continue

            verts = np.asarray(verts)[0]
            cam_t = np.asarray(cam_t)[0]
            is_right = bool(float(det.get("is_right", 1.0)) > 0.5)

            cam_view = self._render_rgba(
                verts,
                cam_t=cam_t,
                focal_length=float(focal),
                render_res_wh=(w, h),
                is_right=is_right,
            )
            alpha = cam_view[:, :, 3:4]
            out_rgb = out_rgb * (1.0 - alpha) + cam_view[:, :, :3] * alpha

        out_bgr = (np.clip(out_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]
        return out_bgr


def main() -> None:
    default_in = (
        REPO_ROOT
        / "recording"
        / "sessions"
        / "human_hands_bimanual"
        / "bird-realsense-data"
        / "mp4"
    )

    parser = argparse.ArgumentParser(description="Run wilor-mini on MP4 video(s) and save predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(default_in),
        help="Either a directory containing .mp4 files, or a single .mp4 file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "recording" / "wilor_mini_out"),
        help="Directory to write *_wilor_mini_pred.npy files",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    parser.add_argument("--dtype", type=str, default="auto", help="auto|fp16|fp32")
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames processed per video")
    parser.add_argument(
        "--save-overlay-video",
        action="store_true",
        help="Also save an MP4 overlay with the predicted hand mesh (requires pyrender + trimesh). "
        "Overlay is written at fps/stride so playback speed matches wall time.",
    )
    parser.add_argument(
        "--overlay-output-dir",
        type=str,
        default=None,
        help="Where to write overlay MP4s (default: same as --output-dir).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    device = _device_from_arg(args.device)
    dtype = _dtype_from_arg(args.dtype, device)

    # Validate OpenCV import early with a helpful message.
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "OpenCV is not installed in this environment (import `cv2` failed).\n"
            "Install one of:\n"
            "  - conda: `conda install -c conda-forge opencv`\n"
            "  - pip:   `python -m pip install opencv-python` (or `opencv-python-headless`)\n"
            "Note: `pip install cv2` will NOT work."
        ) from e

    if args.save_overlay_video:
        try:
            import trimesh  # noqa: F401
            import pyrender  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Mesh overlay video requires `trimesh` and `pyrender`.\n"
                "Install one of:\n"
                "  - conda: `conda install -c conda-forge trimesh pyrender`\n"
                "  - pip:   `python -m pip install trimesh pyrender`\n"
                "If you're on a headless machine and pyrender fails to create an OpenGL context, try:\n"
                "  `export PYOPENGL_PLATFORM=egl`  (or `osmesa` if available)\n"
            ) from e

    try:
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Could not import `wilor_mini`. Make sure you're running inside the conda env "
            "where you installed `wilor-mini` (you mentioned: `conda activate mixed`)."
        ) from e

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

    input_path = Path(args.input)
    if input_path.is_file():
        if input_path.suffix.lower() != ".mp4":
            raise ValueError(f"--input file must be a .mp4, got: {input_path}")
        mp4s = [input_path]
    else:
        in_dir = _require_dir(input_path, "input directory")
        mp4s = sorted(in_dir.glob("*.mp4"))
        if not mp4s:
            raise FileNotFoundError(f"No .mp4 files found in {in_dir}")

    for video_path in mp4s:
        out_path = out_dir / f"{video_path.stem}_wilor_mini_pred.npy"
        overlay_out_path = None
        if args.save_overlay_video:
            overlay_dir = Path(args.overlay_output_dir) if args.overlay_output_dir else out_dir
            overlay_out_path = overlay_dir / f"{video_path.stem}_wilor_mini_overlay.mp4"
        run_wilor_mini_on_video(
            pipe,
            video_path,
            out_path=out_path,
            stride=int(args.stride),
            max_frames=args.max_frames,
            save_overlay_video=bool(args.save_overlay_video),
            overlay_out_path=overlay_out_path,
        )


if __name__ == "__main__":
    main()
