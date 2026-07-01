#!/usr/bin/env python3
"""Stereo triangulation utilities for RGB landmark + RealSense IR pair."""

from __future__ import annotations

import cv2
import numpy as np
import pyrealsense2 as rs

from hand_pose_track import (
    INDEX_MCP,
    PINKY_MCP,
    WRIST,
    landmark_to_pixel,
    rotation_matrix_to_quaternion,
)


def intrinsics_to_matrix(intrin: rs.intrinsics) -> np.ndarray:
    return np.array(
        [[intrin.fx, 0.0, intrin.ppx], [0.0, intrin.fy, intrin.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def extrinsics_to_rt(extrin: rs.extrinsics) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.array(extrin.rotation, dtype=np.float64).reshape(3, 3)
    translation = np.array(extrin.translation, dtype=np.float64)
    return rotation, translation


class RealSenseStereoCalibration:
    """Calibration for one RealSense device (color + rectified IR stereo pair)."""

    def __init__(self, profile: rs.pipeline_profile):
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ir1_stream = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        ir2_stream = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()

        self.color_intrin = color_stream.get_intrinsics()
        self.ir1_intrin = ir1_stream.get_intrinsics()
        self.ir2_intrin = ir2_stream.get_intrinsics()

        self.color_to_ir1 = color_stream.get_extrinsics_to(ir1_stream)
        self.ir1_to_color = ir1_stream.get_extrinsics_to(color_stream)
        self.ir1_to_ir2 = ir1_stream.get_extrinsics_to(ir2_stream)

        r12, t12 = extrinsics_to_rt(self.ir1_to_ir2)
        self.baseline_m = float(np.linalg.norm(t12))
        self.fx_ir1 = self.ir1_intrin.fx

        k1 = intrinsics_to_matrix(self.ir1_intrin)
        k2 = intrinsics_to_matrix(self.ir2_intrin)
        self.p1 = k1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.p2 = k2 @ np.hstack([r12, t12.reshape(3, 1)])


class StereoMatcher:
    """OpenCV SGBM disparity on RealSense IR1/IR2 (stereo matching)."""

    def __init__(self, num_disparities=128, block_size=5):
        nd = max(16, int(np.ceil(num_disparities / 16) * 16))
        self._stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=nd,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def compute_disparity(self, ir_left: np.ndarray, ir_right: np.ndarray) -> np.ndarray:
        if ir_left.ndim == 3:
            ir_left = cv2.cvtColor(ir_left, cv2.COLOR_BGR2GRAY)
        if ir_right.ndim == 3:
            ir_right = cv2.cvtColor(ir_right, cv2.COLOR_BGR2GRAY)
        disp = self._stereo.compute(ir_left, ir_right).astype(np.float32) / 16.0
        disp[disp <= 0] = np.nan
        return disp


def _sample_disparity(disp: np.ndarray, u: float, v: float, radius: int = 3) -> float | None:
    h, w = disp.shape
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or vi < 0 or ui >= w or vi >= h:
        return None
    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    patch = disp[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def color_pixel_to_ir1_pixel(
    color_u: float,
    color_v: float,
    depth_m: float,
    calib: RealSenseStereoCalibration,
) -> tuple[float, float]:
    point_color = rs.rs2_deproject_pixel_to_point(
        calib.color_intrin, [float(color_u), float(color_v)], float(depth_m)
    )
    point_ir1 = rs.rs2_transform_point_to_point(calib.color_to_ir1, point_color)
    ir_u, ir_v = rs.rs2_project_point_to_pixel(calib.ir1_intrin, point_ir1)
    return float(ir_u), float(ir_v)


def triangulate_ir_pair(
    u1: float,
    v1: float,
    u2: float,
    v2: float,
    calib: RealSenseStereoCalibration,
) -> np.ndarray | None:
    pts1 = np.array([[u1], [v1]], dtype=np.float64)
    pts2 = np.array([[u2], [v2]], dtype=np.float64)
    homog = cv2.triangulatePoints(calib.p1, calib.p2, pts1, pts2)
    if abs(homog[3, 0]) < 1e-8:
        return None
    point_ir1 = (homog[:3, 0] / homog[3, 0]).astype(np.float64)
    if not np.all(np.isfinite(point_ir1)):
        return None
    return point_ir1


def ir1_point_to_color(point_ir1: np.ndarray, calib: RealSenseStereoCalibration) -> np.ndarray:
    point_color = rs.rs2_transform_point_to_point(calib.ir1_to_color, point_ir1.tolist())
    return np.asarray(point_color, dtype=np.float64)


def triangulate_color_landmark(
    color_u: float,
    color_v: float,
    disp: np.ndarray,
    calib: RealSenseStereoCalibration,
    depth_search=(0.25, 1.5),
    depth_steps=24,
) -> np.ndarray | None:
    """
    Map an RGB landmark to the IR stereo pair and triangulate 3D in the color frame.

    Depth along the color ray is searched to align the RGB projection with SGBM disparity.
    """
    best_point = None
    best_err = np.inf

    for depth_m in np.linspace(depth_search[0], depth_search[1], depth_steps):
        ir_u, ir_v = color_pixel_to_ir1_pixel(color_u, color_v, depth_m, calib)
        d = _sample_disparity(disp, ir_u, ir_v)
        if d is None:
            continue

        ir_u2 = ir_u - d
        point_ir1 = triangulate_ir_pair(ir_u, ir_v, ir_u2, ir_v, calib)
        if point_ir1 is None:
            continue

        z_from_disp = (calib.baseline_m * calib.fx_ir1) / d
        err = abs(point_ir1[2] - z_from_disp)
        if err < best_err:
            best_err = err
            best_point = ir1_point_to_color(point_ir1, calib)

    if best_point is None or best_err > 0.08:
        return None
    return best_point


def deproject_color_landmark_from_disp(
    color_u: float,
    color_v: float,
    disp: np.ndarray,
    calib: RealSenseStereoCalibration,
    depth_search=(0.25, 1.5),
    depth_steps=24,
) -> np.ndarray | None:
    """
    RGBD-style 3D: pick depth along the color ray using disparity consistency,
    then deproject (no IR triangulation).
    """
    best_depth = None
    best_err = np.inf

    for depth_m in np.linspace(depth_search[0], depth_search[1], depth_steps):
        ir_u, ir_v = color_pixel_to_ir1_pixel(color_u, color_v, depth_m, calib)
        d = _sample_disparity(disp, ir_u, ir_v)
        if d is None:
            continue
        z_from_disp = (calib.baseline_m * calib.fx_ir1) / d
        err = abs(depth_m - z_from_disp)
        if err < best_err:
            best_err = err
            best_depth = depth_m

    if best_depth is None or best_err > 0.08:
        return None
    point = rs.rs2_deproject_pixel_to_point(
        calib.color_intrin, [float(color_u), float(color_v)], float(best_depth)
    )
    return np.asarray(point, dtype=np.float64)


def estimate_hand_pose_6dof_rgbd_from_disp(landmarks, disp, calib, width, height):
    """RGBD-style pose from disparity map (depth lookup + deproject, not triangulation)."""
    wrist_u, wrist_v = landmark_to_pixel(landmarks[WRIST], width, height)
    index_u, index_v = landmark_to_pixel(landmarks[INDEX_MCP], width, height)
    pinky_u, pinky_v = landmark_to_pixel(landmarks[PINKY_MCP], width, height)

    wrist = deproject_color_landmark_from_disp(wrist_u, wrist_v, disp, calib)
    index_mcp = deproject_color_landmark_from_disp(index_u, index_v, disp, calib)
    pinky_mcp = deproject_color_landmark_from_disp(pinky_u, pinky_v, disp, calib)
    if wrist is None or index_mcp is None or pinky_mcp is None:
        return None

    x_axis = index_mcp - wrist
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return None
    x_axis /= x_norm

    palm_span = pinky_mcp - wrist
    z_axis = np.cross(x_axis, palm_span)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return None
    z_axis /= z_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    quaternion = rotation_matrix_to_quaternion(rotation)
    return wrist, quaternion


def triangulate_color_landmark_from_depth(
    color_u: float,
    color_v: float,
    depth_m: float,
    calib: RealSenseStereoCalibration,
) -> np.ndarray | None:
    """
    Stereo triangulation using depth sampled on the same recorded frame as RGBD.
    Projects the color landmark into IR1/IR2, then triangulates the ray pair.
    """
    if depth_m <= 0 or not np.isfinite(depth_m):
        return None

    point_color = rs.rs2_deproject_pixel_to_point(
        calib.color_intrin, [float(color_u), float(color_v)], float(depth_m)
    )
    point_ir1 = rs.rs2_transform_point_to_point(calib.color_to_ir1, point_color)
    z_ir1 = float(point_ir1[2])
    if z_ir1 <= 1e-6:
        return None

    ir_u, ir_v = rs.rs2_project_point_to_pixel(calib.ir1_intrin, point_ir1)
    d = (calib.baseline_m * calib.fx_ir1) / z_ir1
    ir_u2 = float(ir_u) - d
    point_tri = triangulate_ir_pair(float(ir_u), float(ir_v), ir_u2, float(ir_v), calib)
    if point_tri is None:
        return None
    return ir1_point_to_color(point_tri, calib)


def sample_depth_at_color(depth_image: np.ndarray, u: float, v: float, depth_scale: float, radius=4):
    h, w = depth_image.shape
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or vi < 0 or ui >= w or vi >= h:
        return None
    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    patch = depth_image[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid) * depth_scale)


def estimate_hand_pose_6dof_stereo_from_depth(
    landmarks, depth_image, depth_scale, calib, width, height
):
    """Stereo triangulation using depth from the same recorded frame as RGBD."""
    wrist_u, wrist_v = landmark_to_pixel(landmarks[WRIST], width, height)
    index_u, index_v = landmark_to_pixel(landmarks[INDEX_MCP], width, height)
    pinky_u, pinky_v = landmark_to_pixel(landmarks[PINKY_MCP], width, height)

    wrist_d = sample_depth_at_color(depth_image, wrist_u, wrist_v, depth_scale)
    index_d = sample_depth_at_color(depth_image, index_u, index_v, depth_scale)
    pinky_d = sample_depth_at_color(depth_image, pinky_u, pinky_v, depth_scale)
    if wrist_d is None or index_d is None or pinky_d is None:
        return None

    wrist = triangulate_color_landmark_from_depth(wrist_u, wrist_v, wrist_d, calib)
    index_mcp = triangulate_color_landmark_from_depth(index_u, index_v, index_d, calib)
    pinky_mcp = triangulate_color_landmark_from_depth(pinky_u, pinky_v, pinky_d, calib)
    if wrist is None or index_mcp is None or pinky_mcp is None:
        return None

    x_axis = index_mcp - wrist
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return None
    x_axis /= x_norm

    palm_span = pinky_mcp - wrist
    z_axis = np.cross(x_axis, palm_span)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return None
    z_axis /= z_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    quaternion = rotation_matrix_to_quaternion(rotation)
    return wrist, quaternion


def estimate_hand_pose_6dof_stereo(landmarks, disp, calib, width, height):
    """
    RGB landmarks + IR stereo triangulation -> wrist 6DOF in color camera frame.
    """
    wrist_u, wrist_v = landmark_to_pixel(landmarks[WRIST], width, height)
    index_u, index_v = landmark_to_pixel(landmarks[INDEX_MCP], width, height)
    pinky_u, pinky_v = landmark_to_pixel(landmarks[PINKY_MCP], width, height)

    wrist = triangulate_color_landmark(wrist_u, wrist_v, disp, calib)
    index_mcp = triangulate_color_landmark(index_u, index_v, disp, calib)
    pinky_mcp = triangulate_color_landmark(pinky_u, pinky_v, disp, calib)
    if wrist is None or index_mcp is None or pinky_mcp is None:
        return None

    x_axis = index_mcp - wrist
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return None
    x_axis /= x_norm

    palm_span = pinky_mcp - wrist
    z_axis = np.cross(x_axis, palm_span)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return None
    z_axis /= z_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    quaternion = rotation_matrix_to_quaternion(rotation)
    return wrist, quaternion
