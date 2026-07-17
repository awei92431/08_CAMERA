"""Validate geometric Depth-to-Color alignment against MuJoCo oracle depth."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fourc2.camera_geometry import (  # noqa: E402
    MUJOCO_FROM_OPTICAL,
    align_depth_to_color,
    camera_world_pose_optical,
    intrinsics_from_fovy,
    relative_optical_transform,
)


WIDTH, HEIGHT = 640, 360
COLOR_NAME = "eye_in_hand_color"
DEPTH_NAME = "eye_in_hand_depth"
OUTPUT_DIR = ROOT / "outputs" / "depth_to_color_alignment"


def get_camera_id(model, name):
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if camera_id < 0:
        raise RuntimeError(f"camera not found: {name}")
    return camera_id


def render_rgb(renderer, data, camera_id):
    renderer.update_scene(data, camera=camera_id)
    return renderer.render().copy()


def render_depth(renderer, data, camera_id):
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=camera_id)
        return renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()


def depth_preview(depth):
    valid = np.isfinite(depth) & (depth > 0.0)
    result = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2.0, 98.0])
        if high > low:
            values = np.clip((depth[valid] - low) / (high - low), 0.0, 1.0)
            result[valid] = np.asarray(255.0 * (1.0 - values), dtype=np.uint8)
    return result


def depth_edges(depth, threshold=0.01):
    valid = depth > 0.0
    edges = np.zeros(depth.shape, dtype=bool)
    dx = valid[:, 1:] & valid[:, :-1] & (np.abs(depth[:, 1:] - depth[:, :-1]) > threshold)
    dy = valid[1:, :] & valid[:-1, :] & (np.abs(depth[1:, :] - depth[:-1, :]) > threshold)
    edges[:, 1:] |= dx
    edges[:, :-1] |= dx
    edges[1:, :] |= dy
    edges[:-1, :] |= dy
    # One-pixel dilation removes bilinear/rasterization ambiguity near silhouettes.
    padded = np.pad(edges, 1)
    return np.logical_or.reduce([
        padded[y:y + depth.shape[0], x:x + depth.shape[1]]
        for y in range(3) for x in range(3)
    ])


def metrics(aligned, oracle, mask):
    errors = np.abs(aligned[mask] - oracle[mask])
    return {
        "count": int(errors.size),
        "mae": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
    }


def overlay_color_depth(color, aligned):
    valid = aligned > 0.0
    preview = depth_preview(aligned)
    heat = np.zeros_like(color)
    heat[..., 0] = preview
    heat[..., 1] = 255 - preview
    heat[..., 2] = 64
    result = color.astype(np.float32)
    result[valid] = 0.60 * result[valid] + 0.40 * heat[valid]
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene_cube3cm.xml"))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    color_id = get_camera_id(model, COLOR_NAME)
    depth_id = get_camera_id(model, DEPTH_NAME)
    color_k = intrinsics_from_fovy(WIDTH, HEIGHT, model.cam_fovy[color_id])
    depth_k = intrinsics_from_fovy(WIDTH, HEIGHT, model.cam_fovy[depth_id])
    color_from_depth = relative_optical_transform(
        camera_world_pose_optical(data, color_id),
        camera_world_pose_optical(data, depth_id),
    )
    expected = np.eye(4)
    expected[0, 3] = 0.015
    if not np.allclose(color_from_depth, expected, atol=1e-6):
        raise AssertionError(f"Color_T_Depth direction mismatch:\n{color_from_depth}")

    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
    try:
        color = render_rgb(renderer, data, color_id)
        source_depth = render_depth(renderer, data, depth_id)
        # Test-only simulation truth. It is never passed to the alignment function.
        oracle_color_depth = render_depth(renderer, data, color_id)
    finally:
        renderer.close()

    aligned = align_depth_to_color(
        source_depth, depth_k, color_k, color_from_depth
    )
    valid_aligned = aligned > 0.0
    valid_oracle = oracle_color_depth > 0.0
    overlap = valid_aligned & valid_oracle
    non_edge_overlap = overlap & ~(
        depth_edges(oracle_color_depth) | depth_edges(aligned)
    )

    coverage = float(np.mean(valid_aligned))
    oracle_coverage = float(np.sum(overlap) / np.sum(valid_oracle))
    all_stats = metrics(aligned, oracle_color_depth, overlap)
    clean_stats = metrics(aligned, oracle_color_depth, non_edge_overlap)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(OUTPUT_DIR / "color.png")
    Image.fromarray(depth_preview(source_depth)).save(
        OUTPUT_DIR / "source_depth_preview.png"
    )
    np.save(OUTPUT_DIR / "aligned_depth_to_color.npy", aligned)
    Image.fromarray(depth_preview(aligned)).save(
        OUTPUT_DIR / "aligned_depth_preview.png"
    )
    Image.fromarray(overlay_color_depth(color, aligned)).save(
        OUTPUT_DIR / "color_with_aligned_depth_overlay.png"
    )
    np.save(OUTPUT_DIR / "oracle_color_depth.npy", oracle_color_depth)

    print("Color K:\n", color_k.matrix)
    print("Depth K:\n", depth_k.matrix)
    print("MuJoCo camera <- optical axis transform:\n", MUJOCO_FROM_OPTICAL)
    print("Color_T_Depth:\n", color_from_depth)
    print("MuJoCo renderer depth interpretation: positive optical Z (-MuJoCo camera Z), metres")
    print(f"valid coverage: {coverage:.6%}")
    print(f"oracle valid coverage: {oracle_coverage:.6%}")
    print(f"overlap pixels: {all_stats['count']}")
    print(
        "all overlap: "
        f"MAE={all_stats['mae']:.6f} m, median={all_stats['median']:.6f} m, "
        f"max={all_stats['max']:.6f} m"
    )
    print(
        f"non-edge overlap ({clean_stats['count']} pixels): "
        f"MAE={clean_stats['mae']:.6f} m, median={clean_stats['median']:.6f} m, "
        f"max={clean_stats['max']:.6f} m"
    )
    print(f"saved: {OUTPUT_DIR}")
    print("PASS: geometric depth-to-color alignment")


if __name__ == "__main__":
    main()
