"""Render the nominal D435i color and depth streams without alignment."""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
XML_PATH = ROOT / "scene_cube3cm.xml"
OUTPUT_DIR = ROOT / "outputs" / "dual_camera_render"
WIDTH = 640
HEIGHT = 360
COLOR_CAMERA = "eye_in_hand_color"
DEPTH_CAMERA = "eye_in_hand_depth"


def camera_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if value < 0:
        raise RuntimeError(f"MuJoCo camera not found: {name}")
    return value


def camera_pose(data, cam_id):
    position = data.cam_xpos[cam_id].copy()
    rotation = data.cam_xmat[cam_id].reshape(3, 3).copy()
    return position, rotation


def depth_to_color_transform(depth_pose, color_pose):
    """Return the coordinate transform p_color = R p_depth + t."""
    p_depth, r_depth = depth_pose
    p_color, r_color = color_pose
    rotation = r_color.T @ r_depth
    translation = r_color.T @ (p_depth - p_color)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def save_depth_preview(depth, path):
    finite = np.isfinite(depth) & (depth > 0.0)
    values = depth[finite]
    low, high = np.percentile(values, [2.0, 98.0])
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    if high > low:
        normalized = np.clip((depth[finite] - low) / (high - low), 0.0, 1.0)
        scaled[finite] = np.asarray(255.0 * (1.0 - normalized), dtype=np.uint8)
    Image.fromarray(scaled).save(path)


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    color_id = camera_id(model, COLOR_CAMERA)
    depth_id = camera_id(model, DEPTH_CAMERA)
    color_pose = camera_pose(data, color_id)
    depth_pose = camera_pose(data, depth_id)

    renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
    try:
        renderer.update_scene(data, camera=color_id)
        color = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=depth_id)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    assert color.shape == (HEIGHT, WIDTH, 3), color.shape
    assert depth.shape == (HEIGHT, WIDTH), depth.shape
    assert np.isfinite(depth).all(), "depth contains NaN or Inf"
    assert np.all(depth > 0.0), "depth contains non-positive values"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(OUTPUT_DIR / "color.png")
    np.save(OUTPUT_DIR / "depth.npy", depth)
    save_depth_preview(depth, OUTPUT_DIR / "depth_preview.png")

    mount_body = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[color_id]
    )
    print(f"resolution: {WIDTH}x{HEIGHT}")
    print(f"parent body: {mount_body}")
    for name, pose in ((COLOR_CAMERA, color_pose), (DEPTH_CAMERA, depth_pose)):
        print(f"{name} world position:\n{pose[0]}")
        print(f"{name} world rotation:\n{pose[1]}")
    print("Depth->Color coordinate transform (p_color = T @ p_depth):")
    print(depth_to_color_transform(depth_pose, color_pose))
    print(
        f"depth: shape={depth.shape}, min={depth.min():.6f} m, "
        f"max={depth.max():.6f} m"
    )
    print(f"saved: {OUTPUT_DIR}")
    print("PASS: dual camera render")


if __name__ == "__main__":
    main()
