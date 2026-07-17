import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_xml_path(xml_path):
    path = Path(xml_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def reset_to_keyframe(model, data, keyframe_name):
    if not keyframe_name:
        mujoco.mj_forward(model, data)
        return

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id < 0:
        raise ValueError(f"Keyframe '{keyframe_name}' was not found in the model.")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def render_camera(model, data, camera_id, width, height, render_depth):
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera_id)
        rgb = renderer.render()

        depth = None
        if render_depth:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=camera_id)
            depth = renderer.render()
            renderer.disable_depth_rendering()
    finally:
        close = getattr(renderer, "close", None)
        if close is not None:
            close()

    return rgb, depth


def describe_rgb(rgb):
    return (
        f"rgb: shape={rgb.shape} dtype={rgb.dtype} "
        f"min={np.min(rgb)} max={np.max(rgb)}"
    )


def describe_depth(depth):
    finite = np.isfinite(depth)
    if not np.any(finite):
        return f"depth: shape={depth.shape} dtype={depth.dtype} finite=0"

    finite_depth = depth[finite]
    return (
        f"depth: shape={depth.shape} dtype={depth.dtype} "
        f"finite={int(np.sum(finite))}/{depth.size} "
        f"min={float(np.min(finite_depth)):.6f} "
        f"max={float(np.max(finite_depth)):.6f}"
    )


def save_arrays(save_dir, rgb, depth):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    np.save(save_path / "rgb.npy", rgb)
    if depth is not None:
        np.save(save_path / "depth.npy", depth)
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one RGB/depth frame from the 4C2 eye-in-hand MuJoCo camera."
    )
    parser.add_argument("--xml", default="scene_cube3cm.xml", help="XML path relative to the project root.")
    parser.add_argument("--camera-name", default="eye_in_hand_color", help="MuJoCo camera name.")
    parser.add_argument("--keyframe", default="home", help="Keyframe used before rendering. Empty string disables it.")
    parser.add_argument("--width", type=int, default=640, help="Rendered image width.")
    parser.add_argument("--height", type=int, default=480, help="Rendered image height.")
    parser.add_argument("--no-depth", action="store_true", help="Only render RGB.")
    parser.add_argument("--save-dir", default=None, help="Optional directory for rgb.npy and depth.npy.")
    return parser.parse_args()


def main():
    args = parse_args()
    xml_path = resolve_xml_path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file does not exist: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    reset_to_keyframe(model, data, args.keyframe)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    if camera_id < 0:
        names = []
        for cam_id in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            names.append(name or f"camera#{cam_id}")
        raise ValueError(f"Camera '{args.camera_name}' was not found. Available cameras: {names}")

    rgb, depth = render_camera(
        model,
        data,
        camera_id=camera_id,
        width=args.width,
        height=args.height,
        render_depth=not args.no_depth,
    )

    print(f"xml: {xml_path}")
    print(f"camera: name={args.camera_name} id={camera_id} fovy={float(model.cam_fovy[camera_id]):.3f}")
    print(describe_rgb(rgb))
    if depth is not None:
        print(describe_depth(depth))

    if args.save_dir:
        save_path = save_arrays(args.save_dir, rgb, depth)
        print(f"saved: {save_path}")


if __name__ == "__main__":
    main()
