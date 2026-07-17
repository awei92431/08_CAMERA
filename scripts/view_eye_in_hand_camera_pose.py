"""Open a paused third-person view of the fixed eye-in-hand cameras."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "scene.xml"  # includes the current ur5e_4c2.xml
OUTPUT = ROOT / "outputs" / "camera_pose_view" / "color.png"
MOUNT_NAME = "d435i_mount_frame"
COLOR_NAME = "eye_in_hand_color"
DEPTH_NAME = "eye_in_hand_depth"


def named_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return object_id


def print_camera_pose(model, data, name):
    camera_id = named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    position = data.cam_xpos[camera_id].copy()
    rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
    forward = -rotation[:, 2]  # MuJoCo cameras look along local -Z.
    print(f"{name} world position: {position}")
    print(f"{name} world rotation matrix (camera -> world):\n{rotation}")
    print(f"{name} world forward axis (-Z): {forward}")
    return camera_id


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    home_id = named_id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    mount_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, MOUNT_NAME)
    print(f"model: {SCENE_XML} (includes ur5e_4c2.xml)")
    print(f"{MOUNT_NAME} local pos: {model.body_pos[mount_id].copy()}")
    print(f"{MOUNT_NAME} local quat [w x y z]: "
          f"{model.body_quat[mount_id].copy()}")
    color_id = print_camera_pose(model, data, COLOR_NAME)
    print_camera_pose(model, data, DEPTH_NAME)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, width=640, height=360)
    try:
        renderer.update_scene(data, camera=color_id)
        Image.fromarray(renderer.render()).save(OUTPUT)
    finally:
        renderer.close()
    print(f"saved Color frame: {OUTPUT}")

    # launch_passive does not advance physics.  With no mj_step call below,
    # qpos, qvel and every body/camera pose remain exactly at the home state.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_CAMERA

        mount_world = data.xpos[mount_id]
        object_id = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        object_world = data.xpos[object_id]
        viewer.cam.lookat[:] = 0.5 * (mount_world + object_world)
        viewer.cam.distance = 1.05
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        viewer.sync()
        print("Viewer ready: camera frustums and camera frames enabled; "
              "simulation is paused. Close the Viewer window to exit.")
        try:
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 30.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
