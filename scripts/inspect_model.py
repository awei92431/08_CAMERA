from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "scene.xml"


def names(model, obj_type, count):
    result = []
    for obj_id in range(count):
        name = mujoco.mj_id2name(model, obj_type, obj_id)
        result.append(name)
    return result


def print_joint_table(model):
    print("\n=== joints ===")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_id = model.jnt_qposadr[joint_id]
        qvel_id = model.jnt_dofadr[joint_id]
        joint_type = model.jnt_type[joint_id]
        joint_range = model.jnt_range[joint_id]
        print(
            f"{joint_id:02d} name={name} "
            f"type={joint_type} qpos_id={qpos_id} qvel_id={qvel_id} "
            f"range={np.round(joint_range, 4)}"
        )


def print_actuator_table(model):
    print("\n=== actuators ===")
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        ctrl_range = model.actuator_ctrlrange[actuator_id]
        print(
            f"{actuator_id:02d} name={name} "
            f"ctrlrange={np.round(ctrl_range, 4)}"
        )


def print_site_table(model, data):
    print("\n=== sites ===")
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        xpos = data.site_xpos[site_id]
        print(f"{site_id:02d} name={name} xpos={np.round(xpos, 4)}")


def print_keyframe_table(model):
    print("\n=== keyframes ===")
    if model.nkey == 0:
        print("No keyframes.")
        return

    for key_id in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
        qpos = model.key_qpos[key_id]
        ctrl = model.key_ctrl[key_id]
        print(f"{key_id:02d} name={name}")
        print(f"    qpos={np.round(qpos, 4)}")
        print(f"    ctrl={np.round(ctrl, 4)}")


def main():
    print(f"model_path: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print("\n=== summary ===")
    print(f"nq={model.nq}")
    print(f"nv={model.nv}")
    print(f"nu={model.nu}")
    print(f"nbody={model.nbody}")
    print(f"ngeom={model.ngeom}")
    print(f"nsite={model.nsite}")
    print(f"njnt={model.njnt}")
    print(f"nkey={model.nkey}")
    print(f"neq={model.neq}")
    print(f"ntendon={model.ntendon}")

    print_joint_table(model)
    print_actuator_table(model)
    print_keyframe_table(model)

    if model.nkey > 0:
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_id)
            mujoco.mj_forward(model, data)
            print("\nreset to keyframe: home")
        else:
            mujoco.mj_forward(model, data)
    else:
        mujoco.mj_forward(model, data)

    print_site_table(model, data)

    print("\n=== important ids ===")
    for site_name in ["attachment_site", "pinch"]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        print(f"site {site_name}: id={site_id}")

    for actuator_name in [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "fingers_actuator",
    ]:
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        print(f"actuator {actuator_name}: id={actuator_id}")

    print("\n=== short simulation check ===")
    data.ctrl[:] = model.key_ctrl[0] if model.nkey > 0 else 0.0
    for _ in range(10):
        mujoco.mj_step(model, data)
    print(f"qpos={np.round(data.qpos, 4)}")
    print(f"qvel={np.round(data.qvel, 4)}")
    print(f"ctrl={np.round(data.ctrl, 4)}")


if __name__ == "__main__":
    main()
