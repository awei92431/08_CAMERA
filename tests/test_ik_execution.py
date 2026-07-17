import numpy as np
import inspect
import ast
import mujoco
import fourc2  # noqa: F401
import gymnasium as gym
from stable_baselines3 import PPO
from pathlib import Path

ENV_ID = "My4C2AllStageSinglePPOV22Cube3cm-v0"

def make_env():
    env = gym.make(ENV_ID).unwrapped
    env.reset(seed=123)
    return env

def test_weld_absent_and_action_shape():
    env=make_env()
    assert env.action_space.shape == (4,)
    assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "mocap_tcp_weld") == -1
    assert not env.tcp_weld_enabled
    assert env.max_tcp_lead == 0.03
    assert env.ik_axis_weight == 0.35
    assert env.ik_posture_mode == "off"
    assert env.arm_kp_scale == 2.0
    env.close()

def test_default_execution_uses_ik_without_raw_posture_or_arm_qpos_write():
    env=make_env()
    calls=env.ik_solve_calls
    env.step(np.array([0.2, -0.1, 0.3, 0.0], dtype=np.float32))
    assert env.ik_solve_calls == calls + 1
    assert env.diag_ik["posture_mode"] == "off"
    assert np.array_equal(
        env.diag_ik["posture_projected_increment"], np.zeros(6)
    )
    # Execution must command position actuators; only the separate ik_data
    # scratch state may receive candidate qpos assignments inside _solve_ik.
    source=inspect.getsource(type(env)._apply_tcp_action)
    tree=ast.parse(source.lstrip())
    assigned=[]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets=node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned.extend(ast.unparse(target) for target in targets)
    assert not any("self.data.qpos" in target for target in assigned)
    assert "self.data.ctrl[self.arm_actuator_ids]" in source
    env.close()

def test_all_posture_modes_remain_selectable():
    for mode in ("raw", "off", "nullspace"):
        env=gym.make(ENV_ID, ik_posture_mode=mode).unwrapped
        assert env.ik_posture_mode == mode
        env.close()

def test_arm_position_servo_is_fixed_at_validated_two_x_gain():
    env=make_env()
    assert np.array_equal(
        env.model.actuator_gainprm[env.arm_actuator_ids, 0],
        np.array([2000., 2000., 2000., 500., 500., 500.]),
    )
    assert np.array_equal(
        env.model.actuator_biasprm[env.arm_actuator_ids, 1],
        np.array([-2000., -2000., -2000., -500., -500., -500.]),
    )
    env.close()

def test_zero_and_cartesian_directions_use_ik():
    directions=np.vstack([np.zeros(3), np.eye(3), -np.eye(3)])
    for direction in directions:
        env=make_env()
        # Reach safety correctly rejects downward targets below pregrasp. Move
        # up first so the -Z command has legal workspace in which to act.
        if direction[2] < 0:
            for _ in range(5): env.step(np.array([0,0,1,0], np.float32))
            for _ in range(30): env.step(np.zeros(4, np.float32))
        before=env.data.site_xpos[env.pinch_site_id].copy(); q0=env.data.qpos[env.arm_qpos_ids].copy()
        env.step(np.r_[direction, 0.0].astype(np.float32)); after=env.data.site_xpos[env.pinch_site_id].copy()
        assert env.ik_solve_calls >= 1
        if np.allclose(direction, 0): assert np.linalg.norm(after-before) < 0.003
        else: assert np.dot(after-before, direction) > 0
        assert np.all(env.last_ik_joint_target >= env.arm_ctrl_low-1e-9)
        assert np.all(env.last_ik_joint_target <= env.arm_ctrl_high+1e-9)
        assert not np.array_equal(env.data.qpos[env.arm_qpos_ids], q0) or np.allclose(direction,0)
        env.close()

def test_old_checkpoint_loads():
    checkpoint=Path(__file__).resolve().parents[1]/"checkpoints/best_full_flow_v22.zip"
    model=PPO.load(checkpoint, device="cpu")
    env=make_env()
    obs,_=env.reset(seed=0)
    action,_=model.predict(obs, deterministic=True)
    assert action.shape == (4,)
    env.step(action)
    env.close()
