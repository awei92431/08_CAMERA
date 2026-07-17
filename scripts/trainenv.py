import os
import sys
import time
import json
import hashlib
import importlib.metadata
import platform
import shutil
from pathlib import Path
import gymnasium as gym
import argparse
import numpy as np



#.resolve()表示转化为绝对路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#手动把项目根目录加进 sys.path
if str(PROJECT_ROOT) not in sys.path:
    #把项目根目录插入到搜索路径列表的最前面。
    sys.path.insert(0,str(PROJECT_ROOT))


MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True , exist_ok=True)

#如果环境变量 MPLCONFIGDIR 还没有设置，就把它设置成 MPL_CACHE_DIR；如果已经设置了，就保持原样。
os.environ.setdefault("MPLCONFIGDIR",str(MPL_CACHE_DIR))

from stable_baselines3 import PPO
#导入评估回调函数。
#EvalCallback是指训练过程中定期拿当前模型去评估环境里跑几轮，看表现怎么样，并保存最好的模型。
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
#可以统一设置 Python、NumPy、PyTorch 等随机种子，让实验结果更容易复现。
from stable_baselines3.common.utils import FloatSchedule, set_random_seed

#创建并行环境
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

import fourc2


class ConciseEvalCallback(EvalCallback):
    def __init__(
        self,
        *args,
        retention_eval_env=None,
        retention_eval_episodes=0,
        min_primary_success=0.0,
        min_retention_success=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._final_eval_infos = []
        self.best_success_rate = -np.inf
        self.best_success_reward = -np.inf
        self.retention_eval_env = retention_eval_env
        self.retention_eval_episodes = int(retention_eval_episodes)
        self.min_primary_success = float(min_primary_success)
        self.min_retention_success = float(min_retention_success)
        self.best_handoff_score = None

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            self._final_eval_infos.append(locals_["info"].copy())

    @staticmethod
    def _mean_bool(infos, key):
        if not infos:
            return 0.0
        return float(np.mean([float(info.get(key, False)) for info in infos]))

    @staticmethod
    def _mean_float(infos, key):
        values = [float(info[key]) for info in infos if key in info]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _max_float(infos, key):
        values = [float(info[key]) for info in infos if key in info]
        return float(np.max(values)) if values else 0.0

    def _record_eval_details(self):
        infos = self._final_eval_infos
        if not infos:
            return {}

        metrics = {
            "stage_success_rate": self._mean_bool(infos, "stage_success"),
            "reach_success_rate": self._mean_bool(infos, "reach_success"),
            "grasp_success_rate": self._mean_bool(infos, "grasp_success"),
            "coarse_grasp_success_rate": self._mean_bool(infos, "coarse_grasp_success"),
            "strict_grasp_success_rate": self._mean_bool(infos, "strict_grasp_success"),
            "lift_success_rate": self._mean_bool(infos, "lift_success"),
            "place_success_rate": self._mean_bool(infos, "place_success"),
            "place_height_ok_rate": self._mean_bool(infos, "place_height_ok"),
            "place_opened_rate": self._mean_bool(infos, "place_opened"),
            "latched_rate": self._mean_bool(infos, "is_grasp_latched"),
            "bilateral_rate": self._mean_bool(infos, "has_bilateral_contact"),
            "raw_bilateral_rate": self._mean_bool(infos, "has_raw_bilateral_contact"),
            "object_lift_mean": self._mean_float(infos, "object_lift"),
            "object_lift_max": self._max_float(infos, "object_lift"),
            "lift_distance": self._mean_float(infos, "lift_distance"),
            "object_to_goal_xy_distance": self._mean_float(infos, "object_to_goal_xy_distance"),
            "object_to_goal_distance": self._mean_float(infos, "object_to_goal_distance"),
            "grasp_distance": self._mean_float(infos, "pinch_to_grasp_distance"),
            "grasp_xy_error": self._mean_float(infos, "grasp_xy_error"),
            "grasp_z_error": self._mean_float(infos, "grasp_z_error"),
            "pad_object_penetration": self._mean_float(infos, "pad_object_penetration"),
            "max_pad_object_penetration": self._mean_float(infos, "max_pad_object_penetration"),
            "table_contacts": self._mean_float(infos, "table_contact_count"),
            "tcp_target_error": self._mean_float(infos, "tcp_target_error"),
        }

        self.logger.record("eval_watch/success", metrics["stage_success_rate"])
        self.logger.record("eval_watch/reach", metrics["reach_success_rate"])
        self.logger.record("eval_watch/strict", metrics["strict_grasp_success_rate"])
        self.logger.record("eval_watch/raw_bi", metrics["raw_bilateral_rate"])
        self.logger.record("eval_watch/lift", metrics["lift_success_rate"])
        self.logger.record("eval_watch/lift_m", metrics["object_lift_mean"])
        self.logger.record("eval_watch/place", metrics["place_success_rate"])
        self.logger.record("eval_watch/place_opened", metrics["place_opened_rate"])
        self.logger.record("eval_watch/goal_xy", metrics["object_to_goal_xy_distance"])
        self.logger.record("eval_watch/pad_pen", metrics["max_pad_object_penetration"])
        self.logger.record("eval_watch/best_success", max(0.0, self.best_success_rate))
        self.logger.dump(self.num_timesteps)
        return metrics

    def _save_best_success_model(self, metrics):
        if not metrics or self.best_model_save_path is None:
            return False

        success_rate = metrics["stage_success_rate"]
        mean_reward = float(self.last_mean_reward)
        if success_rate < self.best_success_rate:
            return False
        if (
            success_rate == self.best_success_rate
            and mean_reward <= self.best_success_reward
        ):
            return False

        self.best_success_rate = success_rate
        self.best_success_reward = mean_reward
        path = Path(self.best_model_save_path) / "best_success_model"
        self.model.save(str(path))
        return True

    def _evaluate_retention(self):
        if self.retention_eval_env is None or self.retention_eval_episodes <= 0:
            return None
        final_infos = []

        def collect_final_info(locals_, globals_):
            if locals_.get("done", False):
                final_infos.append(locals_["info"].copy())

        episode_rewards, _ = evaluate_policy(
            self.model,
            self.retention_eval_env,
            n_eval_episodes=self.retention_eval_episodes,
            deterministic=True,
            render=False,
            callback=collect_final_info,
            return_episode_rewards=True,
            warn=False,
        )
        return {
            "stage_success_rate": self._mean_bool(final_infos, "stage_success"),
            "reach_success_rate": self._mean_bool(final_infos, "reach_success"),
            "grasp_success_rate": self._mean_bool(final_infos, "grasp_success"),
            "lift_success_rate": self._mean_bool(final_infos, "lift_success"),
            "place_success_rate": self._mean_bool(final_infos, "place_success"),
            "mean_reward": float(np.mean(episode_rewards)),
        }

    def _save_best_handoff_model(self, primary, retention):
        if retention is None or not primary or self.best_model_save_path is None:
            return False
        primary_success = primary["stage_success_rate"]
        retention_success = retention["stage_success_rate"]
        if (
            primary_success < self.min_primary_success
            or retention_success < self.min_retention_success
        ):
            return False
        # Select the model by its weakest side first.  A high new-stage score
        # cannot hide catastrophic forgetting of the already accepted chain.
        score = (
            min(primary_success, retention_success),
            retention_success,
            primary_success,
            float(self.last_mean_reward),
        )
        if self.best_handoff_score is not None and score <= self.best_handoff_score:
            return False
        self.best_handoff_score = score
        self.model.save(str(Path(self.best_model_save_path) / "best_handoff_model"))
        return True

    def _on_step(self):
        should_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        previous_best = self.best_mean_reward
        if should_eval:
            self._final_eval_infos = []
        continue_training = super()._on_step()

        if should_eval:
            metrics = self._record_eval_details()
            success_best = self._save_best_success_model(metrics)
            retention = self._evaluate_retention()
            handoff_best = self._save_best_handoff_model(metrics, retention)
            success_text = ""
            if len(self._is_success_buffer) > 0:
                success_text = f" succ={np.mean(self._is_success_buffer):.2f}"
            detail_text = ""
            if metrics:
                detail_text = (
                    f" strict={metrics['strict_grasp_success_rate']:.2f}"
                    f" raw_bi={metrics['raw_bilateral_rate']:.2f}"
                    f" lift={metrics['lift_success_rate']:.2f}"
                    f" place={metrics['place_success_rate']:.2f}"
                    f" lift_m={metrics['object_lift_mean']:.3f}"
                    f" goal_xy={metrics['object_to_goal_xy_distance']:.3f}"
                )

            best_text = " best" if self.best_mean_reward > previous_best else ""
            success_best_text = " best_success" if success_best else ""
            retention_text = ""
            if retention is not None:
                self.logger.record(
                    "retention_watch/success",
                    retention["stage_success_rate"],
                )
                self.logger.record(
                    "retention_watch/reach", retention["reach_success_rate"]
                )
                self.logger.record(
                    "retention_watch/grasp", retention["grasp_success_rate"]
                )
                self.logger.record(
                    "retention_watch/lift", retention["lift_success_rate"]
                )
                self.logger.record(
                    "retention_watch/place", retention["place_success_rate"]
                )
                self.logger.dump(self.num_timesteps)
                retention_text = f" retain={retention['stage_success_rate']:.2f}"
            handoff_text = " best_handoff" if handoff_best else ""
            print(
                f"[eval] step={self.num_timesteps:,} "
                f"rew={self.last_mean_reward:.1f} "
                f"best={self.best_mean_reward:.1f}"
                f"{success_text}{retention_text}{detail_text}{best_text}"
                f"{success_best_text}{handoff_text}",
                flush=True,
            )

        return continue_training


class EpisodeDiagnosticsCallback(BaseCallback):
    def __init__(self, print_interval=1, print_to_terminal=True):
        super().__init__()
        self.episode_infos = []
        self.print_interval = max(int(print_interval), 1)
        self.print_to_terminal = print_to_terminal
        self.rollout_count = 0
        self.total_episodes = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                self.episode_infos.append(info.copy())
        return True

    def _on_rollout_end(self):
        self.rollout_count += 1
        if not self.episode_infos:
            if (
                self.print_to_terminal
                and self.rollout_count % self.print_interval == 0
            ):
                print(
                    f"[train] step={self.num_timesteps:,} "
                    "waiting for completed episodes",
                    flush=True,
                )
            return

        def mean_bool(key):
            values = [float(info.get(key, False)) for info in self.episode_infos]
            return float(sum(values) / len(values))

        def mean_float(key):
            values = [
                float(info[key])
                for info in self.episode_infos
                if key in info and not isinstance(info[key], (list, tuple))
            ]
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        def max_float(key):
            values = [float(info[key]) for info in self.episode_infos if key in info]
            return float(max(values)) if values else 0.0

        def mean_episode_value(key):
            values = []
            for info in self.episode_infos:
                episode = info.get("episode")
                if isinstance(episode, dict) and key in episode:
                    values.append(float(episode[key]))
            if not values:
                return 0.0
            return float(sum(values) / len(values))

        episode_count = len(self.episode_infos)
        self.total_episodes += episode_count

        stage_success_rate = mean_bool("stage_success")
        reach_success_rate = mean_bool("reach_success")
        reach_centered_rate = mean_bool("reach_centered")
        reach_xy_centered_rate = mean_bool("reach_xy_centered")
        reach_z_centered_rate = mean_bool("reach_z_centered")
        reach_tcp_tracked_rate = mean_bool("reach_tcp_tracked")
        grasp_success_rate = mean_bool("grasp_success")
        coarse_grasp_success_rate = mean_bool("coarse_grasp_success")
        strict_grasp_success_rate = mean_bool("strict_grasp_success")
        lift_success_rate = mean_bool("lift_success")
        place_success_rate = mean_bool("place_success")
        place_height_ok_rate = mean_bool("place_height_ok")
        place_opened_rate = mean_bool("place_opened")
        latched_rate = mean_bool("is_grasp_latched")
        any_contact_rate = mean_bool("has_any_contact")
        bilateral_rate = mean_bool("has_bilateral_contact")
        unilateral_rate = mean_bool("has_unilateral_contact")
        raw_bilateral_rate = mean_bool("has_raw_bilateral_contact")
        final_stage = mean_float("active_stage")
        object_lift_mean = mean_float("object_lift")
        object_lift_max = max_float("object_lift")
        object_to_goal_xy_distance = mean_float("object_to_goal_xy_distance")
        object_to_goal_distance = mean_float("object_to_goal_distance")
        gripper_mean = mean_float("gripper_state")
        pregrasp_distance = mean_float("pinch_to_pregrasp_distance")
        pregrasp_xy_error = mean_float("pregrasp_xy_error")
        pregrasp_z_error = mean_float("pregrasp_z_error")
        grasp_distance = mean_float("pinch_to_grasp_distance")
        lift_distance = mean_float("lift_distance")
        task_distance = mean_float("distance")
        grasp_phase = mean_float("grasp_phase")
        xy_drift = mean_float("object_horizontal_drift")
        object_xy_speed = mean_float("object_xy_speed")
        table_contacts = mean_float("table_contact_count")
        table_clearance = mean_float("table_clearance_penalty")
        table_side = mean_float("table_side_penalty")
        low_away = mean_float("low_away_from_object_penalty")
        tcp_target_error = mean_float("tcp_target_error")
        stage_failure_rate = mean_bool("stage_failure")
        grasp_close_allowed_rate = mean_bool("grasp_close_allowed")
        grasp_xy_aligned_rate = mean_bool("grasp_xy_aligned")
        grasp_descent_allowed_rate = mean_bool("grasp_descent_allowed")
        grasp_xy_error = mean_float("grasp_xy_error")
        grasp_z_error = mean_float("grasp_z_error")
        pad_object_penetration = mean_float("pad_object_penetration")

        self.logger.record("watch/success", stage_success_rate)
        self.logger.record("watch/reach", reach_success_rate)
        self.logger.record("watch/strict", strict_grasp_success_rate)
        self.logger.record("watch/raw_bi", raw_bilateral_rate)
        self.logger.record("watch/lift", lift_success_rate)
        self.logger.record("watch/lift_m", object_lift_mean)
        self.logger.record("watch/place", place_success_rate)
        self.logger.record("watch/place_opened", place_opened_rate)
        self.logger.record("watch/goal_xy", object_to_goal_xy_distance)
        self.logger.record("watch/pad_pen", pad_object_penetration)

        if (
            self.print_to_terminal
            and self.rollout_count % self.print_interval == 0
        ):
            print(
                f"[train] step={self.num_timesteps:,} "
                f"eps={self.total_episodes}(+{episode_count}) "
                f"rew={mean_episode_value('r'):.1f} "
                f"len={mean_episode_value('l'):.0f} "
                f"succ={stage_success_rate:.2f} "
                f"detail R/G/L/P={reach_success_rate:.2f}/{grasp_success_rate:.2f}/{lift_success_rate:.2f}/{place_success_rate:.2f} "
                f"coarse/strict={coarse_grasp_success_rate:.2f}/{strict_grasp_success_rate:.2f} "
                f"stage={final_stage:.1f} "
                f"phase={grasp_phase:.1f} "
                f"dist={task_distance:.3f} "
                f"pre={pregrasp_distance:.3f} "
                f"pre_xy/z={pregrasp_xy_error:.3f}/{pregrasp_z_error:.3f} "
                f"r_ok={reach_xy_centered_rate:.2f}/{reach_z_centered_rate:.2f}/{reach_tcp_tracked_rate:.2f} "
                f"gd={grasp_distance:.3f} "
                f"lift={object_lift_mean:.3f}/{object_lift_max:.3f} "
                f"goal_xy={object_to_goal_xy_distance:.3f} "
                f"latch={latched_rate:.2f} "
                f"any={any_contact_rate:.2f} "
                f"bi={bilateral_rate:.2f} "
                f"raw_bi={raw_bilateral_rate:.2f} "
                f"pen={pad_object_penetration:.4f} "
                f"grip={gripper_mean:.2f} "
                f"drift={xy_drift:.3f} "
                f"obj_v={object_xy_speed:.3f} "
                f"tc={table_contacts:.1f} "
                f"clear={table_clearance:.3f} "
                f"side={table_side:.3f} "
                f"low={low_away:.3f} "
                f"tcp_err={tcp_target_error:.3f} "
                f"fail={stage_failure_rate:.2f} "
                f"close={grasp_close_allowed_rate:.2f} "
                f"align={grasp_xy_aligned_rate:.2f} "
                f"desc={grasp_descent_allowed_rate:.2f} "
                f"xy={grasp_xy_error:.3f} "
                f"z={grasp_z_error:.3f}",
                flush=True,
            )

        self.episode_infos.clear()

def make_env(env_id, seed):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

#并行训练时，SB3 需要的是“创建环境的函数”，不是直接创建好的 env。
def make_env_fn(env_id,seed,rank):
    def _init():
        return make_env(env_id, seed + rank)
    return _init


def resolve_project_path(path_arg):
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_model_path(path_arg):
    path = resolve_project_path(path_arg)
    candidates = [path]
    if path.suffix != ".zip":
        candidates.append(path.with_suffix(".zip"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"找不到模型文件，检查过: {checked}")


def sha256_file(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_code_hashes():
    files = [
        PROJECT_ROOT / "fourc2" / "__init__.py",
        PROJECT_ROOT / "fourc2" / "envs" / "allstage.py",
        PROJECT_ROOT / "scripts" / "trainenv.py",
        PROJECT_ROOT / "scripts" / "eval.py",
    ]
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in files
        if path.exists()
    }


def collect_reproducibility_files():
    files = set((PROJECT_ROOT / "fourc2").rglob("*.py"))
    files.update(
        {
            PROJECT_ROOT / "scripts" / "trainenv.py",
            PROJECT_ROOT / "scripts" / "eval.py",
            PROJECT_ROOT / "scene.xml",
            PROJECT_ROOT / "scene_cube3cm.xml",
            PROJECT_ROOT / "ur5e_4c2.xml",
        }
    )
    assets_dir = PROJECT_ROOT / "assets"
    if assets_dir.exists():
        files.update(path for path in assets_dir.rglob("*") if path.is_file())
    return sorted(path for path in files if path.is_file())


def collect_reproducibility_hashes():
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in collect_reproducibility_files()
    }


def collect_dependency_versions():
    packages = [
        "gymnasium",
        "mujoco",
        "numpy",
        "stable-baselines3",
        "torch",
    ]
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def snapshot_training_sources(run_dir):
    snapshot_root = run_dir / "source_snapshot"
    source_files = [
        PROJECT_ROOT / "fourc2" / "__init__.py",
        PROJECT_ROOT / "fourc2" / "envs" / "__init__.py",
        PROJECT_ROOT / "fourc2" / "envs" / "allstage.py",
        PROJECT_ROOT / "scripts" / "trainenv.py",
        PROJECT_ROOT / "scripts" / "eval.py",
        PROJECT_ROOT / "scene.xml",
        PROJECT_ROOT / "scene_cube3cm.xml",
        PROJECT_ROOT / "ur5e_4c2.xml",
    ]
    for source in source_files:
        if not source.exists():
            continue
        destination = snapshot_root / source.relative_to(PROJECT_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return snapshot_root


PLACE_HOLD_MIGRATION_ID = "v2-lift-to-v2.1-place-hold"
PLACE_HOLD_SOURCE_VERSION = "cube3cm-stageobs-v2"
PLACE_HOLD_SOURCE_CODE_SHA256 = {
    "fourc2/__init__.py": "321f6c49190159765bb5ea6f1722b559f406ed49ad4801e90464d587a32675ae",
    "fourc2/envs/allstage.py": "efb6cfd88665eae95983a18d2868493a0669bf6ad0e7babcc84067ff454bc7de",
    "scripts/trainenv.py": "972e2f7d0cb5a4dfe92b2a40da156b93ffef191eeb4173dcd5f3ceceb8c7c69d",
    "scripts/eval.py": "71f81efeb4bc8bc86fe854b0951d0ff8f828f09ddfbd3c6cdfb54230b8222a17",
}
PLACE_HOLD_CHANGED_FILES = {
    "fourc2/__init__.py",
    "fourc2/envs/allstage.py",
    "scripts/trainenv.py",
}

SINGLE_PPO_V22_MIGRATION_ID = "v2-reachgrasp-to-v2.2-lift-single"
SINGLE_PPO_V22_SOURCE_MODEL_SHA256 = (
    "1bd97128c7f132c1ea4dbdf962964746a938f8a165414cfcd0ab8ff6bdf7ee6d"
)
PRECISION_XY6_MIGRATION_ID = "v2.2-lift-upmap-to-xy6mm-grasp-lift"
PRECISION_XY6_SOURCE_MODEL_SHA256 = (
    "cc96d12a9f1c13e70fcd1fcd950d5eaf70db2a56fe2f1b06485cf4abb4dd8815"
)


def validate_precision_xy6_migration(
    source_config,
    model_path,
    target_env_id,
    current_dependencies,
    actual_model_hash,
):
    if fourc2.PIPELINE_VERSION != "cube3cm-single-ppo-v2.2-xy6mm":
        return None
    if source_config.get("pipeline_version") != "cube3cm-single-ppo-v2.2-lift-upmap":
        return None
    if source_config.get("env_id") != "My4C2LiftSingleV22Cube3cm-v0":
        return None
    if target_env_id != "My4C2GraspLiftTransitionV22Cube3cm-v0":
        return None
    if model_path.name != "best_handoff_model.zip":
        return None
    if actual_model_hash != PRECISION_XY6_SOURCE_MODEL_SHA256:
        return None
    if source_config.get("output_model_sha256", {}).get(model_path.name) != actual_model_hash:
        return None
    if source_config.get("dependency_versions") != current_dependencies:
        return None
    return {
        "id": PRECISION_XY6_MIGRATION_ID,
        "source_version": source_config.get("pipeline_version"),
        "target_version": fourc2.PIPELINE_VERSION,
        "source_env_id": source_config.get("env_id"),
        "target_env_id": target_env_id,
        "model_name": model_path.name,
        "model_sha256": actual_model_hash,
        "reason": (
            "Explicit migration of the accepted 39-D Lift handoff. Only the "
            "Grasp stable/strict XY acceptance is tightened from 15/12 mm to "
            "6/4 mm; observation, network and Lift behavior are unchanged."
        ),
    }


def validate_single_ppo_v22_migration(
    source_config,
    model_path,
    target_env_id,
    current_dependencies,
    actual_model_hash,
):
    if fourc2.PIPELINE_VERSION != "cube3cm-single-ppo-v2.2-xy6mm":
        return None
    if source_config.get("pipeline_version") != "cube3cm-stageobs-v2":
        return None
    if source_config.get("env_id") != "My4C2ReachGraspStageCube3cm-v0":
        return None
    if target_env_id != "My4C2LiftSingleV22Cube3cm-v0":
        return None
    if model_path.name != "best_success_model.zip":
        return None
    if actual_model_hash != SINGLE_PPO_V22_SOURCE_MODEL_SHA256:
        return None
    if (
        source_config.get("output_model_sha256", {}).get(model_path.name)
        != actual_model_hash
    ):
        return None
    if source_config.get("dependency_versions") != current_dependencies:
        return None
    return {
        "id": SINGLE_PPO_V22_MIGRATION_ID,
        "source_version": source_config.get("pipeline_version"),
        "target_version": fourc2.PIPELINE_VERSION,
        "source_env_id": source_config.get("env_id"),
        "target_env_id": target_env_id,
        "model_name": model_path.name,
        "model_sha256": actual_model_hash,
        "reason": (
            "Explicit one-time migration of the accepted 39-D ReachGrasp "
            "checkpoint. v2.2 changes Place control and adds opt-in sequential "
            "transition environments; Reach and Grasp execution is unchanged."
        ),
    }


def validate_place_hold_compatible_migration(
    source_config,
    model_path,
    target_env_id,
    current_hashes,
    current_dependencies,
    actual_model_hash,
):
    if fourc2.PIPELINE_VERSION != "cube3cm-stageobs-v2.1-place-hold":
        return None
    if source_config.get("pipeline_version") != PLACE_HOLD_SOURCE_VERSION:
        return None
    if source_config.get("env_id") != "My4C2GraspLiftStageCube3cm-v0":
        return None
    if target_env_id != "My4C2CurriculumCube3cm-v0":
        return None
    if model_path.name != "final_model.zip":
        return None
    if source_config.get("args", {}).get("allow_legacy_model", True):
        return None
    if source_config.get("load_model_provenance", {}).get("status") != "matched":
        return None
    if source_config.get("code_sha256") != PLACE_HOLD_SOURCE_CODE_SHA256:
        return None
    if source_config.get("dependency_versions") != current_dependencies:
        return None
    if (
        source_config.get("output_model_sha256", {}).get(model_path.name)
        != actual_model_hash
    ):
        return None

    source_hashes = source_config.get("reproducibility_sha256", {})
    if set(source_hashes) != set(current_hashes):
        return None
    for path, source_hash in source_hashes.items():
        if path in PLACE_HOLD_CHANGED_FILES:
            continue
        if current_hashes[path] != source_hash:
            return None

    return {
        "id": PLACE_HOLD_MIGRATION_ID,
        "source_version": PLACE_HOLD_SOURCE_VERSION,
        "target_version": fourc2.PIPELINE_VERSION,
        "source_env_id": source_config.get("env_id"),
        "target_env_id": target_env_id,
        "model_name": model_path.name,
        "model_sha256": actual_model_hash,
        "changed_files": sorted(PLACE_HOLD_CHANGED_FILES),
        "reason": (
            "Place-only controller change: hold TCP stationary while the "
            "rate-limited gripper opens. Reach/Grasp/Lift branches are unchanged."
        ),
    }


def validate_load_model_provenance(
    model_path,
    allow_legacy_model=False,
    target_env_id=None,
):
    run_config_path = model_path.parent.parent / "run_config.json"
    if not run_config_path.exists():
        if allow_legacy_model:
            return {
                "status": "legacy_override",
                "run_config": None,
                "pipeline_version": None,
            }
        raise RuntimeError(
            f"Input model has no run_config.json: {model_path}. "
            "Strict pipeline handoff refused. Use --allow-legacy-model only "
            "for an explicitly non-reproducible diagnostic run."
        )

    with run_config_path.open("r", encoding="utf-8") as file_obj:
        source_config = json.load(file_obj)
    source_version = source_config.get("pipeline_version")
    current_hashes = collect_reproducibility_hashes()
    source_hashes = source_config.get("reproducibility_sha256")
    current_dependencies = collect_dependency_versions()
    source_dependencies = source_config.get("dependency_versions")
    output_hashes = source_config.get("output_model_sha256", {})
    expected_model_hash = output_hashes.get(model_path.name)
    actual_model_hash = sha256_file(model_path)
    mismatch_reasons = []
    if source_version != fourc2.PIPELINE_VERSION:
        mismatch_reasons.append(
            f"pipeline_version {source_version!r} != {fourc2.PIPELINE_VERSION!r}"
        )
    if source_hashes != current_hashes:
        mismatch_reasons.append("code/XML/mesh hashes differ")
    if source_dependencies != current_dependencies:
        mismatch_reasons.append("dependency versions differ")
    if expected_model_hash is None:
        mismatch_reasons.append("input ZIP is not recorded by the source run")
    elif expected_model_hash != actual_model_hash:
        mismatch_reasons.append("input ZIP SHA256 differs from the source run")

    compatible_migration = None
    if mismatch_reasons:
        compatible_migration = validate_precision_xy6_migration(
            source_config=source_config,
            model_path=model_path,
            target_env_id=target_env_id,
            current_dependencies=current_dependencies,
            actual_model_hash=actual_model_hash,
        )
        if compatible_migration is None:
            compatible_migration = validate_single_ppo_v22_migration(
            source_config=source_config,
            model_path=model_path,
            target_env_id=target_env_id,
            current_dependencies=current_dependencies,
            actual_model_hash=actual_model_hash,
            )
        if compatible_migration is None:
            compatible_migration = validate_place_hold_compatible_migration(
            source_config=source_config,
            model_path=model_path,
            target_env_id=target_env_id,
            current_hashes=current_hashes,
            current_dependencies=current_dependencies,
            actual_model_hash=actual_model_hash,
            )
        if compatible_migration is not None:
            status = "compatible_migration"
        elif allow_legacy_model:
            status = "legacy_override"
        else:
            raise RuntimeError(
                "Strict pipeline provenance mismatch for input model "
                f"{model_path}: " + "; ".join(mismatch_reasons) + ". "
                "Rebuild the previous stage with the current frozen version."
            )
    else:
        status = "matched"
    return {
        "status": status,
        "run_config": str(run_config_path),
        "pipeline_version": source_version,
        "mismatch_reasons": mismatch_reasons,
        "compatible_migration": compatible_migration,
    }


def record_output_model_hashes(run_config_path, model_dir):
    with run_config_path.open("r", encoding="utf-8") as file_obj:
        run_config = json.load(file_obj)
    run_config["output_model_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(model_dir.glob("*.zip"))
    }
    with run_config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(run_config, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def build_new_ppo(env, args, tensorboard_dir):
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        verbose=args.verbose,
        tensorboard_log=str(tensorboard_dir),
    )


def load_ppo_with_optional_observation_expansion(
    model_path,
    env,
    args,
    tensorboard_dir,
):
    old_model = PPO.load(str(model_path))
    old_shape = old_model.observation_space.shape
    new_shape = env.observation_space.shape

    if old_shape == new_shape:
        lr_schedule = FloatSchedule(args.learning_rate)
        model = PPO.load(
            str(model_path),
            env=env,
            seed=args.seed,
            verbose=args.verbose,
            tensorboard_log=str(tensorboard_dir),
            custom_objects={
                "learning_rate": args.learning_rate,
                "lr_schedule": lr_schedule,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "gamma": args.gamma,
            },
        )
        return model, False

    if old_shape != (39,) or new_shape != (43,):
        raise ValueError(
            "Unsupported observation-space migration: "
            f"model={old_shape}, env={new_shape}"
        )

    # Full/curriculum environments append a four-value stage one-hot.  Create
    # a fresh optimizer and copy every learned parameter.  The four new input
    # columns are initialized to zero so the expanded policy initially
    # produces exactly the same actions and values as the 39-D source model.
    model = build_new_ppo(env, args, tensorboard_dir)
    old_state = old_model.policy.state_dict()
    new_state = model.policy.state_dict()
    expanded_keys = []
    for key, old_tensor in old_state.items():
        if key not in new_state:
            raise KeyError(f"Missing policy parameter during migration: {key}")
        new_tensor = new_state[key]
        if old_tensor.shape == new_tensor.shape:
            new_state[key] = old_tensor.clone()
            continue
        if (
            old_tensor.ndim == 2
            and new_tensor.ndim == 2
            and old_tensor.shape[0] == new_tensor.shape[0]
            and old_tensor.shape[1] == 39
            and new_tensor.shape[1] == 43
        ):
            expanded = new_tensor.clone()
            expanded.zero_()
            expanded[:, :39] = old_tensor
            new_state[key] = expanded
            expanded_keys.append(key)
            continue
        raise ValueError(
            f"Unsupported policy parameter migration for {key}: "
            f"{tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)}"
        )

    model.policy.load_state_dict(new_state, strict=True)
    if not expanded_keys:
        raise RuntimeError("Observation expanded but no input layers were migrated")
    print(
        "expanded observation model: 39 -> 43; zero-initialized stage inputs in "
        + ", ".join(expanded_keys),
        flush=True,
    )
    return model, True



def main():
    parser = argparse.ArgumentParser(description="用PPO训练4C2夹爪抓取")
    parser.add_argument("--env-id", default="My4C2AllStageCube3cm-v0")
    parser.add_argument("--eval-env-id", default=None)
    parser.add_argument("--retention-eval-env-id", default=None)
    parser.add_argument("--retention-eval-episodes", type=int, default=None)
    parser.add_argument("--min-primary-success", type=float, default=0.80)
    parser.add_argument("--min-retention-success", type=float, default=0.80)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--rehearsal-env-id", default=None)
    parser.add_argument("--rehearsal-envs", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--terminal-log-interval", type=int, default=1)
    parser.add_argument("--no-terminal-log", action="store_true")
    parser.add_argument("--load-model", default=None)
    parser.add_argument("--allow-legacy-model", action="store_true")
    args = parser.parse_args()

    set_random_seed(args.seed)
    if args.rehearsal_envs < 0 or args.rehearsal_envs >= args.n_envs:
        raise ValueError("--rehearsal-envs must be in [0, n_envs)")
    if bool(args.rehearsal_env_id) != bool(args.rehearsal_envs):
        raise ValueError(
            "--rehearsal-env-id and a positive --rehearsal-envs must be used together"
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.env_id}_ppo_{timestamp}"
    run_dir = PROJECT_ROOT / "runs" / run_name
    model_dir = run_dir / "models"
    log_dir = run_dir / "logs"
    tensorboard_dir = run_dir / "tensorboard"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. "
            "Use a new --run-name; strict runs are never overwritten or appended."
        )
    resolved_load_model = (
        resolve_model_path(args.load_model) if args.load_model else None
    )
    load_model_provenance = (
        validate_load_model_provenance(
            resolved_load_model,
            allow_legacy_model=args.allow_legacy_model,
            target_env_id=args.env_id,
        )
        if resolved_load_model
        else None
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot_dir = snapshot_training_sources(run_dir)

    primary_env_count = args.n_envs - args.rehearsal_envs
    training_env_ids = (
        [args.env_id] * primary_env_count
        + [args.rehearsal_env_id] * args.rehearsal_envs
    )
    training_env_fns = [
        make_env_fn(env_id, args.seed, rank)
        for rank, env_id in enumerate(training_env_ids)
    ]
    if args.n_envs == 1:
        env = DummyVecEnv(training_env_fns)
        vec_env_type = "DummyVecEnv"
    else:
        env = SubprocVecEnv(training_env_fns)
        vec_env_type = "SubprocVecEnv"
    #这里 VecMonitor 类似 Monitor，只是给并行环境用的。
    env = VecMonitor(env)


    eval_env_id = args.eval_env_id or args.env_id
    eval_env = VecMonitor(
        DummyVecEnv([make_env_fn(eval_env_id, args.seed + 10_000, 0)])
    )
    retention_eval_env = None
    if args.retention_eval_env_id:
        retention_eval_env = VecMonitor(
            DummyVecEnv(
                [make_env_fn(args.retention_eval_env_id, args.seed + 20_000, 0)]
            )
        )


    print("available env ids:", fourc2.ENV_IDS)
    print("env_id:", args.env_id)
    print("eval_env_id:", eval_env_id)
    print("retention_eval_env_id:", args.retention_eval_env_id or "off")
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)
    print("run_dir:", run_dir)
    print("tensorboard_log:", tensorboard_dir)
    print("total_timesteps:", args.total_timesteps)
    print("learning_rate:", args.learning_rate)
    print("vec_env:", vec_env_type)
    print("training_env_mix:", {
        args.env_id: primary_env_count,
        **(
            {args.rehearsal_env_id: args.rehearsal_envs}
            if args.rehearsal_env_id
            else {}
        ),
    })
    if args.no_terminal_log:
        print("terminal_log: off")
    else:
        print("terminal_log: concise line every", args.terminal_log_interval, "rollout(s)")
        print("columns: succ=selected stage success, detail R/G/L/P=reach/grasp/lift/place probes, goal_xy=object-goal xy distance")
        print("tensorboard watch: watch/success, watch/reach, watch/strict, watch/raw_bi, watch/lift, watch/place, watch/place_opened, watch/goal_xy, watch/pad_pen")
    if eval_env_id == args.env_id:
        print("eval_env: same env_id as training; staged env evaluates the selected stage")
    else:
        print("eval_env: separate env_id; best model is selected by eval env performance")
    print("eval tensorboard watch: eval_watch/success, eval_watch/reach, eval_watch/strict, eval_watch/raw_bi, eval_watch/lift, eval_watch/place, eval_watch/place_opened, eval_watch/goal_xy")
    if resolved_load_model:
        print("load_model:", resolved_load_model)
        print("load_model_provenance_status:", load_model_provenance["status"])
        if load_model_provenance.get("compatible_migration"):
            print(
                "compatible_migration_id:",
                load_model_provenance["compatible_migration"]["id"],
            )

    run_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "log_dir": str(log_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "env_id": args.env_id,
        "eval_env_id": eval_env_id,
        "pipeline_version": fourc2.PIPELINE_VERSION,
        "args": vars(args),
        "load_model_resolved": str(resolved_load_model)
        if resolved_load_model
        else None,
        "load_model_sha256": sha256_file(resolved_load_model)
        if resolved_load_model
        else None,
        "load_model_provenance": load_model_provenance,
        "dependency_versions": collect_dependency_versions(),
        "code_sha256": collect_code_hashes(),
        "reproducibility_sha256": collect_reproducibility_hashes(),
        "source_snapshot": str(source_snapshot_dir),
    }
    run_config_path = run_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as file_obj:
        json.dump(run_config, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    print("run_config:", run_config_path)

    eval_callback = ConciseEvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path= str(log_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes= args.eval_episodes,
        deterministic=True,
        render = False,
        verbose=args.verbose,
        retention_eval_env=retention_eval_env,
        retention_eval_episodes=(
            args.retention_eval_episodes
            if args.retention_eval_episodes is not None
            else args.eval_episodes
        ),
        min_primary_success=args.min_primary_success,
        min_retention_success=args.min_retention_success,

    )
    diagnostics_callback = EpisodeDiagnosticsCallback(
        print_interval=args.terminal_log_interval,
        print_to_terminal=not args.no_terminal_log,
    )
    callbacks = CallbackList([eval_callback, diagnostics_callback])

    expanded_observation_model = False
    if args.load_model:
        model, expanded_observation_model = load_ppo_with_optional_observation_expansion(
            resolved_load_model,
            env,
            args,
            tensorboard_dir,
        )
    else:
        model = build_new_ppo(env, args, tensorboard_dir)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=args.log_interval,
        progress_bar=False,
        tb_log_name="ppo",
        reset_num_timesteps=args.load_model is None or expanded_observation_model,
    )

    final_model_path = model_dir / "final_model"
    model.save(final_model_path)
    print("saved final model:", final_model_path.with_suffix(".zip"))
    record_output_model_hashes(run_config_path, model_dir)
    print("recorded output model hashes:", run_config_path)

    env.close()
    eval_env.close()

if __name__ =="__main__":
    main()
