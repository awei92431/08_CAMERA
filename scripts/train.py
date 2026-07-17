import os
import sys
import time
from pathlib import Path
import gymnasium as gym
import argparse

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
from stable_baselines3.common.callbacks import EvalCallback
#Monitor 的作用是记录 episode 的奖励、长度、成功率等信息
from stable_baselines3.common.monitor import Monitor
#可以统一设置 Python、NumPy、PyTorch 等随机种子，让实验结果更容易复现。
from stable_baselines3.common.utils import set_random_seed

import fourc2

def make_env(env_id,seed):
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

def main():
    parser = argparse.ArgumentParser(description="用PPO训练4C2夹爪抓取")
    parser.add_argument("--env-id", default="My4C2AllStageCube3cm-v0")
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
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()

    set_random_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.env_id}_ppo_{timestamp}"
    run_dir = PROJECT_ROOT / "runs" / run_name
    model_dir = run_dir / "models"
    log_dir = run_dir / "logs"
    tensorboard_dir = run_dir / "tensorboard"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_id, args.seed)
    eval_env = make_env(args.env_id, args.seed + 10_000)

    print("available env ids:", fourc2.ENV_IDS)
    print("env_id:", args.env_id)
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)
    print("run_dir:", run_dir)
    print("tensorboard_log:", tensorboard_dir)
    print("total_timesteps:", args.total_timesteps)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path= str(log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes= args.eval_episodes,
        deterministic=True,
        render = False,

    )

    model = PPO(
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

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        log_interval=args.log_interval,
        progress_bar=False,
        tb_log_name="ppo",
    )

    final_model_path = model_dir / "final_model"
    model.save(final_model_path)
    print("saved final model:", final_model_path.with_suffix(".zip"))

    env.close()
    eval_env.close()

if __name__ =="__main__":
    main()
