"""
train_dqn_improved.py - Train and validate a stronger DQN policy for RouteMATE.

This script trains DQN on the improved ride-sharing environment, evaluates it
against a greedy baseline on the same seeds, and promotes the model only if it
actually performs better.

Usage:
  python src/agents/train_dqn_improved.py
  python src/agents/train_dqn_improved.py --timesteps 300000 --episodes 30
"""

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add project root for local imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.gym_environment import RideSharingEnvImproved


def create_env(request_rate: float, max_steps: int, seed: int | None = None):
    reward_config = {
        "step_penalty": -0.005,
        "pickup_reward": 4.0,
        "dropoff_reward": 10.0,
        "failed_assignment": -2.5,
        "distance_bonus_scale": 0.35,
        "capacity_bonus_scale": 0.8,
        "utilization_bonus_scale": 2.5,
    }

    env = RideSharingEnvImproved(
        city_size=10,
        num_vehicles=5,
        vehicle_capacity=4,
        request_rate=request_rate,
        max_steps=max_steps,
        reward_config=reward_config,
    )
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def greedy_action_from_obs(observation: np.ndarray, num_vehicles: int) -> int:
    # Observation layout: 4 request fields + 8 fields per vehicle; index 4 in each block is distance.
    distances = [observation[4 + i * 8 + 4] for i in range(num_vehicles)]
    return int(np.argmin(distances))


def evaluate_policy(model: DQN, episodes: int, request_rate: float, max_steps: int) -> Dict[str, float]:
    rewards = []
    completions = []

    env = create_env(request_rate=request_rate, max_steps=max_steps)

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward

        metrics = info["metrics"]
        total_requests = metrics.get("total_requests", 0)
        completed = metrics.get("completed_requests", 0)
        completion_pct = (completed / total_requests * 100.0) if total_requests > 0 else 0.0

        rewards.append(total_reward)
        completions.append(completion_pct)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_completion": float(np.mean(completions)),
        "std_completion": float(np.std(completions)),
    }


def evaluate_greedy(episodes: int, request_rate: float, max_steps: int) -> Dict[str, float]:
    rewards = []
    completions = []

    env = create_env(request_rate=request_rate, max_steps=max_steps)

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0

        while not done:
            action = greedy_action_from_obs(obs, env.unwrapped.num_vehicles)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += reward

        metrics = info["metrics"]
        total_requests = metrics.get("total_requests", 0)
        completed = metrics.get("completed_requests", 0)
        completion_pct = (completed / total_requests * 100.0) if total_requests > 0 else 0.0

        rewards.append(total_reward)
        completions.append(completion_pct)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_completion": float(np.mean(completions)),
        "std_completion": float(np.std(completions)),
    }


def promote_model_if_better(trained: Dict[str, float], greedy: Dict[str, float], source_zip: Path, dest_zip: Path) -> bool:
    better_reward = trained["mean_reward"] > greedy["mean_reward"]
    better_completion = trained["mean_completion"] >= greedy["mean_completion"]

    if better_reward and better_completion:
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_zip, dest_zip)
        return True

    return False


def train_and_evaluate(
    timesteps: int,
    episodes: int,
    request_rate: float,
    max_steps: int,
    init_model: str | None,
) -> Tuple[Path, Dict[str, object]]:
    models_dir = PROJECT_ROOT / "outputs" / "models"
    logs_dir = PROJECT_ROOT / "outputs" / "logs" / "DQN_improved"
    reports_dir = PROJECT_ROOT / "outputs" / "training_reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_env = create_env(request_rate=request_rate, max_steps=max_steps)
    eval_env = create_env(request_rate=request_rate, max_steps=max_steps)

    init_model_path = Path(init_model) if init_model else None
    if init_model_path and init_model_path.exists():
        model = DQN.load(str(init_model_path).replace(".zip", ""), env=train_env)
        model.verbose = 1
    else:
        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=1e-4,
            buffer_size=200_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.02,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs={"net_arch": [256, 256, 128]},
            verbose=1,
            tensorboard_log=str(logs_dir),
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(logs_dir),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=str(models_dir),
        name_prefix="dqn_routemate_improved",
    )

    model.learn(total_timesteps=timesteps, callback=[eval_callback, checkpoint_callback])

    model_base_path = models_dir / "dqn_routemate_improved"
    model.save(str(model_base_path))

    trained_metrics = evaluate_policy(model, episodes=episodes, request_rate=request_rate, max_steps=max_steps)
    greedy_metrics = evaluate_greedy(episodes=episodes, request_rate=request_rate, max_steps=max_steps)

    promoted = promote_model_if_better(
        trained=trained_metrics,
        greedy=greedy_metrics,
        source_zip=model_base_path.with_suffix(".zip"),
        dest_zip=models_dir / "dqn_routemate_final.zip",
    )

    reward_delta = trained_metrics["mean_reward"] - greedy_metrics["mean_reward"]
    completion_delta = trained_metrics["mean_completion"] - greedy_metrics["mean_completion"]

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "timesteps": timesteps,
        "episodes": episodes,
        "request_rate": request_rate,
        "max_steps": max_steps,
        "init_model": str(init_model_path) if init_model_path and init_model_path.exists() else None,
        "trained": trained_metrics,
        "greedy": greedy_metrics,
        "delta": {
            "reward": reward_delta,
            "completion": completion_delta,
        },
        "promoted_to_final": promoted,
        "model_path": str(model_base_path.with_suffix(".zip")),
    }

    report_path = reports_dir / f"dqn_training_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Keep this file in sync for quick inspection in existing workflows.
    final_eval_path = PROJECT_ROOT / "outputs" / "final_evaluation_results.json"
    with open(final_eval_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return report_path, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train stronger DQN and benchmark vs greedy")
    parser.add_argument("--timesteps", type=int, default=250_000, help="Total training timesteps")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per policy")
    parser.add_argument("--request-rate", type=float, default=1.5, help="Requests generated per step")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument(
        "--init-model",
        type=str,
        default="outputs/models/dqn_routemate_final.zip",
        help="Optional existing DQN checkpoint to continue training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("ROUTEMATE DQN IMPROVED TRAINING")
    print("=" * 80)
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Eval episodes: {args.episodes}")
    print(f"Request rate: {args.request_rate}")
    print(f"Max steps: {args.max_steps}")
    print("-" * 80)

    report_path, summary = train_and_evaluate(
        timesteps=args.timesteps,
        episodes=args.episodes,
        request_rate=args.request_rate,
        max_steps=args.max_steps,
        init_model=args.init_model,
    )

    print("\nTraining complete.")
    print(f"DQN reward:   {summary['trained']['mean_reward']:+.3f}")
    print(f"Greedy reward:{summary['greedy']['mean_reward']:+.3f}")
    print(f"Delta reward: {summary['delta']['reward']:+.3f}")
    print(f"DQN comp:     {summary['trained']['mean_completion']:.2f}%")
    print(f"Greedy comp:  {summary['greedy']['mean_completion']:.2f}%")
    print(f"Delta comp:   {summary['delta']['completion']:+.2f}%")
    print(f"Promoted:     {summary['promoted_to_final']}")
    print(f"Report:       {report_path}")


if __name__ == "__main__":
    main()
