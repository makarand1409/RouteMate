"""
auto_train_search.py - PPO hyperparameter search for RouteMATE.

Purpose:
- Train multiple PPO configurations automatically.
- Evaluate each model on multiple traffic scenarios.
- Rank configs by completion rate and reward.
- Report whether 90% completion was reached.

Usage:
  python src/agents/auto_train_search.py
  python src/agents/auto_train_search.py --timesteps 300000 --episodes 30
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Add project root for local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from src.environment.gym_environment import RideSharingEnv


DEFAULT_CONFIGS = [
    {
        "name": "ppo_balanced",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    {
        "name": "ppo_stable_small_lr",
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "clip_range": 0.2,
    },
    {
        "name": "ppo_fast_updates",
        "learning_rate": 4e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 6,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.25,
    },
]

SCENARIO_PROFILES = {
    "easy": [
        {"name": "easy_low", "request_rate": 0.4},
        {"name": "easy_mid", "request_rate": 0.6},
        {"name": "easy_high", "request_rate": 0.8},
    ],
    "standard": [
        {"name": "low", "request_rate": 1.0},
        {"name": "mid", "request_rate": 1.5},
        {"name": "high", "request_rate": 2.0},
    ],
    "hard": [
        {"name": "hard_low", "request_rate": 1.5},
        {"name": "hard_mid", "request_rate": 2.0},
        {"name": "hard_high", "request_rate": 2.5},
    ],
}


def create_env(request_rate=1.5, max_steps=200, num_vehicles=5, vehicle_capacity=4, seed=None):
    env = RideSharingEnv(
        city_size=10,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        request_rate=request_rate,
        max_steps=max_steps,
    )
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def train_model(
    config,
    total_timesteps,
    model_dir,
    log_dir,
    num_vehicles,
    vehicle_capacity,
    train_request_rate,
):
    train_env = create_env(
        request_rate=train_request_rate,
        max_steps=200,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        verbose=0,
        tensorboard_log=log_dir,
    )

    start = time.time()
    model.learn(total_timesteps=total_timesteps)
    duration_sec = time.time() - start

    model_path = os.path.join(model_dir, config["name"])
    model.save(model_path)

    return model, model_path + ".zip", duration_sec


def evaluate_model(model, eval_episodes, num_vehicles, vehicle_capacity, scenarios):

    scenario_results = []
    all_completion = []
    all_rewards = []

    for scenario in scenarios:
        env = create_env(
            request_rate=scenario["request_rate"],
            max_steps=200,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
        )

        rewards = []
        completions = []

        for episode in range(eval_episodes):
            obs, _info = env.reset(seed=episode)
            done = False
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                total_reward += reward

            metrics = info["metrics"]
            total_requests = metrics["total_requests"]
            completed = metrics["completed_requests"]
            completion_pct = (completed / total_requests * 100.0) if total_requests > 0 else 0.0

            rewards.append(total_reward)
            completions.append(completion_pct)

        scenario_summary = {
            "scenario": scenario["name"],
            "request_rate": scenario["request_rate"],
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_completion": float(np.mean(completions)),
            "std_completion": float(np.std(completions)),
        }

        scenario_results.append(scenario_summary)
        all_rewards.extend(rewards)
        all_completion.extend(completions)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_completion": float(np.mean(all_completion)),
        "std_completion": float(np.std(all_completion)),
        "scenarios": scenario_results,
    }


def run_search(
    total_timesteps,
    eval_episodes,
    target_completion,
    output_root,
    num_vehicles,
    vehicle_capacity,
    train_request_rate,
    scenario_profile,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"ppo_search_{timestamp}")
    model_dir = os.path.join(run_dir, "models")
    log_dir = os.path.join(run_dir, "logs")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    all_results = []
    scenarios = SCENARIO_PROFILES[scenario_profile]

    print("=" * 80)
    print("ROUTEMATE PPO AUTO TRAIN + SEARCH")
    print("=" * 80)
    print(f"Target completion: {target_completion:.1f}%")
    print(f"Timesteps per config: {total_timesteps:,}")
    print(f"Eval episodes per scenario: {eval_episodes}")
    print(f"Fleet size: {num_vehicles} vehicles (capacity {vehicle_capacity})")
    print(f"Training request rate: {train_request_rate}")
    print(f"Evaluation profile: {scenario_profile}")
    print("Evaluation rates: " + ", ".join(str(s["request_rate"]) for s in scenarios))
    print("-" * 80)

    for idx, config in enumerate(DEFAULT_CONFIGS, start=1):
        print(f"\n[{idx}/{len(DEFAULT_CONFIGS)}] Training {config['name']} ...")

        model, model_file, duration_sec = train_model(
            config=config,
            total_timesteps=total_timesteps,
            model_dir=model_dir,
            log_dir=log_dir,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            train_request_rate=train_request_rate,
        )

        eval_metrics = evaluate_model(
            model,
            eval_episodes=eval_episodes,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            scenarios=scenarios,
        )

        result = {
            "config": config,
            "model_file": model_file,
            "train_minutes": round(duration_sec / 60.0, 2),
            "metrics": eval_metrics,
            "hits_target": eval_metrics["mean_completion"] >= target_completion,
        }
        all_results.append(result)

        print(
            f"  completion={eval_metrics['mean_completion']:.2f}% | "
            f"reward={eval_metrics['mean_reward']:+.2f} | "
            f"time={result['train_minutes']:.1f} min"
        )

    all_results.sort(key=lambda r: (r["metrics"]["mean_completion"], r["metrics"]["mean_reward"]), reverse=True)

    summary = {
        "timestamp": timestamp,
        "target_completion": target_completion,
        "total_timesteps_per_config": total_timesteps,
        "eval_episodes_per_scenario": eval_episodes,
        "num_vehicles": num_vehicles,
        "vehicle_capacity": vehicle_capacity,
        "train_request_rate": train_request_rate,
        "scenario_profile": scenario_profile,
        "scenario_rates": [s["request_rate"] for s in scenarios],
        "best": all_results[0],
        "all_results": all_results,
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SEARCH COMPLETE")
    print("=" * 80)
    print(f"Best config: {summary['best']['config']['name']}")
    print(f"Best completion: {summary['best']['metrics']['mean_completion']:.2f}%")
    print(f"Best reward: {summary['best']['metrics']['mean_reward']:+.2f}")
    print(f"Target reached: {'YES' if summary['best']['hits_target'] else 'NO'}")
    print(f"Summary file: {summary_path}")

    if not summary["best"]["hits_target"]:
        print("\nTip: increase timesteps to 1_000_000+ and/or increase fleet size to improve completion.")

    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-train and evaluate PPO configs for RouteMATE")
    parser.add_argument("--timesteps", type=int, default=300000, help="Training timesteps per config")
    parser.add_argument("--episodes", type=int, default=20, help="Evaluation episodes per scenario")
    parser.add_argument("--target", type=float, default=90.0, help="Target completion percentage")
    parser.add_argument("--output", type=str, default="outputs/training_sweeps", help="Output directory")
    parser.add_argument("--vehicles", type=int, default=5, help="Number of vehicles in fleet")
    parser.add_argument("--capacity", type=int, default=4, help="Vehicle capacity")
    parser.add_argument("--train-rate", type=float, default=1.0, help="Training request rate")
    parser.add_argument(
        "--profile",
        type=str,
        default="standard",
        choices=list(SCENARIO_PROFILES.keys()),
        help="Evaluation scenario profile",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search(
        total_timesteps=args.timesteps,
        eval_episodes=args.episodes,
        target_completion=args.target,
        output_root=args.output,
        num_vehicles=args.vehicles,
        vehicle_capacity=args.capacity,
        train_request_rate=args.train_rate,
        scenario_profile=args.profile,
    )
