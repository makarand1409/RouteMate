"""
train_ppo_agent.py - Train PPO Agent for RouteMATE

This script trains a Proximal Policy Optimization (PPO) agent
to learn optimal vehicle-request matching.

PPO is a state-of-the-art RL algorithm that:
- Learns gradually and stably
- Works well for discrete action spaces
- Widely used in industry (OpenAI, DeepMind)

Training process:
1. Agent interacts with environment
2. Collects experiences (states, actions, rewards)
3. Updates policy to maximize rewards
4. Repeat for many iterations
5. Save trained model
"""

import sys
import os

# Add parent directory to path to find src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from src.environment.gym_environment import RideSharingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
from datetime import datetime


def create_environment(max_steps=200):
    """
    Create and wrap the environment for training.
    
    Args:
        max_steps: Maximum steps per episode
        
    Returns:
        Wrapped environment ready for training
    """
    env = RideSharingEnv(
        city_size=10,
        num_vehicles=5,
        vehicle_capacity=4,
        request_rate=1.5,
        max_steps=max_steps
    )
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    return env


def train_ppo_agent(
    total_timesteps=200000,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    save_path="outputs/models",
    log_path="outputs/logs"
):
    """
    Train a PPO agent.
    
    Args:
        total_timesteps: Total training steps (100k is good start)
        learning_rate: How fast agent learns (0.0003 is standard)
        n_steps: Steps before policy update
        batch_size: Mini-batch size for training
        n_epochs: Epochs per policy update
        save_path: Where to save trained model
        log_path: Where to save training logs
        
    Returns:
        Trained PPO model
    """
    print("=" * 70)
    print("TRAINING PPO AGENT FOR ROUTEMATE")
    print("=" * 70)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Create training environment
    print("\n1. Creating environment...")
    env = create_environment(max_steps=200)
    print(f"   ✓ Environment created")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space.n} vehicles")
    
    # Create evaluation environment (separate from training)
    print("\n2. Creating evaluation environment...")
    eval_env = create_environment(max_steps=200)
    print(f"   ✓ Evaluation environment ready")
    
    # Configure training parameters
    print(f"\n3. Training configuration:")
    print(f"   - Total timesteps: {total_timesteps:,}")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Steps per update: {n_steps}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Epochs per update: {n_epochs}")
    
    # Create PPO model
    print(f"\n4. Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",  # Multi-Layer Perceptron (neural network)
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,  # Discount factor (how much to value future rewards)
        gae_lambda=0.95,  # Generalized Advantage Estimation
        clip_range=0.2,  # PPO clipping parameter
        verbose=1,  # Print training progress
        tensorboard_log=log_path
    )
    print(f"   ✓ PPO model created")
    print(f"   - Policy network: MLP (Multi-Layer Perceptron)")
    print(f"   - Total parameters: ~{model.policy.features_extractor.features_dim}")
    
    # Set up callbacks
    print(f"\n5. Setting up callbacks...")
    
    # Evaluation callback - evaluates agent every N steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=5000,  # Evaluate every 5000 steps
        n_eval_episodes=5,  # Use 5 episodes for evaluation
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Checkpoint callback - saves model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=save_path,
        name_prefix="ppo_routemate"
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    print(f"   ✓ Callbacks configured")
    print(f"   - Evaluation every 5,000 steps")
    print(f"   - Checkpoints every 10,000 steps")
    
    # Train the agent!
    print(f"\n6. Starting training...")
    print(f"   This will take 30-60 minutes depending on your CPU.")
    print(f"   Progress will be shown below.")
    print(f"   " + "-" * 66)
    
    timestamp_start = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks
    )
    
    timestamp_end = datetime.now()
    training_duration = (timestamp_end - timestamp_start).total_seconds()
    
    print(f"   " + "-" * 66)
    print(f"   ✓ Training complete!")
    print(f"   - Duration: {training_duration/60:.1f} minutes")
    
    # Save final model
    final_model_path = os.path.join(save_path, "ppo_routemate_final")
    model.save(final_model_path)
    print(f"\n7. Saving model...")
    print(f"   ✓ Model saved to: {final_model_path}.zip")
    
    return model, final_model_path


def evaluate_trained_agent(model_path, num_episodes=10):
    """
    Evaluate the trained agent.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "=" * 70)
    print("EVALUATING TRAINED AGENT")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    print("✓ Model loaded")
    
    # Create environment
    env = create_environment(max_steps=200)
    
    # Run evaluation episodes
    print(f"\nRunning {num_episodes} evaluation episodes...")
    print("-" * 70)
    
    all_rewards = []
    all_completion_rates = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()  # Unpack the tuple from Gymnasium API
        done = False
        total_reward = 0
        
        while not done:
            # Use trained policy (deterministic)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # Episode ends if either is True
            total_reward += reward
        
        # Collect metrics
        metrics = info['metrics']
        completion_rate = 0
        if metrics['total_requests'] > 0:
            completion_rate = metrics['completed_requests'] / metrics['total_requests']
        
        all_rewards.append(total_reward)
        all_completion_rates.append(completion_rate * 100)
        
        print(f"Episode {episode+1:2d}: "
              f"Reward={total_reward:+7.2f}, "
              f"Completed={metrics['completed_requests']:3d}/{metrics['total_requests']:3d}, "
              f"Rate={completion_rate*100:5.1f}%")
    
    # Print summary
    print("-" * 70)
    print("\nEVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Reward Statistics:")
    print(f"  Average reward: {np.mean(all_rewards):+.2f}")
    print(f"  Std deviation: {np.std(all_rewards):.2f}")
    print(f"  Best episode: {np.max(all_rewards):+.2f}")
    print(f"  Worst episode: {np.min(all_rewards):+.2f}")
    
    print(f"\n📈 Completion Rate:")
    print(f"  Average: {np.mean(all_completion_rates):.1f}%")
    print(f"  Std deviation: {np.std(all_completion_rates):.1f}%")
    
    results = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_completion': np.mean(all_completion_rates),
        'std_completion': np.std(all_completion_rates)
    }
    
    return results


def compare_with_baselines(trained_results):
    """
    Compare trained agent with baseline policies.
    
    Args:
        trained_results: Results from trained agent evaluation
    """
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    
    # Approximate baseline results (you can update these with your actual results)
    random_reward = -20.0
    greedy_reward = +6.5
    
    random_completion = 15.0
    greedy_completion = 30.0
    
    trained_reward = trained_results['mean_reward']
    trained_completion = trained_results['mean_completion']
    
    print(f"\n📊 Average Reward Comparison:")
    print(f"  Random baseline:  {random_reward:+7.2f}")
    print(f"  Greedy baseline:  {greedy_reward:+7.2f}")
    print(f"  Trained RL agent: {trained_reward:+7.2f}")
    
    improvement_over_greedy = ((trained_reward - greedy_reward) / abs(greedy_reward) * 100)
    print(f"\n  Improvement over greedy: {improvement_over_greedy:+.1f}%")
    
    print(f"\n📈 Completion Rate Comparison:")
    print(f"  Random baseline:  {random_completion:.1f}%")
    print(f"  Greedy baseline:  {greedy_completion:.1f}%")
    print(f"  Trained RL agent: {trained_completion:.1f}%")
    
    improvement_comp = trained_completion - greedy_completion
    print(f"\n  Improvement over greedy: {improvement_comp:+.1f} percentage points")
    
    # Success check
    print(f"\n🎯 Training Success:")
    if trained_reward > greedy_reward:
        print(f"  ✅ SUCCESS! RL agent beats greedy baseline!")
        print(f"  ✅ Reward improvement: {improvement_over_greedy:+.1f}%")
    else:
        print(f"  ⚠️  Agent needs more training or hyperparameter tuning")
    
    if trained_completion > greedy_completion:
        print(f"  ✅ Completion rate improved by {improvement_comp:.1f}%")
    
    return improvement_over_greedy, improvement_comp


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ROUTEMATE - PPO TRAINING PIPELINE")
    print("=" * 70)
    
    # Step 1: Train agent
    print("\nSTEP 1: Training PPO agent...")
    model, model_path = train_ppo_agent(
        total_timesteps=100000,  # 100k steps - good starting point
        learning_rate=0.0003,
        save_path="outputs/models",
        log_path="outputs/logs"
    )
    
    # Step 2: Evaluate trained agent
    print("\n\nSTEP 2: Evaluating trained agent...")
    trained_results = evaluate_trained_agent(
        model_path=model_path + ".zip",
        num_episodes=10
    )
    
    # Step 3: Compare with baselines
    print("\n\nSTEP 3: Comparing with baselines...")
    improvement_reward, improvement_completion = compare_with_baselines(trained_results)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n📁 Saved Files:")
    print(f"  - Trained model: {model_path}.zip")
    print(f"  - Training logs: outputs/logs/")
    print(f"  - Checkpoints: outputs/models/")
    
    print(f"\n📊 Final Results:")
    print(f"  - RL agent reward: {trained_results['mean_reward']:+.2f}")
    print(f"  - RL completion rate: {trained_results['mean_completion']:.1f}%")
    print(f"  - Improvement over greedy: {improvement_reward:+.1f}%")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. View training curves: tensorboard --logdir outputs/logs")
    print(f"  2. Run comparison plots: python tests/compare_all_policies.py")
    print(f"  3. Use trained model in your app!")
    
    print("\n" + "=" * 70)