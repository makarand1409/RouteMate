"""
evaluate_final_model.py - Evaluate Trained ML Model vs Baselines

Run this to get final performance comparison after training completes.
"""

import sys
sys.path.insert(0, '.')

from src.environment.gym_environment import RideSharingEnvImproved
from stable_baselines3 import PPO
import numpy as np

def evaluate_policy(env, policy_name, model=None, n_episodes=10):
    """Evaluate a policy over multiple episodes"""
    episode_rewards = []
    episode_completions = []
    episode_requests = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action based on policy
            if policy_name == "ml" and model is not None:
                action, _ = model.predict(obs, deterministic=True)
            elif policy_name == "greedy":
                # Greedy: select nearest vehicle
                if env.current_request is not None:
                    distances = []
                    for vehicle in env.vehicles:
                        dist = env.city.manhattan_distance(
                            vehicle.current_location,
                            env.current_request.pickup
                        )
                        distances.append(dist)
                    action = np.argmin(distances)
                else:
                    action = 0
            else:  # random
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_completions.append(info['metrics']['completed_requests'])
        episode_requests.append(info['metrics']['total_requests'])
        
        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Completed={info['metrics']['completed_requests']}/{info['metrics']['total_requests']}, "
              f"Rate={info['metrics']['completed_requests']/info['metrics']['total_requests']*100:.1f}%")
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_completion = np.mean(episode_completions)
    avg_requests = np.mean(episode_requests)
    completion_rate = (avg_completion / avg_requests * 100) if avg_requests > 0 else 0
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_completion': avg_completion,
        'avg_requests': avg_requests,
        'completion_rate': completion_rate
    }

def main():
    print("=" * 70)
    print("FINAL MODEL EVALUATION - ALL POLICIES")
    print("=" * 70)
    
    # Create environment
    env = RideSharingEnvImproved(max_steps=200)
    
    # Load trained model
    print("\nLoading trained ML model...")
    try:
        model = PPO.load("outputs/models/ppo_routemate_improved_1M")
        print("✓ Model loaded successfully!")
    except:
        print("✗ Model not found. Make sure training completed.")
        return
    
    # Evaluate all policies
    results = {}
    
    print("\n" + "=" * 70)
    print("POLICY 1: RANDOM (Baseline)")
    print("=" * 70)
    results['random'] = evaluate_policy(env, 'random', n_episodes=10)
    
    print("\n" + "=" * 70)
    print("POLICY 2: GREEDY (Nearest Vehicle)")
    print("=" * 70)
    results['greedy'] = evaluate_policy(env, 'greedy', n_episodes=10)
    
    print("\n" + "=" * 70)
    print("POLICY 3: ML AGENT (Trained PPO - 1M Steps)")
    print("=" * 70)
    results['ml'] = evaluate_policy(env, 'ml', model=model, n_episodes=10)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Policy':<20} {'Avg Reward':<15} {'Completion Rate':<20} {'Status':<10}")
    print("-" * 70)
    
    for policy_name, stats in results.items():
        status = ""
        if policy_name == 'ml':
            if stats['avg_reward'] > results['greedy']['avg_reward']:
                status = "🏆 WINNER!"
            else:
                status = "⚠️ Needs improvement"
        
        print(f"{policy_name.upper():<20} "
              f"{stats['avg_reward']:>7.2f} ± {stats['std_reward']:<5.2f} "
              f"{stats['completion_rate']:>17.1f}%  "
              f"{status}")
    
    # Calculate improvement
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    ml_reward = results['ml']['avg_reward']
    greedy_reward = results['greedy']['avg_reward']
    improvement = ((ml_reward - greedy_reward) / greedy_reward) * 100
    
    print(f"\nML vs Greedy:")
    print(f"  ML Reward:     {ml_reward:.2f}")
    print(f"  Greedy Reward: {greedy_reward:.2f}")
    print(f"  Improvement:   {improvement:+.1f}%")
    
    if improvement > 0:
        print(f"\n✅ SUCCESS! ML agent outperforms greedy baseline by {improvement:.1f}%")
    else:
        print(f"\n⚠️  ML agent underperforms greedy by {abs(improvement):.1f}%")
        print("   Consider: more training steps, different hyperparameters, or reward tuning")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    
    # Save results
    import json
    with open('outputs/final_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Results saved to: outputs/final_evaluation_results.json")

if __name__ == "__main__":
    main()
