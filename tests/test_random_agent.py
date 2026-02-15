"""
test_random_agent.py - Test Random Agent Performance

This script tests how well a completely random agent performs.
This establishes the WORST-CASE baseline.

Random agent: Picks any vehicle randomly, ignoring all information.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.gym_environment import RideSharingEnv


def test_random_agent(num_episodes=10, max_steps=200):
    """
    Test random agent over multiple episodes.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print("=" * 70)
    print("TESTING RANDOM AGENT (Worst-Case Baseline)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Strategy: Completely random vehicle selection")
    
    # Create environment
    env = RideSharingEnv(
        city_size=10,
        num_vehicles=5,
        vehicle_capacity=4,
        request_rate=1.5,
        max_steps=max_steps
    )
    
    # Track results across episodes
    all_rewards = []
    all_completion_rates = []
    all_assigned = []
    all_failed = []
    
    print(f"\n" + "-" * 70)
    print("Running episodes...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        obs, info = env.reset()  # Unpack Gymnasium tuple
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # RANDOM ACTION - just pick any vehicle
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
        
        # Episode complete - collect metrics
        metrics = info['metrics']
        completion_rate = 0
        if metrics['total_requests'] > 0:
            completion_rate = metrics['completed_requests'] / metrics['total_requests']
        
        all_rewards.append(total_reward)
        all_completion_rates.append(completion_rate * 100)
        all_assigned.append(metrics['assigned_requests'])
        all_failed.append(metrics['failed_assignments'])
        
        print(f"Episode {episode+1:2d}: "
              f"Reward={total_reward:+7.2f}, "
              f"Completed={metrics['completed_requests']:3d}/{metrics['total_requests']:3d}, "
              f"Rate={completion_rate*100:5.1f}%, "
              f"Failed={metrics['failed_assignments']:2d}")
    
    # Print summary statistics
    print(f"\n" + "=" * 70)
    print("RANDOM AGENT SUMMARY")
    print("=" * 70)
    
    import numpy as np
    
    print(f"\n📊 Reward Statistics:")
    print(f"  Average reward per episode: {np.mean(all_rewards):+.2f}")
    print(f"  Std deviation: {np.std(all_rewards):.2f}")
    print(f"  Best episode: {np.max(all_rewards):+.2f}")
    print(f"  Worst episode: {np.min(all_rewards):+.2f}")
    
    print(f"\n📈 Completion Rate:")
    print(f"  Average: {np.mean(all_completion_rates):.1f}%")
    print(f"  Std deviation: {np.std(all_completion_rates):.1f}%")
    
    print(f"\n🎯 Assignment Statistics:")
    print(f"  Average assigned: {np.mean(all_assigned):.1f}")
    print(f"  Average failed: {np.mean(all_failed):.1f}")
    print(f"  Failure rate: {np.mean(all_failed) / np.mean(all_assigned) * 100:.1f}%")
    
    print(f"\n💡 Key Insight:")
    print(f"  Random selection is inefficient because it doesn't consider:")
    print(f"  - Vehicle distance to pickup")
    print(f"  - Vehicle capacity/occupancy")
    print(f"  - Current vehicle routes")
    print(f"\n  This is why we need smarter policies!")
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_completion_rate': np.mean(all_completion_rates)
    }


if __name__ == "__main__":
    results = test_random_agent(num_episodes=10, max_steps=200)
    
    print(f"\n" + "=" * 70)
    print("✓ Random agent testing complete!")
    print("=" * 70)
    print(f"\nNext: Test greedy baseline to see improvement!")