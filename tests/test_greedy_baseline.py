"""
test_greedy_baseline.py - Test Greedy (Nearest Vehicle) Agent

This script tests a simple greedy policy: always pick the nearest vehicle.
This is similar to your Phase 1 baseline but adapted for the Gym environment.

Greedy agent: Picks vehicle closest to pickup, ignoring capacity and routes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.gym_environment import RideSharingEnv
import numpy as np


def greedy_nearest_policy(observation, num_vehicles=5):
    """
    Greedy policy: Select vehicle with minimum distance to pickup.
    
    Observation structure:
    [pickup_x, pickup_y, dropoff_x, dropoff_y,  (indices 0-3)
     v1_x, v1_y, v1_occ, v1_queue, v1_distance,  (indices 4-8)
     v2_x, v2_y, v2_occ, v2_queue, v2_distance,  (indices 9-13)
     ...]
    
    Distance to pickup is at indices: 8, 13, 18, 23, 28
    (Every 5th element starting from index 8)
    
    Args:
        observation: Current observation from environment
        num_vehicles: Number of vehicles in fleet
        
    Returns:
        action: Index of vehicle with minimum distance
    """
    distances = []
    for i in range(num_vehicles):
        # Distance is at index: 4 + i*5 + 4
        distance_idx = 4 + (i * 5) + 4
        distances.append(observation[distance_idx])
    
    # Return index of vehicle with minimum distance
    return int(np.argmin(distances))


def test_greedy_agent(num_episodes=10, max_steps=200):
    """
    Test greedy nearest-vehicle agent over multiple episodes.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print("=" * 70)
    print("TESTING GREEDY BASELINE (Nearest Vehicle Policy)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Strategy: Always pick nearest vehicle to pickup")
    
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
        obs, info = env.reset()  # Unpack Gymnasium API tuple
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            # GREEDY ACTION - pick nearest vehicle
            action = greedy_nearest_policy(obs, num_vehicles=env.num_vehicles)
            
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
    print("GREEDY BASELINE SUMMARY")
    print("=" * 70)
    
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
    
    print(f"\n💡 Strengths:")
    print(f"  ✅ Better than random (considers distance)")
    print(f"  ✅ Simple and fast")
    print(f"  ✅ Easy to explain")
    
    print(f"\n⚠️  Limitations:")
    print(f"  ❌ Doesn't consider vehicle capacity")
    print(f"  ❌ Doesn't consider current passengers")
    print(f"  ❌ Doesn't plan ahead")
    print(f"  ❌ Myopic (only looks at immediate request)")
    
    print(f"\n🎯 This is what RL will try to beat!")
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_completion_rate': np.mean(all_completion_rates)
    }


if __name__ == "__main__":
    results = test_greedy_agent(num_episodes=10, max_steps=200)
    
    print(f"\n" + "=" * 70)
    print("✓ Greedy baseline testing complete!")
    print("=" * 70)
    print(f"\nNext: Train RL agent to beat this baseline!")