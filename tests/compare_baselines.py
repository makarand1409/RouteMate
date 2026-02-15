"""
compare_baselines.py - Compare Random vs Greedy Baseline

This script runs both baseline policies and compares their performance.
Use this to establish your baselines before training the RL agent.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.gym_environment import RideSharingEnv
import numpy as np
import matplotlib.pyplot as plt


def random_policy(observation, num_vehicles=5):
    """Random policy: pick any vehicle."""
    return np.random.randint(0, num_vehicles)


def greedy_policy(observation, num_vehicles=5):
    """Greedy policy: pick nearest vehicle."""
    distances = []
    for i in range(num_vehicles):
        distance_idx = 4 + (i * 5) + 4
        distances.append(observation[distance_idx])
    return int(np.argmin(distances))


def test_policy(policy_func, policy_name, num_episodes=10, max_steps=200):
    """
    Test a policy over multiple episodes.
    
    Args:
        policy_func: Function that takes observation and returns action
        policy_name: Name of the policy (for printing)
        num_episodes: Number of episodes to run
        max_steps: Steps per episode
        
    Returns:
        Dictionary with results
    """
    print(f"\nTesting {policy_name}...")
    
    env = RideSharingEnv(
        city_size=10,
        num_vehicles=5,
        vehicle_capacity=4,
        request_rate=1.5,
        max_steps=max_steps
    )
    
    all_rewards = []
    all_completion_rates = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()  # Unpack Gymnasium tuple
        total_reward = 0
        done = False
        
        while not done:
            action = policy_func(obs, num_vehicles=env.num_vehicles)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        metrics = info['metrics']
        completion_rate = 0
        if metrics['total_requests'] > 0:
            completion_rate = metrics['completed_requests'] / metrics['total_requests']
        
        all_rewards.append(total_reward)
        all_completion_rates.append(completion_rate * 100)
    
    return {
        'name': policy_name,
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_completion': np.mean(all_completion_rates),
        'std_completion': np.std(all_completion_rates),
        'all_rewards': all_rewards,
        'all_completions': all_completion_rates
    }


def compare_policies(num_episodes=10, max_steps=200):
    """
    Compare random and greedy policies.
    """
    print("=" * 70)
    print("BASELINE COMPARISON: Random vs Greedy")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Episodes per policy: {num_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Vehicles: 5")
    print(f"  Request rate: 1.5/step")
    
    # Test both policies
    random_results = test_policy(random_policy, "Random Policy", num_episodes, max_steps)
    greedy_results = test_policy(greedy_policy, "Greedy (Nearest) Policy", num_episodes, max_steps)
    
    # Print comparison
    print(f"\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Average Reward per Episode:")
    print(f"  Random:  {random_results['mean_reward']:+7.2f} ± {random_results['std_reward']:.2f}")
    print(f"  Greedy:  {greedy_results['mean_reward']:+7.2f} ± {greedy_results['std_reward']:.2f}")
    improvement = ((greedy_results['mean_reward'] - random_results['mean_reward']) / 
                   abs(random_results['mean_reward']) * 100)
    print(f"  Improvement: {improvement:+.1f}%")
    
    print(f"\n📈 Average Completion Rate:")
    print(f"  Random:  {random_results['mean_completion']:.1f}% ± {random_results['std_completion']:.1f}%")
    print(f"  Greedy:  {greedy_results['mean_completion']:.1f}% ± {greedy_results['std_completion']:.1f}%")
    improvement_comp = greedy_results['mean_completion'] - random_results['mean_completion']
    print(f"  Improvement: {improvement_comp:+.1f} percentage points")
    
    # Create visualization
    create_comparison_plot(random_results, greedy_results)
    
    return random_results, greedy_results


def create_comparison_plot(random_results, greedy_results):
    """Create side-by-side comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Rewards
    ax1 = axes[0]
    policies = ['Random', 'Greedy']
    means = [random_results['mean_reward'], greedy_results['mean_reward']]
    stds = [random_results['std_reward'], greedy_results['std_reward']]
    
    bars = ax1.bar(policies, means, yerr=stds, capsize=10, 
                   color=['#e74c3c', '#3498db'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Average Reward per Episode', fontsize=12)
    ax1.set_title('Reward Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=11)
    
    # Plot 2: Completion Rates
    ax2 = axes[1]
    means_comp = [random_results['mean_completion'], greedy_results['mean_completion']]
    stds_comp = [random_results['std_completion'], greedy_results['std_completion']]
    
    bars = ax2.bar(policies, means_comp, yerr=stds_comp, capsize=10,
                   color=['#e74c3c', '#3498db'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Completion Rate (%)', fontsize=12)
    ax2.set_title('Completion Rate Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Baseline Policy Comparison - RouteMATE', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {filepath}")
    
    plt.show()


if __name__ == "__main__":
    random_results, greedy_results = compare_policies(num_episodes=10, max_steps=200)
    
    print(f"\n" + "=" * 70)
    print("✓ Baseline comparison complete!")
    print("=" * 70)
    
    print(f"\n📝 Summary:")
    print(f"  - Greedy policy is better than random (as expected)")
    print(f"  - Greedy reward: {greedy_results['mean_reward']:+.2f}")
    print(f"  - Greedy completion: {greedy_results['mean_completion']:.1f}%")
    
    print(f"\n🎯 RL Agent Goal:")
    print(f"  Beat greedy baseline by 20-30%")
    print(f"  Target reward: >{greedy_results['mean_reward'] * 1.25:+.2f}")
    print(f"  Target completion: >{greedy_results['mean_completion'] * 1.15:.1f}%")
    
    print(f"\n🚀 Next: Train RL agent in Phase 3!")