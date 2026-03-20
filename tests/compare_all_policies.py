"""
compare_all_policies.py - Compare Random, Greedy, and Trained RL Agent

This script evaluates all three policies side-by-side and creates
publication-ready comparison plots for your report.

Policies compared:
1. Random (worst-case baseline)
2. Greedy/Nearest (heuristic baseline)
3. Trained PPO Agent (your ML contribution!)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.gym_environment import RideSharingEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def random_policy(observation, num_vehicles=5):
    """Random vehicle selection."""
    return np.random.randint(0, num_vehicles)


def greedy_policy(observation, num_vehicles=5):
    """Nearest vehicle selection."""
    distances = [observation[4 + i*5 + 4] for i in range(num_vehicles)]
    return int(np.argmin(distances))


def test_policy(policy, policy_name, num_episodes=10, max_steps=200, model=None):
    """
    Test a policy over multiple episodes.
    
    Args:
        policy: Policy function or "model" for trained agent
        policy_name: Name for display
        num_episodes: Number of test episodes
        max_steps: Steps per episode
        model: Trained model (if using RL agent)
        
    Returns:
        Results dictionary
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
    all_completions = []
    all_wait_times = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if model is not None:
                # Use trained model
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Use policy function
                action = policy(obs, num_vehicles=env.num_vehicles)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        # Collect metrics
        metrics = info['metrics']
        completion_rate = 0
        if metrics['total_requests'] > 0:
            completion_rate = metrics['completed_requests'] / metrics['total_requests']
        
        all_rewards.append(total_reward)
        all_completions.append(completion_rate * 100)
    
    return {
        'name': policy_name,
        'rewards': all_rewards,
        'completions': all_completions,
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_completion': np.mean(all_completions),
        'std_completion': np.std(all_completions)
    }


def create_comparison_plots(results_list, save_dir="outputs"):
    """
    Create comprehensive comparison visualizations.
    
    Args:
        results_list: List of results dictionaries
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    policy_names = [r['name'] for r in results_list]
    colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green
    
    # Plot 1: Reward Comparison (Bar chart)
    ax1 = fig.add_subplot(gs[0, :])
    means = [r['mean_reward'] for r in results_list]
    stds = [r['std_reward'] for r in results_list]
    
    bars = ax1.bar(policy_names, means, yerr=stds, capsize=10,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Average Reward per Episode', fontsize=13, fontweight='bold')
    ax1.set_title('Policy Comparison: Episode Rewards', fontsize=15, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=12, fontweight='bold')
    
    # Plot 2: Completion Rate (Bar chart)
    ax2 = fig.add_subplot(gs[1, 0])
    means_comp = [r['mean_completion'] for r in results_list]
    stds_comp = [r['std_completion'] for r in results_list]
    
    bars = ax2.bar(policy_names, means_comp, yerr=stds_comp, capsize=10,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Request Completion Rate', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, max(means_comp) * 1.3])
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Reward Distribution (Box plot)
    ax3 = fig.add_subplot(gs[1, 1])
    reward_data = [r['rewards'] for r in results_list]
    bp = ax3.boxplot(reward_data, labels=policy_names, patch_artist=True,
                     notch=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Reward Distribution', fontsize=12, fontweight='bold')
    ax3.set_title('Reward Variability Across Episodes', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    # Plot 4: Improvement Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Calculate improvements
    greedy_reward = results_list[1]['mean_reward']  # Greedy is index 1
    ml_reward = results_list[2]['mean_reward']      # ML is index 2
    
    greedy_comp = results_list[1]['mean_completion']
    ml_comp = results_list[2]['mean_completion']
    
    improvement_reward = ((ml_reward - greedy_reward) / abs(greedy_reward) * 100)
    improvement_comp = ml_comp - greedy_comp
    
    table_data = [
        ['Policy', 'Avg Reward', 'Avg Completion', 'vs Greedy'],
        ['Random', f"{results_list[0]['mean_reward']:+.2f}", 
         f"{results_list[0]['mean_completion']:.1f}%", '-'],
        ['Greedy (Baseline)', f"{greedy_reward:+.2f}", 
         f"{greedy_comp:.1f}%", 'Baseline'],
        ['ML Agent (PPO)', f"{ml_reward:+.2f}", 
         f"{ml_comp:.1f}%", 
         f"+{improvement_reward:.1f}% reward\n+{improvement_comp:.1f}% completion"]
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center',
                     loc='center', colWidths=[0.25, 0.2, 0.25, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style ML row (highlight)
    for i in range(4):
        cell = table[(3, i)]
        cell.set_facecolor('#d5f4e6')
        cell.set_text_props(weight='bold')
    
    plt.suptitle('RouteMATE: ML Agent vs Baselines - Complete Comparison',
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save plot
    filepath = os.path.join(save_dir, 'ml_complete_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {filepath}")
    
    plt.show()


def save_results_csv(results_list, save_dir="outputs"):
    """Save results to CSV for further analysis."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create summary dataframe
    summary_data = []
    for r in results_list:
        summary_data.append({
            'Policy': r['name'],
            'Mean_Reward': r['mean_reward'],
            'Std_Reward': r['std_reward'],
            'Mean_Completion': r['mean_completion'],
            'Std_Completion': r['std_completion']
        })
    
    df = pd.DataFrame(summary_data)
    filepath = os.path.join(save_dir, 'policy_comparison_summary.csv')
    df.to_csv(filepath, index=False)
    print(f"✓ Saved summary CSV: {filepath}")


if __name__ == "__main__":
    print("=" * 70)
    print("COMPLETE POLICY COMPARISON")
    print("=" * 70)
    
    print("\nThis will compare:")
    print("  1. Random Policy (worst-case baseline)")
    print("  2. Greedy Policy (heuristic baseline)")
    print("  3. Trained ML Agent (your contribution!)")
    
    # Load trained model - try improved model first, fall back to final
    model_path = "outputs/models/ppo_routemate_improved_1M.zip"
    if not os.path.exists(model_path):
        model_path = "outputs/models/best_model.zip"
    if not os.path.exists(model_path):
        model_path = "outputs/models/ppo_routemate_final.zip"
    
    if not os.path.exists(model_path):
        print(f"\n❌ Error: No trained model found")
        print(f"   Checked for: ppo_routemate_improved_1M.zip, best_model.zip, ppo_routemate_final.zip")
        print(f"   Please train the model first: python quick_train_improved.py")
        sys.exit(1)
    
    print(f"\n✓ Loading trained model from: {model_path}")
    trained_model = PPO.load(model_path)
    print("✓ Model loaded successfully")
    
    # Test all policies
    print(f"\n" + "=" * 70)
    print("Running evaluations (this will take a few minutes)...")
    print("=" * 70)
    
    results = []
    
    # Test Random
    results.append(test_policy(
        random_policy, "Random", num_episodes=10, max_steps=200
    ))
    
    # Test Greedy
    results.append(test_policy(
        greedy_policy, "Greedy", num_episodes=10, max_steps=200
    ))
    
    # Test ML Agent
    results.append(test_policy(
        None, "ML Agent", num_episodes=10, max_steps=200, model=trained_model
    ))
    
    # Print summary
    print(f"\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Reward: {r['mean_reward']:+.2f} ± {r['std_reward']:.2f}")
        print(f"  Completion: {r['mean_completion']:.1f}% ± {r['std_completion']:.1f}%")
    
    # Calculate improvements
    greedy_reward = results[1]['mean_reward']
    ml_reward = results[2]['mean_reward']
    improvement = ((ml_reward - greedy_reward) / abs(greedy_reward) * 100)
    
    print(f"\n🎯 ML Agent Performance:")
    print(f"  Improvement over greedy: {improvement:+.1f}%")
    
    if improvement > 20:
        print(f"  ✅ EXCELLENT! Significant improvement!")
    elif improvement > 10:
        print(f"  ✅ GOOD! Clear improvement over baseline")
    elif improvement > 0:
        print(f"  ✅ POSITIVE! ML agent is better")
    else:
        print(f"  ⚠️  Consider more training or hyperparameter tuning")
    
    # Create visualizations
    print(f"\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)
    
    create_comparison_plots(results)
    save_results_csv(results)
    
    print(f"\n" + "=" * 70)
    print("✓ COMPARISON COMPLETE!")
    print("=" * 70)
    
    print(f"\n📁 Output files:")
    print(f"  - ml_complete_comparison.png (charts)")
    print(f"  - policy_comparison_summary.csv (data)")
    
    print(f"\n🎓 For your report/presentation:")
    print(f"  - Show the comparison chart")
    print(f"  - Highlight {improvement:+.1f}% improvement")
    print(f"  - Explain why ML is better (considers capacity, plans ahead)")
    
    print(f"\n🚀 Your ML project is complete!")
