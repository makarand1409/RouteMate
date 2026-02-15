"""
metrics_and_viz.py - Metrics Collection and Visualization for RouteMATE

This module provides tools to:
1. Collect and save simulation metrics to CSV
2. Visualize performance with matplotlib charts
3. Compare different matching policies
4. Generate report-ready plots

Use this to analyze your baseline and compare with ML later!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

# Handle both package import and standalone testing
try:
    from .simulation_engine import SimulationEngine
    from .matching_policy import NearestVehiclePolicy, RandomPolicy
except ImportError:
    from simulation_engine import SimulationEngine
    from matching_policy import NearestVehiclePolicy, RandomPolicy


class MetricsCollector:
    """
    Collects and saves simulation metrics.
    
    This class records detailed metrics during simulation runs
    and saves them to CSV files for later analysis.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize metrics collector.
        
        Args:
            output_dir: Directory to save CSV files (created if doesn't exist)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.runs_data = []  # Store data from multiple runs
    
    def record_run(self, 
                   policy_name: str, 
                   results: Dict, 
                   parameters: Dict = None):
        """
        Record results from a single simulation run.
        
        Args:
            policy_name: Name of the matching policy used
            results: Results dictionary from SimulationEngine.run()
            parameters: Optional dict of simulation parameters
        """
        run_data = {
            'timestamp': datetime.now().isoformat(),
            'policy': policy_name,
            **results
        }
        
        if parameters:
            run_data['parameters'] = json.dumps(parameters)
        
        self.runs_data.append(run_data)
    
    def save_to_csv(self, filename: str = None):
        """
        Save all recorded runs to CSV.
        
        Args:
            filename: CSV filename (auto-generated if None)
            
        Returns:
            Path to saved CSV file
        """
        if not self.runs_data:
            print("No data to save!")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        df = pd.DataFrame(self.runs_data)
        df.to_csv(filepath, index=False)
        
        print(f"✓ Saved {len(self.runs_data)} runs to {filepath}")
        return filepath
    
    def get_summary_stats(self) -> pd.DataFrame:
        """
        Get summary statistics grouped by policy.
        
        Returns:
            DataFrame with mean, std, min, max for each policy
        """
        if not self.runs_data:
            return None
        
        df = pd.DataFrame(self.runs_data)
        
        # Group by policy and calculate statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df.groupby('policy')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
        
        return summary
    
    def clear(self):
        """Clear all recorded data."""
        self.runs_data = []


class SimulationVisualizer:
    """
    Creates visualizations of simulation results.
    
    Generates matplotlib plots for:
    - Performance comparison across policies
    - Time series of key metrics
    - Distribution plots
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plot images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set nice default style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    def plot_policy_comparison(self, 
                               results_dict: Dict[str, Dict],
                               metrics: List[str] = None,
                               save_path: str = None):
        """
        Create bar chart comparing different policies.
        
        Args:
            results_dict: Dict mapping policy name to results dict
                         e.g., {'Nearest': {...}, 'Random': {...}}
            metrics: List of metric names to plot
                    If None, plots key metrics
            save_path: Path to save figure (optional)
        
        Example:
            >>> results = {
            ...     'Nearest': nearest_results,
            ...     'Random': random_results
            ... }
            >>> viz.plot_policy_comparison(results)
        """
        if metrics is None:
            metrics = [
                'avg_wait_time',
                'avg_trip_time', 
                'completion_rate',
                'avg_distance_per_vehicle'
            ]
        
        # Prepare data
        policies = list(results_dict.keys())
        num_metrics = len(metrics)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = [results_dict[policy].get(metric, 0) for policy in policies]
            
            # Create bar chart
            bars = ax.bar(policies, values, alpha=0.7, edgecolor='black')
            
            # Color bars
            colors = plt.cm.Set3(np.linspace(0, 1, len(policies)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Formatting
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Policy Comparison - RouteMATE Simulator', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {filepath}")
        
        plt.show()
        return fig
    
    def plot_time_series(self,
                        engine: SimulationEngine,
                        save_path: str = None):
        """
        Plot metrics over time from simulation history.
        
        Args:
            engine: SimulationEngine with completed run (has history)
            save_path: Path to save figure (optional)
        """
        history = engine.history
        
        if not history or 'time' not in history:
            print("No history data available. Run simulation first!")
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Pending requests over time
        ax1 = axes[0]
        ax1.plot(history['time'], history['pending_requests'], 
                linewidth=2, color='#e74c3c', label='Pending Requests')
        ax1.fill_between(history['time'], history['pending_requests'], 
                        alpha=0.3, color='#e74c3c')
        ax1.set_xlabel('Time Step', fontsize=11)
        ax1.set_ylabel('Pending Requests', fontsize=11)
        ax1.set_title('Pending Requests Over Time', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Plot 2: Active vehicles over time
        ax2 = axes[1]
        ax2.plot(history['time'], history['active_vehicles'],
                linewidth=2, color='#3498db', label='Active Vehicles')
        ax2.fill_between(history['time'], history['active_vehicles'],
                        alpha=0.3, color='#3498db')
        ax2.set_xlabel('Time Step', fontsize=11)
        ax2.set_ylabel('Active Vehicles', fontsize=11)
        ax2.set_title('Active Vehicles Over Time', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()
        
        plt.suptitle(f'RouteMATE Simulation - {engine.matching_policy}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {filepath}")
        
        plt.show()
        return fig
    
    def plot_summary_dashboard(self,
                              results: Dict,
                              policy_name: str = "Policy",
                              save_path: str = None):
        """
        Create a comprehensive dashboard with multiple metrics.
        
        Args:
            results: Results dictionary from simulation
            policy_name: Name of the policy for title
            save_path: Path to save figure (optional)
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'RouteMATE Dashboard - {policy_name}',
                    fontsize=16, fontweight='bold')
        
        # 1. Completion Rate (big number)
        ax1 = fig.add_subplot(gs[0, 0])
        completion_rate = results.get('completion_rate', 0) * 100
        ax1.text(0.5, 0.5, f"{completion_rate:.1f}%",
                ha='center', va='center', fontsize=40, fontweight='bold',
                color='#27ae60' if completion_rate > 80 else '#e67e22')
        ax1.text(0.5, 0.2, "Completion Rate",
                ha='center', va='center', fontsize=12)
        ax1.axis('off')
        
        # 2. Avg Wait Time
        ax2 = fig.add_subplot(gs[0, 1])
        wait_time = results.get('avg_wait_time', 0)
        ax2.text(0.5, 0.5, f"{wait_time:.2f}",
                ha='center', va='center', fontsize=40, fontweight='bold',
                color='#3498db')
        ax2.text(0.5, 0.2, "Avg Wait Time (steps)",
                ha='center', va='center', fontsize=12)
        ax2.axis('off')
        
        # 3. Total Requests
        ax3 = fig.add_subplot(gs[0, 2])
        total_req = results.get('total_requests', 0)
        ax3.text(0.5, 0.5, f"{total_req}",
                ha='center', va='center', fontsize=40, fontweight='bold',
                color='#9b59b6')
        ax3.text(0.5, 0.2, "Total Requests",
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
        
        # 4. Request breakdown (pie chart)
        ax4 = fig.add_subplot(gs[1, :2])
        completed = results.get('completed_requests', 0)
        pending = results.get('pending_requests', 0)
        in_progress = total_req - completed - pending
        
        sizes = [completed, in_progress, pending]
        labels = [f'Completed\n({completed})', 
                 f'In Progress\n({in_progress})',
                 f'Pending\n({pending})']
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax4.set_title('Request Status Breakdown', fontsize=12, fontweight='bold')
        
        # 5. Distance metrics (bar)
        ax5 = fig.add_subplot(gs[1, 2])
        total_dist = results.get('total_distance_traveled', 0)
        avg_dist = results.get('avg_distance_per_vehicle', 0)
        
        ax5.bar(['Total', 'Avg/Vehicle'], [total_dist, avg_dist], 
               color=['#3498db', '#e67e22'], alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Distance (blocks)', fontsize=10)
        ax5.set_title('Distance Traveled', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Key metrics table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('tight')
        ax6.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['Total Requests', f"{total_req}"],
            ['Completed Requests', f"{completed}"],
            ['Completion Rate', f"{completion_rate:.1f}%"],
            ['Avg Wait Time', f"{wait_time:.2f} steps"],
            ['Avg Trip Time', f"{results.get('avg_trip_time', 0):.2f} steps"],
            ['Total Distance', f"{total_dist} blocks"],
            ['Customers Served', f"{results.get('total_customers_served', 0)}"],
        ]
        
        table = ax6.table(cellText=table_data, cellLoc='left',
                         colWidths=[0.3, 0.2],
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        if save_path:
            filepath = self.output_dir / save_path
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved dashboard to {filepath}")
        
        plt.show()
        return fig


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing RouteMATE Metrics & Visualization")
    print("=" * 60)
    
    # Create metrics collector
    collector = MetricsCollector(output_dir="outputs")
    
    # Create visualizer
    viz = SimulationVisualizer(output_dir="outputs")
    
    print("\n1. Running simulations with different policies...")
    print("-" * 60)
    
    # Run with Nearest Vehicle Policy
    print("\nRunning with NearestVehiclePolicy...")
    engine_nearest = SimulationEngine(
        num_vehicles=5,
        request_rate=1.5
    )
    results_nearest = engine_nearest.run(max_steps=100, verbose=False)
    collector.record_run("Nearest", results_nearest)
    print("✓ Nearest policy complete")
    
    # Run with Random Policy
    print("\nRunning with RandomPolicy...")
    from matching_policy import RandomPolicy
    engine_random = SimulationEngine(
        num_vehicles=5,
        request_rate=1.5,
        matching_policy=RandomPolicy(engine_nearest.city)
    )
    results_random = engine_random.run(max_steps=100, verbose=False)
    collector.record_run("Random", results_random)
    print("✓ Random policy complete")
    
    # Save to CSV
    print("\n2. Saving results to CSV...")
    print("-" * 60)
    csv_path = collector.save_to_csv("policy_comparison.csv")
    
    # Show summary statistics
    print("\n3. Summary Statistics:")
    print("-" * 60)
    summary = collector.get_summary_stats()
    print(summary)
    
    # Create visualizations
    print("\n4. Creating visualizations...")
    print("-" * 60)
    
    # Policy comparison
    results_dict = {
        'Nearest Vehicle': results_nearest,
        'Random': results_random
    }
    
    print("\nGenerating policy comparison plot...")
    viz.plot_policy_comparison(results_dict, save_path="policy_comparison.png")
    
    # Time series
    print("\nGenerating time series plot...")
    viz.plot_time_series(engine_nearest, save_path="time_series_nearest.png")
    
    # Dashboard
    print("\nGenerating dashboard...")
    viz.plot_summary_dashboard(results_nearest, 
                               policy_name="Nearest Vehicle Policy",
                               save_path="dashboard_nearest.png")
    
    print("\n" + "=" * 60)
    print("✓ All visualizations complete!")
    print("=" * 60)
    print(f"\nCheck the 'outputs' folder for:")
    print("  - policy_comparison.csv (data)")
    print("  - policy_comparison.png (chart)")
    print("  - time_series_nearest.png (chart)")
    print("  - dashboard_nearest.png (dashboard)")
