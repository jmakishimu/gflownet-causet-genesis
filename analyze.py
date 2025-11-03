#
# gflownet-causet-genesis/analyze.py
#
import pandas as pd
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from causet_reward import CausalSetRewardProxy

sns.set_theme(style="whitegrid")

def load_results(run_dir):
    """Loads log and ensemble from a run directory."""
    log_path = os.path.join(run_dir, "training_log.csv")
    ensemble_path = os.path.join(run_dir, "final_ensemble.pkl")

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Ensemble file not found: {ensemble_path}")

    logs = pd.read_csv(log_path)

    with open(ensemble_path, 'rb') as f:
        ensemble = pickle.load(f)

    return logs, ensemble

def load_baseline(baseline_file):
    """Loads the random baseline ensemble."""
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"Baseline file not found: {baseline_file}")

    with open(baseline_file, 'rb') as f:
        baseline_ensemble = pickle.load(f)
    return baseline_ensemble

def plot_1_loss(logs, save_path):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=logs, x='step', y='loss')
    plt.title("Plot 1: GFlowNet Loss Curve")
    plt.xlabel("Training Step")
    plt.ylabel("GFlowNet Loss (Trajectory Balance)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "1_loss_curve.png"), dpi=150)
    plt.close()
    print("✓ Generated Plot 1: Loss Curve")

def plot_2_reward(logs, save_path):
    """Plot reward acquisition"""
    plt.figure(figsize=(10, 6))

    # Plot both energy and reward
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Energy
    sns.lineplot(data=logs, x='step', y='avg_energy', ax=ax1)
    ax1.set_title("Average Energy")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Energy (S_BD or (MMD-4)²)")

    # Reward
    sns.lineplot(data=logs, x='step', y='avg_reward', ax=ax2)
    ax2.set_title("Average Reward")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Reward (exp(-β·Energy))")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "2_reward_acquisition.png"), dpi=150)
    plt.close()
    print("✓ Generated Plot 2: Reward Acquisition")

def plot_3_training_metrics(logs, save_path):
    """Plot additional training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Valid fraction
    if 'valid_fraction' in logs.columns:
        sns.lineplot(data=logs, x='step', y='valid_fraction', ax=axes[0,0])
        axes[0,0].set_title("Valid Trajectory Fraction")
        axes[0,0].set_ylabel("Fraction Complete")

    # Diversity
    if 'diversity' in logs.columns:
        sns.lineplot(data=logs, x='step', y='diversity', ax=axes[0,1])
        axes[0,1].set_title("Sample Diversity")
        axes[0,1].set_ylabel("Unique Causets / Total")

    # Epsilon
    if 'epsilon' in logs.columns:
        sns.lineplot(data=logs, x='step', y='epsilon', ax=axes[1,0])
        axes[1,0].set_title("Exploration Rate (ε)")
        axes[1,0].set_ylabel("Epsilon")

    # Temperature
    if 'temperature' in logs.columns:
        sns.lineplot(data=logs, x='step', y='temperature', ax=axes[1,1])
        axes[1,1].set_title("Sampling Temperature")
        axes[1,1].set_ylabel("Temperature")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "3_training_metrics.png"), dpi=150)
    plt.close()
    print("✓ Generated Plot 3: Training Metrics")

def analyze_ensemble(ensemble, metric_type):
    """Calculate metrics for ensemble"""
    if metric_type == 'mmd':
        proxy = CausalSetRewardProxy(reward_type='mmd')
    else:
        proxy = CausalSetRewardProxy(reward_type='bd')

    mmds = []
    bds = []

    if not ensemble:
        print("Warning: Ensemble is empty.")
        return {'mmds': [], 'bds': [], 'valid': []}

    # Filter valid graphs
    target_n = ensemble[0].number_of_nodes()
    valid_ensemble = [g for g in ensemble if g.number_of_nodes() == target_n]

    if len(valid_ensemble) < len(ensemble):
        print(f"Warning: Filtered {len(ensemble) - len(valid_ensemble)} malformed graphs.")

    desc = f"Analyzing {len(valid_ensemble)} graphs (N={target_n})"
    for g in tqdm(valid_ensemble, desc=desc):
        try:
            if metric_type == 'mmd':
                mmd = proxy.calculate_avg_mmd(g)
                mmds.append(mmd)
            else:
                bd = proxy.get_bd_energy(g)
                bds.append(bd)
        except Exception as e:
            print(f"Error analyzing graph: {e}")
            continue

    return {
        'mmds': mmds,
        'bds': bds,
        'valid': valid_ensemble
    }

def plot_4_bd_comparison(ensemble, baseline_ensemble, save_path):
    """Compare BD action between GFN and random"""
    print("\nAnalyzing BD action for GFN ensemble...")
    gfn_stats = analyze_ensemble(ensemble, 'bd')
    gfn_bds = gfn_stats['bds']

    print("Analyzing BD action for baseline ensemble...")
    baseline_stats = analyze_ensemble(baseline_ensemble, 'bd')
    baseline_bds = baseline_stats['bds']

    if not gfn_bds or not baseline_bds:
        print("Skipping Plot 4: Insufficient data")
        return

    # Create comparison plot
    plt.figure(figsize=(12, 6))

    df1 = pd.DataFrame({'S_BD': gfn_bds, 'Source': 'GFN (Trained)'})
    df2 = pd.DataFrame({'S_BD': baseline_bds, 'Source': 'Random (Baseline)'})
    plot_df = pd.concat([df1, df2])

    sns.histplot(data=plot_df, x='S_BD', hue='Source', kde=True,
                 element="step", stat="density", common_norm=False)

    plt.title("Plot 4: BD Action Distribution")
    plt.xlabel("Benincasa-Dowker Action (S_BD)")
    plt.ylabel("Density")

    # Add mean lines
    gfn_mean = np.mean(gfn_bds)
    baseline_mean = np.mean(baseline_bds)
    plt.axvline(x=gfn_mean, color='blue', linestyle='--',
                label=f'GFN Mean: {gfn_mean:.1f}')
    plt.axvline(x=baseline_mean, color='orange', linestyle='--',
                label=f'Baseline Mean: {baseline_mean:.1f}')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "4_bd_action_comparison.png"), dpi=150)
    plt.close()

    print(f"✓ Generated Plot 4: BD Action Comparison")
    print(f"  GFN Mean S_BD: {gfn_mean:.2f} ± {np.std(gfn_bds):.2f}")
    print(f"  Baseline Mean S_BD: {baseline_mean:.2f} ± {np.std(baseline_bds):.2f}")
    print(f"  Improvement: {baseline_mean - gfn_mean:.2f} ({((baseline_mean - gfn_mean)/baseline_mean*100):.1f}%)")

def plot_5_mmd_distribution(ensemble, baseline_ensemble, save_path):
    """The 'Money Shot' - MMD distribution showing emergent dimension"""
    print("\nAnalyzing MMD for GFN ensemble...")
    gfn_stats = analyze_ensemble(ensemble, 'mmd')
    gfn_mmds = gfn_stats['mmds']

    print("Analyzing MMD for baseline ensemble...")
    baseline_stats = analyze_ensemble(baseline_ensemble, 'mmd')
    baseline_mmds = baseline_stats['mmds']

    if not gfn_mmds or not baseline_mmds:
        print("Skipping Plot 5: Insufficient data")
        return

    # Create the money shot
    plt.figure(figsize=(14, 7))

    df1 = pd.DataFrame({'MMD': gfn_mmds, 'Source': 'GFN (Trained on BD)'})
    df2 = pd.DataFrame({'MMD': baseline_mmds, 'Source': 'Random (Baseline)'})
    plot_df = pd.concat([df1, df2])

    sns.histplot(data=plot_df, x='MMD', hue='Source', kde=True,
                 element="step", stat="density", common_norm=False, bins=30)

    plt.title("Plot 5: Emergent Dimension (The 'Money Shot')", fontsize=16, fontweight='bold')
    plt.xlabel("Myrheim-Meyer Dimension (MMD)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Add d=4 reference line
    plt.axvline(x=4.0, color='red', linestyle=':', linewidth=2, label="Target d=4")

    # Add mean lines
    gfn_mean = np.mean(gfn_mmds)
    baseline_mean = np.mean(baseline_mmds)
    plt.axvline(x=gfn_mean, color='blue', linestyle='--', alpha=0.7,
                label=f'GFN Mean: {gfn_mean:.2f}')
    plt.axvline(x=baseline_mean, color='orange', linestyle='--', alpha=0.7,
                label=f'Baseline Mean: {baseline_mean:.2f}')

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "5_money_shot_mmd.png"), dpi=150)
    plt.close()

    print(f"✓ Generated Plot 5: The Money Shot")
    print(f"  GFN Mean MMD: {gfn_mean:.2f} ± {np.std(gfn_mmds):.2f}")
    print(f"  Baseline Mean MMD: {baseline_mean:.2f} ± {np.std(baseline_mmds):.2f}")

    # Calculate fraction near d=4
    gfn_near_4 = sum(3.5 <= d <= 4.5 for d in gfn_mmds) / len(gfn_mmds)
    baseline_near_4 = sum(3.5 <= d <= 4.5 for d in baseline_mmds) / len(baseline_mmds)
    print(f"  GFN near d=4 [3.5, 4.5]: {gfn_near_4*100:.1f}%")
    print(f"  Baseline near d=4: {baseline_near_4*100:.1f}%")

def main_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to experiment run directory"
    )
    parser.add_argument(
        "--baseline_file", type=str, required=True,
        help="Path to baseline ensemble pickle"
    )
    parser.add_argument(
        "--reward_type", type=str, required=True, choices=["bd", "mmd"],
        help="The reward type used in training"
    )
    args = parser.parse_args()

    # Create output directory
    plot_save_dir = os.path.join(args.run_dir, "plots")
    os.makedirs(plot_save_dir, exist_ok=True)

    print("="*60)
    print("CAUSAL SET GFLOWNET ANALYSIS")
    print("="*60)
    print(f"Run directory: {args.run_dir}")
    print(f"Baseline file: {args.baseline_file}")
    print(f"Reward type: {args.reward_type}")
    print()

    # Load data
    print("Loading results...")
    logs, ensemble = load_results(args.run_dir)
    print(f"✓ Loaded {len(ensemble)} graphs from ensemble")

    print("Loading baseline...")
    baseline_ensemble = load_baseline(args.baseline_file)
    print(f"✓ Loaded {len(baseline_ensemble)} baseline graphs")
    print()

    # Generate plots
    print("Generating plots...")
    print()

    plot_1_loss(logs, plot_save_dir)
    plot_2_reward(logs, plot_save_dir)
    plot_3_training_metrics(logs, plot_save_dir)

    # Reward-specific plots
    if args.reward_type == 'bd':
        plot_4_bd_comparison(ensemble, baseline_ensemble, plot_save_dir)
        # Also compute MMD for emergent dimension
        plot_5_mmd_distribution(ensemble, baseline_ensemble, plot_save_dir)
    else:
        # For MMD training, show how well it locked to target dimension
        print("\nNote: For MMD training, Plot 4 shows dimension locking")
        # Could add MMD-specific plot here

    print()
    print("="*60)
    print(f"All plots saved to: {plot_save_dir}")
    print("="*60)

if __name__ == "__main__":
    main_analysis()
